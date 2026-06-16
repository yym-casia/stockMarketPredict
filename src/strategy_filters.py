# -*- coding: utf-8 -*-
"""统一策略筛选器 - 提升胜率与盈利能力"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


DEFAULT_STRATEGY = {
    'min_change': 2.0,
    'max_change': 5.0,
    'min_tech_score': 65,
    'min_volume_ratio': 1.2,
    'min_rsi': 35,
    'max_rsi': 55,
    'min_conditions': 3,
    'require_bullish_candle': True,
    'require_macd': True,
    'require_ma_trend': True,
    'require_macd_golden': False,
    'close_strength_min': 0.0,
    'max_prev_drop': 5.0,
    'min_composite_score': 0,
    'market_min_score': 45,
    'market_full_score': 55,
    'weak_market_max_positions': 2,
    'market_defensive_enabled': True,
    'filter_cold_sector_in_weak': True,
    'exit_cold_sector_in_weak': True,
    'trailing_start': 4.0,
    'trailing_pct': 2.5,
    'stop_loss': 3.0,
    'stop_loss_on_close': True,
    'trend_stop_loss_partial_ratio': 0,
    'trend_dip_refill_pct': None,
    'trend_half_hold_exit_pct': None,
    'late_trend_filter_enabled': False,
    'max_consecutive_up_days': 3,
    'max_limit_ups_before_entry': 1,
    'limit_up_lookback_days': 10,
    'late_trend_block_mode': 'both',
    'shakeout_rebuy_enabled': True,
    'shakeout_rebuy_days': 10,
    'shakeout_rebuy_mode': 'consecutive_yang',
    'shakeout_rebuy_min_up_days': 3,
    'shakeout_rebuy_require_recover': True,
    'shakeout_rebuy_recover_pct': 0.0,
    'shakeout_rebuy_skip_weak_regime': True,
    'shakeout_rebuy_stop_loss': 4.5,
    'shakeout_rebuy_stop_grace_days': 5,
    'shakeout_rebuy_delayed_confirm': False,
    'shakeout_rebuy_size_ratio': 0.5,
    'shakeout_rebuy_max_daily': 1,
    'shakeout_rebuy_min_volume_ratio': 1.1,
    'shakeout_reserve_pct': 20.0,
    'shakeout_bypass_max_positions': True,
    'min_ml_score': 0.0,
    'ml_weight': 0.25,
}

# 小亏大赚默认止盈档位: 少分批、高目标
TP_SMALL_LOSS_BIG_WIN = [
    {'pct': 6.0, 'ratio': 0.20},
    {'pct': 12.0, 'ratio': 0.30},
    {'pct': 20.0, 'ratio': 0.50},
]

# v9: 推迟首档止盈、减小早期减仓，让盈利单跑更远
TP_V9_RUN_FURTHER = [
    {'pct': 10.0, 'ratio': 0.15},
    {'pct': 18.0, 'ratio': 0.25},
    {'pct': 28.0, 'ratio': 0.60},
]

# 大行情: 推迟止盈、抬高目标，配合放宽涨幅筛选
TP_BIG_RUNNER = [
    {'pct': 18.0, 'ratio': 0.10},
    {'pct': 30.0, 'ratio': 0.20},
    {'pct': 45.0, 'ratio': 0.70},
]


def tp_conflicts_with_add(tp_pct: float, add_trigger_pct: float,
                          add_ratio: float) -> bool:
    """强者恒强：加仓档位与阶梯止盈同档时，优先加仓、不减仓。"""
    if add_ratio <= 0 or add_trigger_pct <= 0:
        return False
    return abs(tp_pct % add_trigger_pct) < 0.01


def load_strategy_config(config: dict) -> dict:
    cfg = DEFAULT_STRATEGY.copy()
    cfg.update(config.get('strategy', {}))
    sel = config.get('stock_selection', {})
    if sel.get('sector_standard'):
        cfg['sector_standard'] = sel['sector_standard']
    return cfg


def check_technical_signals_strict(df: pd.DataFrame, cfg: dict) -> Dict:
    """严格版技术信号（比原版更少误入场）"""
    if df is None or len(df) < 20:
        return {'valid': False, 'reason': '数据不足', 'score': 0}

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    rsi = latest.get('rsi', np.nan)
    if pd.isna(rsi):
        return {'valid': False, 'reason': 'RSI无效', 'score': 0}

    rsi_ok = cfg['min_rsi'] <= rsi <= cfg['max_rsi']
    rsi_score = max(0, 30 - abs(rsi - 45) / 2)

    macd_hist = latest.get('macd_hist', np.nan)
    prev_hist = prev.get('macd_hist', np.nan)
    if pd.isna(macd_hist):
        return {'valid': False, 'reason': 'MACD无效', 'score': 0}

    macd_golden = macd_hist > 0 and macd_hist > prev_hist
    macd_turning = macd_hist > 0 and prev_hist <= 0
    macd_ok = macd_golden or macd_turning
    macd_score = 30 if macd_golden else (25 if macd_turning else 0)

    volume_ratio = latest.get('volume_ratio', 1.0)
    if pd.isna(volume_ratio):
        volume_ratio = 1.0
    volume_ok = volume_ratio >= cfg['min_volume_ratio']
    volume_score = min(25, volume_ratio * 10) if volume_ok else 0

    ma5, ma10, ma20 = latest.get('ma5'), latest.get('ma10'), latest.get('ma20')
    ma_ok = False
    ma_score = 0
    if not any(pd.isna(x) for x in [ma5, ma10, ma20]):
        ma_ok = ma5 >= ma10 >= ma20
        ma_score = 30 if ma_ok else (15 if ma5 >= ma10 else 0)

    bullish_candle = True
    if cfg.get('require_bullish_candle'):
        bullish_candle = latest['close'] > latest['open']

    total_score = rsi_score + macd_score + volume_score + ma_score
    conditions = [rsi_ok, macd_ok, volume_ok, ma_ok]
    conditions_met = sum(conditions)

    close_strength = 1.0
    if latest['high'] > latest['low']:
        close_strength = (latest['close'] - latest['low']) / (latest['high'] - latest['low'])
    strength_ok = close_strength >= cfg.get('close_strength_min', 0)

    macd_golden_ok = (macd_golden or macd_turning) if cfg.get('require_macd_golden') else True

    valid = (
        conditions_met >= cfg['min_conditions']
        and total_score >= cfg['min_tech_score']
        and bullish_candle
        and strength_ok
        and macd_golden_ok
        and (macd_ok if cfg.get('require_macd') else True)
        and (ma_ok or ma5 >= ma10 if cfg.get('require_ma_trend') else True)
    )

    return {
        'valid': valid,
        'score': min(100, total_score),
        'close_strength': round(close_strength, 2),
        'rsi': float(rsi),
        'rsi_ok': rsi_ok,
        'macd_ok': macd_ok,
        'macd_golden': macd_golden,
        'volume_ratio': float(volume_ratio),
        'volume_ok': volume_ok,
        'ma_ok': ma_ok,
        'bullish_candle': bullish_candle,
        'conditions_met': conditions_met,
        'reason': f"RSI:{rsi:.1f} 量比:{volume_ratio:.2f} 条件:{conditions_met}/4",
    }


def compute_market_score(all_data: Dict[str, pd.DataFrame], date: str,
                         cfg: dict = None, **kwargs) -> Tuple[float, bool, int]:
    """大盘情绪（广度 + 指数趋势 + 主线），见 market_regime。"""
    from src.market_regime import compute_market_score as _enhanced_score
    return _enhanced_score(all_data, date, cfg, **kwargs)


def trend_peak_ok(hist_df: pd.DataFrame, cfg: dict) -> bool:
    """趋势入场：左侧 N 日内最低到最高涨幅不超过阈值（无大山峰）。"""
    max_peak = cfg.get('trend_max_peak_pct', 0)
    if not max_peak or max_peak <= 0:
        return True
    lookback = int(cfg.get('trend_peak_lookback_days', 30))
    if hist_df is None or len(hist_df) < lookback:
        return False
    window = hist_df.tail(lookback)
    low_min = float(window['low'].min())
    high_max = float(window['high'].max())
    if low_min <= 0:
        return False
    swing_pct = (high_max / low_min - 1) * 100
    return swing_pct <= max_peak


def entry_chase_ok(hist_df: pd.DataFrame, cfg: dict) -> bool:
    """避免趋势末段追高：要求距近期高点有足够回调，可选限制前 N 日累计涨幅。"""
    min_dist = cfg.get('min_dist_from_high_pct', 0)
    max_prior = cfg.get('max_prior_gain_pct', 0)
    if (not min_dist or min_dist >= 0) and (not max_prior or max_prior <= 0):
        return True

    lookback = int(cfg.get('dist_from_high_lookback_days', 30))
    prior_days = int(cfg.get('max_prior_gain_days', 20))
    need_len = max(lookback, prior_days + 1) if max_prior and max_prior > 0 else lookback
    if hist_df is None or len(hist_df) < need_len:
        return False

    close = float(hist_df.iloc[-1]['close'])
    if close <= 0:
        return False

    if min_dist and min_dist < 0:
        high_max = float(hist_df.tail(lookback)['high'].max())
        if high_max <= 0:
            return False
        dist_pct = (close / high_max - 1) * 100
        if dist_pct > min_dist:
            return False

    if max_prior and max_prior > 0 and len(hist_df) > prior_days:
        prior_close = float(hist_df.iloc[-1 - prior_days]['close'])
        if prior_close > 0:
            prior_gain = (close / prior_close - 1) * 100
            if prior_gain > max_prior:
                return False

    return True


def entry_late_trend_ok(hist_df: pd.DataFrame, cfg: dict, code: str = '') -> bool:
    """排除趋势末期：买入前连涨过多且近期已有多个涨停。"""
    if not cfg.get('late_trend_filter_enabled', False):
        return True
    if hist_df is None or len(hist_df) < 3:
        return False

    max_consec = int(cfg.get('max_consecutive_up_days', 3))
    max_lu = int(cfg.get('max_limit_ups_before_entry', 1))
    lookback = int(cfg.get('limit_up_lookback_days', 10))
    mode = cfg.get('late_trend_block_mode', 'both')

    hist = hist_df.copy()
    if 'pct_change' not in hist.columns:
        hist['pct_change'] = hist['close'].pct_change() * 100

    consec = 0
    for j in range(len(hist) - 2, max(len(hist) - 22, -1), -1):
        ch = float(hist.iloc[j].get('pct_change', 0) or 0)
        if ch > 0:
            consec += 1
        else:
            break

    from src.limit_up_filters import limit_up_threshold
    th = limit_up_threshold(str(code).replace('sh', '').replace('sz', '')) - 0.5
    start = max(0, len(hist) - 1 - lookback)
    end = len(hist) - 1
    lu_count = sum(
        1 for j in range(start, end)
        if float(hist.iloc[j].get('pct_change', 0) or 0) >= th
    )

    over_consec = consec > max_consec
    over_lu = lu_count > max_lu
    if mode == 'any':
        return not (over_consec or over_lu)
    return not (over_consec and over_lu)


def apply_ml_score(result: Dict, ml_proba: float, cfg: dict) -> Optional[Dict]:
    """将 ML 概率融入综合评分"""
    min_ml = cfg.get('min_ml_score', 0)
    if min_ml > 0 and ml_proba < min_ml:
        return None
    w = cfg.get('ml_weight', 0.25)
    result['ml_score'] = round(ml_proba, 4)
    result['score'] = round(result['score'] * (1 - w) + ml_proba * 100 * w, 2)
    return result


def screen_stock_row(row, hist_df: pd.DataFrame, cfg: dict,
                     ml_proba: float = None) -> Optional[Dict]:
    """筛选单只股票（回测/实盘通用）"""
    change = row.get('pct_change', row.get('change', 0))
    if pd.isna(change) or not (cfg['min_change'] <= change <= cfg['max_change']):
        return None

    volume = row.get('volume', 0)
    if volume < 500000:
        return None

    if len(hist_df) >= 2:
        prev_change = hist_df.iloc[-2].get('pct_change', 0)
        if not pd.isna(prev_change) and prev_change < -cfg.get('max_prev_drop', 5.0):
            return None

    if cfg.get('require_trend_peak', True) and not trend_peak_ok(hist_df, cfg):
        return None

    if not entry_chase_ok(hist_df, cfg):
        return None

    code = str(row.get('code', '')).replace('sh', '').replace('sz', '')
    if not entry_late_trend_ok(hist_df, cfg, code):
        return None

    signals = check_technical_signals_strict(hist_df, cfg)
    if not signals['valid']:
        return None

    mid_change = (cfg['min_change'] + cfg['max_change']) / 2
    change_penalty = max(0, (change - mid_change) * 2)
    score = signals['score'] + change * 0.15 - change_penalty
    if signals.get('close_strength', 0) >= 0.7:
        score += 5

    if score < cfg.get('min_composite_score', 0):
        return None

    result = {
        'code': row.get('code', ''),
        'buy_price': float(row['close'] if 'close' in row.index else row.get('price', 0)),
        'change': float(change),
        'tech_score': signals['score'],
        'rsi': signals['rsi'],
        'volume_ratio': signals['volume_ratio'],
        'score': round(score, 2),
        'signals': signals,
    }
    if ml_proba is not None:
        return apply_ml_score(result, ml_proba, cfg)
    if cfg.get('min_ml_score', 0) > 0:
        return None
    return result


def rank_candidates(candidates: List[Dict], top_n: int = 5) -> List[Dict]:
    """按综合评分排序，ML分高且涨幅适中的优先"""
    candidates.sort(
        key=lambda x: (x['score'], x.get('ml_score', 0), -abs(x['change'] - 7.5)),
        reverse=True,
    )
    return candidates[:top_n]
