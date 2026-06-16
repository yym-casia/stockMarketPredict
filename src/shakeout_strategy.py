# -*- coding: utf-8 -*-
"""洗盘震仓策略：止损后接回、半仓止损、收盘止损"""

from typing import Dict, List, Optional
import pandas as pd

from src.trading_calendar import count_trading_days


DEFAULT_SHAKEOUT = {
    'stop_loss_on_close': True,
    'trend_stop_loss_partial_ratio': 0,
    'trend_dip_refill_pct': None,
    'trend_half_hold_exit_pct': None,
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
    'shakeout_rebuy_min_last_yang_pct': 1.0,
    'shakeout_reserve_pct': 20.0,
    'shakeout_bypass_max_positions': True,
}


def merge_shakeout_cfg(strat_cfg: dict) -> dict:
    cfg = dict(DEFAULT_SHAKEOUT)
    cfg.update({k: strat_cfg[k] for k in DEFAULT_SHAKEOUT if k in strat_cfg})
    return cfg


def should_watch_after_stop(reason: str, strategy_type: str, sell_ratio: float,
                            strat_cfg: dict) -> bool:
    cfg = merge_shakeout_cfg(strat_cfg)
    if not cfg['shakeout_rebuy_enabled']:
        return False
    if strategy_type != 'trend' or sell_ratio < 0.99:
        return False
    return '止损' in reason and '半仓' not in reason


def is_bullish_bar(row) -> bool:
    """阳线：收盘 > 开盘。"""
    close = float(row.get('close', 0))
    open_ = float(row.get('open', close))
    return close > open_


def count_consecutive_bullish_bars(hist: pd.DataFrame) -> int:
    """从最近交易日向前统计连续阳线天数。"""
    if hist is None or hist.empty:
        return 0
    count = 0
    for i in range(len(hist) - 1, -1, -1):
        if is_bullish_bar(hist.iloc[i]):
            count += 1
        else:
            break
    return count


def _volume_ratio_ok(hist: pd.DataFrame, cfg: dict) -> bool:
    min_vr = cfg.get('shakeout_rebuy_min_volume_ratio', 0)
    if min_vr <= 0 or len(hist) < 2:
        return True
    today_vol = float(hist.iloc[-1].get('volume', 0))
    prev_vol = float(hist.iloc[-2].get('volume', 0))
    if prev_vol <= 0:
        return True
    return today_vol / prev_vol >= min_vr


def _yang_entry_ok(hist: pd.DataFrame, cfg: dict) -> Optional[str]:
    """连阳入场/延迟确认，通过则返回触发描述。"""
    mode = cfg.get('shakeout_rebuy_mode', 'consecutive_yang')
    if mode != 'consecutive_yang':
        return None
    yang_count = count_consecutive_bullish_bars(hist)
    min_days = int(cfg.get('shakeout_rebuy_min_up_days', 3))
    if len(hist) < min_days:
        return None
    today = hist.iloc[-1]
    if cfg.get('shakeout_rebuy_delayed_confirm'):
        if len(hist) < 2:
            return None
        yesterday = hist.iloc[-2]
        min_last = float(cfg.get('shakeout_rebuy_min_last_yang_pct', 1.0))
        chg = float(today.get('pct_change', today.get('change', 0)) or 0)
        if yang_count >= min_days + 1 and is_bullish_bar(today):
            if float(today['close']) > float(yesterday['close']):
                return f'延迟确认|{yang_count}连阳破前高'
        if yang_count == min_days and chg >= min_last and is_bullish_bar(today):
            return f'{min_days}连阳强阳{chg:.1f}%'
        return None
    if yang_count < min_days:
        return None
    return f'连续{yang_count}连阳'


def eval_shakeout_rebuy(
    watch: Dict,
    row,
    hist: pd.DataFrame,
    strat_cfg: dict,
    regime: dict = None,
) -> Optional[Dict]:
    """判断是否满足止损后接回条件。"""
    cfg = merge_shakeout_cfg(strat_cfg)
    close = float(row['close'])
    sell_p = watch['sell_price']
    mode = cfg.get('shakeout_rebuy_mode', 'consecutive_yang')
    yang_count = count_consecutive_bullish_bars(hist)

    if cfg.get('shakeout_rebuy_skip_weak_regime') and regime:
        reg = regime.get('regime', '')
        if reg in ('weak', 'bearish') or not regime.get('can_enter', True):
            return None

    if mode == 'consecutive_yang':
        trigger_desc = _yang_entry_ok(hist, cfg)
        if not trigger_desc:
            return None
    else:
        threshold = sell_p * (1 + cfg['shakeout_rebuy_recover_pct'] / 100)
        if close < threshold:
            return None
        trigger_desc = f'收复止损价{sell_p:.2f}'

    if cfg.get('shakeout_rebuy_require_recover'):
        recover_pct = float(cfg.get('shakeout_rebuy_recover_pct', 0))
        threshold = sell_p * (1 + recover_pct / 100)
        if close < threshold:
            return None

    if not _volume_ratio_ok(hist, cfg):
        return None

    change = float(row.get('pct_change', row.get('change', 0)))
    score = 75 + min(change, 8) * 2 + yang_count * 3
    return {
        'code': watch['code'],
        'buy_price': close,
        'change': change,
        'score': round(score, 2),
        'strategy_type': 'trend',
        'shakeout_rebuy': True,
        'yang_days': yang_count,
        'signals': {
            'reason': f'止损后接回|{trigger_desc}|收盘{close:.2f}',
        },
    }


def shakeout_rebuy_candidates(
    watches: Dict[str, Dict],
    all_data: Dict[str, pd.DataFrame],
    date: str,
    held: set,
    strat_cfg: dict,
    regime: dict = None,
) -> List[Dict]:
    """实盘/EOD：从关注列表中筛选满足接回条件的标的。"""
    cfg = merge_shakeout_cfg(strat_cfg)
    if not cfg['shakeout_rebuy_enabled'] or not watches:
        return []
    max_watch = int(cfg['shakeout_rebuy_days'])
    max_daily = int(cfg['shakeout_rebuy_max_daily'])
    dt = pd.Timestamp(date)
    cands = []
    for code, watch in list(watches.items()):
        if code in held:
            continue
        sell_date = watch.get('sell_date', '')
        if sell_date and count_trading_days(sell_date, date) > max_watch:
            continue
        df = all_data.get(code)
        if df is None:
            continue
        hist = df.loc[df.index <= dt]
        if hist.empty:
            continue
        row = hist.iloc[-1].copy()
        row['code'] = code
        rebuy = eval_shakeout_rebuy(watch, row, hist, strat_cfg, regime=regime)
        if rebuy:
            rebuy['buy_date'] = date
            cands.append(rebuy)
    cands.sort(key=lambda x: x.get('score', 0), reverse=True)
    return cands[:max_daily]
