# -*- coding: utf-8 -*-
"""涨停板首板选股策略

条件:
  - 30个交易日内首板（今日涨停，此前窗口内无涨停）
  - 当日成交量 > 上一交易日 3 倍
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

DEFAULT_LIMIT_UP = {
    'enabled': True,
    'lookback_days': 30,
    'min_volume_ratio': 3.0,
    'max_daily_picks': 2,
    'score_boost': 15,
    'stop_loss': 5.0,
    'trailing_start': 15.0,
    'trailing_pct': 10.0,
    'take_profit_levels': [],
}


def get_limit_up_exit_params(config: dict = None) -> dict:
    """涨停持仓退出：最高收益>trailing_start% 后，收益回落≥最高收益×trailing_pct% 则清仓"""
    cfg = load_limit_up_config(config or {})
    return {
        'stop_loss_pct': float(cfg.get('stop_loss', 5.0)),
        'trailing_start': float(cfg.get('trailing_start', 15.0)),
        'trailing_pct': float(cfg.get('trailing_pct', 10.0)),
        'take_profit_levels': cfg.get('take_profit_levels', []),
    }


def limit_up_profit_trailing_hit(max_profit_pct: float, current_profit_pct: float,
                                 min_peak_pct: float = 15.0,
                                 drawback_ratio_pct: float = 10.0):
    """判断涨停回撤止盈是否触发。

    例: 最高16.9%，trailing_pct=10 → 阈值1.69%，当前收益≤15.21%时卖出。
    返回 (是否卖出, 收益回落点数, 回落阈值点数)
    """
    if max_profit_pct < min_peak_pct or current_profit_pct <= 0:
        return False, 0.0, 0.0
    profit_drawback = round(max_profit_pct - current_profit_pct, 4)
    threshold = round(max_profit_pct * (drawback_ratio_pct / 100), 4)
    return profit_drawback >= threshold, profit_drawback, threshold


def load_limit_up_config(config: dict) -> dict:
    cfg = DEFAULT_LIMIT_UP.copy()
    cfg.update(config.get('limit_up', {}))
    return cfg


def limit_up_threshold(code: str, name: str = '') -> float:
    """涨跌幅涨停阈值(%)"""
    code = str(code).replace('sh', '').replace('sz', '').strip()
    if name and ('ST' in str(name).upper()):
        return 4.8
    if code.startswith(('300', '688')):
        return 19.5
    if code.startswith(('8', '4')):
        return 29.5
    return 9.8


def is_limit_up_bar(row, code: str, name: str = '', tolerance: float = 0.3) -> bool:
    ch = row.get('pct_change', row.get('change', np.nan))
    if pd.isna(ch):
        return False
    return float(ch) >= limit_up_threshold(code, name) - tolerance


def is_first_limit_up(hist_df: pd.DataFrame, code: str, lookback: int = 30,
                      name: str = '') -> bool:
    if hist_df is None or len(hist_df) < lookback + 1:
        return False
    today = hist_df.iloc[-1]
    if not is_limit_up_bar(today, code, name):
        return False
    window = hist_df.iloc[-(lookback + 1):-1]
    for i in range(len(window)):
        row = window.iloc[i]
        if is_limit_up_bar(row, code, name):
            return False
    return True


def volume_spike(hist_df: pd.DataFrame, min_ratio: float = 3.0) -> tuple:
    if hist_df is None or len(hist_df) < 2:
        return False, 0.0
    today_vol = float(hist_df.iloc[-1].get('volume', 0))
    prev_vol = float(hist_df.iloc[-2].get('volume', 0))
    if prev_vol <= 0 or today_vol <= 0:
        return False, 0.0
    ratio = today_vol / prev_vol
    return ratio >= min_ratio, ratio


def screen_limit_up_row(row, hist_df: pd.DataFrame, cfg: dict,
                        name: str = '') -> Optional[Dict]:
    """回测/历史筛选单只股票"""
    if not cfg.get('enabled', True):
        return None

    code = row.get('code', '')
    lookback = cfg.get('lookback_days', 30)
    min_vol = cfg.get('min_volume_ratio', 3.0)

    if not is_first_limit_up(hist_df, code, lookback, name):
        return None

    vol_ok, vol_ratio = volume_spike(hist_df, min_vol)
    if not vol_ok:
        return None

    change = float(row.get('pct_change', row.get('change', 0)))
    score = 80 + min(vol_ratio, 8) * 2 + cfg.get('score_boost', 15)

    return {
        'code': code,
        'buy_price': float(row['close'] if 'close' in row.index else row.get('price', 0)),
        'change': change,
        'tech_score': 0,
        'volume_ratio': round(vol_ratio, 2),
        'score': round(score, 2),
        'strategy_type': 'limit_up',
        'signals': {
            'reason': f'首板|量比昨{vol_ratio:.1f}x|{lookback}日内首次涨停',
            'vol_vs_prev': round(vol_ratio, 2),
            'lookback_days': lookback,
        },
    }


def screen_limit_up_realtime(quote_row: dict, hist_df: pd.DataFrame,
                             cfg: dict) -> Optional[dict]:
    """实盘筛选"""
    if not cfg.get('enabled', True) or hist_df is None:
        return None

    code = quote_row.get('code_clean', quote_row.get('code', ''))
    name = quote_row.get('name', '')
    change = quote_row.get('change', 0)

    if change < limit_up_threshold(code, name) - 0.5:
        return None

    hist = hist_df.copy()
    if 'pct_change' not in hist.columns:
        hist['pct_change'] = hist['close'].pct_change() * 100
    # 用实时涨幅覆盖最后一行
    if len(hist) > 0:
        last_idx = hist.index[-1]
        hist.loc[last_idx, 'pct_change'] = change
        if quote_row.get('volume'):
            hist.loc[last_idx, 'volume'] = quote_row['volume']

    result = screen_limit_up_row(hist.iloc[-1], hist, cfg, name=name)
    if not result:
        return None

    out = {**quote_row, **result}
    out['code_clean'] = code
    out['strategy_type'] = 'limit_up'
    out['tech_score'] = result['score']
    out['tech_reason'] = result['signals']['reason']
    return out


def merge_strategy_candidates(trend_list: List[Dict], limit_up_list: List[Dict],
                              trend_slots: int = 8, limit_up_slots: int = 2,
                              doubler_list: List[Dict] = None,
                              doubler_slots: int = 0) -> List[Dict]:
    """合并多策略候选"""
    from src.strategy_merge import merge_multi_strategy_candidates
    for c in trend_list:
        c.setdefault('strategy_type', 'trend')
    pools = {
        'limit_up': (limit_up_list or [], limit_up_slots),
        'doubler': (doubler_list or [], doubler_slots),
        'trend': (trend_list, trend_slots),
    }
    return merge_multi_strategy_candidates(pools)
