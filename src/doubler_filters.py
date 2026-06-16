# -*- coding: utf-8 -*-
"""翻倍股模式选股策略 — 基于历史翻倍样本共性特征"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.doubler_analyzer import load_doubler_patterns
from src.features import FeatureEngineer

DEFAULT_DOUBLER = {
    'enabled': False,
    'independent_enabled': False,
    'boost_enabled': True,
    'max_daily_picks': 1,
    'min_change': 2.0,
    'max_change': 7.0,
    'min_volume_ratio': 1.6,
    'min_return_5d_pct': 2.0,
    'max_return_5d_pct': 12.0,
    'min_return_20d_pct': 5.0,
    'max_return_20d_pct': 45.0,
    'min_rsi': 48,
    'max_rsi': 72,
    'max_return_20d_pct': 30.0,
    'min_macd_hist': 0,
    'require_ma_bullish': True,
    'require_new_high_20': True,
    'require_volume_breakout': False,
    'min_boll_position': 0.55,
    'min_momentum_10': 0.03,
    'min_conditions': 4,
    'score_boost': 12,
    'stop_loss': 4.0,
    'trailing_start': 20.0,
    'trailing_pct': 8.0,
    'take_profit_levels': [
        {'pct': 15.0, 'ratio': 0.15},
        {'pct': 30.0, 'ratio': 0.25},
        {'pct': 50.0, 'ratio': 0.60},
    ],
}

_fe = FeatureEngineer()
_feat_cache: Dict[str, pd.DataFrame] = {}


def load_doubler_config(config: dict) -> dict:
    cfg = DEFAULT_DOUBLER.copy()
    patterns = load_doubler_patterns()
    cfg.update(patterns.get('strategy_rules', {}))
    cfg.update(config.get('doubler', {}))
    return cfg


def get_doubler_exit_params(config: dict = None) -> dict:
    cfg = load_doubler_config(config or {})
    return {
        'stop_loss_pct': float(cfg.get('stop_loss', 4.0)),
        'trailing_start': float(cfg.get('trailing_start', 20.0)),
        'trailing_pct': float(cfg.get('trailing_pct', 8.0)),
        'take_profit_levels': cfg.get('take_profit_levels', []),
    }


def _ensure_features(hist_df: pd.DataFrame, code: str) -> pd.DataFrame:
    if code in _feat_cache and len(_feat_cache[code]) == len(hist_df):
        return _feat_cache[code]
    feat = _fe.calculate_technical_indicators(hist_df.copy())
    if 'pct_change' not in feat.columns:
        feat['pct_change'] = feat['close'].pct_change() * 100
    _feat_cache[code] = feat
    return feat


def _independent_enabled(cfg: dict) -> bool:
    return cfg.get('independent_enabled', cfg.get('enabled', False))


def eval_doubler_pattern(row, hist_df: pd.DataFrame, cfg: dict) -> Optional[dict]:
    """评估是否命中翻倍启动模式，返回检查项与加分（不单独建仓）。"""
    code = str(row.get('code', ''))
    if hist_df is None or len(hist_df) < 60:
        return None

    feat = _ensure_features(hist_df, code)
    try:
        last = feat.loc[row.name] if getattr(row, 'name', None) in feat.index else feat.iloc[-1]
    except Exception:
        last = feat.iloc[-1]

    change = float(row.get('pct_change', last.get('pct_change', 0)) or 0)
    if not (cfg['min_change'] <= change <= cfg['max_change']):
        return None

    vol = float(row.get('volume', last.get('volume', 0)) or 0)
    if vol < 500000:
        return None

    vr = float(last.get('volume_ratio', 0) or 0)
    rsi = float(last.get('rsi_14', 50) or 50)
    ret5 = float(last.get('return_5d', 0) or 0) * 100
    ret20 = float(last.get('return_20d', 0) or 0) * 100
    macd_h = float(last.get('macd_hist', 0) or 0)
    ma_bull = int(last.get('ma_bullish', 0) or 0)
    new_high = int(last.get('is_new_high_20', 0) or 0)
    vol_brk = int(last.get('volume_breakout', 0) or 0)
    boll = float(last.get('boll_position', 0.5) or 0.5)
    mom10 = float(last.get('momentum_10', 0) or 0)

    checks = []
    if vr >= cfg['min_volume_ratio']:
        checks.append('量比')
    if cfg['min_return_5d_pct'] <= ret5 <= cfg['max_return_5d_pct']:
        checks.append('5日动量')
    if cfg['min_return_20d_pct'] <= ret20 <= cfg.get('max_return_20d_pct', 45):
        checks.append('20日动量')
    if cfg['min_rsi'] <= rsi <= cfg['max_rsi']:
        checks.append('RSI')
    if macd_h >= cfg['min_macd_hist']:
        checks.append('MACD')
    if not cfg.get('require_ma_bullish') or ma_bull:
        checks.append('均线多头')
    if not cfg.get('require_new_high_20') or new_high:
        checks.append('20日新高')
    if not cfg.get('require_volume_breakout') or vol_brk:
        checks.append('放量突破')
    if boll >= cfg.get('min_boll_position', 0.5):
        checks.append('布林强势')
    if mom10 >= cfg.get('min_momentum_10', 0):
        checks.append('10日动能')

    if len(checks) < cfg.get('min_conditions', 4):
        return None

    boost = float(cfg.get('score_boost', 15))
    reason = f"翻倍特征[{','.join(checks[:4])}] 量比{vr:.1f} RSI{rsi:.0f}"
    return {
        'boost': boost,
        'checks': checks,
        'reason': reason,
        'rsi': rsi,
        'volume_ratio': vr,
    }


def apply_doubler_boost(candidate: dict, row, hist_df: pd.DataFrame,
                        cfg: dict) -> dict:
    """趋势候选命中翻倍模式时加分，不改变 strategy_type。"""
    if not cfg.get('boost_enabled', True):
        return candidate
    hit = eval_doubler_pattern(row, hist_df, cfg)
    if not hit:
        return candidate
    candidate['score'] = round(candidate.get('score', 0) + hit['boost'], 2)
    candidate['doubler_boost'] = hit['boost']
    candidate['doubler_match'] = hit['reason']
    sig = candidate.get('signals', {})
    if isinstance(sig, dict):
        sig['doubler'] = hit['reason']
    return candidate


def screen_doubler_row(row, hist_df: pd.DataFrame, cfg: dict,
                       name: str = '') -> Optional[dict]:
    if not _independent_enabled(cfg):
        return None

    hit = eval_doubler_pattern(row, hist_df, cfg)
    if not hit:
        return None

    code = str(row.get('code', ''))
    price = float(row.get('close', 0))
    score = 70 + len(hit['checks']) * 3 + hit['boost']
    return {
        'code': code,
        'name': name,
        'buy_price': price,
        'score': round(score, 2),
        'strategy_type': 'doubler',
        'signals': {
            'reason': hit['reason'],
            'checks': hit['checks'],
            'volume_ratio': hit['volume_ratio'],
            'rsi': hit['rsi'],
        },
    }


def screen_doubler_realtime(quote_row: dict, hist_df: pd.DataFrame,
                            cfg: dict) -> Optional[dict]:
    if not cfg.get('enabled', True) or hist_df is None:
        return None
    hist = hist_df.copy()
    if 'pct_change' not in hist.columns:
        hist['pct_change'] = hist['close'].pct_change() * 100
    change = quote_row.get('change', 0)
    if len(hist) > 0:
        hist.iloc[-1, hist.columns.get_loc('close')] = quote_row.get('price', hist.iloc[-1]['close'])
        if quote_row.get('volume'):
            hist.iloc[-1, hist.columns.get_loc('volume')] = quote_row['volume']
        hist.iloc[-1, hist.columns.get_loc('pct_change')] = change

    row = hist.iloc[-1].copy()
    row['code'] = quote_row.get('code_clean', quote_row.get('code', ''))
    result = screen_doubler_row(row, hist, cfg, name=quote_row.get('name', ''))
    if not result:
        return None
    out = {**quote_row, **result}
    out['code_clean'] = row['code']
    out['tech_score'] = result['score']
    out['tech_reason'] = result['signals']['reason']
    return out
