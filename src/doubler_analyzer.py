# -*- coding: utf-8 -*-
"""翻倍股样本挖掘与共性特征提炼"""

import json
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features import FeatureEngineer

FEATURE_KEYS = [
    'return_5d', 'return_20d', 'volume_ratio', 'rsi_14', 'macd_hist',
    'ma_bullish', 'break_ma20', 'is_new_high_20', 'volume_breakout',
    'momentum_10', 'boll_position', 'consecutive_up', 'amplitude', 'change_pct',
]

ENTRY_FEATURE_KEYS = [k for k in FEATURE_KEYS if k not in ('change_pct',)]


def _forward_max_return(close: pd.Series, horizon: int = 60) -> pd.Series:
    """未来 horizon 日内最高价相对当日收盘的最大涨幅(%)"""
    n = len(close)
    out = pd.Series(np.nan, index=close.index)
    arr = close.values.astype(float)
    for i in range(n):
        end = min(i + horizon + 1, n)
        if i + 1 >= end:
            continue
        base = arr[i]
        if base <= 0:
            continue
        out.iloc[i] = (arr[i + 1:end].max() / base - 1) * 100
    return out


def find_doubler_entries(df: pd.DataFrame, code: str,
                         min_forward_pct: float = 100.0,
                         horizon: int = 60,
                         min_history: int = 60) -> List[Dict]:
    """找翻倍行情的「启动日」：未来60日内翻倍，且当日具放量突破特征"""
    if df is None or len(df) < min_history + horizon:
        return []
    fe = FeatureEngineer()
    feat = fe.calculate_technical_indicators(df.copy())
    feat['pct_change'] = feat['close'].pct_change() * 100
    n = len(feat)
    entries = []
    used_dates = set()

    for i in range(min_history, n - horizon):
        base = float(feat.iloc[i]['close'])
        if base <= 0:
            continue
        future = feat.iloc[i + 1:i + horizon + 1]['close']
        if future.max() < base * (1 + min_forward_pct / 100):
            continue
        fwd_pct = (future.max() / base - 1) * 100

        # 在翻倍起点后20日内找第一个「启动日」
        launch = None
        for k in range(i, min(i + 20, n)):
            row = feat.iloc[k]
            chg = float(row.get('pct_change', 0) or 0)
            vr = float(row.get('volume_ratio', 0) or 0)
            new_hi = int(row.get('is_new_high_20', 0) or 0)
            brk = int(row.get('break_ma20', 0) or 0)
            ma_bull = int(row.get('ma_bullish', 0) or 0)
            if 1.5 <= chg <= 9.0 and vr >= 1.3 and (new_hi or brk) and ma_bull:
                launch = (k, row)
                break
        if launch is None:
            continue
        k, row = launch
        d = feat.index[k].strftime('%Y-%m-%d')
        if d in used_dates:
            continue
        used_dates.add(d)
        snap = {key: float(row[key]) if key in row.index and pd.notna(row[key]) else None
                for key in FEATURE_KEYS}
        entries.append({
            'code': code,
            'date': d,
            'entry_close': float(row['close']),
            'forward_max_pct': float(fwd_pct),
            'entry_change_pct': float(row.get('pct_change', 0) or 0),
            'features': snap,
        })
    return entries


def pick_top_doublers(all_entries: List[Dict], n: int = 50) -> List[Dict]:
    """每只股票保留最强样本，再取 top N"""
    by_code: Dict[str, Dict] = {}
    for e in all_entries:
        c = e['code']
        if c not in by_code or e['forward_max_pct'] > by_code[c]['forward_max_pct']:
            by_code[c] = e
    ranked = sorted(by_code.values(), key=lambda x: x['forward_max_pct'], reverse=True)
    return ranked[:n]


def _sample_controls(all_data: Dict[str, pd.DataFrame], doubler_dates: set,
                     n: int = 200, horizon: int = 60) -> List[Dict]:
    """随机采样非翻倍入场日作对照"""
    fe = FeatureEngineer()
    samples = []
    codes = list(all_data.keys())
    rng = np.random.default_rng(42)
    tries = 0
    while len(samples) < n and tries < n * 30:
        tries += 1
        code = codes[rng.integers(0, len(codes))]
        df = all_data[code]
        if len(df) < 80:
            continue
        feat = fe.calculate_technical_indicators(df.copy())
        feat['fwd_max_ret'] = _forward_max_return(feat['close'], horizon)
        i = int(rng.integers(60, len(feat) - 10))
        d = feat.index[i].strftime('%Y-%m-%d')
        if (code, d) in doubler_dates:
            continue
        fwd = feat.iloc[i].get('fwd_max_ret')
        if pd.isna(fwd) or fwd >= 100:
            continue
        row = feat.iloc[i]
        samples.append({k: float(row[k]) if k in row.index and pd.notna(row[k]) else None
                        for k in FEATURE_KEYS})
    return samples


def summarize_features(doubler_feats: List[Dict], control_feats: List[Dict]) -> Dict:
    """对比翻倍样本 vs 对照组，提炼阈值"""
    report = {'doubler_count': len(doubler_feats), 'control_count': len(control_feats), 'features': {}}
    thresholds = {}

    for key in ENTRY_FEATURE_KEYS:
        d_vals = [f['features'][key] for f in doubler_feats
                  if f.get('features', {}).get(key) is not None]
        c_vals = [f.get(key) for f in control_feats if f.get(key) is not None]
        if not d_vals:
            continue
        d_arr, c_arr = np.array(d_vals), np.array(c_vals) if c_vals else np.array([0])
        report['features'][key] = {
            'doubler_median': round(float(np.median(d_arr)), 4),
            'doubler_p25': round(float(np.percentile(d_arr, 25)), 4),
            'doubler_p75': round(float(np.percentile(d_arr, 75)), 4),
            'control_median': round(float(np.median(c_arr)), 4) if len(c_arr) else None,
        }
        # 翻倍股特征：取 p25~p75 中偏「启动」一侧
        if key in ('volume_ratio', 'ma_bullish', 'break_ma20', 'is_new_high_20',
                   'volume_breakout', 'macd_hist', 'consecutive_up', 'momentum_10'):
            thresholds[f'min_{key}'] = round(float(np.percentile(d_arr, 25)), 4)
        elif key == 'rsi_14':
            thresholds['min_rsi'] = round(float(np.percentile(d_arr, 20)), 2)
            thresholds['max_rsi'] = round(float(np.percentile(d_arr, 80)), 2)
        elif key == 'return_5d':
            thresholds['min_return_5d'] = round(float(np.percentile(d_arr, 25)) * 100, 2)
            thresholds['max_return_5d'] = round(float(np.percentile(d_arr, 75)) * 100, 2)
        elif key == 'return_20d':
            thresholds['min_return_20d'] = round(float(np.percentile(d_arr, 25)) * 100, 2)
            thresholds['max_return_20d'] = round(float(np.percentile(d_arr, 75)) * 100, 2)
        elif key == 'boll_position':
            thresholds['min_boll_position'] = round(float(np.percentile(d_arr, 25)), 4)

    # 当日涨幅：翻倍股启动日通常温和放量上涨
    chg_vals = [f['entry_change_pct'] for f in doubler_feats]
    if chg_vals:
        thresholds['min_change'] = round(float(np.percentile(chg_vals, 20)), 2)
        thresholds['max_change'] = round(float(np.percentile(chg_vals, 80)), 2)

    report['derived_thresholds'] = thresholds
    return report


def analyze_pool(all_data: Dict[str, pd.DataFrame], n_samples: int = 50,
                 output_path: str = 'data/doubler_patterns.json') -> Dict:
    """全池分析并保存模式"""
    all_entries = []
    for code, df in all_data.items():
        all_entries.extend(find_doubler_entries(df, code))
    doublers = pick_top_doublers(all_entries, n_samples)
    doubler_dates = {(e['code'], e['date']) for e in doublers}
    controls = _sample_controls(all_data, doubler_dates)
    summary = summarize_features(doublers, controls)

    # 共性文字结论
    th = summary['derived_thresholds']
    patterns = []
    if th.get('min_volume_ratio', 0) >= 1.5:
        patterns.append('放量：量比高于常态')
    if th.get('min_ma_bullish', 0) >= 0.5:
        patterns.append('均线多头排列')
    if th.get('min_is_new_high_20', 0) >= 0.5:
        patterns.append('创20日新高')
    if th.get('min_break_ma20', 0) >= 0.3:
        patterns.append('突破20日均线')
    if th.get('min_macd_hist', 0) > 0:
        patterns.append('MACD红柱/动能向上')
    summary['common_patterns'] = patterns

    out = {
        'sample_count': len(doublers),
        'samples': doublers,
        'analysis': summary,
        'strategy_rules': _build_strategy_rules(th),
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


def _safe_th(th: Dict, key: str, default: float, lo=None, hi=None) -> float:
    v = th.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return round(float(v), 4)


def _build_strategy_rules(th: Dict) -> Dict:
    """将统计阈值转为可执行策略参数"""
    return {
        'enabled': True,
        'max_daily_picks': 1,
        'min_change': _safe_th(th, 'min_change', 2.0, lo=1.5),
        'max_change': _safe_th(th, 'max_change', 7.0, lo=5.0, hi=9.0),
        'min_volume_ratio': _safe_th(th, 'min_volume_ratio', 1.5, lo=1.3),
        'min_return_5d_pct': _safe_th(th, 'min_return_5d', 3.0, lo=0),
        'max_return_5d_pct': _safe_th(th, 'max_return_5d', 15.0, lo=8),
        'min_return_20d_pct': _safe_th(th, 'min_return_20d', 5.0, lo=0),
        'max_return_20d_pct': _safe_th(th, 'max_return_20d', 40.0, lo=20),
        'min_rsi': _safe_th(th, 'min_rsi', 45, lo=40),
        'max_rsi': _safe_th(th, 'max_rsi', 70, lo=55, hi=78),
        'min_macd_hist': _safe_th(th, 'min_macd_hist', 0, lo=0),
        'require_ma_bullish': True,
        'require_new_high_20': True,
        'require_volume_breakout': False,
        'min_boll_position': _safe_th(th, 'min_boll_position', 0.5, lo=0.4),
        'min_momentum_10': _safe_th(th, 'min_momentum_10', 0.02, lo=0),
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


def load_doubler_patterns(path: str = 'data/doubler_patterns.json') -> Dict:
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'strategy_rules': _build_strategy_rules({})}
