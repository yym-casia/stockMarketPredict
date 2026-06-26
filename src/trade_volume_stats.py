# -*- coding: utf-8 -*-
"""回测交易：大赢/亏损单买入日及买入前三日量比统计"""

from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd


def _row_volume_ratio(row, prev_row=None) -> Optional[float]:
    vr = row.get('volume_ratio')
    if vr is not None and not (isinstance(vr, float) and pd.isna(vr)):
        return float(vr)
    if prev_row is not None:
        pv = float(prev_row.get('volume', 0))
        v = float(row.get('volume', 0))
        return v / pv if pv > 0 else None
    return None


def _volume_ratio_at_buy(all_data: Dict, code: str, buy_date: str) -> Optional[float]:
    df = all_data.get(code)
    if df is None or df.empty:
        return None
    hist = df.loc[df.index <= pd.Timestamp(buy_date)]
    if hist.empty:
        return None
    row = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else None
    return _row_volume_ratio(row, prev)


def _pre_buy_volume_ratios(all_data: Dict, code: str, buy_date: str, days: int = 3) -> Optional[dict]:
    """买入前 N 个交易日量比（T-1、T-2、T-3，不含买入当日）。"""
    df = all_data.get(code)
    if df is None or df.empty:
        return None
    hist = df.loc[df.index <= pd.Timestamp(buy_date)]
    if len(hist) < days + 1:
        return None
    out = {}
    for i in range(1, days + 1):
        idx = -1 - i
        row = hist.iloc[idx]
        prev = hist.iloc[idx - 1]
        vr = _row_volume_ratio(row, prev)
        if vr is None:
            return None
        out[f't_minus_{i}'] = round(vr, 2)
    out['avg'] = round(sum(out.values()) / days, 2)
    return out


def _match_sell_to_buy(trades: List[dict]) -> List[dict]:
    buys_by_code = defaultdict(list)
    for t in trades:
        if t.get('action') == 'buy':
            buys_by_code[t['code']].append(t)

    rows = []
    for t in trades:
        if t.get('action') != 'sell':
            continue
        prior = [b for b in buys_by_code[t['code']] if b['date'] <= t['date']]
        if not prior:
            continue
        b = prior[-1]
        rows.append({
            'code': t['code'],
            'buy_date': b['date'],
            'sell_date': t['date'],
            'buy_reason': b.get('reason', ''),
            'sell_reason': t.get('reason', ''),
            'profit': t.get('profit', 0),
            'profit_pct': t.get('profit_pct', 0),
            'score': b.get('score'),
            'ml_score': b.get('ml_score'),
        })
    return rows


def _summary(values: List[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    n = len(s)
    return {
        'count': n,
        'mean': round(sum(s) / n, 2),
        'median': round(s[n // 2], 2),
        'min': round(min(s), 2),
        'max': round(max(s), 2),
        'p25': round(s[n // 4], 2),
        'p75': round(s[(3 * n) // 4], 2),
    }


def _distribution(rows: List[dict], buckets: List[tuple], key: str = 'volume_ratio') -> List[dict]:
    out = []
    total = len(rows) or 1
    for lo, hi, label in buckets:
        sub = [r for r in rows if lo <= r[key] < hi]
        if not sub:
            continue
        out.append({
            'label': label,
            'lo': lo,
            'hi': hi,
            'count': len(sub),
            'pct': round(len(sub) / total * 100, 1),
            'avg_profit_pct': round(sum(r['profit_pct'] for r in sub) / len(sub), 2),
        })
    return out


def _attach_volume_ratios(rows: List[dict], all_data: Dict) -> List[dict]:
    valid = []
    for r in rows:
        vr = _volume_ratio_at_buy(all_data, r['code'], r['buy_date'])
        pre = _pre_buy_volume_ratios(all_data, r['code'], r['buy_date'])
        if vr is None:
            continue
        r = dict(r)
        r['volume_ratio'] = round(vr, 2)
        if pre:
            r['pre_buy_vr'] = pre
            r['pre_buy_avg'] = pre['avg']
        valid.append(r)
    return valid


def _pre_buy_summary(rows: List[dict]) -> dict:
    with_pre = [r for r in rows if r.get('pre_buy_vr')]
    if not with_pre:
        return {}
    return {
        'sample_count': len(with_pre),
        't_minus_3': _summary([r['pre_buy_vr']['t_minus_3'] for r in with_pre]),
        't_minus_2': _summary([r['pre_buy_vr']['t_minus_2'] for r in with_pre]),
        't_minus_1': _summary([r['pre_buy_vr']['t_minus_1'] for r in with_pre]),
        'avg_3d': _summary([r['pre_buy_avg'] for r in with_pre]),
    }


def analyze_trade_volume_stats(
    trades: List[dict],
    all_data: Dict,
    big_win_threshold: float = 20.0,
) -> dict:
    """大赢 vs 亏损单：买入日 + 买入前三日量比分析。"""
    matched = _match_sell_to_buy(trades)
    with_vr = _attach_volume_ratios(matched, all_data)

    big = [r for r in with_vr if r.get('profit_pct', 0) >= big_win_threshold]
    loss = [r for r in with_vr if r.get('profit_pct', 0) <= 0]
    small_win = [r for r in with_vr if 0 < r.get('profit_pct', 0) < big_win_threshold]

    buckets = [
        (0, 1.25, '<1.25'),
        (1.25, 1.35, '1.25-1.35'),
        (1.35, 1.5, '1.35-1.5'),
        (1.5, 1.8, '1.5-1.8'),
        (1.8, 2.0, '1.8-2.0'),
        (2.0, 2.2, '2.0-2.2'),
        (2.2, 2.5, '2.2-2.5'),
        (2.5, 99, '2.5+'),
    ]

    unique_big = {}
    for r in big:
        key = (r['code'], r['buy_date'])
        if key not in unique_big or r['profit_pct'] > unique_big[key]['profit_pct']:
            unique_big[key] = r
    big_unique = list(unique_big.values())

    def top_rows(rows, key_fn, n=10):
        return [
            {
                'code': r['code'],
                'buy_date': r['buy_date'],
                'sell_date': r.get('sell_date'),
                'volume_ratio': r['volume_ratio'],
                'pre_buy_vr': r.get('pre_buy_vr'),
                'pre_buy_avg': r.get('pre_buy_avg'),
                'profit_pct': round(r['profit_pct'], 2),
                'score': r.get('score'),
                'sell_reason': (r.get('sell_reason') or '')[:60],
            }
            for r in sorted(rows, key=key_fn)[:n]
        ]

    loss_by_mag = []
    for lo, hi, label in [(-99, -10, '<-10%'), (-10, -5, '-5~-10%'), (-5, -3.5, '-3.5~-5%'), (-3.5, 0, '-3~-3.5%')]:
        sub = [r for r in loss if lo < r['profit_pct'] <= hi]
        if not sub:
            continue
        v = [r['volume_ratio'] for r in sub]
        pre = [r['pre_buy_avg'] for r in sub if r.get('pre_buy_avg') is not None]
        loss_by_mag.append({
            'label': label,
            'count': len(sub),
            'vr_mean': round(sum(v) / len(v), 2),
            'vr_median': round(sorted(v)[len(v) // 2], 2),
            'pre_buy_avg_mean': round(sum(pre) / len(pre), 2) if pre else None,
        })

    loss_by_reason = defaultdict(list)
    for r in loss:
        key = r['sell_reason'].split('(')[0] if '(' in r['sell_reason'] else r['sell_reason']
        loss_by_reason[key].append(r)

    by_reason = []
    for reason, sub in sorted(loss_by_reason.items(), key=lambda x: -len(x[1])):
        v = [r['volume_ratio'] for r in sub]
        pre = [r['pre_buy_avg'] for r in sub if r.get('pre_buy_avg') is not None]
        by_reason.append({
            'reason': reason,
            'count': len(sub),
            'vr_mean': round(sum(v) / len(v), 2),
            'pre_buy_avg_mean': round(sum(pre) / len(pre), 2) if pre else None,
            'avg_loss_pct': round(sum(r['profit_pct'] for r in sub) / len(sub), 2),
        })

    trend_loss = [r for r in loss if '震仓' not in r.get('buy_reason', '')]
    shake_loss = [r for r in loss if '震仓' in r.get('buy_reason', '')]

    def bucket_stat(label, sub):
        if not sub:
            return None
        v = [r['volume_ratio'] for r in sub]
        pre = [r['pre_buy_avg'] for r in sub if r.get('pre_buy_avg') is not None]
        return {
            'label': label,
            'count': len(sub),
            'vr_mean': round(sum(v) / len(v), 2),
            'pre_buy_avg_mean': round(sum(pre) / len(pre), 2) if pre else None,
            'avg_profit_pct': round(sum(r['profit_pct'] for r in sub) / len(sub), 2),
        }

    score_buckets = []
    for lo, hi in [(70, 80), (80, 85), (85, 90), (90, 101)]:
        sub = [r for r in loss if r.get('score') is not None and lo <= r['score'] < hi]
        if not sub:
            continue
        v = [r['volume_ratio'] for r in sub]
        pre = [r['pre_buy_avg'] for r in sub if r.get('pre_buy_avg') is not None]
        score_buckets.append({
            'score_range': f'{lo}-{hi}',
            'count': len(sub),
            'vr_mean': round(sum(v) / len(v), 2),
            'pre_buy_avg_mean': round(sum(pre) / len(pre), 2) if pre else None,
            'avg_loss_pct': round(sum(r['profit_pct'] for r in sub) / len(sub), 2),
        })

    big_with_pre = [r for r in big if r.get('pre_buy_avg') is not None]
    loss_with_pre = [r for r in loss if r.get('pre_buy_avg') is not None]

    return {
        'big_wins': {
            'threshold_pct': big_win_threshold,
            'sell_count': len(big),
            'unique_position_count': len(big_unique),
            'summary': _summary([r['volume_ratio'] for r in big]),
            'unique_summary': _summary([r['volume_ratio'] for r in big_unique]),
            'pre_buy_3d': _pre_buy_summary(big),
            'pre_buy_3d_unique': _pre_buy_summary(big_unique),
            'distribution': _distribution(big, buckets),
            'pre_buy_avg_distribution': _distribution(big_with_pre, buckets, 'pre_buy_avg'),
            'top_profit': top_rows(big_unique, lambda x: -x['profit_pct'], 10),
        },
        'losses': {
            'sell_count': len(loss),
            'summary': _summary([r['volume_ratio'] for r in loss]),
            'pre_buy_3d': _pre_buy_summary(loss),
            'distribution': _distribution(loss, buckets),
            'pre_buy_avg_distribution': _distribution(loss_with_pre, buckets, 'pre_buy_avg'),
            'by_loss_magnitude': loss_by_mag,
            'by_sell_reason': by_reason,
            'by_entry_type': list(filter(None, [
                bucket_stat('趋势选股', trend_loss),
                bucket_stat('震仓接回', shake_loss),
            ])),
            'by_score_bucket': score_buckets,
            'top_high_volume_ratio': top_rows(loss, lambda x: -x['volume_ratio'], 10),
            'top_low_volume_ratio': top_rows(loss, lambda x: x['volume_ratio'], 10),
            'worst_losses': top_rows(loss, lambda x: x['profit_pct'], 10),
        },
        'small_wins': {
            'sell_count': len(small_win),
            'summary': _summary([r['volume_ratio'] for r in small_win]),
            'pre_buy_3d': _pre_buy_summary(small_win),
        },
    }
