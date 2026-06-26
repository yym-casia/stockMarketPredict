# -*- coding: utf-8 -*-
"""买入当日各筛选条件统计（大赢/亏损/全部买入）"""

from collections import defaultdict
from typing import Dict, List

import pandas as pd

ENTRY_NUMERIC = [
    ('change', '涨幅%'),
    ('prev_change', '昨涨幅%'),
    ('rsi', 'RSI'),
    ('volume_ratio', '量比'),
    ('tech_score', '技术分'),
    ('score', '综合分'),
    ('ml_score', 'ML分'),
    ('close_strength', '收盘强度'),
    ('conditions_met', '技术条件数'),
]

ENTRY_FLAGS = [
    ('rsi_ok', 'RSI达标'),
    ('macd_ok', 'MACD达标'),
    ('macd_golden', 'MACD金叉'),
    ('volume_ok', '量比达标'),
    ('ma_ok', '均线达标'),
    ('bullish_candle', '阳线'),
]


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


def _entry_from_data(all_data: Dict, code: str, buy_date: str) -> dict:
    df = all_data.get(code)
    if df is None or df.empty:
        return {}
    hist = df.loc[df.index <= pd.Timestamp(buy_date)]
    if hist.empty:
        return {}
    row = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else None
    change = row.get('pct_change', row.get('change', 0))
    vr = row.get('volume_ratio')
    if vr is None or (isinstance(vr, float) and pd.isna(vr)):
        if prev is not None:
            pv = float(prev.get('volume', 0))
            v = float(row.get('volume', 0))
            vr = v / pv if pv > 0 else None
    prev_change = None
    if prev is not None:
        pc = prev.get('pct_change', 0)
        if not pd.isna(pc):
            prev_change = float(pc)
    cs = 1.0
    if row['high'] > row['low']:
        cs = (row['close'] - row['low']) / (row['high'] - row['low'])
    rsi = row.get('rsi')
    return {
        'change': round(float(change), 2) if not pd.isna(change) else None,
        'prev_change': round(prev_change, 2) if prev_change is not None else None,
        'rsi': round(float(rsi), 2) if rsi is not None and not pd.isna(rsi) else None,
        'volume_ratio': round(float(vr), 2) if vr is not None and not pd.isna(vr) else None,
        'close_strength': round(float(cs), 2),
    }


def _resolve_entry(trade: dict, all_data: Dict) -> dict:
    entry = dict(trade.get('entry') or {})
    fb = _entry_from_data(all_data, trade['code'], trade['date'])
    for k, v in fb.items():
        if entry.get(k) is None and v is not None:
            entry[k] = v
    if trade.get('score') is not None:
        entry.setdefault('score', trade['score'])
    if trade.get('ml_score') is not None:
        entry.setdefault('ml_score', trade['ml_score'])
    return entry


def _stats_for(subset: List[dict]) -> dict:
    if not subset:
        return {'count': 0, 'fields': {}, 'flags': {}}
    fields = {}
    for key, label in ENTRY_NUMERIC:
        vals = [r['entry'].get(key) for r in subset if r['entry'].get(key) is not None]
        if vals:
            fields[key] = {'label': label, 'summary': _summary(vals)}
    flags = {}
    for key, label in ENTRY_FLAGS:
        trues = sum(1 for r in subset if r['entry'].get(key))
        total = sum(1 for r in subset if r['entry'].get(key) is not None)
        if total:
            flags[key] = {
                'label': label,
                'true_count': trues,
                'sample_count': total,
                'true_pct': round(trues / total * 100, 1),
            }
    return {'count': len(subset), 'fields': fields, 'flags': flags}


def analyze_buy_entry_stats(
    trades: List[dict],
    all_data: Dict,
    big_win_threshold: float = 20.0,
) -> dict:
    """统计所有买入在当日的筛选条件分布（按盈亏分组）。"""
    buys = [t for t in trades if t.get('action') == 'buy']
    sells_by_code = defaultdict(list)
    for t in trades:
        if t.get('action') == 'sell':
            sells_by_code[t['code']].append(t)

    rows = []
    for b in buys:
        code = b['code']
        bd = b['date']
        sell = None
        for s in sorted(sells_by_code.get(code, []), key=lambda x: x['date']):
            if s['date'] >= bd:
                sell = s
                break
        profit_pct = sell.get('profit_pct') if sell else None
        if profit_pct is None:
            outcome = 'open'
        elif profit_pct >= big_win_threshold:
            outcome = 'big_win'
        elif profit_pct > 0:
            outcome = 'small_win'
        else:
            outcome = 'loss'
        entry = _resolve_entry(b, all_data)
        rows.append({
            'code': code,
            'buy_date': bd,
            'reason': b.get('reason', ''),
            'strategy_type': b.get('strategy_type', 'trend'),
            'profit_pct': round(profit_pct, 2) if profit_pct is not None else None,
            'outcome': outcome,
            'entry': entry,
        })

    big = [r for r in rows if r['outcome'] == 'big_win']
    loss = [r for r in rows if r['outcome'] == 'loss']
    small = [r for r in rows if r['outcome'] == 'small_win']
    open_pos = [r for r in rows if r['outcome'] == 'open']

    return {
        'all_buys': _stats_for(rows),
        'big_wins': _stats_for(big),
        'losses': _stats_for(loss),
        'small_wins': _stats_for(small),
        'open_positions': _stats_for(open_pos),
        'outcome_counts': {
            'total': len(rows),
            'big_win': len(big),
            'loss': len(loss),
            'small_win': len(small),
            'open': len(open_pos),
        },
        'samples': [
            {
                'code': r['code'],
                'buy_date': r['buy_date'],
                'reason': r['reason'],
                'outcome': r['outcome'],
                'profit_pct': r['profit_pct'],
                **{k: r['entry'].get(k) for k, _ in ENTRY_NUMERIC},
                **{k: r['entry'].get(k) for k, _ in ENTRY_FLAGS},
            }
            for r in rows
        ],
    }
