# -*- coding: utf-8 -*-
"""分析半仓持股深亏特征"""
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

from src.backtest_data_store import BacktestDataStore
from daily_pipeline_v4 import calculate_technical_indicators

data = json.load(open('data/backtest_capital_results.json', encoding='utf-8'))
trades = data['variants']['趋势选股']['trades']

by_code = {}
for t in trades:
    if t.get('strategy_type') != 'trend':
        continue
    by_code.setdefault(t['code'], []).append(t)

rounds = []
for code, ts in by_code.items():
    ts = sorted(ts, key=lambda x: x['date'])
    buy = None
    had_half = False
    for t in ts:
        if t['action'] == 'buy' and buy is None:
            buy = t
            had_half = False
        elif t['action'] == 'sell' and buy is not None:
            if '趋势止损半仓' in t.get('reason', ''):
                had_half = True
            reason = t.get('reason', '')
            terminal = (
                '半仓持有清仓' in reason
                or '回测结束清仓' in reason
                or ('止损' in reason and '半仓' not in reason)
            )
            if terminal:
                rounds.append({
                    'code': code,
                    'buy_date': buy['date'],
                    'buy_price': buy['price'],
                    'sell_date': t['date'],
                    'sell_reason': reason,
                    'final_profit_pct': t.get('profit_pct', 0),
                    'had_half_stop': had_half,
                })
                buy = None
                had_half = False

bad = [r for r in rounds if r['had_half_stop'] and r['final_profit_pct'] <= -20]
mid = [r for r in rounds if r['had_half_stop'] and -5 <= r['final_profit_pct'] <= 10]
good = [r for r in rounds if r['had_half_stop'] and r['final_profit_pct'] > 20]
all_half = [r for r in rounds if r['had_half_stop']]

print(f'半仓止损轮次: {len(all_half)}')
print(f'  最终亏>20%: {len(bad)}')
print(f'  最终-5~10%: {len(mid)}')
print(f'  最终盈>20%: {len(good)}')

codes = list({r['code'] for r in all_half})
raw, _ = BacktestDataStore().load_or_fetch(codes, 300)


def feat_at_buy(r):
    df = raw.get(r['code'])
    if df is None:
        return None
    df = calculate_technical_indicators(df.copy())
    df['pct_change'] = df['close'].pct_change() * 100
    df.index = pd.to_datetime(df.index)
    dt = pd.Timestamp(r['buy_date'])
    loc = df.index.searchsorted(dt)
    if loc >= len(df):
        return None
    row = df.iloc[loc]
    hist = df.iloc[max(0, loc - 29):loc + 1]
    low30 = float(hist['low'].min())
    high30 = float(hist['high'].max())
    close = float(row['close'])
    prior_swing = (high30 - low30) / low30 * 100 if low30 > 0 else 0
    ret5 = (close / float(df.iloc[max(0, loc - 5)]['close']) - 1) * 100 if loc >= 5 else 0
    ret10 = (close / float(df.iloc[max(0, loc - 10)]['close']) - 1) * 100 if loc >= 10 else 0
    ret20 = (close / float(df.iloc[max(0, loc - 20)]['close']) - 1) * 100 if loc >= 20 else 0
    dist_from_high30 = (close / high30 - 1) * 100 if high30 > 0 else 0
    buy_chg = float(row['pct_change'])
    rsi = float(row['rsi']) if 'rsi' in row and not pd.isna(row['rsi']) else None
    return {
        'code': r['code'],
        'buy_date': r['buy_date'],
        'final_profit_pct': r['final_profit_pct'],
        'sell_reason': r['sell_reason'][:40],
        'buy_day_change': round(buy_chg, 2),
        'prior_5d': round(ret5, 1),
        'prior_10d': round(ret10, 1),
        'prior_20d': round(ret20, 1),
        'swing_30d': round(prior_swing, 1),
        'dist_from_30d_high': round(dist_from_high30, 1),
        'rsi': round(rsi, 1) if rsi is not None else None,
    }


def summarize(group, label):
    rows = [feat_at_buy(r) for r in group]
    rows = [x for x in rows if x]
    if not rows:
        print(f'\n=== {label}: 无样本 ===')
        return rows
    df = pd.DataFrame(rows)
    print(f'\n=== {label} (n={len(df)}) ===')
    for col in ['buy_day_change', 'prior_5d', 'prior_10d', 'prior_20d',
                'swing_30d', 'dist_from_30d_high', 'rsi']:
        if col in df.columns and df[col].notna().any():
            print(f'  {col}: 均{df[col].mean():+.1f}  中位{df[col].median():+.1f}  最大{df[col].max():+.1f}')
    print('  明细:')
    for row in df.sort_values('final_profit_pct').to_dict('records'):
        print(
            f"    {row['code']} {row['buy_date']} "
            f"买日涨{row['buy_day_change']:+.1f}% "
            f"前20日{row['prior_20d']:+.1f}% "
            f"距30日高{row['dist_from_30d_high']:+.1f}% "
            f"30日振幅{row['swing_30d']:.1f}% "
            f"-> 终{row['final_profit_pct']:+.1f}%"
        )
    return rows

summarize(bad, '拖累: 半仓后亏>20%')
summarize(mid, '对照: 半仓后持平')
summarize(good, '对照: 半仓后盈>20%')
summarize(all_half, '全部半仓止损')
