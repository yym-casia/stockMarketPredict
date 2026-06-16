# -*- coding: utf-8 -*-
"""统计半仓止损后整轮盈亏"""
import json
import sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

data = json.load(open('data/backtest_capital_results.json', encoding='utf-8'))
trades = data['variants']['趋势选股']['trades']

by_code = defaultdict(list)
for t in trades:
    if t.get('strategy_type') != 'trend':
        continue
    by_code[t['code']].append(t)


def is_half_stop(reason: str, shares_ratio: float) -> bool:
    return ('止损卖出' in reason or '止损半仓' in reason) and (
        '50%' in reason or abs(shares_ratio - 0.5) < 0.01
    )


def is_terminal(reason: str, shares_ratio: float) -> bool:
    if '余仓清仓' in reason or '回测结束' in reason or '移动止盈' in reason:
        return True
    if '止损' in reason and '半仓' not in reason and '卖出50%' not in reason:
        return True
    if shares_ratio >= 0.99 and '阶梯' not in reason:
        return True
    return False


rounds = []
for code, ts in by_code.items():
    ts = sorted(ts, key=lambda x: x['date'])
    buy = None
    had_half = False
    half_date = None
    round_profit = 0.0

    for t in ts:
        if t['action'] == 'buy' and buy is None:
            buy = t
            had_half = False
            half_date = None
            round_profit = 0.0
        elif t['action'] == 'sell' and buy is not None:
            reason = t.get('reason', '')
            if is_half_stop(reason, t.get('shares_ratio', 0)):
                had_half = True
                half_date = t['date']
            round_profit += t.get('profit', 0)
            if had_half and is_terminal(reason, t.get('shares_ratio', 0)):
                rounds.append({
                    'code': code,
                    'buy_date': buy['date'],
                    'half_date': half_date,
                    'sell_date': t['date'],
                    'final_reason': reason,
                    'round_profit': round_profit,
                    'final_pct': t.get('profit_pct', 0),
                })
                buy = None
                had_half = False

half_rounds = [r for r in rounds if r['half_date']]
print(f'半仓止损轮次: {len(half_rounds)}')
if not half_rounds:
    sys.exit(0)

win = [r for r in half_rounds if r['round_profit'] > 0]
loss = [r for r in half_rounds if r['round_profit'] <= 0]
n = len(half_rounds)
print(f'整轮最终盈利: {len(win)} ({len(win)/n*100:.1f}%)')
print(f'整轮最终亏损: {len(loss)} ({len(loss)/n*100:.1f}%)')
print(f'整轮均利润: {sum(r["round_profit"] for r in half_rounds)/n:+.0f} 元')
print(f'整轮总利润: {sum(r["round_profit"] for r in half_rounds):+.0f} 元')

exit_cnt = Counter()
for r in half_rounds:
    reason = r['final_reason']
    if '移动止盈' in reason:
        exit_cnt['移动止盈'] += 1
    elif '余仓清仓' in reason:
        exit_cnt['余仓清仓'] += 1
    elif '回测结束' in reason:
        exit_cnt['回测结束'] += 1
    else:
        exit_cnt['其他'] += 1
print('最终退出方式:', dict(exit_cnt))

print('\n=== 半仓后最终盈利 ===')
for r in sorted(win, key=lambda x: -x['round_profit']):
    print(f"  {r['code']} {r['buy_date']} 半仓{r['half_date']} -> {r['sell_date']} "
          f"整轮+{r['round_profit']:.0f} | {r['final_reason'][:28]}")

print('\n=== 半仓后最终亏损 ===')
for r in sorted(loss, key=lambda x: x['round_profit']):
    print(f"  {r['code']} {r['buy_date']} 半仓{r['half_date']} -> {r['sell_date']} "
          f"整轮{r['round_profit']:.0f} | {r['final_reason'][:28]}")

half_sells = [
    t for t in trades
    if t.get('action') == 'sell' and is_half_stop(t.get('reason', ''), t.get('shares_ratio', 0))
]
print(f'\n半仓卖出单笔: 共{len(half_sells)}笔, 该笔全亏(约-4%)')
