# -*- coding: utf-8 -*-
import json
from datetime import datetime

with open('data/stock_tracking.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'总跟踪股票数: {len(data)}')
print()

results = []
for code, info in data.items():
    daily = info.get('daily_prices', {})
    if not daily:
        continue
    
    dates = sorted(daily.keys())
    first_price = info.get('buy_price', 0)
    latest_price = daily[dates[-1]]['close'] if dates else 0
    take_profit = info.get('take_profit', 0)
    stop_loss = info.get('stop_loss', 0)
    added_date = info.get('added_date', '')
    
    if first_price > 0:
        total_return = (latest_price - first_price) / first_price * 100
    else:
        total_return = 0
    
    # 判断状态
    if latest_price >= take_profit:
        status = '止盈'
    elif latest_price <= stop_loss:
        status = '止损'
    else:
        status = '持有中'
    
    results.append({
        'code': code,
        'name': info.get('name', '?'),
        'added_date': added_date,
        'buy_price': first_price,
        'latest_price': latest_price,
        'total_return': total_return,
        'status': status,
        'days_held': len(daily),
        'confidence': info.get('confidence', 0)
    })

# 统计
status_counts = {}
for r in results:
    s = r['status']
    status_counts[s] = status_counts.get(s, 0) + 1

print('状态分布:')
for s, c in status_counts.items():
    print(f'  {s}: {c}只')
print()

# 按状态分组显示
for status in ['止盈', '止损', '持有中']:
    stocks = [r for r in results if r['status'] == status]
    if not stocks:
        continue
    
    print(f'\n=== {status} ({len(stocks)}只) ===')
    
    if status == '止盈':
        stocks.sort(key=lambda x: x['total_return'], reverse=True)
    elif status == '止损':
        stocks.sort(key=lambda x: x['total_return'])
    else:
        stocks.sort(key=lambda x: x['total_return'], reverse=True)
    
    for s in stocks[:10]:
        print(f"  {s['code']} {s['name']}: 买入{s['buy_price']:.2f} -> 当前{s['latest_price']:.2f} ({s['total_return']:+.2f}%) | 持仓{s['days_held']}天 | 置信度{s['confidence']:.2f}")
    
    if len(stocks) > 10:
        print(f'  ... 还有 {len(stocks)-10} 只')

# 计算整体胜率
tp = status_counts.get('止盈', 0)
sl = status_counts.get('止损', 0)
holding = status_counts.get('持有中', 0)
completed = tp + sl
win_rate = tp / completed * 100 if completed > 0 else 0

print(f'\n=== 整体统计 ===')
print(f'已完成交易: {completed}只 (止盈{tp} / 止损{sl})')
print(f'胜率: {win_rate:.1f}%')
print(f'持有中: {holding}只')

# 计算平均收益
completed_trades = [r for r in results if r['status'] in ['止盈', '止损']]
if completed_trades:
    avg_return = sum(r['total_return'] for r in completed_trades) / len(completed_trades)
    print(f'平均收益率: {avg_return:+.2f}%')

# 分析不同置信度区间的表现
print('\n=== 置信度分析 ===')
confidence_bins = [
    (0.0, 0.7, '低(0-0.7)'),
    (0.7, 0.8, '中(0.7-0.8)'),
    (0.8, 0.9, '较高(0.8-0.9)'),
    (0.9, 1.0, '高(0.9+)')
]

for low, high, label in confidence_bins:
    bin_stocks = [r for r in completed_trades if low <= r['confidence'] < high]
    if not bin_stocks:
        continue
    
    bin_tp = len([r for r in bin_stocks if r['status'] == '止盈'])
    bin_sl = len([r for r in bin_stocks if r['status'] == '止损'])
    bin_win_rate = bin_tp / len(bin_stocks) * 100
    bin_avg_return = sum(r['total_return'] for r in bin_stocks) / len(bin_stocks)
    
    print(f'  {label}: {len(bin_stocks)}只, 胜率{bin_win_rate:.1f}%, 平均收益{bin_avg_return:+.2f}%')

# 分析不同买入日期的表现
print('\n=== 日期分析 ===')
date_groups = {}
for r in completed_trades:
    d = r['added_date']
    if d not in date_groups:
        date_groups[d] = []
    date_groups[d].append(r)

for d in sorted(date_groups.keys()):
    stocks = date_groups[d]
    d_tp = len([r for r in stocks if r['status'] == '止盈'])
    d_sl = len([r for r in stocks if r['status'] == '止损'])
    d_win_rate = d_tp / len(stocks) * 100
    d_avg_return = sum(r['total_return'] for r in stocks) / len(stocks)
    print(f'  {d}: {len(stocks)}只, 胜率{d_win_rate:.1f}%, 平均收益{d_avg_return:+.2f}%')
