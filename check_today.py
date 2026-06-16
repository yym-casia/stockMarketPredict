# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, '.')
from src.data_fetcher_multi import MultiSourceDataFetcher
from datetime import datetime
import json
import os
import csv

fetcher = MultiSourceDataFetcher()

# 从跟踪记录获取昨日推荐的股票
TRACKING_FILE = os.path.join('data', 'stock_tracking.json')
stocks = []
try:
    with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for code in data.keys():
            # 转换格式: sh600654 -> 600654, sz002421 -> 002421
            clean_code = code.replace('sh', '').replace('sz', '')
            stocks.append(clean_code)
except Exception as e:
    print(f"读取跟踪记录失败: {e}")

print(f"跟踪的股票: {stocks[:5]}...")

# 获取实时数据
if stocks:
    data = fetcher.get_realtime_quotes(stocks[:20])

today = datetime.now().strftime('%Y-%m-%d')

print(f"\n=== 今日股票推荐 ({today}) ===")
print("="*60)

if not stocks:
    print("暂无跟踪数据")
else:
    # 读取跟踪数据
    with open(TRACKING_FILE, 'r', encoding='utf-8') as f:
        tracking = json.load(f)
    
    for idx, row in data.iterrows():
        code = row.get('code', '')
        name = row.get('name', '?')
        current = row.get('price', 0)
        close = row.get('last_close', 0)
        open_p = row.get('open', 0)
        high = row.get('high', 0)
        low = row.get('low', 0)
        volume = row.get('volume', 0)
        
        # 今日涨跌
        if close > 0:
            change_pct = (current - close) / close * 100
        else:
            change_pct = 0
        
        # 转换代码格式显示
        display_code = code.replace('sh', '').replace('sz', '')
        
        # 查找对应的跟踪记录
        track_info = tracking.get(display_code, {})
        buy_price = track_info.get('buy_price', current)
        take_profit = track_info.get('take_profit', 0)
        stop_loss = track_info.get('stop_loss', 0)
        
        # 计算收益
        profit_pct = (current - buy_price) / buy_price * 100 if buy_price > 0 else 0
        
        print(f"\n【{display_code}】{name}")
        print(f"  买入价: {buy_price:.2f} → 当前: {current:.2f}")
        print(f"  涨跌: {change_pct:+.2f}% | 收益: {profit_pct:+.2f}%")
        print(f"  止盈: {take_profit:.2f} | 止损: {stop_loss:.2f}")
        
        if current >= take_profit > 0:
            print("  🚀 已触及止盈线！")
        elif current <= stop_loss > 0:
            print("  ⚠️ 已触及止损线！")
        
        # 更新跟踪记录的今日数据
        if display_code in tracking:
            tracking[display_code]['daily_prices'][today] = {
                'close': current,
                'change': change_pct,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }
    
    # 保存更新后的跟踪数据
    with open(TRACKING_FILE, 'w', encoding='utf-8') as f:
        json.dump(tracking, f, ensure_ascii=False, indent=2)
    
    # 导出为CSV格式（东方财富兼容格式）
    csv_file = os.path.join('data', f'eastmoney_import_{today}.csv')
    with open(csv_file, 'w', newline='', encoding='gbk') as f:
        writer = csv.writer(f)
        writer.writerow(['代码', '名称'])  # 东方财富格式
        for code in stocks[:20]:
            if code in tracking:
                name = tracking[code].get('name', '')
                writer.writerow([code, name])
    
    print(f"\n📁 CSV已导出: {csv_file}")
    print("💡 导入方法：东方财富APP → 自选 → 更多 → 导入/导出 → 从文件导入")
    
    print("\n" + "="*60)
    print("✅ 跟踪记录已更新")