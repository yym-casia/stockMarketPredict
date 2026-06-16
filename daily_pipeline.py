# -*- coding: utf-8 -*-
"""
每日股票推荐流水线
- 生成新的每日推荐（5只）
- 更新跟踪股票价格
- 分类显示：新推荐 / 持有中 / 已止盈 / 已止损
"""

import sys
import os
import json
import yaml
import warnings
import csv
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher_multi import MultiSourceDataFetcher
from src.stock_tracker import StockTracker
from stock_pool_manager import get_pool_manager, StockPoolManager

# 初始化股票池管理器
_pool_manager = get_pool_manager('data/stock_pool.json')
CANDIDATE_STOCKS = _pool_manager.get_all_stocks()


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fetch_sector_performance() -> Dict[str, float]:
    """获取板块涨幅数据"""
    # 使用股票池管理器的权重
    manager = get_pool_manager()
    return manager.SECTOR_WEIGHTS


def get_stock_sector(code: str) -> str:
    """根据股票代码判断所属板块"""
    manager = get_pool_manager()
    return manager.get_stock_sector(code)


def fetch_and_screen_stocks(fetcher: MultiSourceDataFetcher, candidate_pool: List[str]) -> List[Dict]:
    """获取并筛选候选股票，增加板块轮动权重"""
    print("\n📊 正在获取候选股票数据...")
    
    all_data = []
    batch_size = 30
    
    for i in range(0, len(candidate_pool), batch_size):
        batch = candidate_pool[i:i+batch_size]
        print(f"  批次 {i//batch_size + 1}: {len(batch)} 只")
        
        try:
            df = fetcher.get_realtime_quotes(batch)
            if df is not None and not df.empty:
                all_data.append(df)
        except Exception as e:
            print(f"  批次获取失败: {e}")
    
    if not all_data:
        return []
    
    combined = pd.concat(all_data, ignore_index=True) if len(all_data) > 1 else all_data[0]
    
    # 清理代码格式
    combined['code_clean'] = combined['code'].str.replace('sh', '').str.replace('sz', '')
    
    # 计算涨跌幅
    combined['change'] = combined.apply(
        lambda r: ((r['price'] - r['last_close']) / r['last_close'] * 100) if r['last_close'] > 0 else 0,
        axis=1
    )
    
    # 获取板块热度
    sector_hot = fetch_sector_performance()
    
    # 添加板块信息并计算板块权重得分
    combined['sector'] = combined['code_clean'].apply(get_stock_sector)
    combined['sector_weight'] = combined['sector'].map(lambda x: sector_hot.get(x, 1.0))
    
    # 综合得分 = 涨幅 * 板块权重
    combined['weighted_score'] = combined['change'] * combined['sector_weight']
    
    # 筛选条件：涨幅 2%-7%，成交量充足
    filtered = combined[
        (combined['change'] >= 2.0) & 
        (combined['change'] <= 7.0) &
        (combined['volume'] > 1000000)
    ].copy()
    
    # 排除ST股
    filtered = filtered[~filtered['name'].str.contains('ST|退', na=False, case=False)]
    
    # 按综合得分排序（优先热门板块）
    filtered = filtered.sort_values('weighted_score', ascending=False)
    
    # 打印板块分布
    sector_counts = filtered['sector'].value_counts().head(10)
    print(f"\n📈 热门板块分布:")
    for sector, count in sector_counts.items():
        hot_mark = "🔥" if sector in ['AI/算力', '机器人', '低空经济', '半导体'] else ""
        print(f"   {sector}: {count}只 {hot_mark}")
    
    print(f"\n✅ 筛选后剩余 {len(filtered)} 只股票")
    return filtered.to_dict('records')


def generate_recommendations(
    candidates: List[Dict], 
    daily_picks: int = 5,
    take_profit_pct: float = 5.0,
    stop_loss_pct: float = 3.0,
    min_confidence: float = 0.85
) -> List[Dict]:
    """生成推荐列表"""
    if not candidates:
        return []
    
    recommendations = []
    
    for row in candidates[:daily_picks * 3]:  # 扩大筛选池
        code = str(row.get('code', '')).replace('sh', '').replace('sz', '')
        name = row.get('name', '?')
        price = row.get('price', 0)
        last_close = row.get('last_close', 0)
        change = row.get('change', ((price - last_close) / last_close * 100) if last_close > 0 else 0)
        volume = row.get('volume', 0)
        
        if price <= 0:
            continue
        
        # 获取板块信息
        sector = get_stock_sector(code)
        sector_hot = fetch_sector_performance()
        sector_weight = sector_hot.get(sector, 1.0)
        
        # 计算置信度（基于涨幅、成交量、板块热度）
        volume_score = min(volume / 10000000, 1.0)  # 成交量归一化
        change_score = min(change / 10, 1.0)  # 涨幅归一化
        sector_score = (sector_weight - 1.0) / 0.3  # 板块热度归一化
        confidence = 0.4 + change_score * 0.25 + volume_score * 0.2 + sector_score * 0.15
        
        # 过滤低置信度
        if confidence < min_confidence:
            continue
        
        take_profit = price * (1 + take_profit_pct / 100)
        stop_loss = price * (1 - stop_loss_pct / 100)
        
        # 综合评分（加入板块权重）
        score = change * 0.4 + volume_score * 10 * 0.25 + confidence * 10 * 0.2 + sector_weight * 5 * 0.15
        
        recommendations.append({
            'code': code,
            'name': name,
            'buy_price': round(price, 2),
            'change': round(change, 2),
            'volume': int(volume),
            'take_profit': round(take_profit, 2),
            'stop_loss': round(stop_loss, 2),
            'score': round(score, 2),
            'strategy': '短线(5天)',
            'confidence': round(confidence, 3),
            'expected_return': take_profit_pct,
            'sector': sector,
            'sector_weight': round(sector_weight, 2),
        })
    
    # 按评分排序，取前 daily_picks
    recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:daily_picks]
    
    print(f"✅ 生成 {len(recommendations)} 只推荐股票")
    return recommendations


def update_and_categorize(tracker: StockTracker, fetcher: MultiSourceDataFetcher, today: str) -> Dict[str, List]:
    """更新跟踪股票价格并分类"""
    print("\n🔄 正在更新跟踪股票...")
    
    tracked = tracker.get_tracked_stocks()
    if not tracked:
        return {'new': [], 'holding': [], 'take_profit': [], 'stop_loss': [], 'expired': []}
    
    codes = list(tracked.keys())
    
    # 批量获取实时价格
    prices_data = {}
    for i in range(0, len(codes), 50):
        batch = codes[i:i+50]
        try:
            df = fetcher.get_realtime_quotes(batch)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    code = str(row.get('code', '')).replace('sh', '').replace('sz', '')
                    prices_data[code] = {
                        'price': row.get('price', 0),
                        'change': row.get('change', 0),
                        'name': row.get('name', tracked.get(code, {}).get('name', '?'))
                    }
        except Exception as e:
            print(f"  获取价格失败: {e}")
    
    categories = {'new': [], 'holding': [], 'take_profit': [], 'stop_loss': [], 'expired': []}
    
    for code, info in tracked.items():
        added_date = info.get('added_date', today)
        buy_price = info.get('buy_price', 0)
        take_profit = info.get('take_profit', 0)
        stop_loss = info.get('stop_loss', 0)
        
        # 计算持仓天数
        try:
            added = datetime.strptime(added_date, '%Y-%m-%d')
            days_held = (datetime.now() - added).days
        except:
            days_held = 0
        
        # 获取当前价格
        price_info = prices_data.get(code, {})
        current_price = price_info.get('price', 0)
        current_change = price_info.get('change', 0)
        name = price_info.get('name', info.get('name', '?'))
        
        # 更新跟踪数据
        if current_price > 0:
            info['daily_prices'][today] = {
                'close': current_price,
                'change': current_change,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }
        
        # 计算收益
        profit_pct = ((current_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        
        stock_status = {
            'code': code,
            'name': name,
            'buy_price': buy_price,
            'current_price': current_price,
            'profit_pct': round(profit_pct, 2),
            'added_date': added_date,
            'days_held': days_held,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
        }
        
        # 分类
        if added_date == today:
            categories['new'].append(stock_status)
        elif days_held >= 14:
            categories['expired'].append(stock_status)
        elif current_price >= take_profit > 0:
            categories['take_profit'].append(stock_status)
        elif current_price <= stop_loss > 0:
            categories['stop_loss'].append(stock_status)
        else:
            categories['holding'].append(stock_status)
    
    # 保存更新后的跟踪数据
    tracker._save(tracked)
    
    print(f"  新推荐: {len(categories['new'])}, 持有中: {len(categories['holding'])}, "
          f"止盈: {len(categories['take_profit'])}, 止损: {len(categories['stop_loss'])}")
    # 已过期股票保留在跟踪数据中，日常不显示
    if categories['expired']:
        print(f"  (已过期: {len(categories['expired'])} 只 - 保留在跟踪数据中)")
    
    return categories


def print_report(categories: Dict, new_recommendations: List[Dict], today: str, tp_pct: float, sl_pct: float):
    """打印分类报告"""
    print("\n" + "="*80)
    print(f"📊 每日股票推荐报告 ({today})")
    print("="*80)
    
    # 1. 今日新推荐
    print("\n" + "🆕"*15)
    print("【今日新推荐】")
    print("🆕"*15)
    if new_recommendations:
        for i, rec in enumerate(new_recommendations, 1):
            print(f"\n{i}. 【{rec['code']}】{rec['name']}")
            print(f"   买入价: {rec['buy_price']:.2f} | 今日涨幅: {rec['change']:+.2f}%")
            print(f"   板块: {rec.get('sector', '未知')} {'🔥' if rec.get('sector_weight', 1.0) > 1.1 else ''}")
            print(f"   止盈价: {rec['take_profit']:.2f} (+{tp_pct:.0f}%)")
            print(f"   止损价: {rec['stop_loss']:.2f} (-{sl_pct:.0f}%)")
            print(f"   策略: {rec['strategy']} | 置信度: {rec['confidence']*100:.1f}%")
    else:
        print("   今日无新推荐")
    
    # 2. 持有中
    print("\n" + "📈"*15)
    print("【持有中】")
    print("📈"*15)
    if categories['holding']:
        for stock in categories['holding']:
            emoji = "📊" if stock['profit_pct'] >= 0 else "📉"
            print(f"\n{emoji} 【{stock['code']}】{stock['name']}")
            print(f"   买入: {stock['buy_price']:.2f} → 当前: {stock['current_price']:.2f}")
            print(f"   收益: {stock['profit_pct']:+.2f}% | 持仓: {stock['days_held']}天")
            print(f"   止盈: {stock['take_profit']:.2f} | 止损: {stock['stop_loss']:.2f}")
    else:
        print("   暂无持有中股票")
    
    # 3. 已止盈
    print("\n" + "🚀"*15)
    print("【已止盈 - 建议卖出】")
    print("🚀"*15)
    if categories['take_profit']:
        for stock in categories['take_profit']:
            print(f"\n🎯 【{stock['code']}】{stock['name']}")
            print(f"   买入: {stock['buy_price']:.2f} → 当前: {stock['current_price']:.2f}")
            print(f"   收益: {stock['profit_pct']:+.2f}% 🎉 | 持仓: {stock['days_held']}天")
    else:
        print("   暂无止盈股票")
    
    # 4. 已止损
    print("\n" + "⚠️"*15)
    print("【已止损 - 建议卖出】")
    print("⚠️"*15)
    if categories['stop_loss']:
        for stock in categories['stop_loss']:
            print(f"\n⛔ 【{stock['code']}】{stock['name']}")
            print(f"   买入: {stock['buy_price']:.2f} → 当前: {stock['current_price']:.2f}")
            print(f"   收益: {stock['profit_pct']:+.2f}% | 持仓: {stock['days_held']}天")
    else:
        print("   暂无止损股票")
    
    # 5. 已过期（日常不显示，保留在跟踪数据中供回测使用）
    # if categories['expired']:
    #     print("\n" + "⏰"*15)
    #     print("【已过期(14天) - 建议卖出】")
    #     print("⏰"*15)
    #     for stock in categories['expired']:
    #         emoji = "🟢" if stock['profit_pct'] >= 0 else "🔴"
    #         print(f"\n{emoji} 【{stock['code']}】{stock['name']}")
    #         print(f"   买入: {stock['buy_price']:.2f} → 当前: {stock['current_price']:.2f}")
    #         print(f"   收益: {stock['profit_pct']:+.2f}% | 持仓: {stock['days_held']}天")
    
    # 汇总
    print("\n" + "="*80)
    print("📋 汇总")
    print("="*80)
    # 日常显示不包含已过期股票（过期股票保留在跟踪数据中供回测使用）
    total = len(new_recommendations) + len(categories['holding']) + len(categories['take_profit']) + len(categories['stop_loss'])
    print(f"   新推荐: {len(new_recommendations)} 只")
    print(f"   持有中: {len(categories['holding'])} 只")
    print(f"   已止盈: {len(categories['take_profit'])} 只")
    print(f"   已止损: {len(categories['stop_loss'])} 只")
    # print(f"   已过期: {len(categories['expired'])} 只")  # 日常不显示
    print(f"   总计: {total} 只")
    print("="*80)


def export_csv(categories: Dict, new_recommendations: List[Dict], today: str):
    """导出CSV供东方财富导入"""
    csv_file = os.path.join('data', f'eastmoney_import_{today}.csv')
    
    all_codes = []
    # 新推荐
    for rec in new_recommendations:
        all_codes.append((rec['code'], rec['name']))
    # 持有中
    for stock in categories['holding']:
        all_codes.append((stock['code'], stock['name']))
    
    with open(csv_file, 'w', newline='', encoding='gbk') as f:
        writer = csv.writer(f)
        writer.writerow(['代码', '名称'])
        for code, name in all_codes:
            writer.writerow([code, name])
    
    print(f"\n📁 CSV已导出: {csv_file}")
    print("💡 导入方法：东方财富APP → 自选 → 更多 → 导入/导出 → 从文件导入")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🚀 每日股票推荐流水线启动")
    print("="*80)
    
    # 打印股票池信息
    manager = get_pool_manager()
    manager.print_pool_info()
    
    # 加载配置
    config = load_config()
    daily_picks = config['stock_selection']['daily_picks']
    take_profit_pct = config['data']['take_profit']
    stop_loss_pct = config['data']['stop_loss']
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 初始化
    fetcher = MultiSourceDataFetcher()
    tracker = StockTracker()
    
    # 1. 获取候选股票
    candidates = fetch_and_screen_stocks(fetcher, CANDIDATE_STOCKS)
    
    # 2. 生成新推荐
    new_recommendations = generate_recommendations(
        candidates, daily_picks, take_profit_pct, stop_loss_pct, 
        config['data'].get('min_confidence', 0.85)
    )
    
    # 3. 添加新推荐到跟踪
    if new_recommendations:
        tracker.add_recommendations(new_recommendations, today)
        print(f"✅ 已添加 {len(new_recommendations)} 只新推荐")
    
    # 4. 更新跟踪股票并分类
    categories = update_and_categorize(tracker, fetcher, today)
    
    # 5. 打印报告
    print_report(categories, new_recommendations, today, take_profit_pct, stop_loss_pct)
    
    # 6. 导出CSV
    export_csv(categories, new_recommendations, today)
    
    print("\n✅ 流水线执行完成")


if __name__ == "__main__":
    main()
