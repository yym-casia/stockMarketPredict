# -*- coding: utf-8 -*-
"""
每日股票推荐流水线 - 优化版（阶段1：提高胜率）
主要优化：
1. 增加技术指标过滤（RSI、MACD、均线）
2. 提高入场门槛
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher_multi import MultiSourceDataFetcher
from src.stock_tracker import StockTracker
from stock_pool_manager import get_pool_manager, get_stock_sector, SECTOR_WEIGHTS


# ============ 阶段1优化：技术指标模块 ============

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标（RSI、MACD、成交量）"""
    df = df.copy()
    
    # RSI (14日)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 成交量指标
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # 均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    return df


def get_stock_history(stock_code: str, days: int = 30) -> Optional[pd.DataFrame]:
    """获取股票历史数据用于技术指标计算"""
    try:
        import akshare as ak
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 20)
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code, 
            period="daily",
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            adjust="qfq"
        )
        
        if df is not None and not df.empty and len(df) >= 20:
            # 处理列名（兼容不同版本的akshare）
            expected_cols = ['date', 'open', 'close', 'high', 'low', 'volume',
                           'amount', 'amplitude', 'pct_change', 'change', 'turnover']
            
            if len(df.columns) >= 11:
                df.columns = expected_cols[:len(df.columns)]
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 计算技术指标
            df = calculate_technical_indicators(df)
            return df
            
    except Exception as e:
        print(f"  获取 {stock_code} 历史数据失败: {e}")
    
    return None


def check_technical_signals(stock_code: str, df: pd.DataFrame) -> Dict:
    """检查技术指标信号，返回评分和判断结果"""
    if df is None or len(df) < 20:
        return {'valid': False, 'reason': '数据不足', 'score': 0}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    # 1. RSI检查 (30-60为合理区间)
    rsi = latest['rsi']
    if pd.isna(rsi):
        return {'valid': False, 'reason': 'RSI计算失败', 'score': 0}
    
    rsi_ok = 30 <= rsi <= 60
    rsi_score = max(0, 25 - abs(rsi - 45) / 3)  # 45为最佳，偏离扣分
    
    # 2. MACD检查 (金叉或柱状图转正)
    macd_hist = latest['macd_hist']
    macd_signal = latest['macd_signal']
    
    if pd.isna(macd_hist) or pd.isna(macd_signal):
        return {'valid': False, 'reason': 'MACD计算失败', 'score': 0}
    
    # MACD金叉判断：当前柱状图>0 或 柱状图由负转正
    macd_golden_cross = macd_hist > 0
    macd_turning = macd_hist > prev['macd_hist'] and prev['macd_hist'] < 0
    macd_ok = macd_golden_cross or macd_turning
    macd_score = 25 if macd_golden_cross else (20 if macd_turning else 0)
    
    # 3. 成交量检查 (成交量放大)
    volume_ratio = latest['volume_ratio']
    if pd.isna(volume_ratio):
        volume_ok = True
        volume_score = 15
    else:
        volume_ok = volume_ratio > 0.8
        volume_score = min(25, volume_ratio * 12.5)  # 量比2.0得满分
    
    # 4. 均线排列 (5日线上穿10日线)
    ma5 = latest['ma5']
    ma10 = latest['ma10']
    ma20 = latest['ma20']
    
    ma_ok = True
    ma_score = 0
    if not pd.isna(ma5) and not pd.isna(ma10):
        ma_ok = ma5 >= ma10
        ma_score = 25 if ma_ok else 0
        # 多头排列加分
        if not pd.isna(ma20) and ma5 >= ma10 >= ma20:
            ma_score = 30  # 多头排列额外加分
    
    # 综合评分 (满分100)
    total_score = rsi_score + macd_score + volume_score + ma_score
    
    # 通过条件：至少满足2个主要条件且总分>=50
    conditions_met = sum([rsi_ok, macd_ok, volume_ok, ma_ok])
    is_valid = conditions_met >= 2 and total_score >= 50
    
    # 生成原因说明
    reasons = []
    if rsi_ok: reasons.append(f"RSI:{rsi:.1f}")
    else: reasons.append(f"RSI不佳:{rsi:.1f}")
    
    if macd_ok: reasons.append("MACD金叉" if macd_golden_cross else "MACD转多")
    else: reasons.append("MACD非金叉")
    
    if volume_ok: reasons.append(f"量比:{volume_ratio:.2f}")
    else: reasons.append("量不足")
    
    if ma_ok: reasons.append("均线多头排列" if ma_score > 25 else "均线向上")
    else: reasons.append("均线向下")
    
    return {
        'valid': is_valid,
        'score': min(100, total_score),
        'rsi': rsi,
        'rsi_ok': rsi_ok,
        'macd_ok': macd_ok,
        'macd_golden': macd_golden_cross,
        'volume_ratio': volume_ratio if not pd.isna(volume_ratio) else 1.0,
        'volume_ok': volume_ok,
        'ma_ok': ma_ok,
        'conditions_met': conditions_met,
        'reason': ' | '.join(reasons)
    }


# ============ 原有代码（保持不变） ============

def load_config(config_path: str = 'config/config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def fetch_sector_performance() -> Dict[str, float]:
    """获取板块表现数据"""
    try:
        import akshare as ak
        sector_df = ak.stock_board_industry_name_em()
        if sector_df is not None and not sector_df.empty:
            return sector_df.set_index('板块名称')['涨跌幅'].to_dict()
    except Exception as e:
        print(f"获取板块数据失败: {e}")
    return SECTOR_WEIGHTS


def fetch_and_screen_stocks(fetcher: MultiSourceDataFetcher, candidate_pool: List[str]) -> List[Dict]:
    """获取并筛选候选股票，增加技术指标过滤（阶段1优化）"""
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
    
    # 基础筛选条件：涨幅 2%-7%，成交量充足
    filtered = combined[
        (combined['change'] >= 2.0) & 
        (combined['change'] <= 7.0) &
        (combined['volume'] > 1000000)
    ].copy()
    
    # 排除ST股
    filtered = filtered[~filtered['name'].str.contains('ST|退', na=False, case=False)]
    
    # ===== 阶段1优化：增加技术指标筛选 =====
    print("\n🔍 进行技术指标筛选（RSI、MACD、均线）...")
    
    tech_passed = []
    tech_failed = []
    
    for idx, row in filtered.iterrows():
        code = row['code_clean']
        name = row.get('name', '')
        
        # 获取历史数据
        hist_df = get_stock_history(code, days=30)
        
        # 检查技术指标
        signals = check_technical_signals(code, hist_df)
        
        if signals['valid']:
            row_dict = row.to_dict()
            row_dict['tech_score'] = signals['score']
            row_dict['rsi'] = signals['rsi']
            row_dict['volume_ratio'] = signals['volume_ratio']
            row_dict['tech_reason'] = signals['reason']
            tech_passed.append(row_dict)
            print(f"  ✅ {code} {name}: 通过 (评分:{signals['score']:.0f}/100)")
        else:
            tech_failed.append({
                'code': code,
                'name': name,
                'reason': signals['reason'],
                'score': signals['score']
            })
            print(f"  ❌ {code} {name}: 跳过 ({signals['reason']})")
        
        # 避免请求过快
        time.sleep(0.1)
    
    print(f"\n📊 技术指标筛选结果:")
    print(f"   通过: {len(tech_passed)} 只")
    print(f"   未通过: {len(tech_failed)} 只")
    
    if not tech_passed:
        print("⚠️ 没有股票通过技术指标筛选，返回基础筛选结果")
        filtered = filtered.sort_values('weighted_score', ascending=False)
        return filtered.to_dict('records')
    
    # 按技术指标评分排序
    tech_df = pd.DataFrame(tech_passed)
    tech_df = tech_df.sort_values(['tech_score', 'weighted_score'], ascending=[False, False])
    
    # 打印板块分布
    sector_counts = tech_df['sector'].value_counts().head(10)
    print(f"\n📈 热门板块分布:")
    for sector, count in sector_counts.items():
        hot_mark = "🔥" if sector in ['AI/算力', '机器人', '低空经济', '半导体'] else ""
        print(f"   {sector}: {count}只 {hot_mark}")
    
    # 打印技术指标统计
    print(f"\n📊 技术指标统计:")
    print(f"   平均RSI: {tech_df['rsi'].mean():.1f}")
    print(f"   平均量比: {tech_df['volume_ratio'].mean():.2f}")
    print(f"   平均技术评分: {tech_df['tech_score'].mean():.1f}")
    
    print(f"\n✅ 最终筛选: {len(tech_df)} 只股票")
    return tech_df.to_dict('records')


def generate_recommendations(
    screened_stocks: List[Dict],
    tp_pct: float = 5.0,
    sl_pct: float = 3.0,
    top_n: int = 10
) -> List[Dict]:
    """生成推荐列表"""
    recommendations = []
    
    for row in screened_stocks[:top_n * 2]:
        try:
            code = row.get('code_clean', row.get('code', ''))
            name = row.get('name', '')
            price = row.get('price', 0)
            last_close = row.get('last_close', price)
            change = row.get('change', ((price - last_close) / last_close * 100) if last_close > 0 else 0)
            volume = row.get('volume', 0)
            sector = row.get('sector', '其他')
            sector_weight = row.get('sector_weight', 1.0)
            
            # 获取技术指标（如果有）
            tech_score = row.get('tech_score', 50)
            rsi = row.get('rsi', 50)
            volume_ratio = row.get('volume_ratio', 1.0)
            
            # 计算止盈止损
            take_profit = price * (1 + tp_pct / 100)
            stop_loss = price * (1 - sl_pct / 100)
            
            # 综合评分（加入技术指标权重）
            volume_score = min(volume / 10000000, 1.0)
            change_score = min(change / 10, 1.0)
            tech_score_norm = tech_score / 100  # 归一化
            
            # 阶段1优化：增加技术指标权重
            confidence = (0.3 + change_score * 0.2 + volume_score * 0.15 + 
                         sector_weight * 0.1 + tech_score_norm * 0.25)
            
            score = (change * 0.3 + volume_score * 10 * 0.15 + 
                    confidence * 10 * 0.2 + sector_weight * 5 * 0.1 +
                    tech_score_norm * 10 * 0.25)
            
            recommendations.append({
                'code': code,
                'name': name,
                'buy_price': round(price, 2),
                'take_profit': round(take_profit, 2),
                'stop_loss': round(stop_loss, 2),
                'take_profit_pct': tp_pct,
                'stop_loss_pct': sl_pct,
                'change': round(change, 2),
                'volume': int(volume),
                'sector': sector,
                'confidence': round(confidence, 4),
                'score': round(score, 2),
                'tech_score': round(tech_score, 1),
                'rsi': round(rsi, 1),
                'volume_ratio': round(volume_ratio, 2),
                'strategy': 'short_term',
                'reason': f"涨幅{change:.1f}% | 板块{sector} | 技术评分{tech_score:.0f}"
            })
        except Exception as e:
            print(f"处理推荐时出错: {e}")
            continue
    
    # 按评分排序
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]


def update_and_categorize(tracker: StockTracker, fetcher: MultiSourceDataFetcher, today: str) -> Dict[str, List]:
    """更新并分类持仓股票"""
    categories = {
        'holding': [],
        'expired': [],
        'take_profit': [],
        'stop_loss': []
    }
    
    tracked_stocks = tracker.get_tracked_stocks()
    
    if not tracked_stocks:
        return categories
    
    codes = list(tracked_stocks.keys())
    
    try:
        price_data = fetcher.get_realtime_quotes(codes)
        if price_data is None or price_data.empty:
            return categories
    except Exception as e:
        print(f"获取价格数据失败: {e}")
        return categories
    
    for code, info in tracked_stocks.items():
        try:
            price_info = price_data[price_data['code'].str.contains(code)].iloc[0].to_dict()
        except (IndexError, KeyError):
            continue
        
        current_price = price_info.get('price', 0)
        buy_price = info.get('buy_price', current_price)
        
        if buy_price <= 0:
            continue
        
        profit_pct = (current_price - buy_price) / buy_price * 100
        days_held = (datetime.strptime(today, '%Y-%m-%d') - 
                    datetime.strptime(info.get('added_date', today), '%Y-%m-%d')).days
        
        current_change = price_info.get('change', 0)
        
        stock_data = {
            'code': code,
            'name': info.get('name', ''),
            'buy_price': buy_price,
            'current_price': current_price,
            'profit_pct': profit_pct,
            'days_held': days_held,
            'current_change': current_change,
            'take_profit': info.get('take_profit', 0),
            'stop_loss': info.get('stop_loss', 0),
            'added_date': info.get('added_date', today)
        }
        
        if profit_pct >= 8:
            categories['take_profit'].append(stock_data)
        elif profit_pct <= -3:
            categories['stop_loss'].append(stock_data)
        elif days_held >= 14:
            categories['expired'].append(stock_data)
        else:
            categories['holding'].append(stock_data)
    
    return categories


def print_report(categories: Dict, new_recommendations: List[Dict], today: str, tp_pct: float, sl_pct: float):
    """打印报告"""
    print("\n" + "=" * 60)
    print(f"📊 每日股票推荐报告 ({today})")
    print("=" * 60)
    
    # 新增推荐
    if new_recommendations:
        print(f"\n🚀 新增推荐 (Top {len(new_recommendations)}):")
        print("-" * 60)
        for i, rec in enumerate(new_recommendations, 1):
            print(f"{i}. {rec['code']} {rec['name']}")
            print(f"   买入价: {rec['buy_price']:.2f} | 今日涨幅: {rec['change']:+.2f}%")
            print(f"   止盈: {rec['take_profit']:.2f} (+{tp_pct}%) | 止损: {rec['stop_loss']:.2f} (-{sl_pct}%)")
            print(f"   技术评分: {rec.get('tech_score', 'N/A')}/100 | RSI: {rec.get('rsi', 'N/A')}")
            print(f"   置信度: {rec['confidence']*100:.1f}% | 板块: {rec['sector']}")
            print()
    
    # 持仓中
    if categories['holding']:
        print(f"\n📈 持仓中 ({len(categories['holding'])}只):")
        print("-" * 60)
        for stock in categories['holding']:
            emoji = "🟢" if stock['profit_pct'] > 0 else "🔴"
            print(f"{emoji} {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天)")
    
    # 已过期
    if categories['expired']:
        print(f"\n⏰ 已过期 ({len(categories['expired'])}只):")
        print("-" * 60)
        for stock in categories['expired']:
            emoji = "🟢" if stock['profit_pct'] >= 0 else "🔴"
            print(f"{emoji} {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天)")
    
    # 已止盈
    if categories['take_profit']:
        print(f"\n✅ 已止盈 ({len(categories['take_profit'])}只):")
        print("-" * 60)
        for stock in categories['take_profit']:
            print(f"🟢 {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天)")
    
    # 已止损
    if categories['stop_loss']:
        print(f"\n❌ 已止损 ({len(categories['stop_loss'])}只):")
        print("-" * 60)
        for stock in categories['stop_loss']:
            print(f"🔴 {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天)")
    
    print("\n" + "=" * 60)


def export_csv(categories: Dict, new_recommendations: List[Dict], today: str):
    """导出CSV"""
    import csv
    
    filename = f"data/recommendations_{today}.csv"
    os.makedirs('data', exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['代码', '名称', '买入价', '当前价', '收益率%', '持有天数', '状态', '止盈价', '止损价', '技术评分', 'RSI'])
        
        for rec in new_recommendations:
            writer.writerow([
                rec['code'], rec['name'], rec['buy_price'], rec['buy_price'],
                rec['change'], 0, '新增推荐',
                rec['take_profit'], rec['stop_loss'],
                rec.get('tech_score', ''), rec.get('rsi', '')
            ])
        
        for status, stocks in categories.items():
            for stock in stocks:
                writer.writerow([
                    stock['code'], stock['name'], stock['buy_price'],
                    stock['current_price'], f"{stock['profit_pct']:.2f}",
                    stock['days_held'], status,
                    stock.get('take_profit', ''), stock.get('stop_loss', ''),
                    '', ''
                ])
    
    print(f"\n💾 CSV已导出: {filename}")


def main():
    """主函数"""
    print("=" * 60)
    print("🚀 每日股票推荐系统 - 优化版（阶段1：提高胜率）")
    print("=" * 60)
    
    config = load_config()
    tp_pct = config['data']['take_profit']
    sl_pct = config['data']['stop_loss']
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 初始化组件
    fetcher = MultiSourceDataFetcher()
    tracker = StockTracker()
    manager = get_pool_manager()
    
    # 获取股票池
    stock_pool = manager.get_all_stocks()
    print(f"\n📊 股票池数量: {len(stock_pool)} 只")
    
    # 获取并筛选股票
    screened = fetch_and_screen_stocks(fetcher, stock_pool)
    
    if not screened:
        print("\n❌ 未找到符合条件的股票")
        return
    
    # 生成推荐
    new_recommendations = generate_recommendations(screened, tp_pct, sl_pct)
    
    # 更新持仓状态
    categories = update_and_categorize(tracker, fetcher, today)
    
    # 打印报告
    print_report(categories, new_recommendations, today, tp_pct, sl_pct)
    
    # 导出CSV
    export_csv(categories, new_recommendations, today)
    
    print("\n✅ 每日分析完成！")


if __name__ == "__main__":
    main()
