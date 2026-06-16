# -*- coding: utf-8 -*-
"""
每日股票推荐流水线 - 阶段2优化版（优化止盈策略）

阶段1优化：技术指标筛选（RSI、MACD、均线）
阶段2优化：
  1. 阶梯止盈 - 5%/10%/15%分批止盈
  2. 移动止盈 - 盈利后提高止损线
  3. 动态持仓周期 - 根据板块热度调整
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
from dataclasses import dataclass, field
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher_multi import MultiSourceDataFetcher
from src.stock_tracker import StockTracker
from stock_pool_manager import get_pool_manager, get_stock_sector, SECTOR_WEIGHTS


# ============ 阶段2优化：止盈策略模块 ============

@dataclass
class Position:
    """持仓记录"""
    code: str
    name: str
    buy_price: float
    buy_date: str
    take_profit_levels: List[Dict] = field(default_factory=list)
    stop_loss: float = 0.0
    highest_price: float = 0.0
    partial_sold: List[Dict] = field(default_factory=list)
    sector: str = '其他'
    
    def __post_init__(self):
        if not self.take_profit_levels:
            # 默认阶梯止盈：5%/10%/15%
            self.take_profit_levels = [
                {'level': 1, 'pct': 5.0, 'ratio': 0.3, 'triggered': False},
                {'level': 2, 'pct': 10.0, 'ratio': 0.3, 'triggered': False},
                {'level': 3, 'pct': 15.0, 'ratio': 0.4, 'triggered': False}
            ]
        if self.highest_price == 0.0:
            self.highest_price = self.buy_price


class TrailingStopManager:
    """移动止盈管理器"""
    
    def __init__(self, 
                 trailing_start: float = 5.0,  # 盈利5%后启动移动止盈
                 trailing_pct: float = 3.0,     # 回撤3%触发止盈
                 max_hold_days: int = 14):      # 最大持仓天数
        self.trailing_start = trailing_start
        self.trailing_pct = trailing_pct
        self.max_hold_days = max_hold_days
        self.positions: Dict[str, Position] = {}
    
    def add_position(self, code: str, name: str, buy_price: float, 
                     buy_date: str, sector: str = '其他',
                     custom_levels: Optional[List] = None):
        """添加新持仓"""
        position = Position(
            code=code,
            name=name,
            buy_price=buy_price,
            buy_date=buy_date,
            sector=sector,
            take_profit_levels=custom_levels or [
                {'level': 1, 'pct': 5.0, 'ratio': 0.3, 'triggered': False},
                {'level': 2, 'pct': 10.0, 'ratio': 0.3, 'triggered': False},
                {'level': 3, 'pct': 15.0, 'ratio': 0.4, 'triggered': False}
            ],
            stop_loss=buy_price * 0.97  # 初始止损 -3%
        )
        self.positions[code] = position
        return position
    
    def update_price(self, code: str, current_price: float, current_date: str) -> Dict:
        """更新价格并检查止盈止损信号"""
        if code not in self.positions:
            return {'action': 'none', 'reason': '持仓不存在'}
        
        position = self.positions[code]
        position.highest_price = max(position.highest_price, current_price)
        
        profit_pct = (current_price - position.buy_price) / position.buy_price * 100
        days_held = (datetime.strptime(current_date, '%Y-%m-%d') - 
                    datetime.strptime(position.buy_date, '%Y-%m-%d')).days
        
        result = {
            'code': code,
            'name': position.name,
            'profit_pct': profit_pct,
            'days_held': days_held,
            'action': 'hold',
            'reason': '',
            'sell_ratio': 0.0,
            'sell_price': current_price
        }
        
        # 1. 检查阶梯止盈
        for level in position.take_profit_levels:
            if not level['triggered'] and profit_pct >= level['pct']:
                level['triggered'] = True
                result['action'] = 'partial_sell'
                result['sell_ratio'] = level['ratio']
                result['reason'] = f"阶梯止盈第{level['level']}档 (+{level['pct']}% 卖出{level['ratio']*100:.0f}%)"
                
                position.partial_sold.append({
                    'date': current_date,
                    'price': current_price,
                    'ratio': level['ratio'],
                    'profit_pct': profit_pct
                })
                return result
        
        # 2. 检查移动止盈（盈利超过trailing_start后，回撤trailing_pct触发）
        if profit_pct >= self.trailing_start:
            max_profit_pct = (position.highest_price - position.buy_price) / position.buy_price * 100
            drawdown_pct = (position.highest_price - current_price) / position.highest_price * 100
            
            if drawdown_pct >= self.trailing_pct:
                result['action'] = 'sell'
                result['sell_ratio'] = 1.0
                result['reason'] = f"移动止盈触发 (最高盈利{max_profit_pct:.1f}%, 回撤{drawdown_pct:.1f}%)"
                return result
        
        # 3. 检查止损
        if current_price <= position.stop_loss:
            result['action'] = 'sell'
            result['sell_ratio'] = 1.0
            result['reason'] = f"止损触发 (亏损{(profit_pct):.1f}%)"
            return result
        
        # 4. 检查最大持仓天数（动态调整）
        hold_days = self.get_dynamic_hold_days(position.sector)
        if days_held >= hold_days:
            result['action'] = 'sell'
            result['sell_ratio'] = 1.0
            result['reason'] = f"持仓期满 ({days_held}天 >= {hold_days}天)"
            return result
        
        # 5. 更新移动止损线（盈利后提高止损）
        if profit_pct >= 3.0:
            new_stop = max(position.stop_loss, position.buy_price)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                result['reason'] = f"移动止损线上移至成本价"
        
        if profit_pct >= 8.0:
            new_stop = max(position.stop_loss, position.buy_price * 1.05)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                result['reason'] = f"移动止损线上移至+5%"
        
        return result
    
    def get_dynamic_hold_days(self, sector: str) -> int:
        """根据板块获取动态持仓天数"""
        multipliers = {
            'AI/算力': 1.2,
            '机器人': 1.2,
            '低空经济': 1.1,
            '半导体': 1.1,
            '新能源': 1.0,
            '消费电子': 1.0,
            '医药': 0.9,
            '食品饮料': 0.9,
            '银行': 0.8,
            '其他': 1.0
        }
        return int(self.max_hold_days * multipliers.get(sector, 1.0))
    
    def get_position_summary(self, code: str) -> Dict:
        """获取持仓摘要"""
        if code not in self.positions:
            return {}
        
        position = self.positions[code]
        total_sold_ratio = sum(s['ratio'] for s in position.partial_sold)
        remaining_ratio = 1.0 - total_sold_ratio
        
        return {
            'code': code,
            'name': position.name,
            'buy_price': position.buy_price,
            'highest_price': position.highest_price,
            'current_stop_loss': position.stop_loss,
            'partial_sold': position.partial_sold,
            'remaining_ratio': remaining_ratio,
            'total_sold_ratio': total_sold_ratio
        }
    
    def remove_position(self, code: str):
        """移除持仓"""
        if code in self.positions:
            del self.positions[code]


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
            expected_cols = ['date', 'open', 'close', 'high', 'low', 'volume',
                           'amount', 'amplitude', 'pct_change', 'change', 'turnover']
            
            if len(df.columns) >= 11:
                df.columns = expected_cols[:len(df.columns)]
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
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
    rsi_score = max(0, 25 - abs(rsi - 45) / 3)
    
    # 2. MACD检查
    macd_hist = latest['macd_hist']
    macd_signal = latest['macd_signal']
    
    if pd.isna(macd_hist) or pd.isna(macd_signal):
        return {'valid': False, 'reason': 'MACD计算失败', 'score': 0}
    
    macd_golden_cross = macd_hist > 0
    macd_turning = macd_hist > prev['macd_hist'] and prev['macd_hist'] < 0
    macd_ok = macd_golden_cross or macd_turning
    macd_score = 25 if macd_golden_cross else (20 if macd_turning else 0)
    
    # 3. 成交量检查
    volume_ratio = latest['volume_ratio']
    if pd.isna(volume_ratio):
        volume_ok = True
        volume_score = 15
    else:
        volume_ok = volume_ratio > 0.8
        volume_score = min(25, volume_ratio * 12.5)
    
    # 4. 均线排列
    ma5 = latest['ma5']
    ma10 = latest['ma10']
    ma20 = latest['ma20']
    
    ma_ok = True
    ma_score = 0
    if not pd.isna(ma5) and not pd.isna(ma10):
        ma_ok = ma5 >= ma10
        ma_score = 25 if ma_ok else 0
        if not pd.isna(ma20) and ma5 >= ma10 >= ma20:
            ma_score = 30
    
    total_score = rsi_score + macd_score + volume_score + ma_score
    
    conditions_met = sum([rsi_ok, macd_ok, volume_ok, ma_ok])
    is_valid = conditions_met >= 2 and total_score >= 50
    
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
    
    combined['code_clean'] = combined['code'].str.replace('sh', '').str.replace('sz', '')
    
    combined['change'] = combined.apply(
        lambda r: ((r['price'] - r['last_close']) / r['last_close'] * 100) if r['last_close'] > 0 else 0,
        axis=1
    )
    
    sector_hot = fetch_sector_performance()
    
    combined['sector'] = combined['code_clean'].apply(get_stock_sector)
    combined['sector_weight'] = combined['sector'].map(lambda x: sector_hot.get(x, 1.0))
    
    combined['weighted_score'] = combined['change'] * combined['sector_weight']
    
    filtered = combined[
        (combined['change'] >= 2.0) & 
        (combined['change'] <= 7.0) &
        (combined['volume'] > 1000000)
    ].copy()
    
    filtered = filtered[~filtered['name'].str.contains('ST|退', na=False, case=False)]
    
    # ===== 阶段1优化：增加技术指标筛选 =====
    print("\n🔍 进行技术指标筛选（RSI、MACD、均线）...")
    
    tech_passed = []
    tech_failed = []
    
    for idx, row in filtered.iterrows():
        code = row['code_clean']
        name = row.get('name', '')
        
        hist_df = get_stock_history(code, days=30)
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
        
        time.sleep(0.1)
    
    print(f"\n📊 技术指标筛选结果:")
    print(f"   通过: {len(tech_passed)} 只")
    print(f"   未通过: {len(tech_failed)} 只")
    
    if not tech_passed:
        print("⚠️ 没有股票通过技术指标筛选，返回基础筛选结果")
        filtered = filtered.sort_values('weighted_score', ascending=False)
        return filtered.to_dict('records')
    
    tech_df = pd.DataFrame(tech_passed)
    tech_df = tech_df.sort_values(['tech_score', 'weighted_score'], ascending=[False, False])
    
    sector_counts = tech_df['sector'].value_counts().head(10)
    print(f"\n📈 热门板块分布:")
    for sector, count in sector_counts.items():
        hot_mark = "🔥" if sector in ['AI/算力', '机器人', '低空经济', '半导体'] else ""
        print(f"   {sector}: {count}只 {hot_mark}")
    
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
    """生成推荐列表（阶段2优化：增加阶梯止盈配置）"""
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
            
            tech_score = row.get('tech_score', 50)
            rsi = row.get('rsi', 50)
            volume_ratio = row.get('volume_ratio', 1.0)
            
            # 阶段2优化：计算阶梯止盈价位
            take_profit_levels = [
                {'level': 1, 'pct': 5.0, 'price': round(price * 1.05, 2), 'ratio': 0.3},
                {'level': 2, 'pct': 10.0, 'price': round(price * 1.10, 2), 'ratio': 0.3},
                {'level': 3, 'pct': 15.0, 'price': round(price * 1.15, 2), 'ratio': 0.4}
            ]
            
            stop_loss = price * (1 - sl_pct / 100)
            
            volume_score = min(volume / 10000000, 1.0)
            change_score = min(change / 10, 1.0)
            tech_score_norm = tech_score / 100
            
            # 阶段2优化：增加技术评分权重
            confidence = (0.3 + change_score * 0.2 + volume_score * 0.15 + 
                         sector_weight * 0.1 + tech_score_norm * 0.25)
            
            score = (change * 0.3 + volume_score * 10 * 0.15 + 
                    confidence * 10 * 0.2 + sector_weight * 5 * 0.1 +
                    tech_score_norm * 10 * 0.25)
            
            recommendations.append({
                'code': code,
                'name': name,
                'buy_price': round(price, 2),
                'take_profit': round(take_profit_levels[-1]['price'], 2),  # 最高止盈价
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
                'take_profit_levels': take_profit_levels,  # 阶段2新增
                'reason': f"涨幅{change:.1f}% | 板块{sector} | 技术评分{tech_score:.0f}"
            })
        except Exception as e:
            print(f"处理推荐时出错: {e}")
            continue
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]


# ============ 阶段2优化：更新持仓管理 ============

def update_and_categorize_v2(tracker: StockTracker, fetcher: MultiSourceDataFetcher, 
                             today: str, stop_manager: TrailingStopManager) -> Dict[str, List]:
    """更新并分类持仓股票（阶段2优化：使用移动止盈管理器）"""
    categories = {
        'holding': [],
        'expired': [],
        'take_profit': [],
        'stop_loss': [],
        'partial_sell': []  # 阶段2新增：部分止盈
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
        
        # 阶段2优化：使用移动止盈管理器
        if code not in stop_manager.positions:
            # 初始化持仓到管理器
            sector = info.get('sector', '其他')
            stop_manager.add_position(
                code=code,
                name=info.get('name', ''),
                buy_price=buy_price,
                buy_date=info.get('added_date', today),
                sector=sector
            )
        
        result = stop_manager.update_price(code, current_price, today)
        
        profit_pct = result['profit_pct']
        days_held = result['days_held']
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
            'added_date': info.get('added_date', today),
            'highest_price': stop_manager.positions[code].highest_price,
            'current_stop_loss': stop_manager.positions[code].stop_loss,
            'partial_sold': stop_manager.positions[code].partial_sold
        }
        
        # 根据管理器结果分类
        if result['action'] == 'partial_sell':
            categories['partial_sell'].append({**stock_data, 'sell_reason': result['reason']})
        elif result['action'] == 'sell':
            if profit_pct >= 0:
                categories['take_profit'].append({**stock_data, 'sell_reason': result['reason']})
            else:
                categories['stop_loss'].append({**stock_data, 'sell_reason': result['reason']})
            # 卖出后从管理器移除
            stop_manager.remove_position(code)
        elif result['action'] == 'hold':
            categories['holding'].append(stock_data)
        else:
            # 默认分类（兼容旧逻辑）
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
    """打印报告（阶段2优化：显示阶梯止盈信息）"""
    print("\n" + "=" * 60)
    print(f"📊 每日股票推荐报告 ({today}) - 阶段2优化版")
    print("=" * 60)
    
    # 新增推荐
    if new_recommendations:
        print(f"\n🚀 新增推荐 (Top {len(new_recommendations)}):")
        print("-" * 60)
        for i, rec in enumerate(new_recommendations, 1):
            print(f"{i}. {rec['code']} {rec['name']}")
            print(f"   买入价: {rec['buy_price']:.2f} | 今日涨幅: {rec['change']:+.2f}%")
            
            # 阶段2优化：显示阶梯止盈
            if 'take_profit_levels' in rec:
                levels = rec['take_profit_levels']
                print(f"   阶梯止盈: L1:{levels[0]['price']:.2f}(+5%卖30%) | L2:{levels[1]['price']:.2f}(+10%卖30%) | L3:{levels[2]['price']:.2f}(+15%卖40%)")
            else:
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
            stop_info = f" | 移动止损:{stock.get('current_stop_loss', 0):.2f}" if 'current_stop_loss' in stock else ""
            print(f"{emoji} {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天){stop_info}")
    
    # 部分止盈
    if categories.get('partial_sell'):
        print(f"\n💰 部分止盈 ({len(categories['partial_sell'])}只):")
        print("-" * 60)
        for stock in categories['partial_sell']:
            print(f"🟡 {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% | {stock.get('sell_reason', '')}")
    
    # 已止盈
    if categories['take_profit']:
        print(f"\n✅ 已止盈 ({len(categories['take_profit'])}只):")
        print("-" * 60)
        for stock in categories['take_profit']:
            print(f"🟢 {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天) | {stock.get('sell_reason', '')}")
    
    # 已止损
    if categories['stop_loss']:
        print(f"\n❌ 已止损 ({len(categories['stop_loss'])}只):")
        print("-" * 60)
        for stock in categories['stop_loss']:
            print(f"🔴 {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天) | {stock.get('sell_reason', '')}")
    
    # 已过期
    if categories['expired']:
        print(f"\n⏰ 已过期 ({len(categories['expired'])}只):")
        print("-" * 60)
        for stock in categories['expired']:
            emoji = "🟢" if stock['profit_pct'] >= 0 else "🔴"
            print(f"{emoji} {stock['code']} {stock['name']}: {stock['profit_pct']:+.2f}% ({stock['days_held']}天)")
    
    print("\n" + "=" * 60)


def export_csv(categories: Dict, new_recommendations: List[Dict], today: str):
    """导出CSV"""
    import csv
    
    filename = f"data/recommendations_{today}.csv"
    os.makedirs('data', exist_ok=True)
    
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['代码', '名称', '买入价', '当前价', '收益率%', '持有天数', '状态', '止盈价', '止损价', '技术评分', 'RSI', '卖出原因'])
        
        for rec in new_recommendations:
            writer.writerow([
                rec['code'], rec['name'], rec['buy_price'], rec['buy_price'],
                rec['change'], 0, '新增推荐',
                rec['take_profit'], rec['stop_loss'],
                rec.get('tech_score', ''), rec.get('rsi', ''), ''
            ])
        
        for status, stocks in categories.items():
            for stock in stocks:
                writer.writerow([
                    stock['code'], stock['name'], stock['buy_price'],
                    stock['current_price'], f"{stock['profit_pct']:.2f}",
                    stock['days_held'], status,
                    stock.get('take_profit', ''), stock.get('stop_loss', ''),
                    '', '', stock.get('sell_reason', '')
                ])
    
    print(f"\n💾 CSV已导出: {filename}")


def main():
    """主函数（阶段2优化版）"""
    print("=" * 60)
    print("🚀 每日股票推荐系统 - 阶段2优化版")
    print("   阶段1: 技术指标筛选 (RSI/MACD/均线)")
    print("   阶段2: 优化止盈策略 (阶梯止盈+移动止盈)")
    print("=" * 60)
    
    config = load_config()
    tp_pct = config['data']['take_profit']
    sl_pct = config['data']['stop_loss']
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 初始化组件
    fetcher = MultiSourceDataFetcher()
    tracker = StockTracker()
    manager = get_pool_manager()
    
    # 阶段2优化：初始化移动止盈管理器
    stop_manager = TrailingStopManager(
        trailing_start=5.0,   # 盈利5%后启动移动止盈
        trailing_pct=3.0,      # 回撤3%触发止盈
        max_hold_days=14       # 最大持仓天数
    )
    
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
    
    # 阶段2优化：使用新版持仓管理
    categories = update_and_categorize_v2(tracker, fetcher, today, stop_manager)
    
    # 打印报告
    print_report(categories, new_recommendations, today, tp_pct, sl_pct)
    
    # 导出CSV
    export_csv(categories, new_recommendations, today)
    
    print("\n✅ 每日分析完成！")


if __name__ == "__main__":
    main()
