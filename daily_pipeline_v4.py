# -*- coding: utf-8 -*-
"""
每日股票推荐流水线 - 阶段3优化版（优化选股时机）

阶段1优化：技术指标筛选（RSI、MACD、均线）
阶段2优化：优化止盈策略（阶梯止盈+移动止盈+动态持仓）
阶段3优化：
  1. 大盘情绪判断 - 只在情绪好时入场
  2. 板块轮动加速信号 - 捕捉板块启动点
  3. 个股资金流向过滤 - 筛选主力资金流入股
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
from src.strategy_filters import load_strategy_config, screen_stock_row, rank_candidates
from src.market_regime import apply_mainline_to_candidate
from src.doubler_filters import apply_doubler_boost
from src.sector_service import build_regime_from_sectors, get_live_sector_boards
from src.limit_up_filters import (
    load_limit_up_config, get_limit_up_exit_params, limit_up_threshold, screen_limit_up_realtime,
    merge_strategy_candidates,
)
from src.doubler_filters import (
    load_doubler_config, get_doubler_exit_params, screen_doubler_realtime,
    eval_doubler_pattern, _independent_enabled,
)
from stock_pool_manager import get_pool_manager, get_screening_pool


# ============ 阶段3优化：选股时机模块 ============

class MarketSentimentAnalyzer:
    """大盘情绪分析器"""
    
    def __init__(self):
        self.sentiment_cache = None
        self.cache_time = None
        self.cache_duration = 300  # 缓存5分钟
    
    def get_market_sentiment(self) -> Dict:
        """获取大盘情绪评分"""
        if (self.sentiment_cache and self.cache_time and 
            (datetime.now() - self.cache_time).seconds < self.cache_duration):
            return self.sentiment_cache
        
        try:
            import akshare as ak
            
            signals = []
            score = 50
            
            # 1. 上证指数涨跌
            try:
                sh_index = ak.stock_zh_index_spot_sina()
                if sh_index is not None and not sh_index.empty:
                    sh_change = sh_index[sh_index['名称'] == '上证指数']['涨跌幅'].values
                    if len(sh_change) > 0:
                        sh_change = float(sh_change[0])
                        if sh_change > 1.5:
                            score += 15
                            signals.append(f"上证大涨+{sh_change:.1f}%")
                        elif sh_change > 0.5:
                            score += 5
                            signals.append(f"上证上涨+{sh_change:.1f}%")
                        elif sh_change < -1.5:
                            score -= 20
                            signals.append(f"上证大跌{sh_change:.1f}%")
                        elif sh_change < -0.5:
                            score -= 10
                            signals.append(f"上证下跌{sh_change:.1f}%")
            except Exception as e:
                print(f"  获取上证指数失败: {e}")
            
            # 2. 涨跌家数比
            try:
                market_breadth = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
                if market_breadth is not None and not market_breadth.empty:
                    zt_count = len(market_breadth)
                    if zt_count > 100:
                        score += 10
                        signals.append(f"涨停{zt_count}家，情绪高涨")
                    elif zt_count > 50:
                        score += 5
                        signals.append(f"涨停{zt_count}家，情绪偏暖")
                    elif zt_count < 20:
                        score -= 10
                        signals.append(f"涨停{zt_count}家，情绪低迷")
            except Exception as e:
                print(f"  获取涨停数据失败: {e}")
            
            # 3. 北向资金流向
            try:
                north_flow = ak.stock_hsgt_hist_em(symbol="北向资金")
                if north_flow is not None and not north_flow.empty:
                    latest_flow = north_flow.iloc[-1]['当日资金流入']
                    if latest_flow > 50:
                        score += 10
                        signals.append(f"北向大幅流入+{latest_flow:.0f}亿")
                    elif latest_flow > 0:
                        score += 5
                        signals.append(f"北向流入+{latest_flow:.0f}亿")
                    elif latest_flow < -30:
                        score -= 15
                        signals.append(f"北向大幅流出{latest_flow:.0f}亿")
            except Exception as e:
                print(f"  获取北向资金失败: {e}")
            
            score = max(0, min(100, score))
            
            if score >= 80:
                level = 'extreme_greed'
                recommendation = 'aggressive'
            elif score >= 60:
                level = 'greed'
                recommendation = 'normal'
            elif score >= 40:
                level = 'neutral'
                recommendation = 'cautious'
            elif score >= 20:
                level = 'fear'
                recommendation = 'avoid'
            else:
                level = 'extreme_fear'
                recommendation = 'avoid'
            
            result = {
                'score': score,
                'level': level,
                'signals': signals,
                'recommendation': recommendation
            }
            
            self.sentiment_cache = result
            self.cache_time = datetime.now()
            return result
            
        except Exception as e:
            print(f"大盘情绪分析失败: {e}")
            return {
                'score': 50,
                'level': 'neutral',
                'signals': ['数据获取失败，使用默认中性评分'],
                'recommendation': 'cautious'
            }


class SectorRotationDetector:
    """板块轮动探测器（委托 sector_service，与回测命名一致）"""

    def __init__(self):
        self.sector_cache = {}
        self.cache_time = None

    def get_sector_momentum(self, top_n: int = 10) -> List[Dict]:
        from src.sector_service import get_live_sector_boards
        sectors = get_live_sector_boards(top_n=top_n)
        result = []
        for s in sectors:
            change = s['change']
            if change > 5.0:
                recommendation = 'strong_buy'
            elif change > 2.0:
                recommendation = 'buy'
            elif change > -1.0:
                recommendation = 'hold'
            else:
                recommendation = 'avoid'
            result.append({
                'sector': s['sector'],
                'change': change,
                'momentum_score': abs(change) * 10,
                'is_accelerating': change > 3.0,
                'recommendation': recommendation,
            })
        return result

    def is_sector_hot(self, sector: str, threshold: float = 2.0) -> bool:
        sectors = self.get_sector_momentum(top_n=50)
        for s in sectors:
            if s['sector'] == sector and s['change'] >= threshold:
                return True
        return False


class FundFlowFilter:
    """资金流向过滤器"""
    
    def get_fund_flow(self, stock_code: str, max_retries: int = 3) -> Dict:
        """获取个股资金流向（带重试）"""
        import time
        
        for attempt in range(max_retries):
            try:
                import akshare as ak
                
                # 获取个股资金流向
                flow_df = ak.stock_individual_fund_flow(stock=stock_code, market="sh")
                if flow_df is None or flow_df.empty:
                    flow_df = ak.stock_individual_fund_flow(stock=stock_code, market="sz")
                
                if flow_df is not None and not flow_df.empty:
                    latest = flow_df.iloc[-1]
                    
                    main_inflow = latest.get('主力净流入', 0)
                    main_ratio = latest.get('主力净流入占比', 0)
                    
                    score = 50
                    if main_inflow > 1000:
                        score += 30
                    elif main_inflow > 0:
                        score += 15
                    elif main_inflow < -1000:
                        score -= 30
                    elif main_inflow < 0:
                        score -= 15
                    
                    if main_ratio > 10:
                        score += 20
                    elif main_ratio > 5:
                        score += 10
                    elif main_ratio < -10:
                        score -= 20
                    elif main_ratio < -5:
                        score -= 10
                    
                    score = max(0, min(100, score))
                    
                    return {
                        'main_flow': main_inflow,
                        'main_ratio': main_ratio,
                        'score': score,
                        'is_good': score >= 50
                    }
                
                return {
                    'main_flow': 0, 'main_ratio': 0, 'score': 50, 'is_good': True
                }
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                print(f"  获取{stock_code}资金流向失败: {e}")
        
        return {
            'main_flow': 0,
            'main_ratio': 0,
            'score': 50,
            'is_good': True
        }
    
    def filter_by_fund_flow(self, stocks: List[Dict], min_score: int = 50) -> List[Dict]:
        """根据资金流向过滤股票"""
        filtered = []
        
        print("\n💰 进行资金流向筛选...")
        for stock in stocks:
            code = stock.get('code_clean', stock.get('code', ''))
            name = stock.get('name', '')
            
            flow = self.get_fund_flow(code)
            
            if flow['is_good'] and flow['score'] >= min_score:
                stock['fund_flow_score'] = flow['score']
                stock['main_flow'] = flow['main_flow']
                stock['main_ratio'] = flow['main_ratio']
                filtered.append(stock)
                print(f"  ✅ {code} {name}: 资金评分{flow['score']}/100")
            else:
                print(f"  ❌ {code} {name}: 资金评分{flow['score']}/100 (不足)")
            
            time.sleep(0.1)
        
        print(f"\n💰 资金流向筛选结果: {len(filtered)}/{len(stocks)} 只通过")
        return filtered


class EntryTimingOptimizer:
    """入场时机优化器"""
    
    def __init__(self):
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.sector_detector = SectorRotationDetector()
        self.fund_flow_filter = FundFlowFilter()
    
    def should_enter_market(self) -> Tuple[bool, str, Dict]:
        """判断是否适合入场"""
        sentiment = self.sentiment_analyzer.get_market_sentiment()
        
        if sentiment['recommendation'] == 'avoid':
            return False, f"大盘情绪极差 ({sentiment['level']}, 评分{sentiment['score']})", sentiment
        
        hot_sectors = self.sector_detector.get_sector_momentum(top_n=5)
        has_hot_sector = any(s['change'] > 1.0 for s in hot_sectors)
        mainline_names = [s['sector'] for s in hot_sectors[:3] if s['change'] > 0.5]

        if sentiment['score'] < 50 and not has_hot_sector:
            return False, "大盘偏弱且无主线板块，建议观望", {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors,
                'mainlines': mainline_names,
            }

        if sentiment['recommendation'] == 'aggressive':
            msg = "市场情绪积极，适合积极入场"
            if mainline_names:
                msg += f" | 主线: {','.join(mainline_names)}"
            return True, msg, {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors,
                'mainlines': mainline_names,
            }
        elif sentiment['recommendation'] == 'normal':
            msg = "市场情绪正常，可以入场"
            if mainline_names:
                msg += f" | 主线: {','.join(mainline_names)}"
            return True, msg, {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors,
                'mainlines': mainline_names,
            }
        else:
            msg = "市场情绪谨慎，控制仓位入场"
            if mainline_names:
                msg += f" | 主线: {','.join(mainline_names)}"
            return True, msg, {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors,
                'mainlines': mainline_names,
            }
    
    def optimize_stock_selection(self, stocks: List[Dict], top_n: int = 10) -> List[Dict]:
        """优化股票选择：结合资金流向和板块动量"""
        # 1. 资金流向过滤
        stocks = self.fund_flow_filter.filter_by_fund_flow(stocks, min_score=50)
        
        if not stocks:
            return []
        
        # 2. 获取板块动量
        sector_momentum = self.sector_detector.get_sector_momentum(top_n=20)
        sector_scores = {s['sector']: s['momentum_score'] for s in sector_momentum}
        
        # 3. 综合评分排序
        for stock in stocks:
            sector = stock.get('sector', '其他')
            sector_score = sector_scores.get(sector, 0)
            
            original_score = stock.get('score', 0)
            fund_score = stock.get('fund_flow_score', 50)
            tech_score = stock.get('tech_score', 50)
            
            # 综合评分 = 原评分 * 0.4 + 技术评分 * 0.2 + 资金评分 * 0.25 + 板块动量 * 0.15
            stock['final_score'] = (
                original_score * 0.4 + 
                tech_score * 0.2 +
                fund_score * 0.25 + 
                sector_score * 0.15
            )
        
        stocks.sort(key=lambda x: x['final_score'], reverse=True)
        return stocks[:top_n]


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
                 trailing_start: float = 5.0,
                 trailing_pct: float = 3.0,
                 max_hold_days: int = 14):
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
            stop_loss=buy_price * 0.97
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
        
        # 2. 检查移动止盈
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
        
        # 4. 检查最大持仓天数
        hold_days = self.get_dynamic_hold_days(position.sector)
        if days_held >= hold_days:
            result['action'] = 'sell'
            result['sell_ratio'] = 1.0
            result['reason'] = f"持仓期满 ({days_held}天 >= {hold_days}天)"
            return result
        
        # 5. 更新移动止损线
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
    
    def remove_position(self, code: str):
        """移除持仓"""
        if code in self.positions:
            del self.positions[code]


# ============ 阶段1优化：技术指标模块 ============

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    df = df.copy()
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    df['ma20'] = df['close'].rolling(window=20).mean()
    
    return df


def _append_realtime_to_hist(hist_df: pd.DataFrame, price: float, volume: float,
                             change: float, open_px: float = None) -> pd.DataFrame:
    """将实时行情并入历史序列，供 screen_stock_row 与回测一致地评估。"""
    hist = hist_df.copy()
    today = pd.Timestamp(datetime.now().date())
    open_px = open_px if open_px else price
    if len(hist) > 0 and pd.Timestamp(hist.index[-1]).normalize() >= today.normalize():
        hist.iloc[-1, hist.columns.get_loc('close')] = price
        if 'open' in hist.columns:
            hist.iloc[-1, hist.columns.get_loc('open')] = open_px
        if 'high' in hist.columns:
            hist.iloc[-1, hist.columns.get_loc('high')] = max(
                float(hist.iloc[-1]['high']), price)
        if 'low' in hist.columns:
            hist.iloc[-1, hist.columns.get_loc('low')] = min(
                float(hist.iloc[-1]['low']), price)
        if 'volume' in hist.columns:
            hist.iloc[-1, hist.columns.get_loc('volume')] = volume
    else:
        base = hist.iloc[-1] if len(hist) > 0 else None
        hist.loc[today] = {
            'open': open_px,
            'high': price,
            'low': price,
            'close': price,
            'volume': volume,
            'ma5': base['ma5'] if base is not None else price,
            'ma10': base['ma10'] if base is not None else price,
            'ma20': base['ma20'] if base is not None else price,
            'rsi': base['rsi'] if base is not None else 50,
            'macd_hist': base['macd_hist'] if base is not None else 0,
            'volume_ratio': base['volume_ratio'] if base is not None else 1,
        }
    hist['pct_change'] = hist['close'].pct_change() * 100
    if len(hist) > 0:
        hist.iloc[-1, hist.columns.get_loc('pct_change')] = change
    return calculate_technical_indicators(hist)


def get_stock_history(stock_code: str, days: int = 30, max_retries: int = 3) -> Optional[pd.DataFrame]:
    """获取股票历史数据（腾讯财经优先，带缓存）"""
    from src.history_fetcher import get_history_fetcher

    fetch_days = max(days + 20, 60)
    for attempt in range(max_retries):
        df = get_history_fetcher().get_history(stock_code, days=fetch_days)
        if df is not None and len(df) >= 20:
            return calculate_technical_indicators(df)
        if attempt < max_retries - 1:
            time.sleep(1)
    print(f"  获取 {stock_code} 历史数据失败")
    return None


def check_technical_signals(stock_code: str, df: pd.DataFrame) -> Dict:
    """检查技术指标信号"""
    if df is None or len(df) < 20:
        return {'valid': False, 'reason': '数据不足', 'score': 0}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    
    rsi = latest['rsi']
    if pd.isna(rsi):
        return {'valid': False, 'reason': 'RSI计算失败', 'score': 0}
    
    rsi_ok = 30 <= rsi <= 60
    rsi_score = max(0, 25 - abs(rsi - 45) / 3)
    
    macd_hist = latest['macd_hist']
    macd_signal = latest['macd_signal']
    
    if pd.isna(macd_hist) or pd.isna(macd_signal):
        return {'valid': False, 'reason': 'MACD计算失败', 'score': 0}
    
    macd_golden_cross = macd_hist > 0
    macd_turning = macd_hist > prev['macd_hist'] and prev['macd_hist'] < 0
    macd_ok = macd_golden_cross or macd_turning
    macd_score = 25 if macd_golden_cross else (20 if macd_turning else 0)
    
    volume_ratio = latest['volume_ratio']
    if pd.isna(volume_ratio):
        volume_ok = True
        volume_score = 15
    else:
        volume_ok = volume_ratio > 0.8
        volume_score = min(25, volume_ratio * 12.5)
    
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
    """获取板块表现数据（东方财富行业板块，与回测一致）"""
    try:
        from src.sector_service import get_live_sector_changes
        changes = get_live_sector_changes()
        if changes:
            return changes
    except Exception as e:
        print(f"获取板块数据失败: {e}")
    pool_manager = get_pool_manager()
    return getattr(pool_manager, 'SECTOR_WEIGHTS', {})


def fetch_and_screen_stocks(fetcher: MultiSourceDataFetcher, candidate_pool: List[str],
                            force_screen: bool = False) -> List[Dict]:
    """获取并筛选候选股票；force_screen=True 时跳过大盘情绪门禁"""
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
    
    pool_manager = get_pool_manager()
    combined['sector'] = combined['code_clean'].apply(pool_manager.get_stock_sector)
    combined['sector_weight'] = combined['sector'].map(lambda x: sector_hot.get(x, 1.0))
    
    combined['weighted_score'] = combined['change'] * combined['sector_weight']
    
    cfg_full = load_config()
    strat_cfg = load_strategy_config(cfg_full)
    db_cfg = load_doubler_config(cfg_full)

    pool_df = combined.copy()
    if strat_cfg.get('require_hot_sector', False):
        from src.sector_service import get_live_sector_boards
        top_n = int(strat_cfg.get('hot_sector_top_n', strat_cfg.get('mainline_top_n', 5)))
        live_sectors = get_live_sector_boards(top_n=top_n)
        hot_sectors = {s['sector'] for s in live_sectors}
        if hot_sectors:
            before = len(pool_df)
            pool_df = pool_df[pool_df['sector'].isin(hot_sectors)].copy()
            sector_detail = pool_df['sector'].value_counts()
            detail_str = ', '.join(
                f'{name}({cnt})' for name, cnt in sector_detail.items()
            )
            names = ','.join(s['sector'] for s in live_sectors[:top_n])
            print(f"\n🔥 热门板块预筛 Top{top_n}: {before} -> {len(pool_df)} 只 | 板块: {names}")
            if detail_str:
                print(f"   各板块股票数: {detail_str}")
            else:
                print("   各板块股票数: (无匹配股票)")

    filtered = pool_df[
        (pool_df['change'] >= strat_cfg['min_change']) &
        (pool_df['change'] <= strat_cfg['max_change']) &
        (pool_df['volume'] > 1000000)
    ].copy()
    
    filtered = filtered[~filtered['name'].str.contains('ST|退', na=False, case=False)]

    # 趋势筛选：与回测共用 screen_stock_row + apply_mainline + rank_candidates
    print("\n🔍 趋势选股（与回测一致: 涨幅5-10% | 热门板块 | 技术+ML评分）...")

    top_n_boards = max(
        int(strat_cfg.get('hot_sector_top_n', 5)),
        int(strat_cfg.get('mainline_top_n', 5)),
    )
    live_regime = build_regime_from_sectors(
        get_live_sector_boards(top_n=top_n_boards * 4), strat_cfg)

    ml_scorer = None
    try:
        from src.ml_scorer import get_ml_scorer
        ml_scorer = get_ml_scorer(load_config())
    except Exception:
        pass

    trend_candidates = []
    for idx, row in filtered.iterrows():
        code = row['code_clean']
        name = row.get('name', '')
        price = float(row['price'])
        volume = float(row.get('volume', 0))
        change = float(row['change'])

        hist_df = get_stock_history(code, days=80)
        if hist_df is None or len(hist_df) < 20:
            continue
        hist_df = _append_realtime_to_hist(
            hist_df, price, volume, change, open_px=row.get('open', price))

        screen_row = pd.Series({
            'code': code,
            'close': price,
            'price': price,
            'volume': volume,
            'change': change,
            'pct_change': change,
        })
        ml_proba = None
        if ml_scorer and ml_scorer.ready:
            ml_proba = ml_scorer.score_stock_code(code)

        hit = screen_stock_row(screen_row, hist_df, strat_cfg, ml_proba=ml_proba)
        if not hit:
            continue

        hit['sector'] = row.get('sector', '其他')
        hit = apply_doubler_boost(hit, screen_row, hist_df, db_cfg)
        hit = apply_mainline_to_candidate(hit, live_regime, strat_cfg)
        if hit is None:
            continue

        row_dict = row.to_dict()
        row_dict.update(hit)
        row_dict['code_clean'] = code
        row_dict['tech_score'] = hit['tech_score']
        row_dict['tech_reason'] = hit.get('signals', {}).get('reason', '')
        row_dict['strategy_type'] = 'trend'
        trend_candidates.append(row_dict)
        ml_tag = f" ML:{hit['ml_score']:.2f}" if hit.get('ml_score') else ''
        print(f"  ✅ {code} {name}: 综合{hit['score']:.1f} 技术{hit['tech_score']:.0f}{ml_tag}")
        time.sleep(0.1)

    print(f"\n📊 趋势筛选通过: {len(trend_candidates)} 只")

    print("\n🌍 大盘情绪检查...")
    sentiment_analyzer = MarketSentimentAnalyzer()
    market_sentiment = sentiment_analyzer.get_market_sentiment()
    sentiment_score = market_sentiment['score']
    print(f"   情绪评分: {sentiment_score:.1f}/100 ({market_sentiment.get('level', '')})")

    market_min = strat_cfg.get('market_min_score', 45)
    if sentiment_score < market_min and not force_screen:
        print(f"   情绪低于{market_min}，跳过新买入")
        return []
    if sentiment_score < market_min and force_screen:
        print(f"   情绪{sentiment_score:.0f}<{market_min}，强制输出筛选结果")

    daily_picks = int(load_config().get('stock_selection', {}).get('daily_picks', 5))
    ranked = rank_candidates(trend_candidates, top_n=max(daily_picks * 2, 10))
    print(f"\n🏆 综合排名 Top{min(10, len(ranked))}:")
    for i, r in enumerate(ranked[:10], 1):
        print(f"   {i}. {r.get('code', r.get('code_clean'))} 综合{r['score']:.1f} "
              f"涨幅{r.get('change', 0):+.1f}%")

    trend_records = ranked
    lu_cfg = load_limit_up_config(load_config())
    if not lu_cfg.get('enabled') and not _independent_enabled(db_cfg):
        return trend_records

    limit_up_records = []
    if lu_cfg.get('enabled'):
        print("\n[涨停板首板] 筛选 30日内首板 + 量能3倍...")
        for idx, row in combined.iterrows():
            code = row['code_clean']
            name = row.get('name', '')
            if 'ST' in str(name).upper() or '退' in str(name):
                continue
            th = limit_up_threshold(code, name)
            if row['change'] < th - 0.5:
                continue
            hist_df = get_stock_history(code, days=lu_cfg.get('lookback_days', 30) + 10)
            hit = screen_limit_up_realtime(row.to_dict(), hist_df, lu_cfg)
            if hit:
                hit['composite_score'] = hit.get('score', 80)
                hit['sector_momentum'] = hit.get('sector_momentum', 50)
                hit['fund_flow_score'] = hit.get('fund_flow_score', 50)
                limit_up_records.append(hit)
                print(f"  🔥 {code} {name}: {hit['tech_reason']}")
        print(f"   涨停首板通过: {len(limit_up_records)} 只")

    doubler_records = []
    if _independent_enabled(db_cfg):
        print("\n[翻倍模式] 筛选放量突破+动量启动...")
        for idx, row in combined.iterrows():
            code = row['code_clean']
            name = row.get('name', '')
            if 'ST' in str(name).upper() or '退' in str(name):
                continue
            hist_df = get_stock_history(code, days=80)
            hit = screen_doubler_realtime(row.to_dict(), hist_df, db_cfg)
            if hit:
                hit['composite_score'] = hit.get('score', 75)
                hit['sector_momentum'] = hit.get('sector_momentum', 50)
                hit['fund_flow_score'] = hit.get('fund_flow_score', 50)
                doubler_records.append(hit)
                print(f"  📈 {code} {name}: {hit['tech_reason']}")
        print(f"   翻倍模式通过: {len(doubler_records)} 只")

    merged = merge_strategy_candidates(
        trend_records, limit_up_records,
        trend_slots=18,
        limit_up_slots=lu_cfg.get('max_daily_picks', 2) if lu_cfg.get('enabled') else 0,
        doubler_list=doubler_records,
        doubler_slots=db_cfg.get('max_daily_picks', 1) if _independent_enabled(db_cfg) else 0,
    )
    return merged


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
            
            tech_score = row.get('tech_score', 50)
            rsi = row.get('rsi', 50)
            volume_ratio = row.get('volume_ratio', 1.0)
            fund_score = row.get('fund_flow_score', 50)
            
            cfg_full = load_config()
            strat_cfg = load_strategy_config(cfg_full)
            tp_cfg = strat_cfg.get('take_profit_levels', [])
            stype = row.get('strategy_type', 'trend')
            lu_exit = get_limit_up_exit_params(cfg_full) if stype == 'limit_up' else None
            db_exit = get_doubler_exit_params(cfg_full) if stype == 'doubler' else None
            exit_cfg = lu_exit or db_exit
            trailing_start = strat_cfg.get('trailing_start', 15.0)
            trailing_pct = strat_cfg.get('trailing_pct', 6.0)
            if exit_cfg:
                tp_lvls = exit_cfg.get('take_profit_levels', [])
                take_profit_levels = [
                    {'level': i + 1, 'pct': lv['pct'],
                     'price': round(price * (1 + lv['pct'] / 100), 2), 'ratio': lv['ratio']}
                    for i, lv in enumerate(tp_lvls)
                ] if tp_lvls else []
                stop_loss = price * (1 - exit_cfg['stop_loss_pct'] / 100)
                sl_pct = exit_cfg['stop_loss_pct']
                trailing_start = exit_cfg.get('trailing_start', trailing_start)
                trailing_pct = exit_cfg.get('trailing_pct', trailing_pct)
            else:
                take_profit_levels = [
                    {'level': i + 1, 'pct': lv['pct'],
                     'price': round(price * (1 + lv['pct'] / 100), 2), 'ratio': lv['ratio']}
                    for i, lv in enumerate(tp_cfg)
                ] if tp_cfg else []
                sl_pct = strat_cfg.get('stop_loss', sl_pct)
                stop_loss = price * (1 - sl_pct / 100)
            
            volume_score = min(volume / 10000000, 1.0)
            change_score = min(change / 10, 1.0)
            tech_score_norm = tech_score / 100
            fund_score_norm = fund_score / 100
            
            # 阶段3优化：增加资金评分权重
            ml_score = row.get('ml_score', 0)
            if ml_score > 0:
                confidence = ml_score
            elif row.get('score'):
                confidence = row['score'] / 100
            else:
                confidence = (0.25 + change_score * 0.15 + volume_score * 0.1 +
                             sector_weight * 0.1 + tech_score_norm * 0.2 +
                             fund_score_norm * 0.2)

            score = row.get('score') if row.get('score') else (
                change * 0.25 + volume_score * 10 * 0.1 +
                confidence * 10 * 0.2 + sector_weight * 5 * 0.1 +
                tech_score_norm * 10 * 0.2 + fund_score_norm * 10 * 0.15
            )
            
            recommendations.append({
                'code': code,
                'name': name,
                'buy_price': round(price, 2),
                'take_profit': (
                    round(take_profit_levels[-1]['price'], 2) if take_profit_levels
                    else round(price * (1 + trailing_start / 100), 2)
                ),
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
                'fund_score': round(fund_score, 1),
                'strategy': 'short_term',
                'take_profit_levels': take_profit_levels,
                'strategy_type': stype,
                'trailing_start': trailing_start,
                'trailing_pct': trailing_pct,
                'shakeout_rebuy': row.get('shakeout_rebuy', False),
                'yang_days': row.get('yang_days', 0),
                'reason': (
                    f"[止损后{row.get('yang_days', 0)}连阳接回] "
                    f"{(row.get('signals') or {}).get('reason', '')}"
                    if row.get('shakeout_rebuy')
                    else f"[涨停首板] {row.get('tech_reason', '')} | 最高>15%后收益回落≥最高收益10%止盈"
                    if stype == 'limit_up'
                    else f"[翻倍模式] {row.get('tech_reason', '')}"
                    if stype == 'doubler'
                    else f"涨幅{change:.1f}% | 板块{sector} | 技术{tech_score:.0f} | 资金{fund_score:.0f}"
                )
            })
        except Exception as e:
            print(f"处理推荐时出错: {e}")
            continue
    
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return recommendations[:top_n]


def update_and_categorize_v2(tracker: StockTracker, fetcher: MultiSourceDataFetcher, 
                             today: str, stop_manager: TrailingStopManager) -> Dict[str, List]:
    """更新并分类持仓股票"""
    categories = {
        'holding': [],
        'expired': [],
        'take_profit': [],
        'stop_loss': [],
        'partial_sell': []
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
        
        if code not in stop_manager.positions:
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
        
        if result['action'] == 'partial_sell':
            categories['partial_sell'].append({**stock_data, 'sell_reason': result['reason']})
        elif result['action'] == 'sell':
            if profit_pct >= 0:
                categories['take_profit'].append({**stock_data, 'sell_reason': result['reason']})
            else:
                categories['stop_loss'].append({**stock_data, 'sell_reason': result['reason']})
            stop_manager.remove_position(code)
        elif result['action'] == 'hold':
            categories['holding'].append(stock_data)
        else:
            if profit_pct >= 8:
                categories['take_profit'].append(stock_data)
            elif profit_pct <= -3:
                categories['stop_loss'].append(stock_data)
            elif days_held >= 14:
                categories['expired'].append(stock_data)
            else:
                categories['holding'].append(stock_data)

    # 同步 partial_sold / current_stop_loss / sell_reason 回 tracker
    data = tracker._load()
    for code in categories['partial_sell']:
        if code['code'] in data:
            pos = stop_manager.positions.get(code['code'])
            if pos:
                data[code['code']]['partial_sold'] = pos.partial_sold
                data[code['code']]['current_stop_loss'] = pos.stop_loss
                data[code['code']]['sell_reason'] = code.get('sell_reason', '部分止盈')
            else:
                data[code['code']]['partial_sold'] = code.get('partial_sold', [])
                data[code['code']]['sell_reason'] = code.get('sell_reason', '部分止盈')
    # 同步持有中的移动止损
    for stock_data in categories['holding']:
        c = stock_data['code']
        if c in data:
            pos = stop_manager.positions.get(c)
            if pos:
                data[c]['current_stop_loss'] = pos.stop_loss
    tracker._save(data)

    return categories


def print_report(categories: Dict, new_recommendations: List[Dict], today: str, 
                 tp_pct: float, sl_pct: float, sentiment: Optional[Dict] = None):
    """打印报告"""
    print("\n" + "=" * 60)
    print(f"📊 每日股票推荐报告 ({today}) - 阶段3优化版")
    print("=" * 60)
    
    # 显示大盘情绪
    if sentiment:
        print(f"\n🌍 大盘情绪: {sentiment.get('level', 'unknown')} ({sentiment.get('score', 0)}/100)")
        print(f"   信号: {', '.join(sentiment.get('signals', []))}")
    
    # 新增推荐
    if new_recommendations:
        print(f"\n🚀 新增推荐 (Top {len(new_recommendations)}):")
        print("-" * 60)
        for i, rec in enumerate(new_recommendations, 1):
            print(f"{i}. {rec['code']} {rec['name']}")
            print(f"   买入价: {rec['buy_price']:.2f} | 今日涨幅: {rec['change']:+.2f}%")
            
            if 'take_profit_levels' in rec:
                levels = rec['take_profit_levels']
                print(f"   阶梯止盈: L1:{levels[0]['price']:.2f}(+5%卖30%) | L2:{levels[1]['price']:.2f}(+10%卖30%) | L3:{levels[2]['price']:.2f}(+15%卖40%)")
            else:
                print(f"   止盈: {rec['take_profit']:.2f} (+{tp_pct}%) | 止损: {rec['stop_loss']:.2f} (-{sl_pct}%)")
            
            print(f"   技术评分: {rec.get('tech_score', 'N/A')}/100 | RSI: {rec.get('rsi', 'N/A')}")
            print(f"   资金评分: {rec.get('fund_score', 'N/A')}/100")
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
        writer.writerow(['代码', '名称', '买入价', '当前价', '收益率%', '持有天数', '状态', '止盈价', '止损价', '技术评分', 'RSI', '资金评分', '卖出原因'])
        
        for rec in new_recommendations:
            writer.writerow([
                rec['code'], rec['name'], rec['buy_price'], rec['buy_price'],
                rec['change'], 0, '新增推荐',
                rec['take_profit'], rec['stop_loss'],
                rec.get('tech_score', ''), rec.get('rsi', ''), rec.get('fund_score', ''), ''
            ])
        
        for status, stocks in categories.items():
            for stock in stocks:
                writer.writerow([
                    stock['code'], stock['name'], stock['buy_price'],
                    stock['current_price'], f"{stock['profit_pct']:.2f}",
                    stock['days_held'], status,
                    stock.get('take_profit', ''), stock.get('stop_loss', ''),
                    '', '', '', stock.get('sell_reason', '')
                ])
    
    print(f"\n💾 CSV已导出: {filename}")


def main():
    """主函数（阶段3优化版）"""
    print("=" * 60)
    print("🚀 每日股票推荐系统 - 阶段3优化版")
    print("   阶段1: 技术指标筛选 (RSI/MACD/均线)")
    print("   阶段2: 优化止盈策略 (阶梯止盈+移动止盈)")
    print("   阶段3: 优化选股时机 (大盘情绪+板块轮动+资金流向)")
    print("=" * 60)
    
    config = load_config()
    tp_pct = config['data']['take_profit']
    sl_pct = config['data']['stop_loss']
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 初始化组件
    fetcher = MultiSourceDataFetcher()
    tracker = StockTracker()
    get_pool_manager()
    
    # 阶段2优化：初始化移动止盈管理器
    stop_manager = TrailingStopManager(
        trailing_start=5.0,
        trailing_pct=3.0,
        max_hold_days=14
    )
    
    # 阶段3优化：初始化入场时机优化器
    timing_optimizer = EntryTimingOptimizer()
    
    # 阶段3优化：判断入场时机
    print("\n🌍 分析大盘情绪和市场环境...")
    should_enter, reason, market_info = timing_optimizer.should_enter_market()
    
    print(f"\n{'='*60}")
    print(f"📊 入场建议: {'✅ 可以入场' if should_enter else '❌ 建议观望'}")
    print(f"📝 原因: {reason}")
    if 'sentiment' in market_info:
        s = market_info['sentiment']
        print(f"📈 情绪评分: {s['score']}/100 ({s['level']})")
        print(f"📢 信号: {', '.join(s['signals'])}")
    print(f"{'='*60}")
    
    if not should_enter:
        print("\n⚠️ 当前市场环境不佳，建议观望")
        return
    
    # 获取股票池（与回测统一）
    stock_pool = get_screening_pool(config)
    print(f"\n📊 股票池数量: {len(stock_pool)} 只")
    
    # 获取并筛选股票
    screened = fetch_and_screen_stocks(fetcher, stock_pool)
    
    if not screened:
        print("\n❌ 未找到符合条件的股票")
        return
    
    # 阶段3优化：使用入场时机优化器优化选股
    print("\n🎯 使用阶段3优化进行最终选股...")
    optimized_stocks = timing_optimizer.optimize_stock_selection(screened, top_n=20)
    
    if not optimized_stocks:
        print("\n❌ 没有股票通过资金流向筛选")
        return
    
    # 生成推荐
    new_recommendations = generate_recommendations(optimized_stocks, tp_pct, sl_pct)

    # 将新推荐写入 tracking
    if new_recommendations:
        tracker.add_recommendations(new_recommendations, today)
        print(f"\n✅ 已写入 {len(new_recommendations)} 只新推荐到 stock_tracking.json")

    # 阶段2优化：使用新版持仓管理
    categories = update_and_categorize_v2(tracker, fetcher, today, stop_manager)
    
    # 打印报告
    sentiment = market_info.get('sentiment') if isinstance(market_info, dict) else None
    print_report(categories, new_recommendations, today, tp_pct, sl_pct, sentiment)
    
    # 导出CSV
    export_csv(categories, new_recommendations, today)
    
    # 打印历史记录（排除活跃持仓）
    print("\n" + "=" * 60)
    print("📜 历史记录（已清仓/过期）")
    print("=" * 60)
    history = tracker.get_history_summary(limit=20, exclude_active=True)
    if history:
        for item in history:
            emoji = "🟢" if item['total_change'] >= 0 else "🔴"
            print(f"{emoji} {item['code']} {item['name']}: {item['total_change']:+.2f}% ({item['holding_days']}天)")
    else:
        print("暂无历史记录")
    
    print("\n✅ 每日分析完成！")


if __name__ == "__main__":
    main()
