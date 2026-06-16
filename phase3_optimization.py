# -*- coding: utf-8 -*-
"""
阶段3优化：优化选股时机
核心改进：
1. 大盘情绪判断 - 只在情绪好时入场
2. 板块轮动加速信号 - 捕捉板块启动点
3. 个股资金流向过滤 - 筛选主力资金流入股
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time


class MarketSentimentAnalyzer:
    """大盘情绪分析器"""
    
    def __init__(self):
        self.sentiment_cache = None
        self.cache_time = None
        self.cache_duration = 300  # 缓存5分钟
    
    def get_market_sentiment(self) -> Dict:
        """
        获取大盘情绪评分
        返回: {
            'score': 0-100,
            'level': 'extreme_fear'|'fear'|'neutral'|'greed'|'extreme_greed',
            'signals': ['signal1', 'signal2', ...],
            'recommendation': 'avoid'|'cautious'|'normal'|'aggressive'
        }
        """
        # 检查缓存
        if (self.sentiment_cache and self.cache_time and 
            (datetime.now() - self.cache_time).seconds < self.cache_duration):
            return self.sentiment_cache
        
        try:
            import akshare as ak
            
            signals = []
            score = 50  # 基础分
            
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
            
            # 4. 成交量判断
            try:
                sh_hist = ak.stock_zh_index_daily(symbol="sh000001")
                if sh_hist is not None and len(sh_hist) >= 5:
                    recent_vol = sh_hist['volume'].tail(5).mean()
                    prev_vol = sh_hist['volume'].tail(10).head(5).mean()
                    if recent_vol > prev_vol * 1.2:
                        score += 5
                        signals.append("成交量放大")
                    elif recent_vol < prev_vol * 0.8:
                        score -= 5
                        signals.append("成交量萎缩")
            except Exception as e:
                print(f"  获取成交量失败: {e}")
            
            # 确定情绪等级
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
    """板块轮动探测器"""
    
    def __init__(self):
        self.sector_cache = {}
        self.cache_time = None
    
    def get_sector_momentum(self, top_n: int = 10) -> List[Dict]:
        """
        获取板块动量排名
        返回: [
            {
                'sector': '板块名称',
                'change': 涨跌幅,
                'momentum_score': 动量评分,
                'is_accelerating': 是否加速,
                'recommendation': 'strong_buy'|'buy'|'hold'|'avoid'
            }
        ]
        """
        try:
            import akshare as ak
            
            # 获取板块涨幅排名
            sector_df = ak.stock_board_industry_name_em()
            if sector_df is None or sector_df.empty:
                return []
            
            sectors = []
            for _, row in sector_df.head(top_n).iterrows():
                sector_name = row['板块名称']
                change = row.get('涨跌幅', 0)
                
                # 计算动量评分
                momentum_score = abs(change) * 10
                
                # 判断是否加速（需要历史数据对比，简化处理）
                is_accelerating = change > 3.0
                
                # 推荐等级
                if change > 5.0:
                    recommendation = 'strong_buy'
                elif change > 2.0:
                    recommendation = 'buy'
                elif change > -1.0:
                    recommendation = 'hold'
                else:
                    recommendation = 'avoid'
                
                sectors.append({
                    'sector': sector_name,
                    'change': change,
                    'momentum_score': momentum_score,
                    'is_accelerating': is_accelerating,
                    'recommendation': recommendation
                })
            
            # 按动量评分排序
            sectors.sort(key=lambda x: x['momentum_score'], reverse=True)
            return sectors
            
        except Exception as e:
            print(f"板块轮动分析失败: {e}")
            return []
    
    def is_sector_hot(self, sector: str, threshold: float = 2.0) -> bool:
        """判断板块是否热门"""
        sectors = self.get_sector_momentum(top_n=50)
        for s in sectors:
            if s['sector'] == sector and s['change'] >= threshold:
                return True
        return False


class FundFlowFilter:
    """资金流向过滤器"""
    
    def __init__(self):
        pass
    
    def get_fund_flow(self, stock_code: str) -> Dict:
        """
        获取个股资金流向
        返回: {
            'main_flow': 主力资金流向(万),
            'main_ratio': 主力占比(%),
            'retail_flow': 散户资金流向(万),
            'score': 资金评分(0-100)
        }
        """
        try:
            import akshare as ak
            
            # 获取个股资金流向
            flow_df = ak.stock_individual_fund_flow(stock=stock_code, market="sh")
            if flow_df is None or flow_df.empty:
                # 尝试深市
                flow_df = ak.stock_individual_fund_flow(stock=stock_code, market="sz")
            
            if flow_df is not None and not flow_df.empty:
                latest = flow_df.iloc[-1]
                
                main_inflow = latest.get('主力净流入', 0)
                main_ratio = latest.get('主力净流入占比', 0)
                
                # 计算资金评分
                score = 50
                if main_inflow > 1000:  # 主力流入超1000万
                    score += 30
                elif main_inflow > 0:
                    score += 15
                elif main_inflow < -1000:
                    score -= 30
                elif main_inflow < 0:
                    score -= 15
                
                if main_ratio > 10:  # 主力占比超10%
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
                    'retail_flow': latest.get('散户净流入', 0),
                    'score': score,
                    'is_good': score >= 60  # 资金评分>=60为良好
                }
            
        except Exception as e:
            print(f"  获取{stock_code}资金流向失败: {e}")
        
        return {
            'main_flow': 0,
            'main_ratio': 0,
            'retail_flow': 0,
            'score': 50,
            'is_good': True  # 默认允许
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
                print(f"  ✅ {code} {name}: 资金评分{flow['score']}/100 (主力流入{flow['main_flow']:.0f}万)")
            else:
                print(f"  ❌ {code} {name}: 资金评分{flow['score']}/100 (主力流出或评分不足)")
            
            time.sleep(0.1)
        
        print(f"\n💰 资金流向筛选结果: {len(filtered)}/{len(stocks)} 只通过")
        return filtered


class EntryTimingOptimizer:
    """入场时机优化器 - 综合判断是否应该入场"""
    
    def __init__(self):
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.sector_detector = SectorRotationDetector()
        self.fund_flow_filter = FundFlowFilter()
    
    def should_enter_market(self) -> Tuple[bool, str, Dict]:
        """
        判断是否适合入场
        返回: (should_enter, reason, details)
        """
        # 1. 检查大盘情绪
        sentiment = self.sentiment_analyzer.get_market_sentiment()
        
        if sentiment['recommendation'] == 'avoid':
            return False, f"大盘情绪极差 ({sentiment['level']}, 评分{sentiment['score']})", sentiment
        
        # 2. 检查是否有热门板块
        hot_sectors = self.sector_detector.get_sector_momentum(top_n=5)
        has_hot_sector = any(s['change'] > 2.0 for s in hot_sectors)
        
        if not has_hot_sector:
            return False, "无热门板块，市场缺乏主线", {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors
            }
        
        # 3. 综合判断
        if sentiment['recommendation'] == 'aggressive':
            return True, "市场情绪积极，适合积极入场", {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors
            }
        elif sentiment['recommendation'] == 'normal':
            return True, "市场情绪正常，可以入场", {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors
            }
        else:
            return True, "市场情绪谨慎，控制仓位入场", {
                'sentiment': sentiment,
                'hot_sectors': hot_sectors
            }
    
    def optimize_stock_selection(self, stocks: List[Dict], top_n: int = 10) -> List[Dict]:
        """
        优化股票选择：结合资金流向和板块动量
        """
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
            
            # 综合评分 = 原评分 * 0.5 + 资金评分 * 0.3 + 板块动量 * 0.2
            original_score = stock.get('score', 0)
            fund_score = stock.get('fund_flow_score', 50)
            
            stock['final_score'] = (
                original_score * 0.5 + 
                fund_score * 0.3 + 
                sector_score * 0.2
            )
        
        # 排序并返回
        stocks.sort(key=lambda x: x['final_score'], reverse=True)
        return stocks[:top_n]


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("阶段3优化测试：选股时机优化")
    print("=" * 60)
    
    optimizer = EntryTimingOptimizer()
    
    print("\n📊 测试1：大盘情绪判断")
    print("-" * 60)
    should_enter, reason, details = optimizer.should_enter_market()
    print(f"入场建议: {'✅ 可以入场' if should_enter else '❌ 建议观望'}")
    print(f"原因: {reason}")
    if 'sentiment' in details:
        s = details['sentiment']
        print(f"情绪评分: {s['score']}/100 ({s['level']})")
        print(f"信号: {', '.join(s['signals'])}")
    
    print("\n📊 测试2：板块轮动检测")
    print("-" * 60)
    sectors = optimizer.sector_detector.get_sector_momentum(top_n=5)
    for s in sectors[:5]:
        accel = "🚀" if s['is_accelerating'] else ""
        print(f"{s['sector']}: {s['change']:+.2f}% | 动量{s['momentum_score']:.1f} | {s['recommendation']} {accel}")
    
    print("\n📊 测试3：资金流向筛选")
    print("-" * 60)
    test_stocks = [
        {'code_clean': '000001', 'name': '平安银行', 'score': 80, 'sector': '银行'},
        {'code_clean': '000002', 'name': '万科A', 'score': 75, 'sector': '房地产'},
        {'code_clean': '600519', 'name': '贵州茅台', 'score': 90, 'sector': '食品饮料'}
    ]
    
    filtered = optimizer.fund_flow_filter.filter_by_fund_flow(test_stocks, min_score=50)
    print(f"\n筛选结果: {len(filtered)}/{len(test_stocks)} 只通过")
    
    print("\n✅ 阶段3优化测试完成！")
