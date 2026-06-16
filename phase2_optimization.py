# -*- coding: utf-8 -*-
"""
阶段2优化：优化止盈策略
核心改进：
1. 阶梯止盈 - 5%/10%/15%分批止盈
2. 移动止盈 - 盈利后提高止损线
3. 动态持仓周期 - 根据板块热度调整
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json


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
                     buy_date: str, custom_levels: Optional[List] = None):
        """添加新持仓"""
        position = Position(
            code=code,
            name=name,
            buy_price=buy_price,
            buy_date=buy_date,
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
                
                # 记录部分卖出
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
        
        # 4. 检查最大持仓天数
        if days_held >= self.max_hold_days:
            result['action'] = 'sell'
            result['sell_ratio'] = 1.0
            result['reason'] = f"持仓期满 ({days_held}天)"
            return result
        
        # 5. 更新移动止损线（盈利后提高止损）
        if profit_pct >= 3.0:
            # 盈利3%后，止损线上移至成本价
            new_stop = max(position.stop_loss, position.buy_price)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                result['reason'] = f"移动止损线上移至成本价"
        
        if profit_pct >= 8.0:
            # 盈利8%后，止损线上移至盈利5%
            new_stop = max(position.stop_loss, position.buy_price * 1.05)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                result['reason'] = f"移动止损线上移至+5%"
        
        return result
    
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


class DynamicHoldingPeriod:
    """动态持仓周期管理"""
    
    def __init__(self, base_days: int = 14):
        self.base_days = base_days
        self.sector_multipliers = {
            'AI/算力': 1.2,      # 热门板块持仓稍长
            '机器人': 1.2,
            '低空经济': 1.1,
            '半导体': 1.1,
            '新能源': 1.0,
            '消费电子': 1.0,
            '医药': 0.9,         # 防御性板块持仓稍短
            '食品饮料': 0.9,
            '银行': 0.8,         # 低波动板块持仓更短
            '其他': 1.0
        }
    
    def get_hold_days(self, sector: str, market_sentiment: str = 'neutral') -> int:
        """根据板块和市场情绪计算持仓天数"""
        multiplier = self.sector_multipliers.get(sector, 1.0)
        
        # 市场情绪调整
        sentiment_multipliers = {
            'bullish': 1.3,    # 牛市持仓更长
            'neutral': 1.0,
            'bearish': 0.7     # 熊市持仓更短
        }
        
        sentiment_mult = sentiment_multipliers.get(market_sentiment, 1.0)
        
        return int(self.base_days * multiplier * sentiment_mult)


# 测试代码
if __name__ == "__main__":
    print("=" * 60)
    print("阶段2优化测试：止盈策略")
    print("=" * 60)
    
    # 测试移动止盈
    manager = TrailingStopManager(trailing_start=5.0, trailing_pct=3.0)
    
    # 添加测试持仓
    manager.add_position('000001', '平安银行', 10.0, '2026-04-01')
    
    print("\n📊 测试场景1：阶梯止盈")
    print("-" * 60)
    
    # 模拟价格上涨
    test_prices = [
        ('2026-04-02', 10.3),   # +3%
        ('2026-04-03', 10.6),   # +6% - 触发第一档止盈
        ('2026-04-04', 10.8),   # +8%
        ('2026-04-05', 11.2),   # +12% - 触发第二档止盈
        ('2026-04-06', 11.0),   # +10% - 回撤
    ]
    
    for date, price in test_prices:
        result = manager.update_price('000001', price, date)
        print(f"{date} 价格:{price:.2f} 盈利:{result['profit_pct']:+.1f}% -> {result['action']}: {result['reason']}")
    
    print("\n📊 测试场景2：移动止盈")
    print("-" * 60)
    
    manager2 = TrailingStopManager(trailing_start=5.0, trailing_pct=3.0)
    manager2.add_position('000002', '万科A', 20.0, '2026-04-01')
    
    test_prices2 = [
        ('2026-04-02', 21.0),   # +5% - 启动移动止盈
        ('2026-04-03', 22.0),   # +10% - 新高
        ('2026-04-04', 23.0),   # +15% - 新高
        ('2026-04-05', 22.0),   # +10% - 从高点回撤4.3% (>3%)，触发移动止盈
    ]
    
    for date, price in test_prices2:
        result = manager2.update_price('000002', price, date)
        print(f"{date} 价格:{price:.2f} 盈利:{result['profit_pct']:+.1f}% -> {result['action']}: {result['reason']}")
    
    print("\n📊 测试场景3：动态持仓周期")
    print("-" * 60)
    
    dynamic = DynamicHoldingPeriod(base_days=14)
    
    test_sectors = ['AI/算力', '机器人', '半导体', '银行', '医药']
    for sector in test_sectors:
        days = dynamic.get_hold_days(sector, 'neutral')
        days_bull = dynamic.get_hold_days(sector, 'bullish')
        days_bear = dynamic.get_hold_days(sector, 'bearish')
        print(f"{sector}: 中性{days}天 | 牛市{days_bull}天 | 熊市{days_bear}天")
    
    print("\n✅ 阶段2优化测试完成！")
