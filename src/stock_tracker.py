# -*- coding: utf-8 -*-
"""
股票跟踪记录模块
- 跟踪每日推荐股票
- 记录14天涨幅数据
- 自动清理超过50只或超过14天的记录
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class StockTracker:
    """股票跟踪器"""
    
    def __init__(self, tracking_file: str = None):
        if tracking_file is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            tracking_file = os.path.join(project_dir, 'data', 'stock_tracking.json')
        
        self.tracking_file = tracking_file
        self.max_stocks = 50          # 最大跟踪股票数
        self.max_days = 14             # 跟踪天数
        self._ensure_file()
    
    def _ensure_file(self):
        """确保跟踪文件存在"""
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        if not os.path.exists(self.tracking_file):
            self._save({})
    
    def _load(self) -> Dict:
        """加载跟踪数据"""
        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def _save(self, data: Dict):
        """保存跟踪数据"""
        def convert_floats(obj):
            """递归转换numpy float32/float64为Python float"""
            if isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_floats(item) for item in obj]
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            else:
                return obj
        
        data = convert_floats(data)
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def add_recommendations(self, recommendations: List[Dict], date: str = None):
        """
        添加新的推荐股票到跟踪列表
        
        Args:
            recommendations: 推荐股票列表，每项包含 code, name, strategy 等
            date: 推荐日期，格式 YYYY-MM-DD
        """
        if not recommendations:
            return
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        data = self._load()
        
        for rec in recommendations:
            code = rec['code']
            
            if code not in data:
                # 新股票
                data[code] = {
                    'name': rec.get('name', code),
                    'added_date': date,
                    'strategy': rec.get('strategy', ''),
                    'buy_price': rec.get('buy_price', 0),
                    'confidence': rec.get('confidence', 0),
                    'expected_return': rec.get('expected_return', 0),
                    'take_profit': rec.get('take_profit', 0),
                    'stop_loss': rec.get('stop_loss', 0),
                    'sector': rec.get('sector', ''),
                    'tech_score': rec.get('tech_score', 0),
                    'fund_score': rec.get('fund_score', 0),
                    'rsi': rec.get('rsi', 0),
                    'take_profit_levels': rec.get('take_profit_levels', []),
                    'partial_sold': [],
                    'sell_reason': '',
                    'change': rec.get('change', 0),
                    'current_stop_loss': rec.get('stop_loss', 0),
                    'daily_prices': {}
                }
            
            # 更新当日价格
            if date not in data[code]['daily_prices']:
                data[code]['daily_prices'][date] = {
                    'close': rec.get('buy_price', 0),
                    'change': rec.get('change', 0),
                    'take_profit': rec.get('take_profit', 0),
                    'stop_loss': rec.get('stop_loss', 0),
                }
        
        # 清理超过14天的数据
        cutoff_date = (datetime.now() - timedelta(days=self.max_days)).strftime('%Y-%m-%d')
        for code in list(data.keys()):
            daily = data[code].get('daily_prices', {})
            data[code]['daily_prices'] = {k: v for k, v in daily.items() if k >= cutoff_date}
        
        # 清理没有价格数据的股票
        data = {k: v for k, v in data.items() if v.get('daily_prices')}
        
        # 超过50只则删除最早的
        if len(data) > self.max_stocks:
            sorted_codes = sorted(data.keys(), key=lambda x: data[x]['added_date'])
            remove_count = len(data) - self.max_stocks
            for code in sorted_codes[:remove_count]:
                del data[code]
        
        self._save(data)
    
    def update_prices(self, stock_prices: Dict[str, Dict], date: str = None):
        """
        更新跟踪股票的当日价格
        
        Args:
            stock_prices: {code: {'close': price, 'change': change_pct}}
            date: 更新日期
        """
        if not stock_prices:
            return
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        data = self._load()
        updated = False
        
        for code, price_info in stock_prices.items():
            if code in data:
                if date not in data[code]['daily_prices']:
                    data[code]['daily_prices'][date] = {
                        'close': price_info.get('close', 0),
                        'change': price_info.get('change', 0),
                    }
                    updated = True
        
        # 清理超过14天的数据
        cutoff_date = (datetime.now() - timedelta(days=self.max_days)).strftime('%Y-%m-%d')
        for code in list(data.keys()):
            daily = data[code].get('daily_prices', {})
            data[code]['daily_prices'] = {k: v for k, v in daily.items() if k >= cutoff_date}
        
        # 清理没有价格数据的股票
        data = {k: v for k, v in data.items() if v.get('daily_prices')}
        
        if updated:
            self._save(data)
    
    def get_tracked_stocks(self) -> Dict:
        """获取所有跟踪股票"""
        return self._load()
    
    def get_history_summary(self, limit: int = 10, exclude_active: bool = False) -> List[Dict]:
        """
        获取历史推荐摘要
        
        Args:
            limit: 返回的最大记录数
            exclude_active: 是否排除仍在持仓中的股票（持仓天数<14天）
        
        Returns:
            按添加日期排序的推荐列表
        """
        data = self._load()
        
        history = []
        for code, info in data.items():
            daily = info.get('daily_prices', {})
            if not daily:
                continue
            
            dates = sorted(daily.keys())
            first_price = daily[dates[0]]['close'] if dates else 0
            latest_price = daily[dates[-1]]['close'] if dates else 0
            
            # 计算累计涨幅
            if first_price > 0:
                total_change = (latest_price - first_price) / first_price * 100
            else:
                total_change = 0
            
            # 计算持仓天数
            holding_days = len(dates)
            
            # 如果要求排除活跃持仓，且持仓天数小于14天，则跳过
            if exclude_active and holding_days < 14:
                continue
            
            history.append({
                'code': code,
                'name': info.get('name', code),
                'added_date': info.get('added_date', ''),
                'strategy': info.get('strategy', ''),
                'buy_price': info.get('buy_price', 0),
                'latest_price': latest_price,
                'total_change': total_change,
                'holding_days': holding_days,
                'daily_prices': daily,
                'take_profit': info.get('take_profit', 0),
                'stop_loss': info.get('stop_loss', 0),
            })
        
        # 按添加日期排序，最新的在前
        history.sort(key=lambda x: x['added_date'], reverse=True)
        return history[:limit]
    
    def format_history_display(self, limit: int = 10, exclude_active: bool = False) -> str:
        """格式化历史推荐显示"""
        history = self.get_history_summary(limit, exclude_active=exclude_active)
        
        if not history:
            return "暂无历史推荐记录"
        
        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("  📊 历史推荐跟踪")
        lines.append("=" * 100)
        
        for item in history:
            lines.append(f"\n【{item['code']}】{item['name']} | {item['strategy']}")
            lines.append(f"  推荐日期: {item['added_date']} | 买入价: {item['buy_price']:.2f} | 最新: {item['latest_price']:.2f}")
            lines.append(f"  累计涨跌: {item['total_change']:+.2f}% | 持仓: {item['holding_days']}天")
            lines.append(f"  止盈: {item['take_profit']:.2f} | 止损: {item['stop_loss']:.2f}")
            
            # 显示每日价格
            daily = item['daily_prices']
            dates = sorted(daily.keys())
            if len(dates) > 0:
                price_str = "  每日: "
                for d in dates:
                    p = daily[d]
                    change = p.get('change', 0)
                    close = p.get('close', 0)
                    price_str += f"{d[-5:]}({close:.2f},{change:+.2f}%) "
                lines.append(price_str)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试跟踪器
    tracker = StockTracker()
    
    # 测试添加推荐
    test_recs = [
        {'code': '600654', 'name': 'ST中安', 'strategy': '短线(5天)', 'buy_price': 4.40, 'confidence': 0.887, 'expected_return': 74.4, 'take_profit': 4.62, 'stop_loss': 4.27, 'change': 10.00},
        {'code': '600743', 'name': '华远地产', 'strategy': '短线(5天)', 'buy_price': 2.19, 'confidence': 0.871, 'expected_return': 73.6, 'take_profit': 2.30, 'stop_loss': 2.12, 'change': 10.05},
    ]
    
    today = datetime.now().strftime('%Y-%m-%d')
    tracker.add_recommendations(test_recs, today)
    
    print("跟踪记录已保存")
    print(tracker.format_history_display())
