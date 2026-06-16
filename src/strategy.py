# 选股策略模块

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StockSelector:
    """股票选择器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.min_market_cap = config.get('min_market_cap', 20)
        self.max_market_cap = config.get('max_market_cap', 5000)
        self.exclude_st = config.get('exclude_st', True)
        self.min_listing_days = config.get('min_listing_days', 60)
        self.daily_picks = config.get('daily_picks', 5)
    
    def filter_stocks(self, stock_list: List[Dict], market_data: Dict) -> List[str]:
        """过滤股票
        
        Args:
            stock_list: 股票列表
            market_data: 市场数据
        
        Returns:
            过滤后的股票代码列表
        """
        filtered = []
        
        for stock in stock_list:
            code = stock.get('code', '')
            name = stock.get('name', '')
            
            # 排除ST股票
            if self.exclude_st and ('ST' in name or 'st' in name):
                continue
            
            # 排除退市股票
            if '退' in name:
                continue
            
            # 排除新股（需要上市天数）
            # TODO: 需要获取上市日期
            
            filtered.append(code)
        
        return filtered
    
    def calculate_score(self, stock_data: pd.DataFrame, predictions: np.ndarray,
                        proba: np.ndarray) -> pd.DataFrame:
        """计算综合得分
        
        Args:
            stock_data: 股票数据
            predictions: 预测结果
            proba: 预测概率
        
        Returns:
            包含得分的DataFrame
        """
        scores = pd.DataFrame({
            'code': stock_data.index if isinstance(stock_data.index, str) else range(len(predictions)),
            'predict_proba': proba,
            'predict_label': predictions
        })
        
        # 只选择预测为正的股票
        scores = scores[scores['predict_label'] == 1]
        
        # 按概率排序
        scores = scores.sort_values('predict_proba', ascending=False)
        
        return scores
    
    def select_top_stocks(self, scores: pd.DataFrame, 
                          market_data: Optional[Dict] = None) -> List[Dict]:
        """选择得分最高的股票
        
        Args:
            scores: 包含得分的数据
            market_data: 市场数据（用于额外过滤）
        
        Returns:
            推荐股票列表
        """
        # 选择前N只股票
        top_stocks = scores.head(self.daily_picks)
        
        recommendations = []
        for _, row in top_stocks.iterrows():
            recommendations.append({
                'code': row.get('code', ''),
                'score': row.get('predict_proba', 0),
                'rank': len(recommendations) + 1
            })
        
        return recommendations


class StopLossProfitCalculator:
    """止盈止损计算器"""
    
    def __init__(self, take_profit_pct: float = 8.0, 
                 stop_loss_pct: float = 3.0):
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
    
    def calculate(self, buy_price: float, 
                  support_level: Optional[float] = None,
                  resistance_level: Optional[float] = None) -> Dict:
        """计算止盈止损点位
        
        Args:
            buy_price: 买入价格
            support_level: 支撑位（可选）
            resistance_level: 阻力位（可选）
        
        Returns:
            止盈止损点位
        """
        # 默认止盈止损
        take_profit = buy_price * (1 + self.take_profit_pct / 100)
        stop_loss = buy_price * (1 - self.stop_loss_pct / 100)
        
        # 如果提供了支撑阻力位，可以调整
        if support_level and support_level < buy_price:
            # 止损设在支撑位下方
            stop_loss = min(stop_loss, support_level * 0.98)
        
        if resistance_level and resistance_level > buy_price:
            # 止盈设在阻力位附近
            take_profit = min(take_profit, resistance_level * 0.98)
        
        return {
            'buy_price': buy_price,
            'take_profit': round(take_profit, 2),
            'stop_loss': round(stop_loss, 2),
            'take_profit_pct': round((take_profit - buy_price) / buy_price * 100, 2),
            'stop_loss_pct': round((buy_price - stop_loss) / buy_price * 100, 2),
            'risk_reward_ratio': round((take_profit - buy_price) / (buy_price - stop_loss + 1e-8), 2)
        }
    
    def dynamic_stop_loss(self, current_price: float, 
                          highest_price: float,
                          trailing_pct: float = 5.0) -> float:
        """动态止损（移动止损）
        
        Args:
            current_price: 当前价格
            highest_price: 持仓期间最高价
            trailing_pct: 回撤比例(%)
        
        Returns:
            止损价格
        """
        trailing_stop = highest_price * (1 - trailing_pct / 100)
        return round(trailing_stop, 2)


class RecommendationEngine:
    """推荐引擎"""
    
    def __init__(self, predictor, selector: StockSelector,
                 stop_loss_calculator: StopLossProfitCalculator):
        self.predictor = predictor
        self.selector = selector
        self.stop_loss_calc = stop_loss_calculator
    
    def generate_daily_recommendations(self, 
                                       stock_pool: List[str],
                                       market_data: Dict[str, pd.DataFrame],
                                       feature_engineer) -> List[Dict]:
        """生成每日推荐
        
        Args:
            stock_pool: 股票池
            market_data: 市场数据
            feature_engineer: 特征工程器
        
        Returns:
            推荐列表
        """
        all_predictions = []
        
        for stock_code in stock_pool:
            if stock_code not in market_data:
                continue
            
            df = market_data[stock_code]
            if df.empty:
                continue
            
            # 计算特征
            df = feature_engineer.calculate_technical_indicators(df)
            df = feature_engineer.create_target(df)
            
            # 准备预测数据
            latest_data = df.iloc[-1:].copy()
            X = latest_data[feature_engineer.feature_names]
            
            # 预测
            proba = self.predictor.predict_proba(X)[0]
            pred = self.predictor.predict(X)[0]
            
            all_predictions.append({
                'code': stock_code,
                'proba': proba,
                'pred': pred,
                'latest_price': df['close'].iloc[-1],
                'df': df
            })
        
        # 计算得分并排序
        scores = pd.DataFrame([
            {'code': p['code'], 'predict_proba': p['proba'], 
             'predict_label': p['pred'], 'latest_price': p['latest_price']}
            for p in all_predictions
        ])
        
        scores = scores[scores['predict_label'] == 1].sort_values(
            'predict_proba', ascending=False
        )
        
        # 选择推荐股票
        recommendations = []
        for rank, (_, row) in enumerate(scores.head(self.selector.daily_picks).iterrows(), 1):
            code = row['code']
            buy_price = row['latest_price']
            
            # 计算止盈止损
            sl_tp = self.stop_loss_calc.calculate(buy_price)
            
            # 查找股票数据
            stock_pred = next((p for p in all_predictions if p['code'] == code), None)
            
            recommendations.append({
                'rank': rank,
                'code': code,
                'buy_price': buy_price,
                'confidence': round(row['predict_proba'] * 100, 2),
                'take_profit': sl_tp['take_profit'],
                'stop_loss': sl_tp['stop_loss'],
                'take_profit_pct': sl_tp['take_profit_pct'],
                'stop_loss_pct': sl_tp['stop_loss_pct'],
                'risk_reward_ratio': sl_tp['risk_reward_ratio']
            })
        
        return recommendations
    
    def format_recommendations(self, recommendations: List[Dict], 
                               date: str = None) -> str:
        """格式化推荐结果"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        lines = [
            f"# 📈 股票推荐 ({date})",
            f"",
            f"## 推荐股票列表",
            f"",
        ]
        
        for rec in recommendations:
            lines.extend([
                f"### {rec['rank']}. {rec['code']}",
                f"- **买入价格**: {rec['buy_price']:.2f}",
                f"- **置信度**: {rec['confidence']}%",
                f"- **止盈价格**: {rec['take_profit']:.2f} (+{rec['take_profit_pct']}%)",
                f"- **止损价格**: {rec['stop_loss']:.2f} (-{rec['stop_loss_pct']}%)",
                f"- **盈亏比**: {rec['risk_reward_ratio']}:1",
                f""
            ])
        
        lines.extend([
            f"---",
            f"*风险提示：以上推荐仅供参考，不构成投资建议。股市有风险，投资需谨慎。*"
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    # 测试止盈止损计算
    calc = StopLossProfitCalculator(take_profit_pct=8.0, stop_loss_pct=3.0)
    result = calc.calculate(buy_price=100.0)
    print(f"止盈止损: {result}")
    
    # 测试动态止损
    trailing = calc.dynamic_stop_loss(current_price=105, highest_price=110, trailing_pct=5)
    print(f"动态止损: {trailing}")
