# -*- coding: utf-8 -*-
"""
Strategy Optimizer - Adjust strategy parameters based on historical performance
"""
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class StrategyOptimizer:
    """Strategy Optimizer"""
    
    def __init__(self, tracking_file: str = 'data/stock_tracking.json'):
        self.tracking_file = tracking_file
        self.data = self._load_data()
        self.results = self._analyze_performance()
    
    def _load_data(self) -> Dict:
        """Load tracking data"""
        with open(self.tracking_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _analyze_performance(self) -> List[Dict]:
        """Analyze performance of each stock"""
        results = []
        for code, info in self.data.items():
            daily = info.get('daily_prices', {})
            if not daily:
                continue
            
            dates = sorted(daily.keys())
            buy_price = info.get('buy_price', 0)
            latest_price = daily[dates[-1]]['close'] if dates else 0
            take_profit = info.get('take_profit', 0)
            stop_loss = info.get('stop_loss', 0)
            
            if buy_price > 0:
                total_return = (latest_price - buy_price) / buy_price * 100
            else:
                total_return = 0
            
            # Determine status
            if latest_price >= take_profit:
                status = 'take_profit'
            elif latest_price <= stop_loss:
                status = 'stop_loss'
            else:
                status = 'holding'
            
            results.append({
                'code': code,
                'name': info.get('name', '?'),
                'added_date': info.get('added_date', ''),
                'buy_price': buy_price,
                'latest_price': latest_price,
                'total_return': total_return,
                'status': status,
                'days_held': len(daily),
                'confidence': info.get('confidence', 0),
                'expected_return': info.get('expected_return', 0),
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'take_profit_pct': (take_profit - buy_price) / buy_price * 100 if buy_price > 0 else 0,
                'stop_loss_pct': (buy_price - stop_loss) / buy_price * 100 if buy_price > 0 else 0,
            })
        return results
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics"""
        completed = [r for r in self.results if r['status'] in ['take_profit', 'stop_loss']]
        tp = [r for r in completed if r['status'] == 'take_profit']
        sl = [r for r in completed if r['status'] == 'stop_loss']
        holding = [r for r in self.results if r['status'] == 'holding']
        
        win_rate = len(tp) / len(completed) * 100 if completed else 0
        avg_return = np.mean([r['total_return'] for r in completed]) if completed else 0
        
        return {
            'total_tracked': len(self.results),
            'completed': len(completed),
            'take_profit': len(tp),
            'stop_loss': len(sl),
            'holding': len(holding),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_tp_return': np.mean([r['total_return'] for r in tp]) if tp else 0,
            'avg_sl_return': np.mean([r['total_return'] for r in sl]) if sl else 0,
        }
    
    def analyze_by_confidence(self) -> List[Dict]:
        """Analyze by confidence level"""
        completed = [r for r in self.results if r['status'] in ['take_profit', 'stop_loss']]
        
        bins = [
            (0.0, 0.75, 'low'),
            (0.75, 0.85, 'medium'),
            (0.85, 0.90, 'high'),
            (0.90, 1.0, 'very_high')
        ]
        
        analysis = []
        for low, high, label in bins:
            bin_stocks = [r for r in completed if low <= r['confidence'] < high]
            if not bin_stocks:
                continue
            
            tp_count = len([r for r in bin_stocks if r['status'] == 'take_profit'])
            win_rate = tp_count / len(bin_stocks) * 100
            avg_return = np.mean([r['total_return'] for r in bin_stocks])
            
            analysis.append({
                'label': label,
                'range': f'{low:.2f}-{high:.2f}',
                'count': len(bin_stocks),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'min_confidence': min(r['confidence'] for r in bin_stocks),
                'max_confidence': max(r['confidence'] for r in bin_stocks),
            })
        
        return analysis
    
    def analyze_by_date(self) -> List[Dict]:
        """Analyze by date"""
        completed = [r for r in self.results if r['status'] in ['take_profit', 'stop_loss']]
        
        date_groups = {}
        for r in completed:
            d = r['added_date']
            if d not in date_groups:
                date_groups[d] = []
            date_groups[d].append(r)
        
        analysis = []
        for d in sorted(date_groups.keys()):
            stocks = date_groups[d]
            tp_count = len([r for r in stocks if r['status'] == 'take_profit'])
            win_rate = tp_count / len(stocks) * 100
            avg_return = np.mean([r['total_return'] for r in stocks])
            
            analysis.append({
                'date': d,
                'count': len(stocks),
                'win_rate': win_rate,
                'avg_return': avg_return,
            })
        
        return analysis
    
    def analyze_stop_loss_profit_settings(self) -> Dict:
        """Analyze if stop loss/profit settings are reasonable"""
        completed = [r for r in self.results if r['status'] in ['take_profit', 'stop_loss']]
        
        # Analyze stop loss/profit percentages
        tp_pcts = [r['take_profit_pct'] for r in completed]
        sl_pcts = [r['stop_loss_pct'] for r in completed]
        
        # Analyze actual max drawdown and max gain
        max_drawdowns = []
        max_gains = []
        
        for code, info in self.data.items():
            daily = info.get('daily_prices', {})
            if not daily:
                continue
            
            buy_price = info.get('buy_price', 0)
            if buy_price <= 0:
                continue
            
            prices = [d['close'] for d in daily.values()]
            if not prices:
                continue
            
            returns = [(p - buy_price) / buy_price * 100 for p in prices]
            max_gains.append(max(returns))
            max_drawdowns.append(min(returns))
        
        return {
            'avg_take_profit_pct': np.mean(tp_pcts) if tp_pcts else 0,
            'avg_stop_loss_pct': np.mean(sl_pcts) if sl_pcts else 0,
            'avg_max_gain': np.mean(max_gains) if max_gains else 0,
            'avg_max_drawdown': np.mean(max_drawdowns) if max_drawdowns else 0,
            'max_gain_ever': max(max_gains) if max_gains else 0,
            'max_drawdown_ever': min(max_drawdowns) if max_drawdowns else 0,
        }
    
    def generate_optimization_report(self) -> str:
        """Generate optimization report"""
        stats = self.get_summary_stats()
        conf_analysis = self.analyze_by_confidence()
        date_analysis = self.analyze_by_date()
        sl_tp_analysis = self.analyze_stop_loss_profit_settings()
        
        lines = []
        lines.append("=" * 80)
        lines.append("Strategy Optimization Analysis Report")
        lines.append("=" * 80)
        
        # 1. Overall performance
        lines.append("\n[1. Overall Performance]")
        lines.append(f"  Total tracked stocks: {stats['total_tracked']}")
        lines.append(f"  Completed trades: {stats['completed']} (TP:{stats['take_profit']} / SL:{stats['stop_loss']})")
        lines.append(f"  Win rate: {stats['win_rate']:.1f}%")
        lines.append(f"  Avg return: {stats['avg_return']:+.2f}%")
        lines.append(f"  Avg TP return: {stats['avg_tp_return']:+.2f}%")
        lines.append(f"  Avg SL return: {stats['avg_sl_return']:+.2f}%")
        
        # 2. Confidence analysis
        lines.append("\n[2. Confidence Level Analysis]")
        for a in conf_analysis:
            lines.append(f"  {a['label']} ({a['range']}): {a['count']} stocks, Win rate {a['win_rate']:.1f}%, Avg return {a['avg_return']:+.2f}%")
        
        # 3. Date analysis
        lines.append("\n[3. Performance by Date]")
        for a in date_analysis:
            lines.append(f"  {a['date']}: {a['count']} stocks, Win rate {a['win_rate']:.1f}%, Avg return {a['avg_return']:+.2f}%")
        
        # 4. Stop loss/profit analysis
        lines.append("\n[4. Stop Loss/Profit Setting Analysis]")
        lines.append(f"  Avg take profit: {sl_tp_analysis['avg_take_profit_pct']:.2f}%")
        lines.append(f"  Avg stop loss: {sl_tp_analysis['avg_stop_loss_pct']:.2f}%")
        lines.append(f"  Actual avg max gain: {sl_tp_analysis['avg_max_gain']:+.2f}%")
        lines.append(f"  Actual avg max drawdown: {sl_tp_analysis['avg_max_drawdown']:+.2f}%")
        lines.append(f"  Historical max gain: {sl_tp_analysis['max_gain_ever']:+.2f}%")
        lines.append(f"  Historical max drawdown: {sl_tp_analysis['max_drawdown_ever']:+.2f}%")
        
        # 5. Optimization suggestions
        lines.append("\n[5. Optimization Suggestions]")
        
        # Suggestions based on win rate
        if stats['win_rate'] < 50:
            lines.append("  [WARNING] Win rate below 50%, suggestions:")
            lines.append("     - Increase confidence threshold (suggest 0.85+)")
            lines.append("     - Add more filters (volume, market cap)")
        elif stats['win_rate'] < 60:
            lines.append("  [WARNING] Win rate 50-60%, suggestions:")
            lines.append("     - Increase confidence threshold to 0.80+")
            lines.append("     - Optimize stop loss/profit ratios")
        else:
            lines.append("  [GOOD] Win rate is good, suggestions:")
            lines.append("     - Keep current strategy")
            lines.append("     - Can try increasing position size")
        
        # Suggestions based on stop loss/profit analysis
        avg_tp = sl_tp_analysis['avg_take_profit_pct']
        avg_max_gain = sl_tp_analysis['avg_max_gain']
        
        if avg_max_gain > avg_tp * 1.5:
            lines.append(f"  [TIP] Actual max gain ({avg_max_gain:.1f}%) much higher than TP ratio ({avg_tp:.1f}%), suggest:")
            lines.append("     - Consider increasing take profit ratio or using trailing stop")
        
        # Suggestions based on confidence analysis
        if conf_analysis:
            best_conf = max(conf_analysis, key=lambda x: x['win_rate'])
            lines.append(f"  [TIP] Confidence {best_conf['range']} performs best (win rate {best_conf['win_rate']:.1f}%), suggest:")
            lines.append(f"     - Prioritize stocks with confidence > {best_conf['range'].split('-')[0]}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def get_optimized_params(self) -> Dict:
        """Get optimized parameter suggestions"""
        stats = self.get_summary_stats()
        conf_analysis = self.analyze_by_confidence()
        sl_tp_analysis = self.analyze_stop_loss_profit_settings()
        
        # Adjust parameters based on analysis
        params = {
            'min_confidence': 0.80,  # default
            'take_profit_pct': 8.0,   # default
            'stop_loss_pct': 3.0,     # default
            'max_position': 20,       # default position
        }
        
        # Adjust min confidence based on confidence analysis
        if conf_analysis:
            # Find the range with highest win rate
            best = max(conf_analysis, key=lambda x: x['win_rate'])
            if best['win_rate'] > 60:
                # Use the lower bound of that range as min confidence
                min_conf = float(best['range'].split('-')[0])
                params['min_confidence'] = min_conf
        
        # Adjust take profit based on analysis
        avg_max_gain = sl_tp_analysis['avg_max_gain']
        if avg_max_gain > 12:
            params['take_profit_pct'] = 10.0
        elif avg_max_gain > 10:
            params['take_profit_pct'] = 8.0
        else:
            params['take_profit_pct'] = 5.0
        
        # Adjust position based on win rate
        if stats['win_rate'] > 60:
            params['max_position'] = 25
        elif stats['win_rate'] > 50:
            params['max_position'] = 20
        else:
            params['max_position'] = 15
        
        return params


def main():
    """Main function"""
    optimizer = StrategyOptimizer()
    
    # Print optimization report
    print(optimizer.generate_optimization_report())
    
    # Print optimized parameters
    params = optimizer.get_optimized_params()
    print("\n[Optimized Parameters]")
    for k, v in params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
