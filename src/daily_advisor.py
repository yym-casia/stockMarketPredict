# -*- coding: utf-8 -*-
"""每日操作建议生成器：优先卖出信号，控制买入仓位"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.portfolio_manager import PortfolioManager
from src.position_manager import PositionManager
from src.strategy_filters import rank_candidates
from src.market_regime import should_defensive_exit
from src.trading_calendar import is_trading_day
from src.data_fetcher_multi import MultiSourceDataFetcher


class DailyAdvisor:
    """生成每日买卖操作建议"""

    def __init__(self, portfolio: PortfolioManager, position_mgr: PositionManager,
                 operations_file: str = None):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.portfolio = portfolio
        self.position_mgr = position_mgr
        self.operations_file = operations_file or os.path.join(
            project_dir, 'data', 'daily_operations.json'
        )

    @staticmethod
    def _extract_change(price_info: Dict) -> float:
        if not price_info:
            return 0.0
        change = price_info.get('change')
        if change is not None and change != 0:
            return round(float(change), 2)
        return MultiSourceDataFetcher.calc_change(
            price_info.get('price', 0), price_info.get('last_close', 0)
        )

    def evaluate_positions(self, price_data, today: str,
                           total_equity: float = None,
                           confirm_stop_close: bool = False,
                           regime: Dict = None,
                           strat_cfg: Dict = None) -> Tuple[Dict, List[Dict]]:
        """评估所有持仓，返回分类结果和紧急告警"""
        categories = {
            'sell': [], 'partial_sell': [], 'add': [], 'dip_refill': [],
            'hold': [], 'expired': [],
        }
        urgent_alerts = []

        active = self.portfolio.get_active_positions()
        if not active:
            return categories, urgent_alerts

        self.position_mgr.load_from_dict(active)

        for code, info in active.items():
            price_info = self._find_price(price_data, code)
            if not price_info:
                continue

            current_price = price_info.get('price', 0)
            today_change = self._extract_change(price_info)
            defensive = bool(regime and regime.get('defensive'))
            profit_pct = (
                (current_price - info.get('buy_price', 0)) / info.get('buy_price', 1) * 100
                if info.get('buy_price') else 0
            )
            if defensive and strat_cfg:
                exit_reason = should_defensive_exit(
                    profit_pct, info.get('sector', '其他'), regime, strat_cfg)
                if exit_reason:
                    result = {
                        'code': code, 'name': info.get('name', code),
                        'profit_pct': profit_pct,
                        'days_held': info.get('days_held', 0),
                        'action': 'sell', 'reason': exit_reason,
                        'sell_ratio': 1.0, 'urgency': 'urgent',
                        'sell_price': current_price,
                    }
                else:
                    result = self.position_mgr.update_price(
                        code, current_price, today, total_equity=total_equity,
                        confirm_stop_close=confirm_stop_close, defensive=defensive)
            else:
                result = self.position_mgr.update_price(
                    code, current_price, today, total_equity=total_equity,
                    confirm_stop_close=confirm_stop_close, defensive=defensive)

            stock_data = {
                'code': code,
                'name': info.get('name', code),
                'buy_price': info.get('buy_price', 0),
                'current_price': current_price,
                'today_change': today_change,
                'profit_pct': result['profit_pct'],
                'days_held': result['days_held'],
                'sector': info.get('sector', '其他'),
                'take_profit': info.get('take_profit', 0),
                'stop_loss': info.get('stop_loss', 0),
                'current_stop_loss': self.position_mgr.positions[code].stop_loss,
                'sell_reason': result.get('reason', ''),
                'urgency': result.get('urgency', 'normal'),
                'sell_ratio': result.get('sell_ratio', 0),
                'add_ratio': result.get('add_ratio', 0),
                'add_amount_pct': result.get('add_amount_pct'),
                'add_amount': result.get('add_amount'),
            }

            self.portfolio.update_position(code, {
                'price': current_price,
                'change': today_change,
                'last_close': price_info.get('last_close', 0),
                'current_stop_loss': stock_data['current_stop_loss'],
                'partial_sold': self.position_mgr.positions[code].partial_sold,
                'add_levels_done': self.position_mgr.positions[code].add_levels_done,
                'stop_loss_partial_done': self.position_mgr.positions[code].stop_loss_partial_done,
                'dip_refill_done': self.position_mgr.positions[code].dip_refill_done,
                'sell_reason': result.get('reason', ''),
            }, today)

            action = result['action']
            if action == 'sell':
                status = 'take_profit' if result['profit_pct'] >= 0 else 'stop_loss'
                if '期满' in result.get('reason', ''):
                    status = 'expired'
                categories['sell'].append({**stock_data, 'close_status': status})
                urgent_alerts.append(self._make_alert('sell', stock_data, result))
            elif action == 'partial_sell':
                categories['partial_sell'].append(stock_data)
                urgent_alerts.append(self._make_alert('partial_sell', stock_data, result))
            elif action == 'add':
                categories['add'].append(stock_data)
            elif action == 'dip_refill':
                categories['dip_refill'].append(stock_data)
            else:
                categories['hold'].append(stock_data)
                if result.get('urgency') == 'high':
                    urgent_alerts.append(self._make_alert('watch', stock_data, result))

        return categories, urgent_alerts

    def _find_price(self, price_data, code: str) -> Optional[Dict]:
        if price_data is None:
            return None
        try:
            if hasattr(price_data, 'iterrows'):
                matches = price_data[price_data['code'].str.contains(code)]
                if not matches.empty:
                    return matches.iloc[0].to_dict()
            elif isinstance(price_data, dict):
                return price_data.get(code)
        except Exception:
            pass
        return None

    def _make_alert(self, alert_type: str, stock: Dict, result: Dict) -> Dict:
        priority = result.get('urgency', 'normal')
        if alert_type == 'sell' and result.get('profit_pct', 0) < -2:
            priority = 'urgent'
        return {
            'type': alert_type,
            'priority': priority,
            'code': stock['code'],
            'name': stock['name'],
            'action': '立即卖出' if alert_type == 'sell' else '部分卖出' if alert_type == 'partial_sell' else '密切关注',
            'reason': result.get('reason', ''),
            'profit_pct': result.get('profit_pct', 0),
            'current_price': stock.get('current_price', 0),
            'sell_ratio': result.get('sell_ratio', 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    def select_buys(self, candidates: List[Dict], max_count: int,
                    min_confidence: float = 0.85,
                    ml_rank_only: bool = True) -> List[Dict]:
        """根据可用仓位筛选买入标的（排序与回测 rank_candidates 一致）"""
        if max_count <= 0:
            return []

        filtered = self.portfolio.filter_already_held(candidates)
        if not ml_rank_only and min_confidence > 0:
            filtered = [c for c in filtered if c.get('confidence', 0) >= min_confidence]
        return rank_candidates(filtered, top_n=max_count)

    def build_operations(self, categories: Dict, buy_signals: List[Dict],
                         market_info: Dict, today: str) -> Dict:
        """构建完整操作建议"""
        active = self.portfolio.get_active_positions()
        stats = self.portfolio.get_stats()
        available = self.portfolio.get_available_slots()
        cap = self.portfolio._effective_max_positions(active, self.portfolio.max_positions)
        slots_used = len(active)

        sell_ops = []
        for s in categories.get('sell', []):
            sell_ops.append({
                'code': s['code'], 'name': s['name'],
                'action': '卖出', 'priority': s.get('urgency', 'high'),
                'reason': s.get('sell_reason', ''),
                'profit_pct': s['profit_pct'],
                'current_price': s['current_price'],
                'buy_price': s['buy_price'],
                'days_held': s['days_held'],
            })

        add_ops = []
        for s in categories.get('add', []):
            add_ops.append({
                'code': s['code'], 'name': s['name'],
                'action': '加仓',
                'add_ratio': s.get('add_ratio', 0.3),
                'add_amount_pct': s.get('add_amount_pct'),
                'add_amount': s.get('add_amount'),
                'reason': s.get('sell_reason', ''),
                'profit_pct': s['profit_pct'],
                'current_price': s['current_price'],
            })

        partial_ops = []
        for s in categories.get('partial_sell', []):
            partial_ops.append({
                'code': s['code'], 'name': s['name'],
                'action': '部分卖出',
                'sell_ratio': s.get('sell_ratio', 0.3),
                'reason': s.get('sell_reason', ''),
                'profit_pct': s['profit_pct'],
                'current_price': s['current_price'],
            })

        buy_ops = []
        for i, rec in enumerate(buy_signals, 1):
            buy_ops.append({
                'code': rec['code'], 'name': rec['name'],
                'action': '买入', 'priority': 'normal',
                'buy_price': rec['buy_price'],
                'take_profit': rec.get('take_profit', 0),
                'stop_loss': rec.get('stop_loss', 0),
                'confidence': rec.get('confidence', 0),
                'tech_score': rec.get('tech_score', 0),
                'fund_score': rec.get('fund_score', 0),
                'reason': rec.get('reason', ''),
                'rank': i,
            })

        hold_ops = []
        for s in categories.get('hold', []):
            hold_ops.append({
                'code': s['code'], 'name': s['name'],
                'action': '继续持有',
                'profit_pct': s['profit_pct'],
                'today_change': s.get('today_change', 0),
                'days_held': s['days_held'],
                'remaining_days': self.position_mgr.max_hold_days - s['days_held'],
                'current_price': s['current_price'],
                'stop_loss': s.get('current_stop_loss', s.get('stop_loss', 0)),
            })

        summary_parts = []
        if sell_ops:
            summary_parts.append(f"卖出{len(sell_ops)}只")
        if partial_ops:
            summary_parts.append(f"部分止盈{len(partial_ops)}只")
        if add_ops:
            summary_parts.append(f"加仓{len(add_ops)}只")
        dip_ops = []
        for s in categories.get('dip_refill', []):
            dip_ops.append({
                'code': s['code'], 'name': s['name'],
                'action': '补半仓',
                'add_ratio': s.get('add_ratio', 0.5),
                'reason': s.get('sell_reason', ''),
                'profit_pct': s['profit_pct'],
                'current_price': s['current_price'],
            })
        if dip_ops:
            summary_parts.append(f"补半仓{len(dip_ops)}只")
        if buy_ops:
            summary_parts.append(f"买入{len(buy_ops)}只")
        if hold_ops:
            summary_parts.append(f"持有{len(hold_ops)}只")
        if not summary_parts:
            summary_parts.append("暂无操作")

        return {
            'date': today,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_trading_day': is_trading_day(),
            'market_sentiment': market_info.get('sentiment', {}),
            'market_enter': market_info.get('should_enter', True),
            'market_reason': market_info.get('reason', ''),
            'portfolio_summary': {
                'active_count': len(active),
                'slots_used': slots_used,
                'effective_max_positions': cap,
                'max_positions': self.portfolio.max_positions,
                'available_slots': available,
                'max_hold_days': self.portfolio.max_hold_days,
                'win_rate': stats.get('win_rate', 0),
                'avg_profit': stats.get('avg_profit', 0),
                'total_trades': stats.get('total_trades', 0),
            },
            'urgent_alerts': [],
            'operations': {
                'sell': sell_ops,
                'partial_sell': partial_ops,
                'add': add_ops,
                'dip_refill': dip_ops,
                'buy': buy_ops,
                'hold': hold_ops,
            },
            'summary': '今日操作：' + '，'.join(summary_parts),
        }

    def save_operations(self, operations: Dict):
        os.makedirs(os.path.dirname(self.operations_file), exist_ok=True)
        with open(self.operations_file, 'w', encoding='utf-8') as f:
            json.dump(operations, f, ensure_ascii=False, indent=2)

    def apply_sell_actions(self, categories: Dict, today: str):
        """执行卖出操作，将持仓移入历史"""
        for stock in categories.get('sell', []):
            status = stock.get('close_status', 'closed')
            self.portfolio.close_position(
                stock['code'], stock['current_price'],
                stock.get('sell_reason', ''), status, today
            )
            self.position_mgr.remove_position(stock['code'])

    def print_operations(self, operations: Dict):
        print("\n" + "=" * 60)
        print(f"📋 今日操作建议 ({operations['date']})")
        print("=" * 60)
        ps = operations['portfolio_summary']
        eff_max = ps.get('effective_max_positions', ps['max_positions'])
        bonus = eff_max > ps['max_positions']
        slot_label = f"{ps['active_count']}/{eff_max}"
        if bonus:
            slot_label += "(半仓止损+1)"
        print(f"持仓: {slot_label} | "
              f"可买入: {ps['available_slots']}只 | "
              f"胜率: {ps['win_rate']}% | 均收益: {ps['avg_profit']:+.2f}%")
        print(f"总结: {operations['summary']}")

        alerts = operations.get('urgent_alerts', [])
        if alerts:
            print(f"\n🚨 紧急提示 ({len(alerts)}条):")
            for a in alerts:
                icon = '🔴' if a['priority'] == 'urgent' else '🟡'
                print(f"  {icon} {a['action']} {a['code']} {a['name']}: {a['reason']}")

        ops = operations['operations']
        if ops['sell']:
            print(f"\n📤 卖出信号 ({len(ops['sell'])}只):")
            for s in ops['sell']:
                print(f"  🔴 卖出 {s['code']} {s['name']} @ {s['current_price']:.2f} "
                      f"({s['profit_pct']:+.2f}%) — {s['reason']}")

        if ops['partial_sell']:
            print(f"\n💰 部分止盈 ({len(ops['partial_sell'])}只):")
            for s in ops['partial_sell']:
                print(f"  🟡 卖出{s['sell_ratio']*100:.0f}% {s['code']} {s['name']} "
                      f"({s['profit_pct']:+.2f}%) — {s['reason']}")

        if ops.get('dip_refill'):
            print(f"\n🔄 趋势补半仓 ({len(ops['dip_refill'])}只):")
            for s in ops['dip_refill']:
                print(f"  🟡 补{s.get('add_ratio', 0.5)*100:.0f}% {s['code']} {s['name']} "
                      f"({s['profit_pct']:+.2f}%) — {s['reason']}")

        if ops.get('add'):
            print(f"\n📈 强者恒强加仓 ({len(ops['add'])}只):")
            for s in ops['add']:
                if s.get('add_amount_pct'):
                    amt = s.get('add_amount', 0)
                    print(f"  🟢 加仓总资产{s['add_amount_pct']*100:.0f}%"
                          f"({amt:,.0f}元) {s['code']} {s['name']} "
                          f"({s['profit_pct']:+.2f}%) — {s['reason']}")
                else:
                    print(f"  🟢 加仓市值{s['add_ratio']*100:.0f}% {s['code']} {s['name']} "
                          f"({s['profit_pct']:+.2f}%) — {s['reason']}")

        if ops['buy']:
            print(f"\n📥 买入信号 ({len(ops['buy'])}只):")
            for b in ops['buy']:
                print(f"  🟢 买入 {b['code']} {b['name']} @ {b['buy_price']:.2f} "
                      f"置信度{b['confidence']*100:.1f}% — {b['reason']}")

        if ops['hold']:
            print(f"\n📈 继续持有 ({len(ops['hold'])}只):")
            for h in ops['hold']:
                today_chg = h.get('today_change', 0)
                print(f"  ⏳ {h['code']} {h['name']}: 今日{today_chg:+.2f}% | "
                      f"持仓{h['profit_pct']:+.2f}% "
                      f"({h['days_held']}天, 剩余{h['remaining_days']}天)")

        print("=" * 60)
