# -*- coding: utf-8 -*-
"""持仓止盈止损管理：阶梯止盈、移动止盈、动态止损"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.trading_calendar import count_trading_days
from src.limit_up_filters import limit_up_profit_trailing_hit
from src.strategy_filters import tp_conflicts_with_add


@dataclass
class Position:
    code: str
    name: str
    buy_price: float
    buy_date: str
    take_profit_levels: List[Dict] = field(default_factory=list)
    stop_loss: float = 0.0
    highest_price: float = 0.0
    partial_sold: List[Dict] = field(default_factory=list)
    add_levels_done: List[float] = field(default_factory=list)
    stop_loss_partial_done: bool = False
    dip_refill_done: bool = False
    sector: str = '其他'
    strategy_type: str = 'trend'
    trailing_start: Optional[float] = None
    trailing_pct: Optional[float] = None

    def __post_init__(self):
        if self.take_profit_levels is None:
            self.take_profit_levels = []
        if self.highest_price == 0.0:
            self.highest_price = self.buy_price


class PositionManager:
    """管理单只股票的止盈止损信号"""

    SECTOR_HOLD_MULTIPLIERS = {
        'AI/算力': 1.0, '机器人': 1.0, '低空经济': 1.0,
        '半导体': 1.0, '新能源': 1.0, '消费电子': 1.0,
        '医药': 0.93, '食品饮料': 0.93, '银行': 0.87, '其他': 1.0,
    }

    def __init__(self, trailing_start: float = 5.0, trailing_pct: float = 3.0,
                 max_hold_days: int = 15, stop_loss_pct: float = 3.0,
                 add_trigger_pct: float = 10.0, add_ratio: float = 0.0,
                 trend_add_amount_pct: float = None,
                 trend_stop_loss_partial_ratio: float = 0,
                 trend_dip_refill_pct: float = None,
                 trend_half_hold_exit_pct: float = None,
                 stop_loss_on_close: bool = True):
        self.trailing_start = trailing_start
        self.trailing_pct = trailing_pct
        self.max_hold_days = max_hold_days
        self.stop_loss_pct = stop_loss_pct
        self.add_trigger_pct = add_trigger_pct
        self.add_ratio = add_ratio
        self.trend_add_amount_pct = trend_add_amount_pct
        self.trend_stop_loss_partial_ratio = trend_stop_loss_partial_ratio
        self.trend_dip_refill_pct = trend_dip_refill_pct
        self.trend_half_hold_exit_pct = trend_half_hold_exit_pct
        self.stop_loss_on_close = stop_loss_on_close
        self.positions: Dict[str, Position] = {}

    def add_position(self, code: str, name: str, buy_price: float,
                     buy_date: str, sector: str = '其他',
                     custom_levels: Optional[List] = None,
                     stop_loss: Optional[float] = None,
                     strategy_type: str = 'trend',
                     trailing_start: Optional[float] = None,
                     trailing_pct: Optional[float] = None):
        if custom_levels is not None:
            levels = [
                {**lv, 'triggered': lv.get('triggered', False)}
                for lv in custom_levels
            ]
        elif strategy_type == 'limit_up':
            levels = []
        else:
            levels = []
        position = Position(
            code=code, name=name, buy_price=buy_price, buy_date=buy_date,
            sector=sector,
            take_profit_levels=levels,
            stop_loss=stop_loss if stop_loss else buy_price * (1 - self.stop_loss_pct / 100),
            strategy_type=strategy_type,
            trailing_start=trailing_start,
            trailing_pct=trailing_pct,
        )
        self.positions[code] = position
        return position

    def get_dynamic_hold_days(self, sector: str) -> int:
        mult = self.SECTOR_HOLD_MULTIPLIERS.get(sector, 1.0)
        return min(self.max_hold_days, int(self.max_hold_days * mult))

    def update_price(self, code: str, current_price: float, current_date: str,
                     total_equity: float = None,
                     confirm_stop_close: bool = False,
                     defensive: bool = False) -> Dict:
        if code not in self.positions:
            return {'action': 'none', 'reason': '持仓不存在'}

        position = self.positions[code]
        position.highest_price = max(position.highest_price, current_price)
        profit_pct = (current_price - position.buy_price) / position.buy_price * 100
        days_held = count_trading_days(position.buy_date, current_date) - 1
        days_held = max(0, days_held)

        result = {
            'code': code, 'name': position.name,
            'profit_pct': profit_pct, 'days_held': days_held,
            'action': 'hold', 'reason': '', 'sell_ratio': 0.0,
            'sell_price': current_price, 'urgency': 'normal',
        }

        for level in position.take_profit_levels:
            if (not level['triggered'] and profit_pct >= level['pct']
                    and not tp_conflicts_with_add(
                        level['pct'], self.add_trigger_pct, self.add_ratio)):
                level['triggered'] = True
                result.update({
                    'action': 'partial_sell', 'sell_ratio': level['ratio'],
                    'reason': f"阶梯止盈第{level['level']}档 (+{level['pct']}% 卖出{level['ratio']*100:.0f}%)",
                    'urgency': 'high',
                })
                position.partial_sold.append({
                    'date': current_date, 'price': current_price,
                    'ratio': level['ratio'], 'profit_pct': profit_pct,
                })
                return result

        trail_start = position.trailing_start if position.trailing_start is not None else self.trailing_start
        trail_pct = position.trailing_pct if position.trailing_pct is not None else self.trailing_pct
        if defensive:
            trail_start = min(trail_start, 8.0)
            trail_pct = min(trail_pct, 3.5)
        max_profit_pct = (position.highest_price - position.buy_price) / position.buy_price * 100
        if profit_pct > 0 and position.strategy_type == 'limit_up':
            hit, drawback, threshold = limit_up_profit_trailing_hit(
                max_profit_pct, profit_pct, trail_start, trail_pct)
            if hit:
                result.update({
                    'action': 'sell', 'sell_ratio': 1.0, 'urgency': 'urgent',
                    'reason': (f"涨停回撤止盈 (最高{max_profit_pct:.1f}% "
                               f"回落{drawback:.1f}%≥{threshold:.1f}%)"),
                })
                return result
        elif max_profit_pct >= trail_start and profit_pct > 0:
            price_drawdown = (position.highest_price - current_price) / position.highest_price * 100
            if price_drawdown >= trail_pct:
                tag = {'doubler': '翻倍移动止盈'}.get(position.strategy_type, '移动止盈')
                result.update({
                    'action': 'sell', 'sell_ratio': 1.0, 'urgency': 'urgent',
                    'reason': f"{tag} (最高{max_profit_pct:.1f}% 价回撤{price_drawdown:.1f}%)",
                })
                return result

        stop_ok = not self.stop_loss_on_close or confirm_stop_close
        if stop_ok and current_price <= position.stop_loss:
            if (position.strategy_type == 'trend'
                    and self.trend_stop_loss_partial_ratio > 0
                    and not position.stop_loss_partial_done):
                position.stop_loss_partial_done = True
                result.update({
                    'action': 'partial_sell',
                    'sell_ratio': self.trend_stop_loss_partial_ratio,
                    'urgency': 'urgent',
                    'reason': f"趋势止损卖出{int(self.trend_stop_loss_partial_ratio * 100)}% (亏损{profit_pct:.1f}%)",
                })
                position.partial_sold.append({
                    'date': current_date, 'price': current_price,
                    'ratio': self.trend_stop_loss_partial_ratio,
                    'profit_pct': profit_pct, 'type': 'stop_loss_half',
                })
                return result
            result.update({
                'action': 'sell', 'sell_ratio': 1.0, 'urgency': 'urgent',
                'reason': (
                    f"趋势止损 (亏损{profit_pct:.1f}%)"
                    if position.strategy_type == 'trend'
                    else f"止损触发 (亏损{profit_pct:.1f}%)"
                ),
            })
            return result

        hold_days = self.get_dynamic_hold_days(position.sector)
        if days_held >= hold_days:
            result.update({
                'action': 'sell', 'sell_ratio': 1.0, 'urgency': 'high',
                'reason': f"持仓期满 ({days_held}天 >= {hold_days}天)",
            })
            return result

        if profit_pct >= 3.0:
            new_stop = max(position.stop_loss, position.buy_price)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                result['reason'] = "移动止损线上移至成本价"

        if profit_pct >= 8.0:
            new_stop = max(position.stop_loss, position.buy_price * 1.05)
            if new_stop > position.stop_loss:
                position.stop_loss = new_stop
                result['reason'] = "移动止损线上移至+5%"

        if (self.trend_dip_refill_pct is not None
                and position.strategy_type == 'trend'
                and position.stop_loss_partial_done
                and not position.dip_refill_done
                and profit_pct <= self.trend_dip_refill_pct):
            position.dip_refill_done = True
            result.update({
                'action': 'dip_refill',
                'add_ratio': self.trend_stop_loss_partial_ratio,
                'reason': f"趋势跌{self.trend_dip_refill_pct:.0f}%补半仓 (当前{profit_pct:.1f}%)",
                'urgency': 'normal',
            })
            return result

        if (position.strategy_type == 'trend'
                and position.stop_loss_partial_done
                and self.trend_half_hold_exit_pct is not None
                and profit_pct <= self.trend_half_hold_exit_pct):
            result.update({
                'action': 'sell', 'sell_ratio': 1.0, 'urgency': 'urgent',
                'reason': f"趋势余仓清仓 (当前{profit_pct:.1f}%≤{self.trend_half_hold_exit_pct:.0f}%)",
            })
            return result

        add_signal = self._check_add_position(position, current_price, total_equity)
        if add_signal:
            result.update(add_signal)
            return result

        remaining_days = hold_days - days_held
        if remaining_days <= 2 and profit_pct < 0:
            result['urgency'] = 'high'
            result['reason'] = result['reason'] or f"临近到期({remaining_days}天)，亏损{profit_pct:.1f}%"

        return result

    def _check_add_position(self, position: Position, current_price: float,
                            total_equity: float = None) -> Optional[Dict]:
        use_equity = (
            position.strategy_type == 'trend'
            and self.trend_add_amount_pct is not None
        )
        if use_equity:
            if not total_equity or total_equity <= 0:
                return None
        elif self.add_ratio <= 0:
            return None
        max_profit_pct = (position.highest_price - position.buy_price) / position.buy_price * 100
        level = int(max_profit_pct // self.add_trigger_pct) * self.add_trigger_pct
        if level < self.add_trigger_pct or level in position.add_levels_done:
            return None
        position.add_levels_done.append(level)
        tp_ratio = self.add_ratio if self.add_ratio > 0 else 0.01
        for tp in position.take_profit_levels:
            if tp_conflicts_with_add(tp['pct'], self.add_trigger_pct, tp_ratio):
                tp['triggered'] = True
        if use_equity:
            return {
                'action': 'add',
                'add_amount_pct': self.trend_add_amount_pct,
                'add_amount': round(total_equity * self.trend_add_amount_pct, 2),
                'reason': (f'强者恒强 盈利+{level:.0f}%加仓'
                           f'总资产{self.trend_add_amount_pct * 100:.0f}%'),
                'urgency': 'normal',
            }
        return {
            'action': 'add',
            'add_ratio': self.add_ratio,
            'reason': f'强者恒强 盈利+{level:.0f}%加仓市值{self.add_ratio * 100:.0f}%',
            'urgency': 'normal',
        }

    def remove_position(self, code: str):
        self.positions.pop(code, None)

    def load_from_dict(self, positions_data: Dict):
        """从持仓数据恢复 PositionManager 状态"""
        for code, info in positions_data.items():
            if code not in self.positions:
                self.add_position(
                    code=code, name=info.get('name', code),
                    buy_price=info.get('buy_price', 0),
                    buy_date=info.get('buy_date', info.get('added_date', '')),
                    sector=info.get('sector', '其他'),
                    custom_levels=info.get('take_profit_levels'),
                    stop_loss=info.get('current_stop_loss') or info.get('stop_loss'),
                    strategy_type=info.get('strategy_type', 'trend'),
                    trailing_start=info.get('trailing_start'),
                    trailing_pct=info.get('trailing_pct'),
                )
            pos = self.positions[code]
            pos.partial_sold = info.get('partial_sold', [])
            pos.add_levels_done = list(info.get('add_levels_done', []))
            pos.stop_loss_partial_done = info.get('stop_loss_partial_done', False)
            pos.dip_refill_done = info.get('dip_refill_done', False)
            pos.highest_price = info.get('highest_price', pos.buy_price)
            pos.strategy_type = info.get('strategy_type', pos.strategy_type)
            if info.get('trailing_start') is not None:
                pos.trailing_start = info['trailing_start']
            if info.get('trailing_pct') is not None:
                pos.trailing_pct = info['trailing_pct']
            for level in pos.take_profit_levels:
                level['triggered'] = any(
                    p.get('profit_pct', 0) >= level['pct'] for p in pos.partial_sold
                )
