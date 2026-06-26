# -*- coding: utf-8 -*-
"""
资金回测引擎

规则:
  - 初始资金 10万，每次新建仓买入 1万
  - 强者恒强：盈利每上涨 add_trigger_pct% 加仓 add_ratio（按当前市值，每档仅加一次）
  - 止盈止损：移动止盈 + 止损（阶梯止盈由 take_profit_levels 配置，默认关闭）
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.strategy_filters import screen_stock_row, rank_candidates, tp_conflicts_with_add
from src.market_regime import (
    analyze_market_regime, apply_mainline_to_candidate, should_defensive_exit,
)
from src.limit_up_filters import (
    load_limit_up_config, limit_up_profit_trailing_hit,
    screen_limit_up_row, merge_strategy_candidates, is_limit_up_bar,
)
from src.doubler_filters import load_doubler_config, screen_doubler_row, apply_doubler_boost, _independent_enabled
from src.shakeout_strategy import (
    should_watch_after_stop, eval_shakeout_rebuy, merge_shakeout_cfg,
)


@dataclass
class CapitalPosition:
    code: str
    shares: float
    cost_total: float
    avg_cost: float
    buy_date: str
    days_held: int = 0
    highest_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_levels: List[Dict] = field(default_factory=list)
    partial_sold: List[Dict] = field(default_factory=list)
    add_levels_done: List[float] = field(default_factory=list)
    stop_loss_partial_done: bool = False
    dip_refill_done: bool = False
    shares_sold_at_stop: float = 0.0
    strategy_type: str = 'trend'
    trailing_start: float = 10.0
    trailing_pct: float = 4.0
    stop_loss_pct: float = 0.03
    shakeout_rebuy: bool = False

    def market_value(self, price: float) -> float:
        return self.shares * price

    def profit_pct(self, price: float) -> float:
        if self.avg_cost <= 0:
            return 0.0
        return (price - self.avg_cost) / self.avg_cost * 100


class CapitalBacktester:
    def __init__(self, initial_capital: float = 100000, buy_amount: float = 10000,
                 buy_amount_pct: float = None,
                 add_trigger_pct: float = 5.0, add_ratio: float = 0.30,
                 stop_loss_pct: float = 3.5, trailing_start: float = 4.0,
                 trailing_pct: float = 2.5, max_hold_days: int = 15,
                 max_positions: int = 5, commission: float = 0.0015,
                 stamp_tax: float = 0.001, compound_buy: bool = False,
                 max_buy_pct: float = 0.12, ml_scorer=None,
                 ml_data: Dict = None, ml_scale_buy: bool = False,
                 compound_only_profit: bool = True,
                 trend_buy_amount_pct: float = None,
                 trend_max_buy_pct: float = None,
                 trend_add_amount_pct: float = None):
        self.initial_capital = initial_capital
        self.buy_amount_pct = buy_amount_pct
        self.trend_buy_amount_pct = trend_buy_amount_pct
        self.trend_max_buy_pct = trend_max_buy_pct or buy_amount_pct
        self.trend_add_amount_pct = trend_add_amount_pct
        self.buy_amount = (
            initial_capital * buy_amount_pct if buy_amount_pct is not None else buy_amount
        )
        self.add_trigger_pct = add_trigger_pct
        self.add_ratio = add_ratio
        self.stop_loss_pct = stop_loss_pct / 100
        self.trailing_start = trailing_start
        self.trailing_pct = trailing_pct
        self.max_hold_days = max_hold_days
        self.max_positions = max_positions
        self.commission = commission
        self.stamp_tax = stamp_tax
        self.compound_buy = compound_buy
        self.max_buy_pct = max_buy_pct
        self.ml_scorer = ml_scorer
        self.ml_data = ml_data or {}
        self.ml_scale_buy = ml_scale_buy
        self.compound_only_profit = compound_only_profit

        self.cash = initial_capital
        self.positions: Dict[str, CapitalPosition] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_log: List[Dict] = []
        self._closed_wins = 0
        self._closed_losses = 0
        self._cumulative_profit = 0.0
        self._tp_template = None
        self._full_config = {}
        self._sector_map: Dict[str, str] = {}
        self._regime_log: List[Dict] = []
        self._shakeout_watch: Dict[str, Dict] = {}
        self._stop_cooldown: Dict[str, int] = {}

    def _count_trend_positions(self) -> int:
        return sum(1 for p in self.positions.values() if not p.shakeout_rebuy)

    def _shakeout_market_value(self, all_data: Dict, date: str) -> float:
        total = 0.0
        for code, pos in self.positions.items():
            if not pos.shakeout_rebuy:
                continue
            row = self._get_bar(all_data, code, date, forward_fill=True, flat_if_missing=False)
            if row is not None:
                total += pos.market_value(float(row['close']))
        return total

    def _shakeout_reserve_target(self, all_data: Dict, date: str, strat_cfg: dict) -> float:
        cfg = merge_shakeout_cfg(strat_cfg)
        pct = float(cfg.get('shakeout_reserve_pct', 0))
        if pct <= 0:
            return 0.0
        return self._equity(all_data, date) * pct / 100

    @staticmethod
    def _entry_snapshot(cand: dict) -> dict:
        sig = cand.get('signals') or {}
        def _n(v, nd=2):
            if v is None:
                return None
            try:
                return round(float(v), nd)
            except (TypeError, ValueError):
                return v
        def _b(v):
            return bool(v) if v is not None else None
        def _i(v):
            return int(v) if v is not None else None
        return {
            'change': _n(cand.get('change')),
            'tech_score': _n(cand.get('tech_score')),
            'rsi': _n(cand.get('rsi') or sig.get('rsi')),
            'volume_ratio': _n(cand.get('volume_ratio') or sig.get('volume_ratio')),
            'score': _n(cand.get('score')),
            'ml_score': _n(cand.get('ml_score'), 4),
            'close_strength': _n(sig.get('close_strength')),
            'conditions_met': _i(sig.get('conditions_met')),
            'rsi_ok': _b(sig.get('rsi_ok')),
            'macd_ok': _b(sig.get('macd_ok')),
            'macd_golden': _b(sig.get('macd_golden')),
            'volume_ok': _b(sig.get('volume_ok')),
            'ma_ok': _b(sig.get('ma_ok')),
            'bullish_candle': _b(sig.get('bullish_candle')),
        }

    def _calc_buy_amount(self, all_data: Dict, date: str, ml_score: float = 0.5,
                         strategy_type: str = 'trend', is_new_position: bool = True,
                         strat_cfg: dict = None, shakeout_rebuy: bool = False) -> float:
        equity = self._equity(all_data, date)
        reserve_pct = 0.0
        if strat_cfg:
            reserve_pct = float(merge_shakeout_cfg(strat_cfg).get('shakeout_reserve_pct', 0))
        trend_equity = equity * (1 - reserve_pct / 100) if reserve_pct > 0 else equity

        if shakeout_rebuy and strat_cfg and reserve_pct > 0:
            reserve_target = equity * reserve_pct / 100
            shakeout_mv = self._shakeout_market_value(all_data, date)
            pool_left = max(0.0, reserve_target - shakeout_mv)
            size_ratio = float(merge_shakeout_cfg(strat_cfg).get('shakeout_rebuy_size_ratio', 0.5))
            return max(0.0, min(pool_left * size_ratio, self.cash))

        pct = self.buy_amount_pct
        cap_pct = self.max_buy_pct
        if strategy_type == 'trend' and is_new_position and self.trend_buy_amount_pct is not None:
            pct = self.trend_buy_amount_pct
            cap_pct = self.trend_max_buy_pct or pct
        base_equity = trend_equity if reserve_pct > 0 and strategy_type == 'trend' else equity
        if pct is not None:
            base = base_equity * pct
        else:
            base = self.buy_amount
            if self.compound_buy:
                if not self.compound_only_profit or equity > self.initial_capital:
                    base = self.buy_amount * max(1.0, equity / self.initial_capital)
        if self.ml_scale_buy and ml_score > 0:
            base *= 0.7 + 0.6 * ml_score
        cap = base_equity * (cap_pct if cap_pct else self.max_buy_pct)
        floor = base_equity * (pct or 0) * 0.5 if pct else self.buy_amount * 0.5
        amount = max(floor, min(base, cap, self.cash))
        if reserve_pct > 0 and strategy_type == 'trend':
            reserve_need = self._shakeout_reserve_target(all_data, date, strat_cfg)
            shakeout_mv = self._shakeout_market_value(all_data, date)
            min_cash = max(0.0, reserve_need - shakeout_mv)
            amount = min(amount, max(0.0, self.cash - min_cash))
        return amount

    def _default_tp_levels(self, template=None):
        tpl = template if template is not None else self._tp_template
        if tpl is not None:
            if not tpl:
                return []
            return [
                {'level': i + 1, 'pct': lv['pct'], 'ratio': lv['ratio'], 'triggered': False}
                for i, lv in enumerate(tpl)
            ]
        return []

    def _exit_params(self, strategy_type: str) -> Dict:
        if strategy_type == 'limit_up':
            lu = load_limit_up_config(self._full_config)
            if not lu.get('use_trend_exit'):
                sl = float(lu.get('stop_loss', 5.0)) / 100
                return {
                    'stop_loss_pct': sl,
                    'take_profit_levels': lu.get('take_profit_levels'),
                    'trailing_start': float(lu.get('trailing_start', 15.0)),
                    'trailing_pct': float(lu.get('trailing_pct', 10.0)),
                }
        if strategy_type == 'doubler':
            db = load_doubler_config(self._full_config)
            sl = float(db.get('stop_loss', 4.0)) / 100
            return {
                'stop_loss_pct': sl,
                'take_profit_levels': db.get('take_profit_levels'),
                'trailing_start': float(db.get('trailing_start', 20.0)),
                'trailing_pct': float(db.get('trailing_pct', 8.0)),
            }
        return {
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_levels': self._tp_template,
            'trailing_start': self.trailing_start,
            'trailing_pct': self.trailing_pct,
        }

    def _buy(self, code: str, price: float, amount: float, date: str, reason: str,
             strategy_type: str = 'trend', shakeout_rebuy: bool = False,
             strat_cfg: dict = None) -> bool:
        if amount <= 0 or self.cash < amount or price <= 0:
            return False
        fee = amount * self.commission
        net = amount - fee
        shares = net / price
        if shares <= 0:
            return False
        self.cash -= amount

        if code in self.positions:
            pos = self.positions[code]
            pos.cost_total += amount
            pos.shares += shares
            pos.avg_cost = pos.cost_total / pos.shares
            pos.stop_loss = pos.avg_cost * (1 - pos.stop_loss_pct)
        else:
            exit_cfg = self._exit_params(strategy_type)
            sl_pct = exit_cfg['stop_loss_pct']
            if shakeout_rebuy and strat_cfg:
                rebuy_sl = merge_shakeout_cfg(strat_cfg).get('shakeout_rebuy_stop_loss')
                if rebuy_sl is not None:
                    sl_pct = float(rebuy_sl) / 100
            self.positions[code] = CapitalPosition(
                code=code, shares=shares, cost_total=amount, avg_cost=price,
                buy_date=date, highest_price=price,
                stop_loss=price * (1 - sl_pct),
                take_profit_levels=self._default_tp_levels(exit_cfg['take_profit_levels']),
                strategy_type=strategy_type,
                trailing_start=exit_cfg['trailing_start'],
                trailing_pct=exit_cfg['trailing_pct'],
                stop_loss_pct=sl_pct,
                shakeout_rebuy=shakeout_rebuy,
            )
        return True

    def _sell(self, code: str, price: float, ratio: float, date: str, reason: str) -> float:
        if code not in self.positions or ratio <= 0:
            return 0.0
        pos = self.positions[code]
        sell_shares = pos.shares * ratio
        if sell_shares <= 0:
            return 0.0

        gross = sell_shares * price
        fee = gross * (self.commission + self.stamp_tax)
        net = gross - fee
        cost_portion = pos.cost_total * (sell_shares / pos.shares)
        profit = net - cost_portion
        profit_pct = profit / cost_portion * 100 if cost_portion > 0 else 0

        self.cash += net
        pos.shares -= sell_shares
        pos.cost_total -= cost_portion
        self._cumulative_profit += profit
        if profit > 0:
            self._closed_wins += 1
        else:
            self._closed_losses += 1

        stype = pos.strategy_type if code in self.positions else 'trend'
        self.trades.append({
            'code': code, 'date': date, 'action': 'sell', 'reason': reason,
            'price': round(price, 2), 'amount': round(gross, 2),
            'profit': round(profit, 2), 'profit_pct': round(profit_pct, 2),
            'shares_ratio': round(ratio, 2),
            'cumulative_profit': round(self._cumulative_profit, 2),
            'strategy_type': stype,
        })

        if pos.shares < 1e-8:
            del self.positions[code]
        return profit

    def _check_trailing_exit(self, code: str, pos: CapitalPosition, close: float, date: str) -> bool:
        max_profit = (pos.highest_price - pos.avg_cost) / pos.avg_cost * 100
        current_profit = pos.profit_pct(close)
        if current_profit <= 0:
            return False

        if pos.strategy_type == 'limit_up':
            hit, drawback, threshold = limit_up_profit_trailing_hit(
                max_profit, current_profit, pos.trailing_start, pos.trailing_pct)
            if not hit:
                return False
            self._sell(code, close, 1.0, date,
                       f'涨停回撤止盈(最高{max_profit:.1f}% 回落{drawback:.1f}%≥{threshold:.1f}%)')
            return True

        if max_profit < pos.trailing_start:
            return False
        price_drawdown = (pos.highest_price - close) / pos.highest_price * 100
        if price_drawdown < pos.trailing_pct:
            return False
        tag = '翻倍移动止盈' if pos.strategy_type == 'doubler' else '移动止盈'
        self._sell(code, close, 1.0, date,
                   f'{tag}(最高{max_profit:.1f}% 价回撤{price_drawdown:.1f}%)')
        return True

    def _calc_add_amount(self, pos: CapitalPosition, close: float,
                        all_data: Dict, date: str) -> Tuple[float, str]:
        """趋势加仓：优先按总资产比例，否则按持仓市值比例。"""
        if pos.strategy_type == 'trend' and self.trend_add_amount_pct is not None:
            equity = self._equity(all_data, date)
            amt = equity * self.trend_add_amount_pct
            label = f'总资产{self.trend_add_amount_pct * 100:.0f}%'
            return min(amt, self.cash), label
        mkt_val = pos.market_value(close)
        amt = mkt_val * self.add_ratio
        return min(amt, self.cash), f'市值{self.add_ratio * 100:.0f}%'

    def _check_add_position(self, pos: CapitalPosition, row, date: str,
                            all_data: Dict, disable_add: bool = False):
        """强者恒强：盈利达触发档位后加仓（趋势可用总资产比例）"""
        if disable_add:
            return
        if pos.strategy_type == 'trend':
            if self.trend_add_amount_pct is None and self.add_ratio <= 0:
                return
        elif self.add_ratio <= 0:
            return
        high = float(row['high'])
        close = float(row['close'])
        profit_high = pos.profit_pct(high)

        level = int(profit_high // self.add_trigger_pct) * self.add_trigger_pct
        if level < self.add_trigger_pct:
            return

        if level in pos.add_levels_done:
            return

        add_amount, add_label = self._calc_add_amount(pos, close, all_data, date)
        if add_amount < 1000:
            return

        if self._buy(pos.code, close, add_amount, date, f'加仓+{level:.0f}%'):
            pos.add_levels_done.append(level)
            add_ratio_for_tp = self.add_ratio if self.add_ratio > 0 else 0.01
            for tp in pos.take_profit_levels:
                if tp_conflicts_with_add(tp['pct'], self.add_trigger_pct, add_ratio_for_tp):
                    tp['triggered'] = True
            self.trades.append({
                'code': pos.code, 'date': date, 'action': 'add',
                'reason': f'盈利+{level:.0f}%加仓({add_label})',
                'price': round(close, 2), 'amount': round(add_amount, 2),
            })

    def _register_shakeout_watch(self, code: str, sell_price: float, cost: float,
                                 date: str, strat_cfg: dict):
        cfg = merge_shakeout_cfg(strat_cfg)
        day_idx = self._day_index.get(date, -1)
        if day_idx < 0:
            return
        self._shakeout_watch[code] = {
            'code': code,
            'sell_date': date,
            'sell_idx': day_idx,
            'sell_price': sell_price,
            'cost_price': cost,
            'expire_idx': day_idx + int(cfg['shakeout_rebuy_days']),
        }

    def _register_stop_cooldown(self, code: str, date: str, strat_cfg: dict,
                               shakeout_rebuy: bool = False):
        if shakeout_rebuy:
            return
        days = int(strat_cfg.get('stop_cooldown_days', 0))
        if days <= 0:
            return
        day_idx = self._day_index.get(date, -1)
        if day_idx < 0:
            return
        self._stop_cooldown[code] = day_idx + days

    def _in_stop_cooldown(self, code: str, day_idx: int) -> bool:
        end = self._stop_cooldown.get(code)
        return end is not None and day_idx <= end

    def _shakeout_rebuy_candidates(self, all_data: Dict, date: str, day_idx: int,
                                   held: set, strat_cfg: dict) -> List[Dict]:
        cfg = merge_shakeout_cfg(strat_cfg)
        if not cfg['shakeout_rebuy_enabled']:
            return []
        dt = pd.Timestamp(date)
        cands = []
        for code in list(self._shakeout_watch.keys()):
            watch = self._shakeout_watch[code]
            if day_idx > watch['expire_idx']:
                del self._shakeout_watch[code]
                continue
            max_from_stop = int(cfg.get('shakeout_rebuy_max_days_from_stop', 0))
            if max_from_stop > 0:
                sell_idx = watch.get('sell_idx', self._day_index.get(watch['sell_date'], -1))
                if sell_idx < 0 or day_idx - sell_idx > max_from_stop:
                    del self._shakeout_watch[code]
                    continue
            if code in held:
                continue
            df = all_data.get(code)
            if df is None:
                continue
            row = self._get_bar(all_data, code, date, forward_fill=False, flat_if_missing=False)
            if row is None:
                continue
            hist = df.loc[df.index <= dt]
            row = row.copy()
            row['code'] = code
            rebuy = eval_shakeout_rebuy(
                watch, row, hist, strat_cfg, regime=self._current_regime,
            )
            if rebuy:
                if cfg.get('shakeout_rebuy_require_hot_sector') and strat_cfg.get('require_hot_sector'):
                    from src.market_regime import get_hot_sector_pool_names, codes_in_hot_sectors
                    hot_names = get_hot_sector_pool_names(self._current_regime, strat_cfg)
                    if hot_names:
                        allowed = codes_in_hot_sectors(self._sector_map, hot_names)
                        if code not in allowed:
                            continue
                rebuy['buy_date'] = date
                cands.append(rebuy)
        cands.sort(key=lambda x: x.get('score', 0), reverse=True)
        return cands[:int(cfg['shakeout_rebuy_max_daily'])]

    def _check_trend_stop_loss(self, pos: CapitalPosition, trigger_px: float, date: str,
                               strat_cfg: dict) -> bool:
        """趋势选股：partial>0 时卖半仓；partial=0 时走下方全仓止损。"""
        if pos.strategy_type != 'trend' or trigger_px > pos.stop_loss:
            return False
        partial = strat_cfg.get('trend_stop_loss_partial_ratio', 0.5)
        if not pos.stop_loss_partial_done and partial > 0:
            pos.shares_sold_at_stop = pos.shares * partial
            pct = int(round(partial * 100))
            self._sell(
                pos.code, pos.stop_loss, partial, date,
                f'趋势止损卖出{pct}%({pos.profit_pct(pos.stop_loss):.1f}%)',
            )
            pos.stop_loss_partial_done = True
            return True
        return False

    def _check_trend_dip_refill(self, pos: CapitalPosition, close: float, date: str,
                                strat_cfg: dict) -> bool:
        """趋势选股：止损半仓后继续下跌超阈值，补回半仓（trend_dip_refill_pct=null 时关闭）。"""
        if pos.strategy_type != 'trend':
            return False
        threshold = strat_cfg.get('trend_dip_refill_pct')
        if threshold is None:
            return False
        if not pos.stop_loss_partial_done or pos.dip_refill_done:
            return False
        if pos.profit_pct(close) > threshold:
            return False
        if pos.shares_sold_at_stop > 0:
            refill_amount = pos.shares_sold_at_stop * close * (1 + self.commission)
        else:
            refill_amount = pos.market_value(close) * strat_cfg.get(
                'trend_stop_loss_partial_ratio', 0.5)
        refill_amount = min(refill_amount, self.cash)
        if refill_amount < 1000:
            return False
        if self._buy(pos.code, close, refill_amount, date,
                     f'趋势跌{threshold:.0f}%补半仓'):
            pos.dip_refill_done = True
            self.trades.append({
                'code': pos.code, 'date': date, 'action': 'dip_refill',
                'reason': f'跌{pos.profit_pct(close):.1f}%补半仓',
                'price': round(close, 2), 'amount': round(refill_amount, 2),
                'strategy_type': 'trend',
            })
            return True
        return False

    def _check_trend_half_hold_exit(self, pos: CapitalPosition, low: float, date: str,
                                    strat_cfg: dict) -> bool:
        """趋势：半仓止损后继续持有，跌幅超阈值则清仓释放资金。"""
        if pos.strategy_type != 'trend' or not pos.stop_loss_partial_done:
            return False
        exit_pct = strat_cfg.get('trend_half_hold_exit_pct')
        if exit_pct is None:
            return False
        if pos.profit_pct(low) > exit_pct:
            return False
        stop_price = low
        cost = pos.avg_cost
        reason = f'趋势余仓清仓({pos.profit_pct(low):.1f}%≤{exit_pct:.0f}%)'
        self._sell(pos.code, stop_price, 1.0, date, reason)
        if should_watch_after_stop(reason, pos.strategy_type, 1.0, strat_cfg):
            self._register_shakeout_watch(pos.code, stop_price, cost, date, strat_cfg)
        return True

    def _process_position(self, code: str, row, date: str, all_data: Dict,
                          regime: dict = None, strat_cfg: dict = None):
        if code not in self.positions:
            return
        pos = self.positions[code]
        if pos.buy_date == date:
            return

        pos.days_held += 1
        low, high, close = float(row['low']), float(row['high']), float(row['close'])
        pos.highest_price = max(pos.highest_price, high)

        eff_trail_start = pos.trailing_start
        eff_trail_pct = pos.trailing_pct
        if regime and regime.get('defensive'):
            eff_trail_start = min(pos.trailing_start, 8.0)
            eff_trail_pct = min(pos.trailing_pct, 3.5)

        strat_cfg = strat_cfg or {}
        disable_add = bool(regime and regime.get('disable_add'))
        self._check_add_position(pos, row, date, all_data, disable_add=disable_add)

        stop_px = close if strat_cfg.get('stop_loss_on_close', False) else low
        if self._check_trend_stop_loss(pos, stop_px, date, strat_cfg):
            return
        trend_half_stopped = (
            pos.strategy_type == 'trend' and pos.stop_loss_partial_done
        )
        skip_stop = False
        if pos.shakeout_rebuy:
            grace = int(merge_shakeout_cfg(strat_cfg).get('shakeout_rebuy_stop_grace_days', 0))
            if pos.days_held <= grace:
                skip_stop = True
        if not skip_stop and stop_px <= pos.stop_loss and not trend_half_stopped:
            stop_price = pos.stop_loss
            cost = pos.avg_cost
            stype = pos.strategy_type
            reason = f'止损({pos.profit_pct(stop_price):.1f}%)'
            shakeout_flag = pos.shakeout_rebuy
            self._sell(code, stop_price, 1.0, date, reason)
            if should_watch_after_stop(reason, stype, 1.0, strat_cfg):
                self._register_shakeout_watch(code, stop_price, cost, date, strat_cfg)
            self._register_stop_cooldown(code, date, strat_cfg, shakeout_rebuy=shakeout_flag)
            return

        if self._check_trend_dip_refill(pos, close, date, strat_cfg):
            return

        if self._check_trend_half_hold_exit(pos, stop_px, date, strat_cfg):
            return

        for level in pos.take_profit_levels:
            if not level['triggered']:
                if tp_conflicts_with_add(level['pct'], self.add_trigger_pct, self.add_ratio):
                    continue
                tp_price = pos.avg_cost * (1 + level['pct'] / 100)
                if high >= tp_price:
                    level['triggered'] = True
                    self._sell(code, tp_price, level['ratio'], date,
                               f"阶梯止盈L{level['level']}(+{level['pct']}%)")
                    if code not in self.positions:
                        return

        if code not in self.positions:
            return

        saved_start, saved_pct = pos.trailing_start, pos.trailing_pct
        pos.trailing_start, pos.trailing_pct = eff_trail_start, eff_trail_pct
        if self._check_trailing_exit(code, pos, close, date):
            return
        pos.trailing_start, pos.trailing_pct = saved_start, saved_pct

        if pos.days_held >= self.max_hold_days:
            self._sell(code, close, 1.0, date, f'持仓期满({pos.days_held}天)')

    @staticmethod
    def _get_bar(all_data: Dict, code: str, date: str,
                 forward_fill: bool = True, flat_if_missing: bool = False) -> Optional[pd.Series]:
        """取当日 K 线；无数据时用上一交易日收盘价前向填充，避免资产估值归零。"""
        if code not in all_data:
            return None
        df = all_data[code]
        if df is None or df.empty:
            return None

        dt = pd.Timestamp(date)
        exact = dt in df.index
        if exact:
            row = df.loc[dt]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            return row

        if not forward_fill:
            return None
        prior = df.loc[df.index <= dt]
        if prior.empty:
            return None
        row = prior.iloc[-1].copy()
        if flat_if_missing:
            c = float(row['close'])
            row['open'] = c
            row['high'] = c
            row['low'] = c
            row['close'] = c
            if 'pct_change' in row.index:
                row['pct_change'] = 0.0
        return row

    def _append_daily(self, all_data: Dict, date: str):
        eq = self._equity(all_data, date)
        profit = eq - self.initial_capital
        closed = self._closed_wins + self._closed_losses
        win_rate = (self._closed_wins / closed * 100) if closed > 0 else 0
        day_trades = [t for t in self.trades if t['date'] == date]
        self.equity_curve.append({'date': date, 'equity': round(eq, 2)})
        self.daily_log.append({
            'date': date,
            'equity': round(eq, 2),
            'cash': round(self.cash, 2),
            'profit': round(profit, 2),
            'profit_pct': round(profit / self.initial_capital * 100, 2),
            'win_rate': round(win_rate, 1),
            'closed_trades': closed,
            'wins': self._closed_wins,
            'losses': self._closed_losses,
            'positions': len(self.positions),
            'effective_max_positions': self._effective_max_positions(self.max_positions),
            'day_buys': sum(1 for t in day_trades if t['action'] == 'buy'),
            'day_sells': sum(1 for t in day_trades if t['action'] == 'sell'),
            'day_adds': sum(1 for t in day_trades if t['action'] == 'add'),
            'market_score': self._current_regime.get('score'),
            'market_regime': self._current_regime.get('regime'),
            'mainlines': self._current_regime.get('mainlines', [])[:3],
        })

    def _has_trend_half_stop(self) -> bool:
        return any(
            p.strategy_type == 'trend' and p.stop_loss_partial_done
            for p in self.positions.values()
        )

    def _effective_max_positions(self, day_max: int) -> int:
        """存在趋势止损半仓时，总持仓上限+1。"""
        return day_max + (1 if self._has_trend_half_stop() else 0)

    def _equity(self, all_data: Dict, date: str) -> float:
        total = self.cash
        for code, pos in self.positions.items():
            row = self._get_bar(all_data, code, date, forward_fill=True, flat_if_missing=False)
            if row is not None:
                total += pos.market_value(float(row['close']))
        return total

    def run(self, all_data: Dict[str, pd.DataFrame], trading_days: List[str],
            strat_cfg: dict, screen_fn=None, full_config: dict = None) -> Dict:
        self._full_config = full_config or {}
        if screen_fn is None:
            screen_fn = self._default_screen
        elif screen_fn == 'default':
            screen_fn = self._default_screen

        self._tp_template = strat_cfg.get('take_profit_levels')
        skipped = 0
        use_market = strat_cfg.get('market_min_score', 0) > 0
        self._regime_log = []
        self._shakeout_watch = {}
        self._stop_cooldown = {}
        self._day_index = {d: i for i, d in enumerate(trading_days)}
        if not self._sector_map:
            try:
                from stock_pool_manager import get_pool_manager
                self._sector_map = get_pool_manager().stock_sectors
            except Exception:
                self._sector_map = {}
        self._current_regime = {}
        total_days = len(trading_days)
        active_label = ','.join(strat_cfg.get('active_strategies') or ['all'])

        for day_idx, date in enumerate(trading_days, 1):
            regime = analyze_market_regime(
                all_data, date, strat_cfg, self._sector_map, trading_days,
            )
            self._current_regime = regime
            from src.market_regime import summarize_hot_sector_pool
            top_n = int(strat_cfg.get('hot_sector_top_n', strat_cfg.get('mainline_top_n', 5)))
            mainlines_top = regime.get('mainlines', [])[:top_n]
            hot_sector_names = [m['sector'] for m in mainlines_top if m.get('sector')]
            hot_pool = summarize_hot_sector_pool(
                self._sector_map, set(hot_sector_names), universe=set(all_data.keys()),
            )
            self._regime_log.append({
                'date': date,
                'score': regime['score'],
                'regime': regime['regime'],
                'trend_5d': regime['trend_5d'],
                'trend_20d': regime['trend_20d'],
                'mainlines': [m['sector'] for m in regime.get('mainlines', [])[:3]],
                'hot_sectors': hot_sector_names,
                'hot_sectors_detail': [
                    {
                        'sector': m.get('sector'),
                        'change': m.get('change'),
                        'momentum_5d': m.get('momentum_5d'),
                    }
                    for m in mainlines_top
                ],
                'hot_pool_size': hot_pool.get('total', 0),
                'hot_pool_by_sector': hot_pool.get('by_sector', {}),
                'defensive': regime['defensive'],
            })

            day_max = self.max_positions
            can_enter = True
            if use_market:
                can_enter = regime['can_enter']
                day_max = regime['max_positions']
                if not can_enter:
                    skipped += 1
            day_max = min(day_max, self.max_positions)

            if regime.get('defensive') and strat_cfg.get('market_defensive_enabled', False):
                for code in list(self.positions.keys()):
                    row = self._get_bar(all_data, code, date,
                                        forward_fill=True, flat_if_missing=True)
                    if row is None:
                        continue
                    pos = self.positions[code]
                    close = float(row['close'])
                    sector = self._sector_map.get(code, '其他')
                    reason = should_defensive_exit(
                        pos.profit_pct(close), sector, regime, strat_cfg,
                    )
                    if reason:
                        self._sell(code, close, 1.0, date, reason)

            for code in list(self.positions.keys()):
                row = self._get_bar(all_data, code, date, forward_fill=True, flat_if_missing=True)
                if row is not None:
                    self._process_position(code, row, date, all_data, regime, strat_cfg)

            held = set(self.positions.keys())
            shakeout_cfg = merge_shakeout_cfg(strat_cfg)
            bypass_max = shakeout_cfg.get('shakeout_bypass_max_positions', False)

            shakeout_cands = []
            if can_enter and shakeout_cfg.get('shakeout_rebuy_enabled'):
                shakeout_cands = self._shakeout_rebuy_candidates(
                    all_data, date, day_idx, held, strat_cfg,
                )

            for cand in shakeout_cands:
                stype = cand.get('strategy_type', 'trend')
                ml_s = cand.get('ml_score', 0.5)
                buy_amt = self._calc_buy_amount(
                    all_data, date, ml_s, strategy_type=stype,
                    strat_cfg=strat_cfg, shakeout_rebuy=True,
                )
                if buy_amt < 1000 or self.cash < buy_amt:
                    continue
                price = cand['buy_price']
                if self._buy(
                    cand['code'], price, buy_amt, date, '震仓接回', stype,
                    shakeout_rebuy=True, strat_cfg=strat_cfg,
                ):
                    self._shakeout_watch.pop(cand['code'], None)
                    self.trades.append({
                        'code': cand['code'], 'date': date, 'action': 'buy',
                        'reason': '震仓接回', 'price': round(price, 2),
                        'amount': round(buy_amt, 2), 'score': cand.get('score', 0),
                        'ml_score': cand.get('ml_score', 0),
                        'strategy_type': stype,
                        'doubler_boost': cand.get('doubler_boost', 0),
                        'entry': self._entry_snapshot(cand),
                    })

            trend_count = self._count_trend_positions()
            slots = self._effective_max_positions(day_max) - trend_count
            if bypass_max:
                slots = max(slots, 0)
            else:
                slots = self._effective_max_positions(day_max) - len(self.positions)

            if can_enter and slots > 0:
                if screen_fn == self._default_screen:
                    candidates = screen_fn(all_data, date, held, strat_cfg, self._full_config)
                else:
                    candidates = screen_fn(all_data, date, held, strat_cfg)
                candidates = [c for c in candidates if not c.get('shakeout_rebuy')]
                for cand in candidates[:slots]:
                    stype = cand.get('strategy_type', 'trend')
                    ml_s = cand.get('ml_score', 0.5)
                    buy_amt = self._calc_buy_amount(
                        all_data, date, ml_s, strategy_type=stype,
                        strat_cfg=strat_cfg, shakeout_rebuy=False,
                    )
                    if buy_amt < 1000 or self.cash < buy_amt:
                        break
                    price = cand['buy_price']
                    if self._buy(
                        cand['code'], price, buy_amt, date, '新建仓', stype,
                        shakeout_rebuy=False, strat_cfg=strat_cfg,
                    ):
                        reason = {
                            'limit_up': '涨停首板',
                            'doubler': '翻倍模式',
                        }.get(stype, '趋势选股')
                        self.trades.append({
                            'code': cand['code'], 'date': date, 'action': 'buy',
                            'reason': reason, 'price': round(price, 2),
                            'amount': round(buy_amt, 2), 'score': cand.get('score', 0),
                            'ml_score': cand.get('ml_score', 0),
                            'strategy_type': stype,
                            'doubler_boost': cand.get('doubler_boost', 0),
                            'entry': self._entry_snapshot(cand),
                        })

            self._append_daily(all_data, date)

            if day_idx == 1 or day_idx == total_days or day_idx % 20 == 0:
                eq = self._equity(all_data, date)
                print(
                    f'  [{day_idx}/{total_days}] {date} | {active_label} | '
                    f'持仓{len(self.positions)} | 净值{eq:,.0f}',
                    flush=True,
                )

        if trading_days:
            last = trading_days[-1]
            for code in list(self.positions.keys()):
                row = self._get_bar(all_data, code, last, forward_fill=True, flat_if_missing=False)
                if row is not None:
                    self._sell(code, float(row['close']), 1.0, last, '回测结束清仓')

        return self._build_result(trading_days, skipped, strat_cfg)

    def _default_screen(self, all_data, date, held, strat_cfg, full_config: dict = None):
        from src.market_regime import get_hot_sector_pool_names, codes_in_hot_sectors

        trend_cands, limit_up_cands, doubler_cands = [], [], []
        lu_cfg = load_limit_up_config(full_config or {})
        db_cfg = load_doubler_config(full_config or {})
        active = strat_cfg.get('active_strategies')
        if active:
            active_set = set(active)
        else:
            active_set = {'trend', 'limit_up', 'doubler'}
        dt = pd.Timestamp(date)
        min_hist = max(60, lu_cfg.get('lookback_days', 30))

        allowed_codes = None
        if strat_cfg.get('require_hot_sector', False) and 'trend' in active_set:
            from src.market_regime import summarize_hot_sector_pool, format_hot_sector_pool_log
            hot_names = get_hot_sector_pool_names(self._current_regime, strat_cfg)
            if hot_names:
                allowed_codes = codes_in_hot_sectors(self._sector_map, hot_names)
                summary = summarize_hot_sector_pool(
                    self._sector_map, hot_names, universe=set(all_data.keys()),
                )
                print(format_hot_sector_pool_log(
                    summary, total_universe=len(all_data), date=date,
                ), flush=True)

        for code, df in all_data.items():
            if code in held:
                continue
            day_idx = getattr(self, '_day_index', {}).get(date, 0)
            if self._in_stop_cooldown(code, day_idx):
                continue
            row = self._get_bar(all_data, code, date, forward_fill=False, flat_if_missing=False)
            if row is None:
                continue
            hist = df.loc[df.index <= dt]
            if len(hist) < min_hist:
                continue
            row = row.copy()
            row['code'] = code
            row.name = dt

            ml_proba = None
            if 'trend' in active_set and self.ml_scorer and self.ml_scorer.ready:
                ml_df = self.ml_data.get(code)
                if ml_df is not None:
                    ml_proba = self.ml_scorer.score_at(ml_df, date, code=code)

            if 'trend' in active_set:
                if allowed_codes is None or code in allowed_codes:
                    r = screen_stock_row(row, hist, strat_cfg, ml_proba=ml_proba)
                    if r:
                        r['sector'] = self._sector_map.get(code, '其他')
                        r = apply_doubler_boost(r, row, hist, db_cfg)
                        r = apply_mainline_to_candidate(r, self._current_regime, strat_cfg)
                        if r is not None:
                            r['buy_date'] = date
                            r['strategy_type'] = 'trend'
                            trend_cands.append(r)

            if 'limit_up' in active_set and lu_cfg.get('enabled', True):
                if is_limit_up_bar(row, code):
                    lu = screen_limit_up_row(row, hist, lu_cfg)
                    if lu:
                        lu['buy_date'] = date
                        limit_up_cands.append(lu)

            if 'doubler' in active_set and _independent_enabled(db_cfg):
                db = screen_doubler_row(row, hist, db_cfg)
                if db:
                    db['buy_date'] = date
                    doubler_cands.append(db)

        trend_slots = self.max_positions if 'trend' in active_set else 0
        limit_up_slots = (
            lu_cfg.get('max_daily_picks', 2)
            if 'limit_up' in active_set and lu_cfg.get('enabled', True) else 0
        )
        doubler_slots = (
            db_cfg.get('max_daily_picks', 1)
            if 'doubler' in active_set and _independent_enabled(db_cfg) else 0
        )
        merged = merge_strategy_candidates(
            trend_cands, limit_up_cands,
            trend_slots=trend_slots,
            limit_up_slots=limit_up_slots,
            doubler_list=doubler_cands,
            doubler_slots=doubler_slots,
        )
        return merged[:self.max_positions]

    def _build_result(self, trading_days, skipped, strat_cfg) -> Dict:
        final = self.cash
        equity_vals = [e['equity'] for e in self.equity_curve] or [self.initial_capital]
        peak = max(equity_vals)
        max_dd = 0.0
        for eq in equity_vals:
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak * 100)

        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        wins = [t for t in sell_trades if t.get('profit', 0) > 0]
        losses = [t for t in sell_trades if t.get('profit', 0) <= 0]
        gross_win = sum(t.get('profit', 0) for t in wins)
        gross_loss = abs(sum(t.get('profit', 0) for t in losses))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 0
        avg_win = gross_win / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        total_return = (final - self.initial_capital) / self.initial_capital * 100

        buy_count = sum(1 for t in self.trades if t['action'] == 'buy')
        add_count = sum(1 for t in self.trades if t['action'] == 'add')
        def _stype_buys(st):
            return [t for t in self.trades if t['action'] == 'buy' and t.get('strategy_type') == st]
        def _stype_profit(st):
            return sum(t.get('profit', 0) for t in sell_trades if t.get('strategy_type') == st)

        lu_buys = _stype_buys('limit_up')
        doubler_buys = _stype_buys('doubler')
        trend_buys = _stype_buys('trend')
        lu_sell_profit = _stype_profit('limit_up')
        doubler_sell_profit = _stype_profit('doubler')
        trend_sell_profit = _stype_profit('trend')

        months = max(len(trading_days) / 21, 1)
        monthly_return = ((final / self.initial_capital) ** (1 / months) - 1) * 100 if final > 0 else -100

        target = 10_000_000
        if final > self.initial_capital and monthly_return > 0:
            months_to_target = np.log(target / final) / np.log(1 + monthly_return / 100)
        else:
            months_to_target = float('inf')

        return {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': round(final, 2),
                'total_return_pct': round(total_return, 2),
                'peak_equity': round(peak, 2),
                'max_drawdown_pct': round(max_dd, 2),
                'total_trades': len(sell_trades),
                'buy_count': buy_count,
                'add_count': add_count,
                'win_rate': round(len(wins) / len(sell_trades) * 100, 1) if sell_trades else 0,
                'profit_factor': round(profit_factor, 2),
                'avg_win': round(avg_win, 2),
                'avg_loss': round(-avg_loss, 2),
                'avg_profit_per_trade': round(np.mean([t.get('profit', 0) for t in sell_trades]), 2) if sell_trades else 0,
                'monthly_return_pct': round(monthly_return, 2),
                'months_to_10m': round(months_to_target, 1) if months_to_target != float('inf') else None,
                'trend_buys': len(trend_buys),
                'limit_up_buys': len(lu_buys),
                'doubler_buys': len(doubler_buys),
                'trend_profit': round(trend_sell_profit, 2),
                'limit_up_profit': round(lu_sell_profit, 2),
                'doubler_profit': round(doubler_sell_profit, 2),
            },
            'params': {
                'buy_amount': self.buy_amount,
                'buy_amount_pct': self.buy_amount_pct,
                'add_trigger_pct': self.add_trigger_pct,
                'add_ratio': self.add_ratio,
                'stop_loss_pct': self.stop_loss_pct * 100,
                'strategy': strat_cfg,
            },
            'equity_curve': self.equity_curve,
            'daily_log': self.daily_log,
            'regime_log': self._regime_log,
            'trades': self.trades,
            'skipped_market_days': skipped,
        }
