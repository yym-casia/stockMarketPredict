# -*- coding: utf-8 -*-
"""投资组合管理：最多5只持仓，15天持有上限，持续盈利追踪"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from src.trading_calendar import count_trading_days
from src.data_fetcher_multi import MultiSourceDataFetcher
from src.shakeout_strategy import should_watch_after_stop, merge_shakeout_cfg


class PortfolioManager:
    """集中管理活跃持仓与历史交易"""

    def __init__(self, portfolio_file: str = None, tracking_file: str = None,
                 max_positions: int = 5, max_hold_days: int = 15,
                 trend_stop_loss_partial_ratio: float = 0):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.portfolio_file = portfolio_file or os.path.join(project_dir, 'data', 'portfolio.json')
        self.tracking_file = tracking_file or os.path.join(project_dir, 'data', 'stock_tracking.json')
        self.max_positions = max_positions
        self.max_hold_days = max_hold_days
        self.trend_stop_loss_partial_ratio = trend_stop_loss_partial_ratio
        self._ensure_file()

    def _ensure_file(self):
        os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)
        if not os.path.exists(self.portfolio_file):
            migrated = self._migrate_from_tracking()
            self._save(migrated or self._empty_state())
        else:
            data = self._load()
            if not data.get('active_positions') and os.path.exists(self.tracking_file):
                migrated = self._migrate_from_tracking()
                if migrated:
                    self._save(migrated)

    def _empty_state(self) -> Dict:
        return {
            'active_positions': {},
            'closed_positions': [],
            'shakeout_watches': {},
            'stats': {'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
                      'avg_profit': 0.0, 'total_profit': 0.0},
        }

    def _migrate_from_tracking(self) -> Optional[Dict]:
        """从旧 stock_tracking.json 迁移数据"""
        if not os.path.exists(self.tracking_file):
            return None
        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                tracking = json.load(f)
        except Exception:
            return None

        state = self._empty_state()
        today = datetime.now().strftime('%Y-%m-%d')
        active = {}

        for code, info in tracking.items():
            daily = info.get('daily_prices', {})
            if not daily:
                continue
            dates = sorted(daily.keys())
            latest = daily[dates[-1]]
            buy_price = info.get('buy_price', 0)
            current = latest.get('close', buy_price)
            profit_pct = ((current - buy_price) / buy_price * 100) if buy_price > 0 else 0
            days_held = count_trading_days(info.get('added_date', dates[0]), today)

            status = 'holding'
            sell_reason = info.get('sell_reason', '')
            if sell_reason:
                if '止损' in sell_reason:
                    status = 'stop_loss'
                elif '止盈' in sell_reason or '部分' in sell_reason:
                    status = 'take_profit' if '部分' not in sell_reason else 'partial_sell'
                else:
                    status = 'closed'
            elif latest.get('close', 0) <= info.get('stop_loss', 0):
                status = 'stop_loss'
            elif latest.get('close', 0) >= info.get('take_profit', float('inf')):
                status = 'take_profit'
            elif days_held >= self.max_hold_days:
                status = 'expired'

            pos = {
                'code': code, 'name': info.get('name', code),
                'buy_price': buy_price, 'buy_date': info.get('added_date', dates[0]),
                'current_price': current, 'profit_pct': round(profit_pct, 2),
                'days_held': days_held, 'status': status,
                'sector': info.get('sector', '其他'),
                'take_profit': info.get('take_profit', 0),
                'stop_loss': info.get('stop_loss', 0),
                'current_stop_loss': info.get('current_stop_loss', info.get('stop_loss', 0)),
                'take_profit_levels': info.get('take_profit_levels', []),
                'partial_sold': info.get('partial_sold', []),
                'confidence': info.get('confidence', 0),
                'tech_score': info.get('tech_score', 0),
                'fund_score': info.get('fund_score', 0),
                'rsi': info.get('rsi', 0),
                'strategy': info.get('strategy', 'short_term'),
                'daily_prices': daily,
                'sell_reason': sell_reason,
                'highest_price': info.get('highest_price', current),
            }

            if status == 'holding':
                active[code] = pos
            else:
                pos['close_date'] = dates[-1]
                pos['close_reason'] = sell_reason or status
                state['closed_positions'].append(pos)

        active_codes = sorted(active.keys(), key=lambda c: active[c]['buy_date'])
        if len(active_codes) > self.max_positions:
            for code in active_codes[:-self.max_positions]:
                pos = active.pop(code)
                pos['status'] = 'closed'
                pos['close_reason'] = '超出持仓上限，自动清仓'
                state['closed_positions'].append(pos)

        state['active_positions'] = active
        self._recalc_stats(state)
        return state

    def _load(self) -> Dict:
        try:
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return self._empty_state()

    def _save(self, data: Dict):
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._sync_tracking(data)

    def _sync_tracking(self, data: Dict):
        """同步到 stock_tracking.json 供 Dashboard 使用"""
        tracking = {}
        for code, pos in data.get('active_positions', {}).items():
            tracking[code] = {
                'name': pos['name'], 'added_date': pos['buy_date'],
                'strategy': pos.get('strategy', 'short_term'),
                'buy_price': pos['buy_price'],
                'confidence': pos.get('confidence', 0),
                'take_profit': pos.get('take_profit', 0),
                'stop_loss': pos.get('stop_loss', 0),
                'current_stop_loss': pos.get('current_stop_loss', pos.get('stop_loss', 0)),
                'sector': pos.get('sector', ''),
                'tech_score': pos.get('tech_score', 0),
                'fund_score': pos.get('fund_score', 0),
                'rsi': pos.get('rsi', 0),
                'take_profit_levels': pos.get('take_profit_levels', []),
                'partial_sold': pos.get('partial_sold', []),
                'add_levels_done': pos.get('add_levels_done', []),
                'sell_reason': pos.get('sell_reason', ''),
                'change': pos.get('change', 0),
                'highest_price': pos.get('highest_price', pos['buy_price']),
                'daily_prices': pos.get('daily_prices', {}),
                'expected_return': pos.get('expected_return', 0),
            }
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        with open(self.tracking_file, 'w', encoding='utf-8') as f:
            json.dump(tracking, f, ensure_ascii=False, indent=2)

    def _recalc_stats(self, data: Dict):
        closed = data.get('closed_positions', [])
        if not closed:
            data['stats'] = {'total_trades': 0, 'wins': 0, 'losses': 0,
                             'win_rate': 0.0, 'avg_profit': 0.0, 'total_profit': 0.0}
            return
        profits = [p.get('profit_pct', 0) for p in closed]
        wins = sum(1 for p in profits if p > 0)
        data['stats'] = {
            'total_trades': len(closed),
            'wins': wins,
            'losses': len(closed) - wins,
            'win_rate': round(wins / len(closed) * 100, 1),
            'avg_profit': round(sum(profits) / len(profits), 2),
            'total_profit': round(sum(profits), 2),
        }

    def get_active_positions(self) -> Dict:
        return self._load().get('active_positions', {})

    def _effective_max_positions(self, active: Dict, base_max: int) -> int:
        """半仓止损策略开启时，存在趋势止损半仓则总持仓上限+1。"""
        if self.trend_stop_loss_partial_ratio <= 0:
            return base_max
        bonus = any(
            p.get('strategy_type', 'trend') == 'trend' and p.get('stop_loss_partial_done')
            for p in active.values()
        )
        return base_max + (1 if bonus else 0)

    def get_available_slots(self) -> int:
        active = self.get_active_positions()
        cap = self._effective_max_positions(active, self.max_positions)
        return max(0, cap - len(active))

    def can_add_position(self, code: str = None) -> bool:
        active = self.get_active_positions()
        if code and code in active:
            return False
        cap = self._effective_max_positions(active, self.max_positions)
        return len(active) < cap

    def get_active_codes(self) -> List[str]:
        return list(self.get_active_positions().keys())

    def add_position(self, recommendation: Dict, date: str = None) -> bool:
        if not self.can_add_position(recommendation.get('code')):
            return False
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        data = self._load()
        code = recommendation['code']
        data['active_positions'][code] = {
            'code': code,
            'name': recommendation.get('name', code),
            'buy_price': recommendation.get('buy_price', 0),
            'buy_date': date,
            'current_price': recommendation.get('buy_price', 0),
            'profit_pct': 0.0,
            'days_held': 0,
            'status': 'holding',
            'sector': recommendation.get('sector', '其他'),
            'take_profit': recommendation.get('take_profit', 0),
            'stop_loss': recommendation.get('stop_loss', 0),
            'current_stop_loss': recommendation.get('stop_loss', 0),
            'take_profit_levels': recommendation.get('take_profit_levels', []),
            'partial_sold': [],
            'add_levels_done': [],
            'stop_loss_partial_done': False,
            'dip_refill_done': False,
            'confidence': recommendation.get('confidence', 0),
            'tech_score': recommendation.get('tech_score', 0),
            'fund_score': recommendation.get('fund_score', 0),
            'rsi': recommendation.get('rsi', 0),
            'strategy': recommendation.get('strategy', 'short_term'),
            'strategy_type': recommendation.get('strategy_type', 'trend'),
            'trailing_start': recommendation.get('trailing_start'),
            'trailing_pct': recommendation.get('trailing_pct'),
            'change': recommendation.get('change', 0),
            'highest_price': recommendation.get('buy_price', 0),
            'daily_prices': {date: {
                'close': recommendation.get('buy_price', 0),
                'change': recommendation.get('change', 0),
            }},
            'sell_reason': '',
        }
        self._save(data)
        return True

    def close_position(self, code: str, close_price: float, reason: str,
                       status: str, date: str = None) -> Optional[Dict]:
        data = self._load()
        if code not in data['active_positions']:
            return None
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        pos = data['active_positions'].pop(code)
        buy_price = pos.get('buy_price', close_price)
        profit_pct = ((close_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
        pos.update({
            'current_price': close_price,
            'profit_pct': round(profit_pct, 2),
            'status': status,
            'close_date': date,
            'close_reason': reason,
            'sell_reason': reason,
        })
        if date not in pos.get('daily_prices', {}):
            pos.setdefault('daily_prices', {})[date] = {'close': close_price, 'change': profit_pct}
        data['closed_positions'].append(pos)
        self._maybe_register_shakeout_watch(data, code, close_price, buy_price, reason, pos, date)
        self._recalc_stats(data)
        self._save(data)
        return pos

    def _maybe_register_shakeout_watch(self, data: Dict, code: str, sell_price: float,
                                       cost: float, reason: str, pos: Dict, date: str):
        try:
            from daily_pipeline_v4 import load_config
            from src.strategy_filters import load_strategy_config
            strat_cfg = load_strategy_config(load_config())
        except Exception:
            strat_cfg = {}
        stype = pos.get('strategy_type', 'trend')
        if not should_watch_after_stop(reason, stype, 1.0, strat_cfg):
            return
        watches = data.setdefault('shakeout_watches', {})
        watches[code] = {
            'code': code,
            'sell_date': date,
            'sell_price': round(sell_price, 2),
            'cost_price': round(cost, 2),
            'name': pos.get('name', code),
        }

    def get_shakeout_watches(self, today: str = None) -> Dict[str, Dict]:
        """返回未过期的止损关注列表。"""
        data = self._load()
        watches = data.get('shakeout_watches', {})
        if not watches:
            return {}
        if today is None:
            today = datetime.now().strftime('%Y-%m-%d')
        try:
            from daily_pipeline_v4 import load_config
            from src.strategy_filters import load_strategy_config
            strat_cfg = load_strategy_config(load_config())
        except Exception:
            strat_cfg = {}
        max_days = int(merge_shakeout_cfg(strat_cfg).get('shakeout_rebuy_days', 10))
        active = {}
        expired = []
        for code, w in watches.items():
            sell_date = w.get('sell_date', '')
            if sell_date and count_trading_days(sell_date, today) > max_days:
                expired.append(code)
            else:
                active[code] = w
        if expired:
            for code in expired:
                watches.pop(code, None)
            data['shakeout_watches'] = watches
            self._save(data)
        return active

    def remove_shakeout_watch(self, code: str):
        data = self._load()
        watches = data.get('shakeout_watches', {})
        if code in watches:
            watches.pop(code)
            data['shakeout_watches'] = watches
            self._save(data)

    def update_position(self, code: str, price_info: Dict, date: str = None) -> Optional[Dict]:
        data = self._load()
        if code not in data['active_positions']:
            return None
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        pos = data['active_positions'][code]
        current = price_info.get('price', price_info.get('close', 0))
        buy_price = pos.get('buy_price', current)
        profit_pct = ((current - buy_price) / buy_price * 100) if buy_price > 0 else 0

        pos['current_price'] = current
        pos['profit_pct'] = round(profit_pct, 2)
        pos['days_held'] = count_trading_days(pos['buy_date'], date)
        pos['highest_price'] = max(pos.get('highest_price', current), current)
        today_change = price_info.get('change')
        if today_change is None or (today_change == 0 and price_info.get('last_close')):
            today_change = MultiSourceDataFetcher.calc_change(
                current, price_info.get('last_close', 0)
            )
        pos['change'] = today_change
        pos.setdefault('daily_prices', {})[date] = {
            'close': current,
            'change': today_change,
        }
        if price_info.get('current_stop_loss'):
            pos['current_stop_loss'] = price_info['current_stop_loss']
        if price_info.get('partial_sold') is not None:
            pos['partial_sold'] = price_info['partial_sold']
        if price_info.get('add_levels_done') is not None:
            pos['add_levels_done'] = price_info['add_levels_done']
        if price_info.get('stop_loss_partial_done') is not None:
            pos['stop_loss_partial_done'] = price_info['stop_loss_partial_done']
        if price_info.get('dip_refill_done') is not None:
            pos['dip_refill_done'] = price_info['dip_refill_done']
        if price_info.get('sell_reason'):
            pos['sell_reason'] = price_info['sell_reason']

        self._save(data)
        return pos

    def get_stats(self) -> Dict:
        return self._load().get('stats', {})

    def get_closed_history(self, limit: int = 20) -> List[Dict]:
        closed = self._load().get('closed_positions', [])
        return sorted(closed, key=lambda x: x.get('close_date', ''), reverse=True)[:limit]

    def filter_already_held(self, candidates: List[Dict]) -> List[Dict]:
        active = set(self.get_active_codes())
        return [c for c in candidates if c.get('code', c.get('code_clean', '')) not in active]
