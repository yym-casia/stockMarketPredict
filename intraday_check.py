# -*- coding: utf-8 -*-
"""
盘中快速检查 - 仅评估卖出信号，及时提示止盈止损

在交易时段内运行，不执行选股买入。
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import yaml
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trading_calendar import is_trading_day, is_market_hours
from src.portfolio_manager import PortfolioManager
from src.position_manager import PositionManager
from src.daily_advisor import DailyAdvisor
from src.data_fetcher_multi import MultiSourceDataFetcher
from src.strategy_filters import load_strategy_config


def load_config():
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    if not is_trading_day() or not is_market_hours():
        return

    config = load_config()
    portfolio_cfg = config.get('portfolio', {})
    max_positions = portfolio_cfg.get('max_positions', 5)
    max_hold_days = portfolio_cfg.get('max_hold_days', 15)
    strat_cfg = load_strategy_config(config)
    sl_pct = strat_cfg.get('stop_loss', config['data']['stop_loss'])
    bt_cfg = config.get('backtest', {})

    portfolio = PortfolioManager(
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        trend_stop_loss_partial_ratio=strat_cfg.get('trend_stop_loss_partial_ratio', 0),
    )
    position_mgr = PositionManager(
        max_hold_days=max_hold_days,
        stop_loss_pct=sl_pct,
        trailing_start=strat_cfg.get('trailing_start', 4.0),
        trailing_pct=strat_cfg.get('trailing_pct', 2.5),
        add_trigger_pct=bt_cfg.get('add_trigger_pct', 10.0),
        add_ratio=bt_cfg.get('add_ratio', 0.0),
        trend_add_amount_pct=bt_cfg.get('trend_add_amount_pct'),
        trend_stop_loss_partial_ratio=strat_cfg.get('trend_stop_loss_partial_ratio', 0),
        trend_dip_refill_pct=strat_cfg.get('trend_dip_refill_pct'),
        trend_half_hold_exit_pct=strat_cfg.get('trend_half_hold_exit_pct'),
        stop_loss_on_close=strat_cfg.get('stop_loss_on_close', True),
    )
    advisor = DailyAdvisor(portfolio, position_mgr)
    fetcher = MultiSourceDataFetcher()

    active = portfolio.get_active_positions()
    if not active:
        return

    today = datetime.now().strftime('%Y-%m-%d')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    codes = list(active.keys())
    price_data = fetcher.get_realtime_quotes(codes)
    total_equity = bt_cfg.get('initial_capital', 100000)
    categories, urgent_alerts = advisor.evaluate_positions(
        price_data, today, total_equity=total_equity)

    sell_count = len(categories['sell']) + len(categories['partial_sell'])
    add_count = len(categories.get('add', []))
    if sell_count == 0 and add_count == 0 and not urgent_alerts:
        return

    print(f"\n🚨 [{now_str}] 盘中告警 — 卖出{sell_count} | 加仓{add_count}")
    for alert in urgent_alerts:
        icon = '🔴' if alert['priority'] == 'urgent' else '🟡'
        print(f"  {icon} {alert['action']} {alert['code']} {alert['name']}: {alert['reason']}")
    for s in categories.get('add', []):
        if s.get('add_amount_pct'):
            print(f"  🟢 加仓总资产{s['add_amount_pct']*100:.0f}%"
                  f"({s.get('add_amount', 0):,.0f}元) {s['code']} {s['name']}: "
                  f"{s.get('sell_reason', '')} ({s['profit_pct']:+.2f}%)")
        else:
            print(f"  🟢 加仓市值{s.get('add_ratio', 0.3) * 100:.0f}% {s['code']} {s['name']}: "
                  f"{s.get('sell_reason', '')} ({s['profit_pct']:+.2f}%)")

    ops_file = advisor.operations_file
    if os.path.exists(ops_file):
        import json
        with open(ops_file, 'r', encoding='utf-8') as f:
            operations = json.load(f)
    else:
        operations = advisor.build_operations(categories, [], {}, today)

    operations['generated_at'] = now_str
    operations['urgent_alerts'] = urgent_alerts
    operations['intraday_check'] = True
    advisor.save_operations(operations)

    advisor.apply_sell_actions(categories, today)


if __name__ == "__main__":
    main()
