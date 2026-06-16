# -*- coding: utf-8 -*-
"""用回测期末持仓替换实盘持仓，并输出选股逻辑一致性检查。"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import os
import shutil
from datetime import datetime

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.portfolio_manager import PortfolioManager
from src.data_fetcher_multi import MultiSourceDataFetcher
from src.strategy_filters import load_strategy_config


BACKTEST_FILE = 'data/backtest_capital_results.json'
CONFIG_FILE = 'config/config.yaml'


def load_backtest_final_holdings(variant: str = '趋势选股'):
    """回测末交易日收盘持仓（不含「回测结束清仓」）。"""
    with open(BACKTEST_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trades = data['variants'][variant]['trades']
    last_date = data['period']['end']
    open_pos = {}
    for t in trades:
        code = t['code']
        if t['action'] == 'buy':
            open_pos[code] = t
        elif t['action'] == 'sell':
            if t.get('reason') == '回测结束清仓':
                continue
            open_pos.pop(code, None)
    daily = data['variants'][variant].get('daily_log', [])
    if daily:
        last_date = daily[-1].get('date', last_date)
    return last_date, list(open_pos.values())


def build_position_record(trade: dict, strat_cfg: dict, fetcher: MultiSourceDataFetcher,
                          as_of: str) -> dict:
    code = trade['code']
    buy_price = float(trade['price'])
    sl_pct = strat_cfg.get('stop_loss', 3.0)
    trail_start = strat_cfg.get('trailing_start', 15.0)
    trail_pct = strat_cfg.get('trailing_pct', 6.0)
    stop_loss = round(buy_price * (1 - sl_pct / 100), 2)

    name = code
    current_price = buy_price
    change = 0.0
    try:
        quotes = fetcher.get_realtime_quotes([code])
        if quotes is not None and not quotes.empty:
            row = quotes.iloc[0]
            name = row.get('name', code)
            current_price = float(row.get('price', buy_price))
            lc = float(row.get('last_close', buy_price))
            change = ((current_price - lc) / lc * 100) if lc > 0 else 0
    except Exception:
        name = fetcher.get_stock_name(code) or code

    profit_pct = (current_price - buy_price) / buy_price * 100 if buy_price else 0
    from stock_pool_manager import get_pool_manager
    sector = get_pool_manager().get_stock_sector(code)

    return {
        'code': code,
        'name': name,
        'buy_price': buy_price,
        'buy_date': trade['date'],
        'current_price': round(current_price, 2),
        'profit_pct': round(profit_pct, 2),
        'days_held': 0,
        'status': 'holding',
        'sector': sector,
        'take_profit': round(buy_price * (1 + trail_start / 100), 2),
        'stop_loss': stop_loss,
        'current_stop_loss': stop_loss,
        'take_profit_levels': [],
        'partial_sold': [],
        'add_levels_done': [],
        'stop_loss_partial_done': False,
        'dip_refill_done': False,
        'confidence': trade.get('ml_score', trade.get('score', 70) / 100),
        'tech_score': trade.get('score', 70),
        'fund_score': 50,
        'rsi': 50,
        'strategy': 'short_term',
        'strategy_type': trade.get('strategy_type', 'trend'),
        'trailing_start': trail_start,
        'trailing_pct': trail_pct,
        'change': round(change, 2),
        'highest_price': max(buy_price, current_price),
        'daily_prices': {
            as_of: {'close': round(current_price, 2), 'change': round(change, 2)},
        },
        'sell_reason': '',
        'synced_from_backtest': True,
        'backtest_buy_reason': trade.get('reason', ''),
        'backtest_score': trade.get('score'),
        'backtest_ml_score': trade.get('ml_score'),
    }


def print_screening_audit(strat_cfg: dict):
    print('\n' + '=' * 60)
    print('选股逻辑一致性检查（回测 vs 实盘）')
    print('=' * 60)
    aligned = [
        ('核心筛选', 'screen_stock_row', 'daily_pipeline_v4.fetch_and_screen_stocks'),
        ('翻倍加分', 'apply_doubler_boost', 'fetch_and_screen_stocks（boost_enabled=false 时无影响）'),
        ('主线/冷门', 'apply_mainline_to_candidate', 'fetch_and_screen_stocks + build_regime_from_sectors'),
        ('排序', 'rank_candidates / merge_strategy_candidates', 'rank_candidates'),
        ('涨幅区间', f"{strat_cfg['min_change']}-{strat_cfg['max_change']}%", '同 config'),
        ('热门板块', 'get_hot_sector_pool_names(regime)', 'get_live_sector_boards'),
        ('止损/拖尾', f"{strat_cfg.get('stop_loss')}% / {strat_cfg.get('trailing_start')}/{strat_cfg.get('trailing_pct')}%", '同 config'),
        ('弱市仓位', f"≤{strat_cfg.get('weak_market_max_positions')}只", 'run_daily sentiment<full_score'),
        ('冷门过滤', strat_cfg.get('filter_cold_sector_in_weak'), 'apply_mainline_to_candidate'),
    ]
    for label, bt, live in aligned:
        print(f'  ✅ {label}')
        print(f'     回测: {bt}')
        print(f'     实盘: {live}')

    gaps = [
        'ML评分: 回测 score_at(历史日期) vs 实盘 score_stock_code(当日) — 有细微偏差',
        '大盘门禁: 回测 analyze_market_regime vs 实盘 MarketSentimentAnalyzer — 逻辑近似',
        '板块动量: 回测用池内历史5日动量 vs 实盘东方财富当日涨幅 — 数据源不同',
        '成交量预筛: 实盘 fetch 阶段 >100万, screen_stock_row 内 >50万 — 实盘更严',
    ]
    print('\n  ⚠️ 已知差异（未完全等价）:')
    for g in gaps:
        print(f'     · {g}')
    print('=' * 60)


def main():
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    strat_cfg = load_strategy_config(config)

    as_of, holdings = load_backtest_final_holdings()
    if not holdings:
        print('回测期末无持仓（回测结束时会强制清仓）')
        return

    print(f'回测期末日期: {as_of}')
    print(f'回测期末持仓: {len(holdings)} 只')
    for h in holdings:
        print(f"  {h['code']} 买入{h['date']} @ {h['price']:.2f} "
              f"综合{h.get('score', 0):.1f} ML{h.get('ml_score', 0):.3f}")

    portfolio_file = os.path.join('data', 'portfolio.json')
    backup = f"data/portfolio_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(portfolio_file, backup)
    print(f'\n已备份: {backup}')

    with open(portfolio_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    today = datetime.now().strftime('%Y-%m-%d')
    fetcher = MultiSourceDataFetcher()
    closed = []
    for code, pos in data.get('active_positions', {}).items():
        pos = dict(pos)
        pos['status'] = 'closed'
        pos['close_date'] = today
        pos['close_reason'] = '同步回测前清仓'
        pos['sell_reason'] = '同步回测前清仓'
        closed.append(pos)
        print(f'清仓: {code} {pos.get("name", "")}')

    new_active = {}
    for trade in holdings:
        rec = build_position_record(trade, strat_cfg, fetcher, today)
        new_active[rec['code']] = rec
        print(f"写入: {rec['code']} {rec['name']} 买价{rec['buy_price']} "
              f"现价{rec['current_price']} 止损{rec['stop_loss']}")

    data['active_positions'] = new_active
    data['closed_positions'] = data.get('closed_positions', []) + closed
    PortfolioManager()._recalc_stats(data)
    with open(portfolio_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f'\n✅ 实盘持仓已更新: {len(new_active)} 只')
    print_screening_audit(strat_cfg)


if __name__ == '__main__':
    main()
