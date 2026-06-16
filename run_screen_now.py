# -*- coding: utf-8 -*-

"""收盘后选股（回测同款逻辑，跳过交易日检查，同步 Dashboard）"""

import sys

import os

import csv

import json

from datetime import datetime



sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



from daily_pipeline_v4 import generate_recommendations, load_config

from src.strategy_filters import load_strategy_config, rank_candidates

from src.eod_screener import (

    screen_like_backtest, enrich_candidate_names, regime_to_sentiment,

)

from src.stock_tracker import StockTracker

from src.portfolio_manager import PortfolioManager





def main():

    today = datetime.now().strftime('%Y-%m-%d')

    config = load_config()

    strat_cfg = load_strategy_config(config)

    tp_pct = config['data']['take_profit']

    sl_pct = strat_cfg.get('stop_loss', config['data']['stop_loss'])

    min_conf = config['data'].get('min_confidence', 0.55)



    print('=' * 60)

    print(f'收盘后选股（回测同款）| {today}')

    print(f'  涨幅{strat_cfg["min_change"]:.0f}-{strat_cfg["max_change"]:.0f}% | '

          f'热门板块Top{strat_cfg.get("hot_sector_top_n", 5)} | '

          f'止损{strat_cfg.get("stop_loss", 3):.0f}%全仓收盘确认 | '

          f'移动止盈{strat_cfg.get("trailing_start", 15):.0f}%/{strat_cfg.get("trailing_pct", 6):.0f}% | '

          f'无阶梯止盈 | 弱市≤{strat_cfg.get("weak_market_max_positions", 2)}仓'

          f' | 冷门过滤{"开" if strat_cfg.get("filter_cold_sector_in_weak") else "关"}')

    print('=' * 60)



    portfolio = PortfolioManager(

        max_positions=config.get('portfolio', {}).get('max_positions', 5),

        max_hold_days=config.get('portfolio', {}).get('max_hold_days', 90),

    )

    held = set(portfolio.get_active_positions().keys())
    shakeout_watches = portfolio.get_shakeout_watches(today)

    result = screen_like_backtest(
        config, held_codes=held, shakeout_watches=shakeout_watches,
        refresh_data=True, verbose=True,
    )

    screen_date = result['date']

    should_enter = result['can_enter']

    reason = result['reason']

    regime = result['regime']

    sentiment = regime_to_sentiment(regime)



    screened = enrich_candidate_names(result['screened_rows'])

    print(f'\n技术筛选通过: {len(screened)} 只（基准日 {screen_date}）')



    if not screened:

        print('未找到符合条件的股票')

        if not should_enter:

            print(f'原因: {reason}')

        _sync_to_dashboard([], screen_date, should_enter, reason, sentiment, config, strat_cfg)

        return



    candidates = generate_recommendations(screened, tp_pct, sl_pct, top_n=10)

    if strat_cfg.get('ml_rank_only', True):

        filtered = rank_candidates(candidates, top_n=5)

        rank_label = '综合评分排序(与回测一致)'

    else:

        filtered = [c for c in candidates if c.get('confidence', 0) >= min_conf]

        if not filtered and candidates:

            filtered = candidates[:5]

            print('⚠️ 无达标置信度标的，展示综合评分前列')

        rank_label = f'置信度>={min_conf * 100:.0f}%'



    print(f'\n{"=" * 60}')

    print(f'推荐买入 TOP {min(5, len(filtered))} ({rank_label})')

    print('=' * 60)

    for i, rec in enumerate(filtered[:5], 1):

        print(f"{i}. {rec['code']} {rec['name']}")

        print(f"   收盘价:{rec['buy_price']:.2f} 涨幅:{rec.get('change', 0):+.2f}% "

              f"置信度:{rec['confidence'] * 100:.1f}%")

        print(f"   技术:{rec.get('tech_score', 0):.0f} RSI:{rec.get('rsi', 0):.1f} "

              f"量比:{rec.get('volume_ratio', 0):.2f}")

        print(f"   移动止盈启动:{rec.get('trailing_start', 15):.0f}% "

              f"回撤:{rec.get('trailing_pct', 6):.0f}% 止损:{rec.get('stop_loss', 0):.2f}")

        print(f"   {rec.get('reason', '')}")



    os.makedirs('data', exist_ok=True)

    csv_path = f'data/recommendations_{screen_date}.csv'

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:

        w = csv.writer(f)

        w.writerow(['排名', '代码', '名称', '收盘价', '涨幅%', '置信度%', '技术分',

                     'RSI', '量比', '止盈', '止损', '原因'])

        for i, rec in enumerate(filtered[:10], 1):

            w.writerow([

                i, rec['code'], rec['name'], f"{rec['buy_price']:.2f}",

                f"{rec.get('change', 0):.2f}", f"{rec['confidence'] * 100:.1f}",

                f"{rec.get('tech_score', 0):.0f}", f"{rec.get('rsi', 0):.1f}",

                f"{rec.get('volume_ratio', 0):.2f}", f"{rec.get('take_profit', 0):.2f}",

                f"{rec.get('stop_loss', 0):.2f}", rec.get('reason', ''),

            ])

    print(f'\n已导出: {csv_path}')



    _sync_to_dashboard(filtered, screen_date, should_enter, reason, sentiment, config, strat_cfg)





def _sync_to_dashboard(recommendations, today, should_enter, reason, sentiment, config, strat_cfg):

    """写入 stock_tracking.json + daily_operations.json 供 Dashboard 读取。"""

    tracker = StockTracker()

    if recommendations:

        tracker.add_recommendations(recommendations, today)

        print(f'✅ 已同步 {len(recommendations)} 只到 stock_tracking.json')



    portfolio = PortfolioManager(

        max_positions=config.get('portfolio', {}).get('max_positions', 5),

        max_hold_days=config.get('portfolio', {}).get('max_hold_days', 90),

        trend_stop_loss_partial_ratio=strat_cfg.get('trend_stop_loss_partial_ratio', 0),

    )

    active = portfolio.get_active_positions()

    max_pos = config.get('portfolio', {}).get('max_positions', 5)

    stats = portfolio.get_stats()



    buy_ops = []

    for i, rec in enumerate(recommendations[:5], 1):

        buy_ops.append({

            'code': rec['code'],

            'name': rec['name'],

            'action': '买入',

            'priority': 'normal' if should_enter else 'watch',

            'buy_price': rec['buy_price'],

            'take_profit': rec.get('take_profit', 0),

            'stop_loss': rec.get('stop_loss', 0),

            'confidence': rec.get('confidence', 0),

            'tech_score': rec.get('tech_score', 0),

            'fund_score': rec.get('fund_score', 0),

            'reason': rec.get('reason', ''),

            'rank': i,

        })



    summary = f"收盘选股{len(recommendations)}只（回测同款，基准日{today}）"

    if not should_enter:

        summary += f"（{reason}，建议观望）"

    elif recommendations:

        summary += f"，可买入{max(0, max_pos - len(active))}只"



    ops = {

        'date': today,

        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

        'is_trading_day': True,

        'screen_only': True,

        'screen_mode': 'eod_backtest',

        'market_sentiment': sentiment,

        'market_enter': should_enter,

        'market_reason': reason,

        'portfolio_summary': {

            'active_count': len(active),

            'max_positions': max_pos,

            'available_slots': max(0, max_pos - len(active)),

            'win_rate': stats.get('win_rate', 0),

            'avg_profit': stats.get('avg_profit', 0),

            'total_trades': stats.get('total_trades', 0),

        },

        'urgent_alerts': [],

        'operations': {

            'sell': [],

            'partial_sell': [],

            'add': [],

            'dip_refill': [],

            'buy': buy_ops,

            'hold': [],

        },

        'summary': summary,

    }



    ops_path = os.path.join('data', 'daily_operations.json')

    os.makedirs('data', exist_ok=True)

    with open(ops_path, 'w', encoding='utf-8') as f:

        json.dump(ops, f, ensure_ascii=False, indent=2)

    print(f'✅ 已同步操作建议到 daily_operations.json')





if __name__ == '__main__':

    main()

