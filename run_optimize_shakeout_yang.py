# -*- coding: utf-8 -*-
"""止损后连续阳线接回：对比 3/4/5 天 vs 旧逻辑(收复止损价)"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_backtest_capital import (
    load_config, get_trading_days, prepare_data, run_variant,
    STRATEGY_VARIANTS, resolve_backtest_cfg,
)
from stock_pool_manager import get_screening_pool
from src.ml_scorer import build_ml_features, get_ml_scorer
from src.hot_sector_analytics import analyze_hot_sectors

BASE = {
    **STRATEGY_VARIANTS['趋势选股'],
    'stop_loss': 3.0,
    'trailing_start': 15.0,
    'trailing_pct': 6.0,
    'take_profit_levels': [],
    'market_defensive_enabled': True,
    'market_defensive_exit_pct': -3.5,
    'filter_cold_sector_in_weak': True,
    'exit_cold_sector_in_weak': True,
    'weak_market_max_positions': 2,
    'shakeout_rebuy_enabled': True,
    'shakeout_rebuy_require_recover': True,
    'shakeout_rebuy_recover_pct': 0.0,
    'shakeout_rebuy_skip_weak_regime': True,
    'shakeout_rebuy_stop_loss': 4.5,
    'shakeout_rebuy_stop_grace_days': 5,
    'shakeout_rebuy_delayed_confirm': False,
}

GRID = [
    ('连阳3天', {'shakeout_rebuy_mode': 'consecutive_yang', 'shakeout_rebuy_min_up_days': 3}),
    ('连阳4天', {'shakeout_rebuy_mode': 'consecutive_yang', 'shakeout_rebuy_min_up_days': 4}),
    ('连阳5天', {'shakeout_rebuy_mode': 'consecutive_yang', 'shakeout_rebuy_min_up_days': 5}),
]


def main():
    config = load_config()
    bt_cfg = resolve_backtest_cfg(config)
    pool = get_screening_pool(config, bt_cfg.get('pool_size', 3000))
    print(f'📥 加载 {len(pool)} 只股票...')
    all_data, raw_data, _ = prepare_data(pool, days=300, use_store=True)
    if not all_data:
        print('❌ 数据失败')
        return

    trading_days = get_trading_days(config.get('targets', {}).get('backtest_days', 200))
    data_start = min(df.index.min() for df in all_data.values()).strftime('%Y-%m-%d')
    trading_days = [d for d in trading_days if d >= data_start]
    period = {'start': trading_days[0], 'end': trading_days[-1], 'days': len(trading_days)}

    print(f'📅 {period["start"]} ~ {period["end"]}')
    ml_data = build_ml_features(raw_data=raw_data)
    ml_scorer = get_ml_scorer(config, auto_train=True, ml_data=ml_data,
                              train_end_date=period['start'])
    if ml_scorer and ml_data:
        ml_scorer.precompute(ml_data, trading_days, persist=True)

    results = []
    runs = []
    print('\n' + '=' * 95)
    print(f'{"方案":<14} {"收益":>8} {"回撤":>8} {"买入":>5} {"胜率":>6} {"盈亏比":>7} {"接回买":>6}')
    print('=' * 95)

    for name, overrides in GRID:
        variant = {**BASE, **overrides}
        r = run_variant(name, variant, all_data, ml_data, trading_days,
                         config, bt_cfg, ml_scorer)
        runs.append(r)
        s = r['summary']
        rebuy_buys = sum(
            1 for t in r.get('trades', []) if t.get('action') == 'buy' and '震仓' in t.get('reason', '')
        )
        results.append({'name': name, 'overrides': overrides, 'summary': s, 'rebuy_buys': rebuy_buys})
        pf = s.get('profit_factor', 0)
        print(f'{name:<14} {s["total_return_pct"]:>+7.1f}% {s["max_drawdown_pct"]:>7.1f}% '
              f'{s.get("buy_count", 0):>5} {s.get("win_rate", 0):>5.1f}% {pf:>7.2f} {rebuy_buys:>6}')

    best_row = max(results, key=lambda x: x['summary']['total_return_pct'])
    best_wr = max(results, key=lambda x: x['summary']['win_rate'])
    print('=' * 95)
    print(f'最高收益: {best_row["name"]} {best_row["summary"]["total_return_pct"]:+.1f}% '
          f'胜率{best_row["summary"]["win_rate"]:.1f}%')
    print(f'最高胜率: {best_wr["name"]} {best_wr["summary"]["win_rate"]:.1f}% '
          f'收益{best_wr["summary"]["total_return_pct"]:+.1f}%')

    os.makedirs('data', exist_ok=True)
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    shakeout_path = 'data/shakeout_yang_optimize_results.json'
    with open(shakeout_path, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': generated_at,
            'period': period,
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    print(f'\n💾 摘要: {shakeout_path}')

    viz_output = {
        'generated_at': generated_at,
        'period': period,
        'capital_rules': bt_cfg,
        'limit_up_rules': config.get('limit_up', {}),
        'doubler_rules': config.get('doubler', {}),
        'target_capital': 10_000_000,
        'comparison_mode': 'shakeout_yang',
        'best_variant': best_row['name'],
        'best_summary': best_row['summary'],
        'variants': {
            r['variant']: {
                'summary': r['summary'],
                'params': r.get('params', {}),
                'equity_curve': r.get('equity_curve', []),
                'daily_log': r.get('daily_log', []),
                'regime_log': r.get('regime_log', []),
                'hot_sector_stats': analyze_hot_sectors(r.get('regime_log', [])),
                'trades': r.get('trades', []),
            }
            for r in runs
        },
    }
    viz_path = 'data/backtest_capital_results.json'
    with open(viz_path, 'w', encoding='utf-8') as f:
        json.dump(viz_output, f, ensure_ascii=False, indent=2)
    print(f'💾 可视化: {viz_path}')
    print(f'📊 打开 http://localhost:8088/backtest_viz.html 下拉切换「连阳3天/4天/5天」')


if __name__ == '__main__':
    main()
