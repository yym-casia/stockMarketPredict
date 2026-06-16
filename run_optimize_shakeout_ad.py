# -*- coding: utf-8 -*-
"""方案A / 方案D / 基准(3连阳) 接回策略对比回测"""

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
    'shakeout_rebuy_mode': 'consecutive_yang',
    'shakeout_rebuy_min_up_days': 3,
    'shakeout_rebuy_require_recover': True,
    'shakeout_rebuy_recover_pct': 0.0,
    'shakeout_rebuy_skip_weak_regime': True,
    'shakeout_rebuy_stop_loss': 4.5,
    'shakeout_rebuy_stop_grace_days': 5,
    'shakeout_rebuy_delayed_confirm': False,
}

GRID = [
    ('基准3连阳', {}),
    ('方案A严进宽出', {
        'shakeout_rebuy_require_recover': True,
        'shakeout_rebuy_recover_pct': 0.0,
        'shakeout_rebuy_skip_weak_regime': True,
        'shakeout_rebuy_stop_loss': 4.5,
        'shakeout_rebuy_stop_grace_days': 5,
        'shakeout_rebuy_delayed_confirm': False,
    }),
    ('方案D延迟确认', {
        'shakeout_rebuy_delayed_confirm': True,
        'shakeout_rebuy_min_last_yang_pct': 1.0,
        'shakeout_rebuy_require_recover': False,
        'shakeout_rebuy_skip_weak_regime': False,
        'shakeout_rebuy_stop_loss': None,
        'shakeout_rebuy_stop_grace_days': 0,
    }),
]


def count_rebuy_buys(trades):
    return sum(
        1 for t in trades if t.get('action') == 'buy' and '震仓' in t.get('reason', '')
    )


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

    runs = []
    print('\n' + '=' * 100)
    print(f'{"方案":<16} {"收益":>8} {"回撤":>8} {"买入":>5} {"胜率":>6} {"盈亏比":>7} {"接回":>5}')
    print('=' * 100)

    for name, overrides in GRID:
        variant = {**BASE, **overrides}
        r = run_variant(name, variant, all_data, ml_data, trading_days,
                         config, bt_cfg, ml_scorer)
        runs.append(r)
        s = r['summary']
        rebuy = count_rebuy_buys(r.get('trades', []))
        pf = s.get('profit_factor', 0)
        print(f'{name:<16} {s["total_return_pct"]:>+7.1f}% {s["max_drawdown_pct"]:>7.1f}% '
              f'{s.get("buy_count", 0):>5} {s.get("win_rate", 0):>5.1f}% {pf:>7.2f} {rebuy:>5}')

    best = max(runs, key=lambda x: x['summary']['total_return_pct'])
    print('=' * 100)
    print(f'最高收益: {best["variant"]} {best["summary"]["total_return_pct"]:+.1f}% '
          f'胜率{best["summary"]["win_rate"]:.1f}%')

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    os.makedirs('data', exist_ok=True)

    summary_rows = []
    for r in runs:
        s = r['summary']
        summary_rows.append({
            'name': r['variant'],
            'summary': s,
            'rebuy_buys': count_rebuy_buys(r.get('trades', [])),
        })

    with open('data/shakeout_ad_compare_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': generated_at,
            'period': period,
            'results': summary_rows,
        }, f, ensure_ascii=False, indent=2)

    viz = {
        'generated_at': generated_at,
        'period': period,
        'capital_rules': bt_cfg,
        'limit_up_rules': config.get('limit_up', {}),
        'doubler_rules': config.get('doubler', {}),
        'target_capital': 10_000_000,
        'comparison_mode': 'shakeout_ad',
        'best_variant': best['variant'],
        'best_summary': best['summary'],
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
        json.dump(viz, f, ensure_ascii=False, indent=2)

    print(f'\n💾 摘要: data/shakeout_ad_compare_results.json')
    print(f'💾 可视化: {viz_path}')
    print('📊 backtest_viz.html 下拉可切换三方案')


if __name__ == '__main__':
    main()
