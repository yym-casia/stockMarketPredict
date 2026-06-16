# -*- coding: utf-8 -*-
"""弱市风控参数网格搜索 — 在 3% 止损基准上优化回撤"""

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


BASE_VARIANT = {
    **STRATEGY_VARIANTS['趋势选股'],
    'stop_loss': 3.0,
    'trailing_start': 15.0,
    'trailing_pct': 6.0,
    'take_profit_levels': [],
}

DEFENSIVE_GRID = [
    ('基准(全关)', {}),
    ('弱市3仓', {'weak_market_max_positions': 2}),
    ('冷门过滤', {'filter_cold_sector_in_weak': True}),
    ('熊市禁入', {'market_block_bearish_entry': True}),
    ('防御清仓', {
        'market_defensive_enabled': True,
        'market_defensive_exit_pct': -3.5,
        'exit_cold_sector_in_weak': True,
    }),
    ('冷门+禁入', {
        'filter_cold_sector_in_weak': True,
        'market_block_bearish_entry': True,
        'weak_market_max_positions': 2,
    }),
    ('防御+冷门', {
        'market_defensive_enabled': True,
        'market_defensive_exit_pct': -3.5,
        'filter_cold_sector_in_weak': True,
        'exit_cold_sector_in_weak': True,
        'weak_market_max_positions': 2,
    }),
    ('全套风控', {
        'market_defensive_enabled': True,
        'market_defensive_exit_pct': -3.5,
        'market_block_bearish_entry': True,
        'filter_cold_sector_in_weak': True,
        'exit_cold_sector_in_weak': True,
        'weak_market_max_positions': 2,
        'disable_add_in_bearish': True,
    }),
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

    print(f'📅 {period["start"]} ~ {period["end"]} ({period["days"]}天)')
    print(f'🤖 ML 预计算...')
    ml_data = build_ml_features(raw_data=raw_data)
    ml_scorer = get_ml_scorer(config, auto_train=True, ml_data=ml_data,
                              train_end_date=period['start'])
    if ml_scorer and ml_data:
        ml_scorer.precompute(ml_data, trading_days, persist=True)

    results = []
    print('\n' + '=' * 95)
    print(f'{"方案":<14} {"收益":>8} {"回撤":>8} {"买入":>5} {"胜率":>6} {"盈亏比":>7} {"评分":>8}')
    print('=' * 95)

    for name, overrides in DEFENSIVE_GRID:
        variant = {**BASE_VARIANT, **overrides}
        r = run_variant(name, variant, all_data, ml_data, trading_days,
                         config, bt_cfg, ml_scorer)
        s = r['summary']
        ret = s['total_return_pct']
        dd = s['max_drawdown_pct']
        # 风险调整评分: 收益 - 0.5*回撤 (偏重控回撤)
        score = ret - 0.5 * dd
        results.append({
            'name': name,
            'overrides': overrides,
            'summary': s,
            'score': round(score, 2),
        })
        pf = s.get('profit_factor', 0)
        print(f'{name:<14} {ret:>+7.1f}% {dd:>7.1f}% {s.get("buy_count", 0):>5} '
              f'{s.get("win_rate", 0):>5.1f}% {pf:>7.2f} {score:>+7.1f}')

    best = max(results, key=lambda x: x['score'])
    baseline = results[0]
    print('=' * 95)
    print(f'🏆 最优(风险调整): {best["name"]} | 收益 {best["summary"]["total_return_pct"]:+.1f}% '
          f'回撤 {best["summary"]["max_drawdown_pct"]:.1f}%')
    print(f'   基准: 收益 {baseline["summary"]["total_return_pct"]:+.1f}% '
          f'回撤 {baseline["summary"]["max_drawdown_pct"]:.1f}%')

    out = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'baseline': baseline,
        'best': best,
        'all': results,
    }
    path = 'data/defensive_optimize_results.json'
    os.makedirs('data', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n💾 {path}')


if __name__ == '__main__':
    main()
