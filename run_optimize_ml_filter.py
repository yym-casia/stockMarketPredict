# -*- coding: utf-8 -*-
"""min_ml_score + min_composite_score 网格搜索"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_backtest_capital import (
    load_config, get_trading_days, prepare_data, run_variant,
    STRATEGY_VARIANTS, resolve_backtest_cfg,
)
from stock_pool_manager import get_screening_pool
from src.ml_scorer import build_ml_features, get_ml_scorer

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
}

ML_SCORES = [0.0, 0.55, 0.60, 0.65]
COMPOSITE_SCORES = [72, 75, 78]


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
    print('🤖 ML 预计算...')
    ml_data = build_ml_features(raw_data=raw_data)
    ml_scorer = get_ml_scorer(config, auto_train=True, ml_data=ml_data,
                              train_end_date=period['start'])
    if ml_scorer and ml_data:
        ml_scorer.precompute(ml_data, trading_days, persist=True)

    results = []
    print('\n' + '=' * 100)
    print(f'{"方案":<18} {"收益":>8} {"回撤":>8} {"买入":>5} {"胜率":>6} {"盈亏比":>7} {"均笔":>8}')
    print('=' * 100)

    for min_ml, min_comp in product(ML_SCORES, COMPOSITE_SCORES):
        name = f'ml{min_ml:.2f}_c{min_comp}'
        variant = {
            **BASE,
            'min_ml_score': min_ml,
            'min_composite_score': min_comp,
        }
        r = run_variant(name, variant, all_data, ml_data, trading_days,
                         config, bt_cfg, ml_scorer)
        s = r['summary']
        results.append({
            'name': name,
            'min_ml_score': min_ml,
            'min_composite_score': min_comp,
            'summary': s,
        })
        pf = s.get('profit_factor', 0)
        avg = s.get('avg_profit_per_trade', 0)
        print(f'{name:<18} {s["total_return_pct"]:>+7.1f}% {s["max_drawdown_pct"]:>7.1f}% '
              f'{s.get("buy_count", 0):>5} {s.get("win_rate", 0):>5.1f}% {pf:>7.2f} {avg:>+7.0f}')

    baseline = results[0]
    best_win = max(results, key=lambda x: x['summary']['win_rate'])
    best_ret = max(results, key=lambda x: x['summary']['total_return_pct'])
    # 收益≥基准80% 且胜率最高
    base_ret = baseline['summary']['total_return_pct']
    candidates = [x for x in results if x['summary']['total_return_pct'] >= base_ret * 0.8]
    best_balanced = max(candidates, key=lambda x: (
        x['summary']['win_rate'], x['summary']['total_return_pct'],
    ))

    print('=' * 100)
    print(f'基准 ml0_c72: 收益 {baseline["summary"]["total_return_pct"]:+.1f}% '
          f'胜率 {baseline["summary"]["win_rate"]:.1f}% '
          f'买入 {baseline["summary"]["buy_count"]}')
    print(f'最高胜率: {best_win["name"]} → {best_win["summary"]["win_rate"]:.1f}% '
          f'收益 {best_win["summary"]["total_return_pct"]:+.1f}%')
    print(f'最高收益: {best_ret["name"]} → {best_ret["summary"]["total_return_pct"]:+.1f}% '
          f'胜率 {best_ret["summary"]["win_rate"]:.1f}%')
    print(f'均衡推荐: {best_balanced["name"]} → 收益 {best_balanced["summary"]["total_return_pct"]:+.1f}% '
          f'胜率 {best_balanced["summary"]["win_rate"]:.1f}% '
          f'买入 {best_balanced["summary"]["buy_count"]}')

    out = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'baseline': baseline,
        'best_win_rate': best_win,
        'best_return': best_ret,
        'best_balanced': best_balanced,
        'all': results,
    }
    path = 'data/ml_filter_optimize_results.json'
    os.makedirs('data', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n💾 {path}')


if __name__ == '__main__':
    main()
