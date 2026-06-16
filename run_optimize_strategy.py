# -*- coding: utf-8 -*-
"""策略参数网格搜索 — 目标: 提升胜率与盈利能力"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import yaml
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_backtest_capital import (
    load_config, get_trading_days, prepare_data, run_variant,
)
from stock_pool_manager import get_pool_manager


# 手工精选变体 + 关键参数组合
STRATEGY_GRID = [
    # v4: 高质量入场 + 关闭加仓
    {
        'name': 'v4_精选',
        'strat': {
            'min_tech_score': 72, 'max_change': 3.5, 'min_change': 1.5,
            'min_volume_ratio': 1.4, 'close_strength_min': 0.6,
            'require_macd_golden': True, 'min_conditions': 4,
            'market_min_score': 52, 'weak_market_max_positions': 2,
            'min_composite_score': 72, 'max_prev_drop': 2.5,
            'trailing_start': 7.0, 'trailing_pct': 3.5, 'stop_loss': 4.0,
        },
        'add_ratio': 0,
    },
    {
        'name': 'v4_蓄势',
        'strat': {
            'min_tech_score': 70, 'max_change': 3.0, 'min_change': 1.0,
            'min_volume_ratio': 1.35, 'close_strength_min': 0.65,
            'require_macd_golden': True, 'min_conditions': 3,
            'market_min_score': 50, 'weak_market_max_positions': 3,
            'min_composite_score': 70, 'max_rsi': 50,
            'trailing_start': 6.0, 'trailing_pct': 3.0, 'stop_loss': 4.5,
        },
        'add_ratio': 0,
    },
    {
        'name': 'v4_守盈',
        'strat': {
            'min_tech_score': 68, 'max_change': 4.0, 'min_change': 1.5,
            'min_volume_ratio': 1.3, 'close_strength_min': 0.55,
            'market_min_score': 48, 'weak_market_max_positions': 3,
            'min_composite_score': 68,
            'trailing_start': 8.0, 'trailing_pct': 4.0, 'stop_loss': 4.0,
        },
        'add_ratio': 0,
    },
    {
        'name': 'v4_强势市',
        'strat': {
            'min_tech_score': 70, 'max_change': 3.5, 'min_change': 2.0,
            'min_volume_ratio': 1.4, 'close_strength_min': 0.6,
            'require_macd_golden': True, 'min_conditions': 4,
            'market_min_score': 55, 'market_full_score': 60,
            'weak_market_max_positions': 2, 'min_composite_score': 73,
            'trailing_start': 7.0, 'trailing_pct': 3.5, 'stop_loss': 3.5,
        },
        'add_ratio': 0,
    },
    {
        'name': 'v4_少交易',
        'strat': {
            'min_tech_score': 75, 'max_change': 3.0, 'min_change': 1.5,
            'min_volume_ratio': 1.5, 'close_strength_min': 0.7,
            'require_macd_golden': True, 'min_conditions': 4,
            'market_min_score': 52, 'weak_market_max_positions': 2,
            'min_composite_score': 75, 'max_prev_drop': 2.0,
            'trailing_start': 6.0, 'trailing_pct': 3.0, 'stop_loss': 4.5,
        },
        'add_ratio': 0,
        'max_positions': 3,
    },
    {
        'name': 'v4_宽止损',
        'strat': {
            'min_tech_score': 72, 'max_change': 3.5, 'min_change': 1.5,
            'min_volume_ratio': 1.35, 'close_strength_min': 0.6,
            'require_macd_golden': True, 'market_min_score': 50,
            'min_composite_score': 71,
            'trailing_start': 8.0, 'trailing_pct': 4.0, 'stop_loss': 5.5,
        },
        'add_ratio': 0,
    },
    {
        'name': 'v4_快止盈',
        'strat': {
            'min_tech_score': 70, 'max_change': 3.5, 'min_change': 1.5,
            'min_volume_ratio': 1.3, 'close_strength_min': 0.55,
            'require_macd_golden': True, 'market_min_score': 50,
            'min_composite_score': 70,
            'trailing_start': 5.0, 'trailing_pct': 2.5, 'stop_loss': 3.5,
        },
        'add_ratio': 0,
        'tp_levels': [
            {'level': 1, 'pct': 3.0, 'ratio': 0.4},
            {'level': 2, 'pct': 6.0, 'ratio': 0.3},
            {'level': 3, 'pct': 10.0, 'ratio': 0.3},
        ],
    },
    {
        'name': 'v4_加仓保守',
        'strat': {
            'min_tech_score': 72, 'max_change': 3.5, 'min_change': 1.5,
            'min_volume_ratio': 1.4, 'close_strength_min': 0.6,
            'require_macd_golden': True, 'min_conditions': 4,
            'market_min_score': 52, 'min_composite_score': 72,
            'trailing_start': 7.0, 'trailing_pct': 3.5, 'stop_loss': 4.0,
        },
        'add_ratio': 0.15,
        'add_trigger_pct': 8.0,
    },
    # 参数扫描: 止损 x 移动止盈
    *[
        {
            'name': f'v4_s{sl}_t{ts}',
            'strat': {
                'min_tech_score': 70, 'max_change': 3.5, 'min_change': 1.5,
                'min_volume_ratio': 1.35, 'close_strength_min': 0.6,
                'require_macd_golden': True, 'market_min_score': 50,
                'min_composite_score': 70,
                'trailing_start': ts, 'trailing_pct': 3.5, 'stop_loss': sl,
            },
            'add_ratio': 0,
        }
        for sl, ts in product([3.5, 4.0, 4.5, 5.0], [6.0, 7.0, 8.0])
    ],
    # 第二轮: 基于 v4_s5.0_t8.0 扩展
    {
        'name': 'v5_复利',
        'strat': {
            'min_change': 1.5, 'max_change': 3.5, 'min_tech_score': 70,
            'min_volume_ratio': 1.35, 'close_strength_min': 0.6,
            'require_macd_golden': True, 'min_composite_score': 68,
            'market_min_score': 50, 'trailing_start': 8.0,
            'trailing_pct': 3.5, 'stop_loss': 5.0,
        },
        'add_ratio': 0, 'compound_buy': True,
    },
    {
        'name': 'v5_大仓',
        'strat': {
            'min_change': 1.5, 'max_change': 3.5, 'min_tech_score': 70,
            'min_volume_ratio': 1.35, 'close_strength_min': 0.6,
            'require_macd_golden': True, 'min_composite_score': 68,
            'market_min_score': 50, 'trailing_start': 8.0,
            'trailing_pct': 3.5, 'stop_loss': 5.0,
        },
        'add_ratio': 0, 'buy_amount': 15000,
    },
    {
        'name': 'v5_活跃',
        'strat': {
            'min_change': 1.0, 'max_change': 4.0, 'min_tech_score': 68,
            'min_volume_ratio': 1.3, 'close_strength_min': 0.55,
            'require_macd_golden': True, 'min_composite_score': 65,
            'market_min_score': 48, 'trailing_start': 7.0,
            'trailing_pct': 3.5, 'stop_loss': 5.0,
        },
        'add_ratio': 0,
    },
    {
        'name': 'v5_复利加仓',
        'strat': {
            'min_change': 1.5, 'max_change': 3.5, 'min_tech_score': 70,
            'min_volume_ratio': 1.35, 'close_strength_min': 0.6,
            'require_macd_golden': True, 'min_composite_score': 68,
            'market_min_score': 50, 'trailing_start': 8.0,
            'trailing_pct': 3.5, 'stop_loss': 5.0,
        },
        'add_ratio': 0.12, 'add_trigger_pct': 10.0, 'compound_buy': True,
    },
]


def run_grid_variant(item, all_data, trading_days, base_cfg, bt_cfg):
    strat_overrides = item['strat']
    local_bt = {**bt_cfg, 'add_ratio': item.get('add_ratio', 0)}
    if 'add_trigger_pct' in item:
        local_bt['add_trigger_pct'] = item['add_trigger_pct']
    if 'buy_amount' in item:
        local_bt['buy_amount'] = item['buy_amount']
    if item.get('compound_buy'):
        local_bt['compound_buy'] = True

    local_cfg = dict(base_cfg)
    if 'max_positions' in item:
        local_cfg = {**base_cfg, 'portfolio': {
            **base_cfg.get('portfolio', {}),
            'max_positions': item['max_positions'],
        }}

    result = run_variant(item['name'], strat_overrides, all_data, trading_days, local_cfg, local_bt)
    return result


def score_result(r):
    """综合评分: 优先正收益，再看终值和胜率"""
    s = r['summary']
    ret = s['total_return_pct']
    if ret > 0:
        return ret * 2 + s['win_rate'] * 0.5 + s['final_capital'] / 1000
    return ret + s['win_rate'] * 0.2


def main():
    print('=' * 70)
    print('🔬 策略参数优化搜索')
    print('=' * 70)

    config = load_config()
    bt_cfg = {
        'initial_capital': config.get('backtest', {}).get('initial_capital', 100000),
        'buy_amount': 10000,
        'add_trigger_pct': 5.0,
        'add_ratio': 0,
        'commission': config.get('backtest', {}).get('commission', 0.15),
        'stamp_tax': config.get('backtest', {}).get('stamp_tax', 0.1),
    }

    pool = get_pool_manager().get_all_stocks()[:200]
    print(f'\n📥 加载历史数据 ({len(pool)}只)...')
    all_data, _, _ = prepare_data(pool, days=250)
    if not all_data:
        print('❌ 数据获取失败')
        return

    trading_days = get_trading_days(200)
    data_start = min(df.index.min() for df in all_data.values()).strftime('%Y-%m-%d')
    trading_days = [d for d in trading_days if d >= data_start]
    period = {'start': trading_days[0], 'end': trading_days[-1], 'days': len(trading_days)}
    print(f"📅 区间: {period['start']} ~ {period['end']} ({period['days']}天)")

    results = []
    total = len(STRATEGY_GRID)
    for i, item in enumerate(STRATEGY_GRID, 1):
        print(f'  [{i}/{total}] {item["name"]}...', end=' ', flush=True)
        r = run_grid_variant(item, all_data, trading_days, config, bt_cfg)
        s = r['summary']
        print(f"终值{s['final_capital']:,.0f} 收益{s['total_return_pct']:+.1f}% 胜率{s['win_rate']:.0f}%")
        results.append(r)

    results.sort(key=score_result, reverse=True)

    print(f"\n{'='*80}")
    print(f"{'排名':<4} {'策略':<16} {'终值':>10} {'收益率':>8} {'胜率':>6} {'交易':>5} {'月化':>7} {'回撤':>7}")
    print('-' * 80)
    for i, r in enumerate(results[:15], 1):
        s = r['summary']
        print(f"{i:<4} {r['variant']:<16} {s['final_capital']:>10,.0f} {s['total_return_pct']:>+7.1f}% "
              f"{s['win_rate']:>5.1f}% {s['total_trades']:>5} {s['monthly_return_pct']:>+6.1f}% "
              f"{s['max_drawdown_pct']:>6.1f}%")

    best = results[0]
    bs = best['summary']
    print(f"\n🏆 最优: {best['variant']}")
    print(f"   {bs['initial_capital']:,.0f} → {bs['final_capital']:,.0f} ({bs['total_return_pct']:+.2f}%)")
    print(f"   胜率 {bs['win_rate']}% | 月化 {bs['monthly_return_pct']:+.2f}% | 回撤 {bs['max_drawdown_pct']:.1f}%")
    if bs.get('months_to_10m'):
        print(f"   达1000万预估: {bs['months_to_10m']:.0f}个月")
    print('=' * 80)

    os.makedirs('data', exist_ok=True)
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'best_variant': best['variant'],
        'best_summary': bs,
        'best_params': best['params']['strategy'],
        'ranking': [
            {'variant': r['variant'], 'summary': r['summary']}
            for r in results[:20]
        ],
        'variants': {
            r['variant']: {
                'summary': r['summary'],
                'params': r['params'],
                'equity_curve': r['equity_curve'],
                'daily_log': r['daily_log'],
                'trades': r['trades'],
            }
            for r in results[:5]
        },
    }
    with open('data/strategy_optimize_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # 同步到可视化数据
    viz = {
        'generated_at': output['generated_at'],
        'period': period,
        'capital_rules': bt_cfg,
        'best_variant': best['variant'],
        'best_summary': bs,
        'variants': output['variants'],
    }
    with open('data/backtest_capital_results.json', 'w', encoding='utf-8') as f:
        json.dump(viz, f, ensure_ascii=False, indent=2)

    # 更新 config.yaml strategy 段
    best_strat = best['params']['strategy']
    strategy_keys = [
        'min_change', 'max_change', 'min_tech_score', 'min_volume_ratio',
        'min_rsi', 'max_rsi', 'min_conditions', 'require_bullish_candle',
        'require_macd', 'require_ma_trend', 'require_macd_golden',
        'close_strength_min', 'max_prev_drop', 'min_composite_score',
        'market_min_score', 'market_full_score', 'weak_market_max_positions',
        'trailing_start', 'trailing_pct', 'stop_loss',
    ]
    config['strategy'] = {k: best_strat[k] for k in strategy_keys if k in best_strat}
    if best['params'].get('add_ratio', 0) == 0:
        config['backtest']['add_ratio'] = 0
    with open('config/config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    print(f'\n💾 优化结果: data/strategy_optimize_results.json')
    print(f'✅ 已更新 config/config.yaml strategy 段 → {best["variant"]}')


if __name__ == '__main__':
    main()
