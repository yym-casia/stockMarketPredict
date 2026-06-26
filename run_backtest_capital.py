# -*- coding: utf-8 -*-
"""
双策略资金回测 — 趋势选股 + 涨停首板
目标: 200个交易日 10万 → 1000万
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import yaml
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_pool_manager import get_screening_pool
from src.trading_calendar import _read_cache, refresh_trading_dates_cache
from src.history_fetcher import get_history_fetcher
from src.capital_backtest import CapitalBacktester
from src.hot_sector_analytics import analyze_hot_sectors
from src.trade_volume_stats import analyze_trade_volume_stats
from src.buy_entry_stats import analyze_buy_entry_stats, ENTRY_NUMERIC, ENTRY_FLAGS
from src.backtest_data_store import BacktestDataStore
from src.ml_scorer import build_ml_features, get_ml_scorer
from daily_pipeline_v4 import calculate_technical_indicators
from src.strategy_filters import (
    load_strategy_config, TP_SMALL_LOSS_BIG_WIN, TP_V9_RUN_FURTHER, TP_BIG_RUNNER,
)


def load_config():
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_backtest_cfg(config: dict) -> dict:
    bt = dict(config.get('backtest', {}))
    initial = bt.get('initial_capital', 100000)
    pct = bt.get('buy_amount_pct')
    if pct is not None:
        bt['buy_amount_pct'] = float(pct)
        bt['buy_amount'] = initial * bt['buy_amount_pct']
    else:
        bt['buy_amount'] = bt.get('buy_amount', 15000)

    port_max = config.get('portfolio', {}).get('max_positions')
    if port_max is not None:
        bt['portfolio_max_positions'] = int(port_max)
        bt['max_position'] = int(port_max)  # JSON/展示用，与 portfolio 对齐

    trend_pct = bt.get('trend_max_buy_pct') or bt.get('trend_buy_amount_pct') or bt.get('buy_amount_pct')
    if trend_pct is not None:
        trend_pct = float(trend_pct)
        cap = bt.get('max_buy_pct')
        if cap is None or float(cap) < trend_pct:
            bt['max_buy_pct'] = trend_pct

    return bt


def get_trading_days(count=200, end_date: str = None):
    cache = _read_cache()
    if not cache or not any(k.isdigit() for k in cache):
        refresh_trading_dates_cache()
        cache = _read_cache()
    dates = []
    for y in sorted(cache.keys()):
        if y.isdigit():
            dates.extend(cache[y])
    end = end_date or datetime.now().strftime('%Y-%m-%d')
    return sorted(set(d for d in dates if d <= end))[-count:]


def prepare_data(pool, days=280, use_store: bool = True):
    if use_store:
        raw, stats = BacktestDataStore().load_or_fetch(pool, days)
    else:
        raw = get_history_fetcher().fetch_batch(pool, days=days)
        stats = {}
    data = {}
    for code, df in raw.items():
        try:
            df = calculate_technical_indicators(df.copy())
            df['pct_change'] = df['close'].pct_change() * 100
            data[code] = df
        except Exception:
            pass
    return data, raw, stats


V7_TIGHT = {
    'min_change': 5.0, 'max_change': 10.0, 'min_tech_score': 70,
    'min_volume_ratio': 1.35, 'require_macd_golden': True,
    'close_strength_min': 0.6, 'min_composite_score': 70,
    'market_min_score': 50, 'trailing_start': 10.0, 'trailing_pct': 4.0,
    'stop_loss': 3.0, 'min_ml_score': 0.0, 'ml_weight': 0.35, 'ml_rank_only': True,
}

V9_HIGH_UTIL = {
    'min_change': 5.0, 'max_change': 10.0, 'min_tech_score': 70,
    'min_volume_ratio': 1.35, 'require_macd_golden': True,
    'close_strength_min': 0.6, 'min_composite_score': 72,
    'market_min_score': 50, 'trailing_start': 12.0, 'trailing_pct': 5.0,
    'stop_loss': 3.5, 'min_ml_score': 0.0, 'ml_weight': 0.35, 'ml_rank_only': True,
}

# 变体仅控制回测运行方式（策略参数以 config.yaml strategy 为准）
_ISOLATED = {
    '_trend_add_amount_pct': None,
    '_add_ratio': 0.0,
    '_add_trigger_pct': 10.0,
    '_disable_doubler_boost': True,
    '_pool_size': 3000,
}

# 优化脚本用：仅趋势、隔离首板（run_optimize_* 引用此键）
STRATEGY_VARIANTS = {
    '趋势选股': {
        **_ISOLATED,
        '_active_strategies': ['trend'],
        '_disable_limit_up': True,
    },
}


def backtest_variants_for_config(config: dict) -> dict:
    """主资金回测：按 config.yaml 的 limit_up / doubler 开关决定参与策略。"""
    lu_on = bool(config.get('limit_up', {}).get('enabled', False))
    db_on = bool(config.get('doubler', {}).get('independent_enabled', False))
    if not lu_on and not db_on:
        return {'趋势选股': STRATEGY_VARIANTS['趋势选股']}

    active = ['trend']
    label_parts = ['趋势']
    if lu_on:
        active.append('limit_up')
        label_parts.append('首板')
    if db_on:
        active.append('doubler')
        label_parts.append('翻倍')
    return {
        '+'.join(label_parts): {
            **_ISOLATED,
            '_active_strategies': active,
        },
    }


def _json_sanitize(obj):
    """将 numpy 标量转为原生类型，避免 json.dump 失败。"""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run_variant(name, variant_cfg, all_data, ml_data, trading_days, base_cfg, bt_cfg, ml_scorer):
    # 策略参数唯一来源：config.yaml（经 load_strategy_config 合并 DEFAULT）
    strat = load_strategy_config(base_cfg)
    if variant_cfg.get('_active_strategies'):
        strat['active_strategies'] = list(variant_cfg['_active_strategies'])
    v_buy_pct = variant_cfg.get('_buy_amount_pct', bt_cfg.get('buy_amount_pct'))
    v_buy = variant_cfg.get('_buy_amount', bt_cfg.get('buy_amount', 15000))
    v_max_pct = variant_cfg.get('_max_buy_pct', bt_cfg.get('max_buy_pct', 0.15))

    run_cfg = dict(base_cfg)
    if variant_cfg.get('_disable_limit_up'):
        run_cfg['limit_up'] = {**run_cfg.get('limit_up', {}), 'enabled': False}
    if variant_cfg.get('_disable_doubler'):
        run_cfg['doubler'] = {**run_cfg.get('doubler', {}), 'independent_enabled': False, 'enabled': False}
    if variant_cfg.get('_disable_doubler_boost'):
        run_cfg['doubler'] = {**run_cfg.get('doubler', {}), 'boost_enabled': False}
    if variant_cfg.get('_limit_up_use_trend_exit'):
        run_cfg['limit_up'] = {**run_cfg.get('limit_up', {}), 'use_trend_exit': True}
    if variant_cfg.get('_limit_up_max_daily_picks') is not None:
        run_cfg['limit_up'] = {
            **run_cfg.get('limit_up', {}),
            'max_daily_picks': variant_cfg['_limit_up_max_daily_picks'],
        }

    add_r = variant_cfg.get('_add_ratio', bt_cfg.get('add_ratio', 0.0))
    tester = CapitalBacktester(
        initial_capital=bt_cfg.get('initial_capital', 100000),
        buy_amount=v_buy,
        buy_amount_pct=v_buy_pct,
        add_trigger_pct=variant_cfg.get(
            '_add_trigger_pct', bt_cfg.get('add_trigger_pct', 10.0)),
        add_ratio=add_r,
        stop_loss_pct=strat.get('stop_loss', 3.0),
        trailing_start=strat.get('trailing_start', 10.0),
        trailing_pct=strat.get('trailing_pct', 4.0),
        max_hold_days=variant_cfg.get(
            '_max_hold_days',
            strat.get('max_hold_days',
                      base_cfg.get('portfolio', {}).get('max_hold_days', 15)),
        ),
        max_positions=variant_cfg.get(
            '_max_positions', base_cfg.get('portfolio', {}).get('max_positions', 5)),
        commission=bt_cfg.get('commission', 0.15) / 100,
        stamp_tax=bt_cfg.get('stamp_tax', 0.1) / 100,
        compound_buy=bt_cfg.get('compound_buy', True),
        max_buy_pct=v_max_pct,
        compound_only_profit=variant_cfg.get(
            '_compound_only_profit', bt_cfg.get('compound_only_profit', True)),
        ml_scorer=ml_scorer,
        ml_data=ml_data,
        ml_scale_buy=bt_cfg.get('ml_scale_buy', True),
        trend_buy_amount_pct=variant_cfg.get(
            '_trend_buy_amount_pct', bt_cfg.get('trend_buy_amount_pct')),
        trend_max_buy_pct=variant_cfg.get(
            '_trend_max_buy_pct', bt_cfg.get('trend_max_buy_pct')),
        trend_add_amount_pct=variant_cfg.get(
            '_trend_add_amount_pct', bt_cfg.get('trend_add_amount_pct')),
    )
    if ml_scorer:
        ml_scorer.clear_cache()
    result = tester.run(all_data, trading_days, strat, full_config=run_cfg)
    result['variant'] = name
    return result


def _variant_stats(r: dict) -> dict:
    sells = [t for t in r.get('trades', []) if t.get('action') == 'sell']
    big = [t for t in sells if t.get('profit_pct', 0) >= 20]
    dl = r.get('daily_log', [])
    inv = [1 - d['cash'] / d['equity'] for d in dl if d.get('equity', 0) > 0]
    return {
        'big_wins': len(big),
        'big_profit': sum(t.get('profit', 0) for t in big),
        'avg_inv': sum(inv) / len(inv) * 100 if inv else 0,
    }


def print_comparison(results, period, bt_cfg=None, config=None):
    bt_cfg = bt_cfg or {}
    config = config or {}
    max_pos = config.get('portfolio', {}).get('max_positions', 10)
    pct = bt_cfg.get('trend_buy_amount_pct', bt_cfg.get('buy_amount_pct', 0.1))
    stop = config.get('strategy', {}).get('stop_loss', 15)
    target = 10_000_000
    print(f"\n{'='*96}")
    print(f"📊 独立策略对比 ({period['start']}~{period['end']}, {period['days']}个交易日)")
    print(f"   持仓≤{max_pos} | 单票{pct*100:.0f}% | 止损{stop}%")
    print(f"{'='*96}")
    print(f"{'策略':<12} {'终值':>11} {'收益':>7} {'胜率':>5} {'盈亏比':>5} "
          f"{'买入':>5} {'仓位':>5} {'月化':>6}")
    print('-' * 96)

    best = None
    for r in sorted(results, key=lambda x: (x['summary']['final_capital'],
                                            x['summary'].get('profit_factor', 0)), reverse=True):
        s = r['summary']
        st = _variant_stats(r)
        buys = s.get('buy_count', 0)
        print(f"{r['variant']:<12} {s['final_capital']:>11,.0f} {s['total_return_pct']:>+6.1f}% "
              f"{s['win_rate']:>4.0f}% {s.get('profit_factor',0):>5.2f} "
              f"{buys:>5} {st['avg_inv']:>4.0f}% {s['monthly_return_pct']:>+5.1f}%")
        if best is None:
            best = r

    if best:
        s = best['summary']
        daily_ret = ((s['final_capital'] / s['initial_capital']) ** (1 / period['days']) - 1) * 100
        need_daily = (target / s['initial_capital']) ** (1 / period['days']) - 1
        print(f"\n🏆 最优: {best['variant']}")
        print(f"   {s['initial_capital']:,.0f} → {s['final_capital']:,.0f} ({s['total_return_pct']:+.2f}%)")
        print(f"   胜率 {s['win_rate']}% | 盈亏比 {s.get('profit_factor',0)} | "
              f"均赢{s.get('avg_win',0):+.0f} 均亏{s.get('avg_loss',0):+.0f}")
        print(f"   趋势: 买{s.get('trend_buys',0)} 盈亏{s.get('trend_profit',0):+,.0f} | "
              f"涨停: 买{s.get('limit_up_buys',0)} 盈亏{s.get('limit_up_profit',0):+,.0f} | "
              f"翻倍: 买{s.get('doubler_buys',0)} 盈亏{s.get('doubler_profit',0):+,.0f}")
        st = _variant_stats(best)
        dl = best.get('daily_log', [])
        if dl:
            print(f"   平均持仓占比: {st['avg_inv']:.1f}% | 均持仓 "
                  f"{sum(d['positions'] for d in dl)/len(dl):.1f}只 | 大赢≥20%: {st['big_wins']}笔")
        print(f"   日均 {daily_ret:+.3f}% | 达1000万需日均 {need_daily*100:.2f}%")
        base = next((x for x in results if x['variant'].startswith('0_')), None)
        if base and best['variant'] != base['variant']:
            diff = s['final_capital'] - base['summary']['final_capital']
            print(f"   较基准: {diff:+,.0f} ({s['total_return_pct']-base['summary']['total_return_pct']:+.1f}%)")
    print('=' * 96)
    return best


def main():
    config = load_config()
    hold = config.get('portfolio', {}).get('max_hold_days', 15)
    print('=' * 88)
    bt_cfg = resolve_backtest_cfg(config)
    pct = bt_cfg.get('buy_amount_pct')
    buy_label = f'总资产 {pct * 100:.0f}%' if pct is not None else f'{bt_cfg["buy_amount"]:,.0f} 元'
    add_r = bt_cfg.get('add_ratio', 0)
    trend_add = bt_cfg.get('trend_add_amount_pct')
    if trend_add:
        add_label = f'趋势加仓 +{bt_cfg.get("add_trigger_pct", 10):.0f}%触发，每次总资产{trend_add*100:.0f}%'
    elif add_r > 0:
        add_label = f'强者恒强 +{bt_cfg.get("add_trigger_pct", 10):.0f}%加仓市值{add_r * 100:.0f}%'
    else:
        add_label = '不加仓'
    initial = bt_cfg.get('initial_capital', 200000)
    max_pos = config.get('portfolio', {}).get('max_positions', 10)
    trend_pct = bt_cfg.get('trend_buy_amount_pct', bt_cfg.get('buy_amount_pct', 0.1))
    stop_loss = config.get('strategy', {}).get('stop_loss', 15)
    min_vr = config.get('strategy', {}).get('min_volume_ratio', 1.2)
    max_vr = config.get('strategy', {}).get('max_volume_ratio', 0)
    vr_label = f'量比{min_vr}-{max_vr}' if max_vr else f'量比≥{min_vr}'
    print(f'🚀 资金回测 | 初始{initial:,.0f} | 持仓≤{max_pos} | 单票{trend_pct*100:.0f}% | 止损{stop_loss}%')
    print(f'   趋势: {max_pos}只×{trend_pct*100:.0f}% | 止损{stop_loss}% 拖尾15%/6% | 震仓方案A | 买入{vr_label}')
    print('=' * 88)
    bt_cfg.setdefault('add_trigger_pct', 10.0)
    bt_cfg.setdefault('add_ratio', 0.30)
    bt_cfg.setdefault('compound_buy', True)
    bt_cfg.setdefault('max_buy_pct', bt_cfg.get('max_buy_pct', 0.05))
    bt_cfg.setdefault('ml_scale_buy', True)
    bt_cfg.setdefault('pool_size', 500)
    bt_cfg.setdefault('use_ml', True)
    bt_cfg['limit_up_enabled'] = config.get('limit_up', {}).get('enabled', True)
    bt_cfg.setdefault('commission', 0.15)
    bt_cfg.setdefault('stamp_tax', 0.1)

    variants = backtest_variants_for_config(config)
    max_pool = bt_cfg['pool_size']
    for v in variants.values():
        max_pool = max(max_pool, v.get('_pool_size', 0))
    pool = get_screening_pool(config, max_pool)
    print(f'\n📥 加载 {len(pool)} 只股票...')
    use_store = bt_cfg.get('use_data_store', True)
    all_data, raw_data, _ = prepare_data(pool, days=300, use_store=use_store)
    if not all_data:
        print('❌ 数据获取失败')
        return
    print(f'   成功: {len(all_data)} 只', flush=True)

    trading_days = get_trading_days(
        config.get('targets', {}).get('backtest_days', 200),
        end_date=config.get('targets', {}).get('backtest_end_date'),
    )
    data_start = min(df.index.min() for df in all_data.values()).strftime('%Y-%m-%d')
    trading_days = [d for d in trading_days if d >= data_start]
    period = {'start': trading_days[0], 'end': trading_days[-1], 'days': len(trading_days)}

    ml_scorer = None
    ml_data = {}
    if bt_cfg['use_ml']:
        print(f'\n🤖 ML模型 (训练截止 {period["start"]})...', flush=True)
        print('   构建 ML 特征...', flush=True)
        ml_data = build_ml_features(raw_data=raw_data)
        print(f'   ML特征: {len(ml_data)} 只', flush=True)
        ml_scorer = get_ml_scorer(config, auto_train=True, ml_data=ml_data,
                                  train_end_date=period['start'])
        if ml_scorer and ml_data:
            est = len(ml_data) * len(trading_days)
            use_ml_cache = bt_cfg.get('use_ml_score_cache', True)
            print(f'   预计算 ML 评分 (约 {est:,} 次, 磁盘缓存={"开" if use_ml_cache else "关"})...',
                  flush=True)
            ml_scorer.precompute(ml_data, trading_days, persist=use_ml_cache)

    print(f"\n📅 回测: {period['start']} ~ {period['end']} ({period['days']}天)")
    print(f"   策略: {', '.join(variants.keys())}")
    if config.get('limit_up', {}).get('enabled'):
        lu = config.get('limit_up', {})
        print(f"   首板: 启用 | 量比≥{lu.get('min_volume_ratio', 3)} | "
              f"每日≤{lu.get('max_daily_picks', 1)}只 | "
              f"lookback {lu.get('lookback_days', 30)}日")

    results = []
    for name, variant in variants.items():
        print(f'  ⏳ {name}...', flush=True)
        result = run_variant(name, variant, all_data, ml_data, trading_days,
                             config, bt_cfg, ml_scorer)
        s = result['summary']
        print(f'  ✅ {name}: {s["final_capital"]:,.0f} ({s["total_return_pct"]:+.1f}%) '
              f'买入{s.get("buy_count", 0)}笔', flush=True)
        results.append(result)

    best = print_comparison(results, period, bt_cfg, config)

    os.makedirs('data', exist_ok=True)
    output = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': period,
        'capital_rules': bt_cfg,
        'limit_up_rules': config.get('limit_up', {}),
        'doubler_rules': config.get('doubler', {}),
        'target_capital': 10_000_000,
        'comparison_mode': 'isolated',
        'best_variant': best['variant'] if best else None,
        'best_summary': best['summary'] if best else None,
        'variants': {
            r['variant']: {
                'summary': r['summary'],
                'params': r['params'],
                'equity_curve': r['equity_curve'],
                'daily_log': r['daily_log'],
                'regime_log': r.get('regime_log', []),
                'hot_sector_stats': analyze_hot_sectors(r.get('regime_log', [])),
                'volume_ratio_stats': analyze_trade_volume_stats(r['trades'], all_data),
                'buy_entry_stats': analyze_buy_entry_stats(r['trades'], all_data),
                'trades': r['trades'],
            }
            for r in results
        },
    }
    path = 'data/backtest_capital_results.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_sanitize(output), f, ensure_ascii=False, indent=2)
    print(f'\n💾 结果: {path}')
    print(f'📊 可视化: python serve_dashboard.py → http://localhost:8088/backtest_viz.html')


if __name__ == '__main__':
    main()
