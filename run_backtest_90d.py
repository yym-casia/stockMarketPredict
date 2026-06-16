# -*- coding: utf-8 -*-
"""近90个交易日胜率回测（策略 v1 vs v2 对比）"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_pool_manager import get_pool_manager
from src.trading_calendar import refresh_trading_dates_cache, _read_cache
from src.position_manager import PositionManager
from src.history_fetcher import get_history_fetcher
from daily_pipeline_v4 import calculate_technical_indicators
from src.strategy_filters import (
    load_strategy_config, screen_stock_row,
    compute_market_score, rank_candidates,
)


def load_config():
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_trading_days(count: int = 90) -> List[str]:
    cache = _read_cache()
    if not cache or not any(k.isdigit() for k in cache):
        refresh_trading_dates_cache()
        cache = _read_cache()
    all_dates = []
    for year in sorted(cache.keys()):
        if year.isdigit():
            all_dates.extend(cache[year])
    today = datetime.now().strftime('%Y-%m-%d')
    return sorted(set(d for d in all_dates if d <= today))[-count:]


def prepare_history_data(pool: List[str], days: int = 200) -> Dict[str, pd.DataFrame]:
    """批量获取并计算技术指标"""
    fetcher = get_history_fetcher()
    raw = fetcher.fetch_batch(pool, days=days)
    all_data = {}
    for code, df in raw.items():
        try:
            df = calculate_technical_indicators(df.copy())
            df['pct_change'] = df['close'].pct_change() * 100
            all_data[code] = df
        except Exception:
            pass
    return all_data


def screen_candidates(all_data, date, held_codes, strat_cfg):
    candidates, dt = [], pd.Timestamp(date)
    for code, df in all_data.items():
        if code in held_codes or dt not in df.index:
            continue
        loc = df.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < 15:
            continue
        row = df.iloc[loc]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        row = row.copy()
        row['code'] = code
        result = screen_stock_row(row, df.iloc[:loc + 1], strat_cfg)
        if result:
            result['buy_date'] = date
            candidates.append(result)
    return rank_candidates(candidates, top_n=10)


def _open_position(code, buy_date, buy_price, pos_mgr):
    pos_mgr.add_position(code=code, name=code, buy_price=buy_price, buy_date=buy_date)
    return {'code': code, 'buy_date': buy_date, 'buy_price': buy_price,
            'remaining_ratio': 1.0, 'realized_return': 0.0, 'days_held': 0, 'closed': False}


def _close_state(state, exit_date, exit_price, reason):
    state['closed'] = True
    ret = state['realized_return']
    return {'code': state['code'], 'buy_date': state['buy_date'], 'exit_date': exit_date,
            'buy_price': round(state['buy_price'], 2), 'exit_price': round(exit_price, 2),
            'return_pct': round(ret, 2), 'days_held': state['days_held'],
            'exit_reason': reason, 'win': ret > 0, 'hit_target': ret >= 5.0}


def _advance_position(state, row, date_str, max_hold, pos_mgr):
    if state['closed']:
        return None
    code, buy_price = state['code'], state['buy_price']
    pos = pos_mgr.positions[code]
    state['days_held'] += 1

    if row['low'] <= pos.stop_loss and state['remaining_ratio'] > 0:
        ret = (pos.stop_loss - buy_price) / buy_price * 100
        state['realized_return'] += ret * state['remaining_ratio']
        state['remaining_ratio'] = 0
        return _close_state(state, date_str, pos.stop_loss, 'stop_loss')

    for level in pos.take_profit_levels:
        if not level['triggered'] and state['remaining_ratio'] > 0:
            if row['high'] >= buy_price * (1 + level['pct'] / 100):
                level['triggered'] = True
                state['realized_return'] += level['pct'] * level['ratio'] * state['remaining_ratio']
                state['remaining_ratio'] *= (1 - level['ratio'])

    pos.highest_price = max(pos.highest_price, float(row['high']))
    if (pos.highest_price - buy_price) / buy_price * 100 >= pos_mgr.trailing_start and state['remaining_ratio'] > 0:
        dd = (pos.highest_price - row['low']) / pos.highest_price * 100
        if dd >= pos_mgr.trailing_pct:
            ret = (float(row['low']) - buy_price) / buy_price * 100
            state['realized_return'] += ret * state['remaining_ratio']
            state['remaining_ratio'] = 0
            return _close_state(state, date_str, float(row['low']), 'trailing_stop')

    result = pos_mgr.update_price(code, float(row['close']), date_str)
    if result['action'] == 'sell' and state['remaining_ratio'] > 0:
        ret = (float(row['close']) - buy_price) / buy_price * 100
        state['realized_return'] += ret * state['remaining_ratio']
        state['remaining_ratio'] = 0
        reason = 'expired' if '期满' in result.get('reason', '') else 'take_profit'
        return _close_state(state, date_str, float(row['close']), reason)

    if state['days_held'] >= max_hold and state['remaining_ratio'] > 0:
        ret = (float(row['close']) - buy_price) / buy_price * 100
        state['realized_return'] += ret * state['remaining_ratio']
        state['remaining_ratio'] = 0
        return _close_state(state, date_str, float(row['close']), 'expired')
    return None


def _simulate(all_data, trading_days, config, strat_cfg, trailing_start, trailing_pct, sl):
    max_pos = config.get('portfolio', {}).get('max_positions', 5)
    max_hold = config.get('portfolio', {}).get('max_hold_days', 15)
    pos_mgr = PositionManager(max_hold_days=max_hold, stop_loss_pct=sl,
                              trailing_start=trailing_start, trailing_pct=trailing_pct)
    active, trades, skipped = {}, [], 0
    use_market = strat_cfg.get('market_min_score', 0) > 0

    for date in trading_days:
        day_max = max_pos
        if use_market:
            _, ok, day_max = compute_market_score(all_data, date)
            if not ok:
                skipped += 1
                continue

        for code in list(active.keys()):
            if code not in all_data or pd.Timestamp(date) not in all_data[code].index:
                continue
            if active[code]['buy_date'] == date:
                continue
            row = all_data[code].loc[pd.Timestamp(date)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            trade = _advance_position(active[code], row, date, max_hold, pos_mgr)
            if trade:
                trades.append(trade)
                pos_mgr.remove_position(code)
                del active[code]

        slots = day_max - len(active)
        for cand in screen_candidates(all_data, date, set(active.keys()), strat_cfg)[:slots]:
            active[cand['code']] = _open_position(cand['code'], date, cand['buy_price'], pos_mgr)

    if trading_days and active:
        last = trading_days[-1]
        for code, state in list(active.items()):
            if code in all_data and pd.Timestamp(last) in all_data[code].index:
                row = all_data[code].loc[pd.Timestamp(last)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                if state['remaining_ratio'] > 0:
                    ret = (float(row['close']) - state['buy_price']) / state['buy_price'] * 100
                    state['realized_return'] += ret * state['remaining_ratio']
                trades.append(_close_state(state, last, float(row['close']), 'backtest_end'))

    return _build_result(trades, skipped, strat_cfg, sl, trailing_start, trailing_pct)


def _build_result(trades, skipped, strat_cfg, sl, ts, tp):
    if not trades:
        return {'error': '无交易记录', 'skipped_market_days': skipped}
    wins = [t for t in trades if t['win']]
    rets = [t['return_pct'] for t in trades]
    by_reason = {}
    for t in trades:
        by_reason.setdefault(t['exit_reason'], []).append(t)
    return {
        'summary': {
            'total_trades': len(trades), 'wins': len(wins),
            'losses': len(trades) - len(wins),
            'win_rate': round(len(wins) / len(trades) * 100, 1),
            'target_hit_rate': round(sum(1 for t in trades if t['hit_target']) / len(trades) * 100, 1),
            'avg_return': round(np.mean(rets), 2),
            'median_return': round(float(np.median(rets)), 2),
            'avg_hold_days': round(np.mean([t['days_held'] for t in trades]), 1),
        },
        'exit_breakdown': {r: {'count': len(ts), 'win_rate': round(sum(1 for t in ts if t['win']) / len(ts) * 100, 1),
                             'avg_return': round(np.mean([t['return_pct'] for t in ts]), 2)}
                           for r, ts in by_reason.items()},
        'skipped_market_days': skipped,
        'trades': trades,
        'params': {'stop_loss': sl, 'trailing_start': ts, 'trailing_pct': tp},
    }


def run_backtest(days=90, max_stocks=150):
    config = load_config()
    strat_cfg = load_strategy_config(config)
    trading_days = get_trading_days(days)

    pool = get_pool_manager().get_all_stocks()[:max_stocks]
    print(f"\n📥 获取历史数据（腾讯财经，最多200天）...")
    all_data = prepare_history_data(pool, days=200)
    if not all_data:
        return {'error': '无法获取历史数据'}

    data_start = min(df.index.min() for df in all_data.values()).strftime('%Y-%m-%d')
    trading_days = [d for d in trading_days if d >= data_start]
    print(f"📅 数据: {data_start}~{trading_days[-1]}，回测 {len(trading_days)} 个交易日，{len(all_data)} 只股票")

    loose = {**strat_cfg, 'min_change': 2.0, 'max_change': 7.0, 'min_tech_score': 50,
             'min_volume_ratio': 0.8, 'min_conditions': 2, 'require_bullish_candle': False,
             'require_macd': False, 'require_ma_trend': False, 'market_min_score': 0}

    print("\n⏳ v1 优化前...")
    v1 = _simulate(all_data, trading_days, config, loose, 5.0, 3.0, 3.0)
    print("⏳ v2 优化后...")
    v2 = _simulate(all_data, trading_days, config, strat_cfg,
                   strat_cfg.get('trailing_start', 4.0), strat_cfg.get('trailing_pct', 2.5),
                   strat_cfg.get('stop_loss', 3.5))

    return {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'period': {'start': trading_days[0], 'end': trading_days[-1], 'trading_days': len(trading_days)},
        'stocks_tested': len(all_data),
        'data_source': 'tencent',
        'v1_baseline': v1,
        'v2_optimized': v2,
    }


def print_report(result):
    if result.get('error'):
        print(f"\n❌ {result['error']}")
        return
    p = result['period']
    print(f"\n{'='*60}\n📊 策略对比 ({p['start']}~{p['end']}, {p['trading_days']}天, "
          f"{result['stocks_tested']}只, 数据源:{result.get('data_source','')})\n{'='*60}")
    for label, key in [('v1 优化前', 'v1_baseline'), ('v2 优化后', 'v2_optimized')]:
        r = result.get(key, {})
        if 'error' in r:
            print(f"\n【{label}】{r['error']}")
            continue
        s = r['summary']
        print(f"\n【{label}】{s['total_trades']}笔 | 胜率{s['win_rate']}% | "
              f"均收益{s['avg_return']:+.2f}% | 止损{r['exit_breakdown'].get('stop_loss',{}).get('count',0)}笔")
    v1s, v2s = result.get('v1_baseline', {}).get('summary', {}), result.get('v2_optimized', {}).get('summary', {})
    if v1s and v2s:
        print(f"\n【提升】胜率 {v1s['win_rate']}%→{v2s['win_rate']}% | "
              f"收益 {v1s['avg_return']:+.2f}%→{v2s['avg_return']:+.2f}%")
    print('='*60)


def main():
    print('='*60 + '\n🧪 策略优化对比回测\n' + '='*60)
    result = run_backtest()
    print_report(result)
    os.makedirs('data', exist_ok=True)
    path = 'data/backtest_90d_results_v2.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 {path}")


if __name__ == '__main__':
    main()
