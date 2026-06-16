# -*- coding: utf-8 -*-
"""收盘后选股：增量更新本地 K 线，复用回测 _default_screen + analyze_market_regime。"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_pool_manager import get_screening_pool, get_pool_manager
from daily_pipeline_v4 import calculate_technical_indicators, load_config
from src.backtest_data_store import BacktestDataStore
from src.capital_backtest import CapitalBacktester
from src.market_regime import analyze_market_regime, get_hot_sector_pool_names, codes_in_hot_sectors
from src.ml_scorer import build_ml_features, get_ml_scorer
from src.shakeout_strategy import shakeout_rebuy_candidates
from src.strategy_filters import load_strategy_config
from src.trading_calendar import _read_cache, get_expected_last_bar_date, get_latest_trading_day, refresh_trading_dates_cache


def get_trading_days_upto(end_date: str = None, min_count: int = 60) -> List[str]:
    """获取截止 end_date（含）的交易日列表。"""
    cache = _read_cache()
    if not cache or not any(k.isdigit() for k in cache):
        refresh_trading_dates_cache()
        cache = _read_cache()
    dates = []
    for y in sorted(cache.keys()):
        if y.isdigit():
            dates.extend(cache[y])
    end = end_date or datetime.now().strftime('%Y-%m-%d')
    out = sorted(set(d for d in dates if d <= end))
    return out[-max(min_count, len(out)):] if out else []


def prepare_eod_data(
    config: dict = None,
    days: int = 300,
    pool: List[str] = None,
    refresh: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], dict, List[str]]:
    """
    增量更新本地快照并计算技术指标（与 run_backtest_capital.prepare_data 一致）。
    Returns: (all_data, raw_data, store_stats, pool)
    """
    config = config or load_config()
    bt_cfg = config.get('backtest', {})
    max_pool = bt_cfg.get('pool_size', 500)
    pool = pool or get_screening_pool(config, max_pool)

    if bt_cfg.get('use_data_store', True):
        raw, stats = BacktestDataStore().load_or_fetch(pool, days)
    else:
        from src.history_fetcher import get_history_fetcher
        raw = get_history_fetcher().fetch_batch(pool, days=days)
        stats = {}

    all_data = {}
    for code, df in raw.items():
        try:
            df = calculate_technical_indicators(df.copy())
            df['pct_change'] = df['close'].pct_change() * 100
            all_data[code] = df
        except Exception:
            pass
    return all_data, raw, stats, pool


def resolve_screen_date(as_of: str = None) -> str:
    """解析选股基准日：默认取行情应更新到的最近交易日。"""
    if as_of:
        return get_latest_trading_day(
            datetime.strptime(as_of, '%Y-%m-%d').date()
        )
    return get_expected_last_bar_date()


def _setup_ml(
    config: dict,
    raw_data: dict,
    screen_date: str,
    use_ml: bool,
    feature_codes: Optional[Set[str]] = None,
):
    if not use_ml:
        return None, {}
    from src.ml_scorer import MLScorer, build_ml_features, get_ml_scorer

    scorer = MLScorer()
    if scorer.load():
        subset = raw_data
        if feature_codes:
            subset = {c: raw_data[c] for c in feature_codes if c in raw_data}
        if subset:
            ml_data = build_ml_features(raw_data=subset)
            return scorer, ml_data
        return scorer, {}

    ml_data = build_ml_features(raw_data=raw_data)
    scorer = get_ml_scorer(
        config, auto_train=True, ml_data=ml_data, train_end_date=screen_date,
    )
    return scorer, ml_data or {}


def screen_like_backtest(
    config: dict = None,
    screen_date: str = None,
    held_codes: Set[str] = None,
    shakeout_watches: Dict[str, dict] = None,
    refresh_data: bool = True,
    verbose: bool = True,
) -> dict:
    """
    按回测逻辑对指定交易日选股。

    Returns:
        date, regime, can_enter, max_positions, candidates, screened_rows, data_stats
    """
    config = config or load_config()
    strat_cfg = load_strategy_config(config)
    strat_cfg['active_strategies'] = ['trend']

    screen_date = resolve_screen_date(screen_date)
    portfolio_cfg = config.get('portfolio', {})
    max_positions = portfolio_cfg.get('max_positions', 5)
    bt_cfg = config.get('backtest', {})
    days = max(bt_cfg.get('history_days', 300), 280)

    if verbose:
        print(f'\n📥 收盘后数据更新（目标交易日 {screen_date}）...')
    all_data, raw_data, store_stats, _ = prepare_eod_data(
        config, days=days, refresh=refresh_data,
    )
    if not all_data:
        return {
            'date': screen_date,
            'regime': {},
            'can_enter': False,
            'max_positions': 0,
            'candidates': [],
            'screened_rows': [],
            'data_stats': store_stats,
            'reason': '本地/联网数据为空',
        }

    if verbose:
        fetched = store_stats.get('fetched', 0)
        cached = store_stats.get('cached', len(all_data))
        print(f'   成功 {len(all_data)} 只 | 缓存 {cached} | 新拉 {fetched}')

    trading_days = get_trading_days_upto(screen_date)
    data_start = min(df.index.min() for df in all_data.values()).strftime('%Y-%m-%d')
    trading_days = [d for d in trading_days if d >= data_start and d <= screen_date]
    if screen_date not in trading_days:
        trading_days.append(screen_date)
        trading_days.sort()

    try:
        sector_map = get_pool_manager().stock_sectors
    except Exception:
        sector_map = {}

    regime = analyze_market_regime(
        all_data, screen_date, strat_cfg, sector_map, trading_days,
    )
    can_enter = regime.get('can_enter', False)
    day_max = regime.get('max_positions', max_positions)
    reason = _regime_reason(regime, can_enter, strat_cfg)

    if verbose:
        mainlines = [m['sector'] for m in regime.get('mainlines', [])[:5]]
        print(f'\n🌍 市场环境（回测同款 analyze_market_regime）')
        print(f'   日期: {screen_date} | 状态: {regime.get("regime", "?")} | '
              f'评分: {regime.get("score", 0):.1f}')
        print(f'   入场: {"✅ 可入场" if can_enter else "❌ 观望"} | '
              f'最大持仓: {day_max} | {reason}')
        if mainlines:
            print(f'   热门板块: {", ".join(mainlines)}')

    held = set(held_codes or [])
    shakeout_cands = []
    if shakeout_watches and can_enter:
        shakeout_cands = shakeout_rebuy_candidates(
            shakeout_watches, all_data, screen_date, held, strat_cfg, regime=regime,
        )
        if verbose and shakeout_cands:
            print(f'\n📌 止损后连阳接回: {len(shakeout_cands)} 只')
            for c in shakeout_cands:
                print(f'   {c["code"]} {c.get("yang_days", 0)}连阳 收盘{c["buy_price"]:.2f}')

    candidates = []
    if can_enter and day_max > 0:
        ml_codes = set(all_data.keys())
        if strat_cfg.get('require_hot_sector', False):
            hot_names = get_hot_sector_pool_names(regime, strat_cfg)
            if hot_names:
                ml_codes = codes_in_hot_sectors(sector_map, hot_names)
        ml_codes |= held
        if verbose and len(ml_codes) < len(all_data):
            print(f'   ML特征范围: {len(ml_codes)}/{len(all_data)} 只（热门板块+持仓）')
        ml_scorer, ml_data = _setup_ml(
            config, raw_data, screen_date, bt_cfg.get('use_ml', True),
            feature_codes=ml_codes,
        )
        bt = CapitalBacktester(max_positions=day_max)
        bt._sector_map = sector_map
        bt._current_regime = regime
        bt.ml_scorer = ml_scorer
        bt.ml_data = ml_data or {}
        bt._full_config = config
        if verbose:
            print(f'\n🔍 回测同款选股（_default_screen）...')
        candidates = bt._default_screen(
            all_data, screen_date, held, strat_cfg, config,
        )
        if verbose:
            print(f'   候选: {len(candidates)} 只')

    candidates = shakeout_cands + candidates

    screened_rows = candidates_to_screened_rows(candidates, all_data, screen_date)

    return {
        'date': screen_date,
        'regime': regime,
        'can_enter': can_enter,
        'max_positions': day_max,
        'candidates': candidates,
        'screened_rows': screened_rows,
        'shakeout_candidates': shakeout_cands,
        'data_stats': store_stats,
        'reason': reason,
        'all_data': all_data,
    }


def candidates_to_screened_rows(
    candidates: List[dict],
    all_data: Dict[str, pd.DataFrame],
    screen_date: str,
    name_map: Dict[str, str] = None,
) -> List[dict]:
    """将 _default_screen 候选转为 generate_recommendations 所需格式。"""
    rows = []
    dt = pd.Timestamp(screen_date)
    for c in candidates:
        code = str(c.get('code', '')).replace('sh', '').replace('sz', '')
        df = all_data.get(code)
        volume = 0.0
        if df is not None:
            hist = df.loc[df.index <= dt]
            if not hist.empty:
                volume = float(hist.iloc[-1].get('volume', 0))
        name = (name_map or {}).get(code, code)
        rows.append({
            'code': code,
            'code_clean': code,
            'name': name,
            'price': c.get('buy_price', 0),
            'buy_price': c.get('buy_price', 0),
            'change': c.get('change', 0),
            'volume': volume,
            'sector': c.get('sector', '其他'),
            'tech_score': c.get('tech_score', 0),
            'rsi': c.get('rsi', (c.get('signals') or {}).get('rsi', 50)),
            'volume_ratio': c.get('volume_ratio', (c.get('signals') or {}).get('volume_ratio', 1)),
            'score': c.get('score', 0),
            'ml_score': c.get('ml_score', 0),
            'strategy_type': c.get('strategy_type', 'trend'),
            'shakeout_rebuy': c.get('shakeout_rebuy', False),
            'yang_days': c.get('yang_days', 0),
            'signals': c.get('signals', {}),
        })
    return rows


def enrich_candidate_names(rows: List[dict]) -> List[dict]:
    """为候选补充股票名称（仅少量标的，联网开销可接受）。"""
    if not rows:
        return rows
    try:
        from src.data_fetcher_multi import MultiSourceDataFetcher
        fetcher = MultiSourceDataFetcher()
        for row in rows:
            if row.get('name') == row.get('code'):
                row['name'] = fetcher.get_stock_name(row['code']) or row['code']
    except Exception:
        pass
    return rows


def regime_to_sentiment(regime: dict) -> dict:
    """转为 Dashboard / daily_operations 兼容的 sentiment 结构。"""
    if not regime:
        return {'score': 50, 'level': 'neutral'}
    score = regime.get('score', 50)
    level = regime.get('regime', 'neutral')
    return {
        'score': score,
        'level': level,
        'breadth_score': regime.get('breadth_score', score),
        'trend_5d': regime.get('trend_5d', 0),
        'trend_20d': regime.get('trend_20d', 0),
        'mainlines': [m['sector'] for m in regime.get('mainlines', [])[:5]],
        'signals': regime.get('signals', []),
        'source': 'eod_backtest_regime',
    }


def _regime_reason(regime: dict, can_enter: bool, strat_cfg: dict) -> str:
    if not regime:
        return '无市场数据'
    if not can_enter:
        min_score = strat_cfg.get('market_min_score', 50)
        entry_score = regime.get('breadth_score', regime.get('score', 0))
        if entry_score < min_score:
            return f'广度评分{entry_score:.0f}低于阈值{min_score}'
        return f'市场状态{regime.get("regime", "weak")}，暂不建议入场'
    if regime.get('regime') in ('weak', 'bearish'):
        return f'弱市严控仓位≤{regime.get("max_positions", 2)}'
    return '市场条件允许入场'
