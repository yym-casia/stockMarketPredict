# -*- coding: utf-8 -*-
"""大盘情绪、指数趋势与主线板块识别（回测/实盘通用）"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.strategy_filters import DEFAULT_STRATEGY


def _row_at(df: pd.DataFrame, dt: pd.Timestamp) -> Optional[pd.Series]:
    if df is None or df.empty or dt not in df.index:
        return None
    row = df.loc[dt]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    return row


def _pool_changes(all_data: Dict[str, pd.DataFrame], date: str) -> List[float]:
    dt = pd.Timestamp(date)
    changes = []
    for df in all_data.values():
        row = _row_at(df, dt)
        if row is None:
            continue
        c = row.get('pct_change', np.nan)
        if not pd.isna(c):
            changes.append(float(c))
    return changes


def _synthetic_index_return(all_data: Dict[str, pd.DataFrame],
                            end_date: str, trading_days: List[str],
                            window: int) -> float:
    if window <= 0:
        return 0.0
    dt = pd.Timestamp(end_date)
    prior = [d for d in trading_days if d <= end_date]
    if len(prior) < 2:
        return 0.0
    segment = prior[-window:]
    daily_avg = []
    for d in segment:
        chg = _pool_changes(all_data, d)
        if chg:
            daily_avg.append(np.mean(chg))
    return float(sum(daily_avg)) if daily_avg else 0.0


def compute_sector_momentum(all_data: Dict[str, pd.DataFrame], date: str,
                            sector_map: Dict[str, str],
                            trading_days: List[str],
                            top_n: int = 5,
                            eastmoney_boards: Set[str] = None) -> List[Dict]:
    """按板块聚合涨跌幅，识别当日主线（板块名为东方财富行业板块）。"""
    dt = pd.Timestamp(date)
    sector_changes: Dict[str, List[float]] = {}

    for code, df in all_data.items():
        row = _row_at(df, dt)
        if row is None:
            continue
        c = row.get('pct_change', np.nan)
        if pd.isna(c):
            continue
        sector = sector_map.get(code, '其他')
        if sector in ('其他', '扩展池', ''):
            continue
        if eastmoney_boards and sector not in eastmoney_boards:
            continue
        sector_changes.setdefault(sector, []).append(float(c))

    results = []
    for sector, changes in sector_changes.items():
        if len(changes) < 2:
            continue
        today_chg = float(np.mean(changes))
        prior = [d for d in trading_days if d < date][-4:]
        hist = []
        for d in prior:
            dt2 = pd.Timestamp(d)
            day_vals = []
            for code, df in all_data.items():
                if sector_map.get(code, '其他') != sector:
                    continue
                row = _row_at(df, dt2)
                if row is None:
                    continue
                c = row.get('pct_change', np.nan)
                if not pd.isna(c):
                    day_vals.append(float(c))
            if day_vals:
                hist.append(np.mean(day_vals))
        momentum_5d = float(sum(hist) + today_chg) if hist else today_chg
        results.append({
            'sector': sector,
            'change': round(today_chg, 2),
            'momentum_5d': round(momentum_5d, 2),
            'count': len(changes),
            'momentum_score': round(today_chg * 2 + momentum_5d, 2),
        })

    results.sort(key=lambda x: x['momentum_score'], reverse=True)
    return results[:top_n]


def analyze_market_regime(all_data: Dict[str, pd.DataFrame], date: str,
                          cfg: dict = None,
                          sector_map: Dict[str, str] = None,
                          trading_days: List[str] = None) -> Dict:
    """
    综合广度、指数趋势、情绪动量判断市场状态，并识别主线板块。
    """
    cfg = {**DEFAULT_STRATEGY, **(cfg or {})}
    changes = _pool_changes(all_data, date)

    if len(changes) < 10:
        return {
            'score': 50.0, 'breadth_score': 50.0,
            'trend_5d': 0.0, 'trend_20d': 0.0,
            'regime': 'neutral', 'can_enter': True, 'max_positions': 5,
            'mainlines': [], 'hot_sectors': set(), 'cold_sectors': set(),
            'defensive': False, 'disable_add': False,
            'signals': ['样本不足'],
        }

    avg_change = float(np.mean(changes))
    up_ratio = sum(1 for c in changes if c > 0) / len(changes)
    strong_up = sum(1 for c in changes if c > 2) / len(changes)

    breadth = 50.0 + avg_change * 8 + (up_ratio - 0.5) * 30 + strong_up * 20
    breadth = max(0, min(100, breadth))

    tdays = trading_days or sorted({str(d)[:10] for df in all_data.values()
                                    for d in df.index if str(d)[:10] <= date})
    trend_5d = _synthetic_index_return(all_data, date, tdays, 5)
    trend_20d = _synthetic_index_return(all_data, date, tdays, 20)

    score = breadth
    score += trend_5d * 2.5
    score += trend_20d * 1.2
    if trend_5d < -2.0 and breadth < 45:
        score -= 10
    score = max(0, min(100, score))

    bearish_score = cfg.get('market_bearish_score', 42)
    bearish_trend = cfg.get('market_bearish_trend_pct', -3.0)
    min_score = cfg.get('market_min_score', 50)
    full_score = cfg.get('market_full_score', 55)

    if score < bearish_score and trend_20d <= bearish_trend:
        regime = 'bearish'
    elif score < min_score or (trend_5d < -2.5 and breadth < 48):
        regime = 'weak'
    elif score >= full_score and trend_5d > 0:
        regime = 'strong'
    else:
        regime = 'neutral'

    mainlines = []
    hot_sectors: Set[str] = set()
    cold_sectors: Set[str] = set()
    em_boards = None
    if cfg.get('sector_standard', 'eastmoney') == 'eastmoney':
        try:
            from src.sector_service import load_eastmoney_board_names
            em_boards = load_eastmoney_board_names() or None
        except Exception:
            em_boards = None
    if cfg.get('mainline_enabled', True) and sector_map:
        mainlines = compute_sector_momentum(
            all_data, date, sector_map, tdays,
            top_n=cfg.get('mainline_top_n', 5),
            eastmoney_boards=em_boards,
        )
        min_sec = cfg.get('mainline_min_sector_change', 0.3)
        for m in mainlines:
            if m['change'] >= min_sec and m['momentum_5d'] > 0:
                hot_sectors.add(m['sector'])
        if mainlines:
            cold_cut = min(len(mainlines), 3)
            for m in mainlines[-cold_cut:]:
                if m['change'] < -0.5:
                    cold_sectors.add(m['sector'])

    entry_score = breadth if cfg.get('market_entry_use_breadth', True) else score
    can_enter = entry_score >= min_score and avg_change > -1.0
    if cfg.get('market_block_bearish_entry', False):
        can_enter = can_enter and regime != 'bearish'
    if cfg.get('require_mainline_weak', False) and regime in ('weak', 'bearish'):
        if not hot_sectors:
            can_enter = False

    if entry_score >= full_score:
        max_pos = 5
    elif entry_score >= min_score and can_enter:
        max_pos = cfg.get('weak_market_max_positions', 3)
    else:
        max_pos = 0

    defensive_on = cfg.get('market_defensive_enabled', False)
    defensive = defensive_on and regime == 'bearish'
    disable_add = defensive_on and regime == 'bearish' and cfg.get('disable_add_in_bearish', False)

    signals = [
        f"广度{breadth:.0f}",
        f"5日趋势{trend_5d:+.1f}%",
        f"20日趋势{trend_20d:+.1f}%",
    ]
    if mainlines:
        signals.append(f"主线:{','.join(m['sector'] for m in mainlines[:3])}")

    return {
        'score': round(score, 1),
        'breadth_score': round(breadth, 1),
        'trend_5d': round(trend_5d, 2),
        'trend_20d': round(trend_20d, 2),
        'regime': regime,
        'can_enter': can_enter,
        'max_positions': max_pos,
        'mainlines': mainlines,
        'hot_sectors': hot_sectors,
        'cold_sectors': cold_sectors,
        'defensive': defensive,
        'disable_add': disable_add,
        'signals': signals,
    }


def compute_market_score(all_data: Dict[str, pd.DataFrame], date: str,
                         cfg: dict = None,
                         sector_map: Dict[str, str] = None,
                         trading_days: List[str] = None) -> Tuple[float, bool, int]:
    """兼容旧接口，内部走增强版情绪分析。"""
    r = analyze_market_regime(all_data, date, cfg, sector_map, trading_days)
    return r['score'], r['can_enter'], r['max_positions']


def get_hot_sector_pool_names(regime: Dict, cfg: dict) -> Set[str]:
    """当日选股池限定的热门板块（按动量排名前 N，用于预筛股票）。"""
    n = int(cfg.get('hot_sector_top_n', cfg.get('mainline_top_n', 5)))
    return {m['sector'] for m in regime.get('mainlines', [])[:n] if m.get('sector')}


def codes_in_hot_sectors(sector_map: Dict[str, str], hot_sectors: Set[str]) -> Set[str]:
    """属于热门板块的股票代码集合。"""
    if not hot_sectors:
        return set()
    bad = {'扩展池', '其他', ''}
    try:
        from src.sector_service import is_eastmoney_sector
        return {
            code for code, sector in sector_map.items()
            if sector in hot_sectors and sector not in bad
            and is_eastmoney_sector(sector)
        }
    except Exception:
        return {
            code for code, sector in sector_map.items()
            if sector in hot_sectors and sector not in bad
        }


def summarize_hot_sector_pool(
    sector_map: Dict[str, str],
    hot_sectors: Set[str],
    universe: Optional[Set[str]] = None,
) -> Dict:
    """统计热门板块预筛后的股票池数量（可按 universe 限定，如当日有行情的股票）。"""
    allowed = codes_in_hot_sectors(sector_map, hot_sectors)
    if universe is not None:
        allowed = allowed & universe
    by_sector: Dict[str, int] = {}
    for code in allowed:
        sec = sector_map.get(code, '其他')
        by_sector[sec] = by_sector.get(sec, 0) + 1
    return {
        'total': len(allowed),
        'by_sector': by_sector,
        'hot_sectors': list(hot_sectors),
    }


def format_hot_sector_pool_log(
    summary: Dict,
    total_universe: int = 0,
    date: str = '',
    prefix: str = '🔥 热门板块预筛',
) -> str:
    """格式化热门板块预筛日志。"""
    parts = []
    if date:
        parts.append(f'[{date}]')
    parts.append(prefix)
    if total_universe > 0:
        parts.append(f'{summary["total"]}/{total_universe} 只')
    else:
        parts.append(f'{summary["total"]} 只')
    sectors = summary.get('by_sector') or {}
    if sectors:
        detail = ', '.join(
            f'{name}({cnt})'
            for name, cnt in sorted(sectors.items(), key=lambda x: -x[1])
        )
        parts.append(f'| {detail}')
    return ' '.join(parts)


def resolve_hot_sectors(regime: Dict, cfg: dict) -> Set[str]:
    """当日热门板块集合（用于硬过滤或加分）。"""
    pool_names = get_hot_sector_pool_names(regime, cfg)
    if pool_names:
        return pool_names
    hot = set(regime.get('hot_sectors') or [])
    if hot:
        return hot
    top_n = cfg.get('hot_sector_top_n', cfg.get('mainline_top_n', 5))
    min_chg = cfg.get('mainline_min_sector_change', 0.4)
    for m in regime.get('mainlines', [])[:top_n]:
        if m.get('change', 0) >= min_chg * 0.5:
            hot.add(m['sector'])
    return hot


def apply_mainline_to_candidate(cand: Dict, regime: Dict, cfg: dict) -> Optional[Dict]:
    """主线加分 / 冷门板块过滤 / 热门板块硬过滤。"""
    if not cfg.get('mainline_enabled', True):
        cand['mainline'] = False
        return cand
    sector = cand.get('sector', '其他')
    boost = cfg.get('mainline_hot_boost', 12)
    min_mom = cfg.get('mainline_min_momentum_5d', 1.5)
    min_chg = cfg.get('mainline_min_sector_change', 0.4)
    hot_sectors = resolve_hot_sectors(regime, cfg)

    is_mainline = False
    for m in regime.get('mainlines', []):
        if (m['sector'] == sector and m['change'] >= min_chg
                and m.get('momentum_5d', 0) >= min_mom):
            cand['score'] = round(cand.get('score', 0) + boost, 2)
            cand['mainline'] = True
            is_mainline = True
            break

    if cfg.get('require_hot_sector', False) and hot_sectors:
        if sector not in hot_sectors or sector in ('扩展池', '其他'):
            return None

    if not is_mainline:
        if (cfg.get('filter_cold_sector_in_weak', False)
                and regime.get('regime') in ('weak', 'bearish')
                and sector in regime.get('cold_sectors', set())):
            return None
        cand['mainline'] = False
    return cand


def build_live_regime(sentiment_score: float, cfg: dict) -> Dict:
    """实盘：用情绪分 + 实时板块构建与回测兼容的 regime。"""
    from src.sector_service import get_live_sector_boards, build_regime_from_sectors
    top_n = max(
        int(cfg.get('hot_sector_top_n', cfg.get('mainline_top_n', 5))),
        5,
    )
    sectors = get_live_sector_boards(top_n=top_n * 4)
    regime = build_regime_from_sectors(sectors, cfg)
    regime['score'] = float(sentiment_score)
    bearish_score = cfg.get('market_bearish_score', 35)
    min_score = cfg.get('market_min_score', 50)
    full_score = cfg.get('market_full_score', 55)
    if sentiment_score < bearish_score:
        regime['regime'] = 'bearish'
    elif sentiment_score < min_score:
        regime['regime'] = 'weak'
    elif sentiment_score >= full_score:
        regime['regime'] = 'strong'
    else:
        regime['regime'] = 'neutral'
    regime['defensive'] = (
        cfg.get('market_defensive_enabled', False)
        and regime['regime'] == 'bearish'
    )
    return regime


def should_defensive_exit(profit_pct: float, sector: str, regime: Dict,
                          cfg: dict) -> Optional[str]:
    """弱市持仓防御：亏损超阈值或冷门板块亏损则退出。"""
    if not cfg.get('market_defensive_enabled', False):
        return None
    if regime.get('regime') != 'bearish':
        return None
    loss_thr = cfg.get('market_defensive_exit_pct', -2.0)
    if profit_pct <= loss_thr:
        return f"弱市防御止损({profit_pct:.1f}%)"
    if (sector in regime.get('cold_sectors', set()) and profit_pct < 0
            and cfg.get('exit_cold_sector_in_weak', True)):
        return f"弱市冷门板块清仓({sector} {profit_pct:.1f}%)"
    return None
