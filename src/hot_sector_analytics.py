# -*- coding: utf-8 -*-
"""热门板块出现规律统计（回测分析）"""

from typing import Dict, List, Optional


def _group_episodes(
    appearance_dates: List[str],
    date_idx: Dict[str, int],
    max_gap: int = 5,
) -> List[Dict]:
    """
    将板块出现日期合并为「轮次」。
    两次出现间隔 <= max_gap 个交易日视为连续（同一轮次）。
    """
    if not appearance_dates:
        return []
    ordered = sorted(appearance_dates, key=lambda d: date_idx.get(d, 0))
    episodes: List[Dict] = []
    ep_start = ordered[0]
    ep_end = ordered[0]
    ep_days = [ordered[0]]

    for d in ordered[1:]:
        gap = date_idx.get(d, 0) - date_idx.get(ep_end, 0)
        if gap <= max_gap:
            ep_end = d
            ep_days.append(d)
        else:
            episodes.append({
                'start': ep_start,
                'end': ep_end,
                'duration': date_idx[ep_end] - date_idx[ep_start] + 1,
                'days': ep_days,
            })
            ep_start = d
            ep_end = d
            ep_days = [d]

    episodes.append({
        'start': ep_start,
        'end': ep_end,
        'duration': date_idx[ep_end] - date_idx[ep_start] + 1,
        'days': ep_days,
    })
    return episodes


def analyze_hot_sectors(
    regime_log: List[Dict],
    continuity_gap: int = 5,
) -> Dict:
    """
    分析热门板块规律。

    返回:
      - daily: 每日热门板块记录（表格用）
      - sector_summary: 各板块持续天数、出现间隔等汇总
    """
    if not regime_log:
        return {
            'continuity_gap_days': continuity_gap,
            'daily': [],
            'sector_summary': [],
            'total_trading_days': 0,
        }

    dates = [r['date'] for r in regime_log if r.get('date')]
    date_idx = {d: i for i, d in enumerate(dates)}

    daily_records = []
    sector_dates: Dict[str, List[str]] = {}

    for r in regime_log:
        date = r.get('date', '')
        hot = r.get('hot_sectors') or r.get('mainlines') or []
        daily_records.append({
            'date': date,
            'hot_sectors': hot,
            'hot_pool_size': r.get('hot_pool_size', 0),
            'hot_pool_by_sector': r.get('hot_pool_by_sector') or {},
            'hot_sectors_detail': r.get('hot_sectors_detail') or [],
        })
        for sec in hot:
            if sec:
                sector_dates.setdefault(sec, []).append(date)

    sector_summary = []
    for sector, appearances in sector_dates.items():
        episodes = _group_episodes(appearances, date_idx, continuity_gap)
        durations = [e['duration'] for e in episodes]
        intervals: List[int] = []
        for i in range(len(episodes) - 1):
            gap = date_idx[episodes[i + 1]['start']] - date_idx[episodes[i]['end']] - 1
            if gap > 0:
                intervals.append(gap)

        sector_summary.append({
            'sector': sector,
            'episodes': len(episodes),
            'total_hot_days': sum(durations),
            'avg_duration': round(sum(durations) / len(durations), 1) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'durations': durations,
            'intervals': intervals,
            'avg_interval': round(sum(intervals) / len(intervals), 1) if intervals else None,
            'min_interval': min(intervals) if intervals else None,
            'max_interval': max(intervals) if intervals else None,
            'episode_detail': [
                {'start': e['start'], 'end': e['end'], 'duration': e['duration']}
                for e in episodes
            ],
        })

    sector_summary.sort(key=lambda x: (-x['total_hot_days'], -x['episodes']))

    return {
        'continuity_gap_days': continuity_gap,
        'daily': daily_records,
        'sector_summary': sector_summary,
        'total_trading_days': len(dates),
        'unique_sectors': len(sector_summary),
    }
