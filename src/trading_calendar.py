# -*- coding: utf-8 -*-
"""A股交易日历：跳过周末与法定节假日（本地缓存，避免 py_mini_racer 多进程崩溃）"""

import json
import os
from datetime import datetime, date, timedelta, time
from typing import Optional, Set

_CACHE_FILE = None


def _cache_path() -> str:
    global _CACHE_FILE
    if _CACHE_FILE is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _CACHE_FILE = os.path.join(project_dir, 'data', 'trading_dates_cache.json')
    return _CACHE_FILE


def _read_cache() -> dict:
    path = _cache_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _write_cache(data: dict):
    path = _cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _dates_for_year(year: int) -> Set[str]:
    cache = _read_cache()
    return set(cache.get(str(year), []))


def refresh_trading_dates_cache() -> bool:
    """从 akshare 刷新交易日缓存（仅在子进程中调用，如 run_daily.py）"""
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        if df is None or df.empty:
            return False
        col = 'trade_date' if 'trade_date' in df.columns else df.columns[0]
        by_year: dict = {}
        for d in df[col]:
            if isinstance(d, date):
                ds = d.strftime('%Y-%m-%d')
            else:
                ds = str(d)[:10]
            by_year.setdefault(ds[:4], []).append(ds)
        cache = _read_cache()
        cache.update(by_year)
        cache['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        _write_cache(cache)
        return True
    except Exception as e:
        print(f"  交易日历缓存刷新失败: {e}")
        return False


def is_trading_day(dt: Optional[datetime] = None, use_network: bool = False) -> bool:
    """判断是否为交易日。默认只读本地缓存，不加载 akshare。"""
    if dt is None:
        dt = datetime.now()
    d = dt.date() if isinstance(dt, datetime) else dt
    if d.weekday() >= 5:
        return False

    if use_network:
        refresh_trading_dates_cache()

    trading_dates = _dates_for_year(d.year)
    if trading_dates:
        return d.strftime('%Y-%m-%d') in trading_dates
    return True


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """判断是否在交易时段内（9:30-11:30, 13:00-15:00）"""
    if dt is None:
        dt = datetime.now()
    if not is_trading_day(dt):
        return False
    t = dt.time()
    morning = time(9, 30) <= t <= time(11, 30)
    afternoon = time(13, 0) <= t <= time(15, 0)
    return morning or afternoon


def count_trading_days(start: str, end: str) -> int:
    """计算两个日期之间的交易日数"""
    s = datetime.strptime(start, '%Y-%m-%d').date()
    e = datetime.strptime(end, '%Y-%m-%d').date()
    if s > e:
        s, e = e, s
    count = 0
    cur = s
    while cur <= e:
        if is_trading_day(datetime.combine(cur, datetime.min.time())):
            count += 1
        cur += timedelta(days=1)
    return max(1, count)


def get_latest_trading_day(on_or_before: Optional[date] = None) -> str:
    """获取某日（默认今天）当日或之前最近的交易日"""
    d = on_or_before or date.today()
    for _ in range(30):
        if is_trading_day(datetime.combine(d, datetime.min.time())):
            return d.strftime('%Y-%m-%d')
        d -= timedelta(days=1)
    return (on_or_before or date.today()).strftime('%Y-%m-%d')


def get_expected_last_bar_date() -> str:
    """行情数据应更新到的最近交易日（收盘后视为当日，否则昨日）"""
    now = datetime.now()
    d = now.date()
    if is_trading_day(now) and now.hour >= 16:
        return get_latest_trading_day(d)
    prev = d - timedelta(days=1)
    return get_latest_trading_day(prev)


def get_next_trading_day(dt: Optional[datetime] = None) -> str:
    """获取下一个交易日"""
    if dt is None:
        dt = datetime.now()
    d = dt.date() if isinstance(dt, datetime) else dt
    for _ in range(30):
        d += timedelta(days=1)
        if is_trading_day(datetime.combine(d, datetime.min.time())):
            return d.strftime('%Y-%m-%d')
    return (dt.date() + timedelta(days=1)).strftime('%Y-%m-%d')
