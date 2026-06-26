# -*- coding: utf-8 -*-
"""回测/可视化：按买入日拉取后续 K 线（优先本地缓存）"""

import json
import os
import sys
import contextlib
import io

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any, Dict, List, Optional

import pandas as pd

from src.history_fetcher import get_history_fetcher


def _norm_code(code: str) -> str:
    return str(code).replace('sh', '').replace('sz', '').strip()


def fetch_kline_after_buy(
    code: str,
    start_date: str,
    bar_count: int = 30,
    history_days: int = 320,
) -> Dict[str, Any]:
    """
    从 start_date（买入日）起取 bar_count 个交易日 K 线。
    Returns: {ok, code, start_date, bars: [{date,open,high,low,close,volume,pct_change}], error?}
    """
    code = _norm_code(code)
    start_date = str(start_date)[:10]
    bar_count = max(1, min(int(bar_count), 60))

    try:
        fetcher = get_history_fetcher()
        df = fetcher.get_history(code, days=history_days, use_cache=True, incremental=True)
    except Exception as e:
        return {'ok': False, 'code': code, 'start_date': start_date, 'error': str(e)}

    if df is None or len(df) < 5:
        return {
            'ok': False,
            'code': code,
            'start_date': start_date,
            'error': '无行情数据（缓存未命中且拉取失败）',
        }

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    start_ts = pd.Timestamp(start_date)
    sub = df.loc[df.index >= start_ts]
    if sub.empty:
        # 买入日非交易日：取之后第一个交易日
        sub = df.loc[df.index > start_ts]
    if sub.empty:
        return {
            'ok': False,
            'code': code,
            'start_date': start_date,
            'error': f'买入日 {start_date} 之后无 K 线',
        }

    window = sub.head(bar_count)
    bars: List[Dict[str, Any]] = []
    prev_close = None
    for dt, row in window.iterrows():
        close = float(row['close'])
        pct = None
        if prev_close and prev_close > 0:
            pct = round((close / prev_close - 1) * 100, 2)
        elif 'pct_change' in row.index and pd.notna(row.get('pct_change')):
            pct = round(float(row['pct_change']), 2)
        bars.append({
            'date': dt.strftime('%Y-%m-%d'),
            'open': round(float(row['open']), 3),
            'high': round(float(row['high']), 3),
            'low': round(float(row['low']), 3),
            'close': close,
            'volume': int(row.get('volume', 0) or 0),
            'pct_change': pct,
        })
        prev_close = close

    name = ''
    try:
        from stock_pool_manager import get_pool_manager
        pm = get_pool_manager()
        name = pm.stock_names.get(code, '') or pm.stock_names.get(_norm_code(code), '')
    except Exception:
        pass

    return {
        'ok': True,
        'code': code,
        'name': name,
        'start_date': start_date,
        'bar_count': len(bars),
        'bars': bars,
    }


def main_cli():
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) < 3:
        print(json.dumps({'ok': False, 'error': 'usage: trade_kline CODE START_DATE [DAYS]'}))
        return
    code = sys.argv[1]
    start = sys.argv[2]
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    # 拉取过程会 print 日志，重定向到 stderr 避免污染 stdout 的 JSON
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fetch_kline_after_buy(code, start, days)
    junk = buf.getvalue().strip()
    if junk:
        print(junk, file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main_cli()
