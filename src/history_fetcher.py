# -*- coding: utf-8 -*-
"""
统一历史K线获取器

优先级: 腾讯财经(稳定) → 新浪财经 → akshare
带本地磁盘缓存，避免重复请求
"""

import os
import json
import time
import pickle
import hashlib
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests


class HistoryFetcher:
    """多源历史K线获取 + 本地缓存"""

    CACHE_VERSION = 3
    CACHE_TTL_HOURS = None  # None = 持久缓存，按交易日增量更新

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(project_dir, 'data', 'history_cache')
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        })
        self._last_request = 0.0
        self._min_interval = 0.08

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    @staticmethod
    def normalize_code(code: str) -> str:
        return code.replace('.SH', '').replace('.SZ', '').replace('sh', '').replace('sz', '')

    @staticmethod
    def to_tencent_code(code: str) -> str:
        c = HistoryFetcher.normalize_code(code)
        return f"sh{c}" if c.startswith('6') else f"sz{c}"

    @staticmethod
    def to_sina_code(code: str) -> str:
        c = HistoryFetcher.normalize_code(code)
        return f"sh{c}" if c.startswith('6') else f"sz{c}"

    def _cache_path(self, code: str, days: int) -> str:
        c = self.normalize_code(code)
        return os.path.join(self.cache_dir, f"{c}_{days}d_v{self.CACHE_VERSION}.pkl")

    def _load_cache(self, code: str, days: int, check_ttl: bool = True) -> Optional[pd.DataFrame]:
        path = self._cache_path(code, days)
        if not os.path.exists(path):
            return None
        try:
            if check_ttl and self.CACHE_TTL_HOURS is not None:
                mtime = os.path.getmtime(path)
                if (time.time() - mtime) > self.CACHE_TTL_HOURS * 3600:
                    return None
            with open(path, 'rb') as f:
                df = pickle.load(f)
            if df is not None and len(df) >= min(15, days * 0.3):
                return df
        except Exception:
            pass
        return None

    def load_persistent_cache(self, code: str, days: int) -> Optional[pd.DataFrame]:
        return self._load_cache(code, days, check_ttl=False)

    @staticmethod
    def is_cache_fresh(df: pd.DataFrame, days: int) -> bool:
        """缓存是否已包含最近交易日数据。"""
        if df is None or df.empty:
            return False
        try:
            from src.trading_calendar import get_expected_last_bar_date
            expected = pd.Timestamp(get_expected_last_bar_date())
            last = pd.Timestamp(df.index.max())
            return last >= expected and len(df) >= min(15, days * 0.3)
        except Exception:
            return len(df) >= min(15, days * 0.3)

    @staticmethod
    def _merge_history(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        return combined.dropna(subset=['close'])

    def _save_cache(self, code: str, days: int, df: pd.DataFrame):
        try:
            with open(self._cache_path(code, days), 'wb') as f:
                pickle.dump(df, f)
        except Exception:
            pass

    def fetch_tencent(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        """腾讯财经前复权日K"""
        tcode = self.to_tencent_code(code)
        url = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        params = {
            'param': f'{tcode},day,,,{days},qfq',
            '_': int(time.time() * 1000),
        }
        for attempt in range(3):
            try:
                self._throttle()
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                stock_data = data.get('data', {}).get(tcode, {})
                rows = stock_data.get('qfqday') or stock_data.get('day') or []
                if not rows:
                    continue
                parsed = [r[:6] for r in rows if len(r) >= 6]
                if not parsed:
                    continue
                df = pd.DataFrame(parsed, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                df = df[~df.index.duplicated(keep='last')]
                df = df.dropna(subset=['close'])
                if len(df) >= 15:
                    return df
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        return None

    def fetch_sina(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        """新浪财经日K（数据较短，作备用）"""
        scode = self.to_sina_code(code)
        url = 'https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData'
        params = {'symbol': scode, 'scale': '60', 'ma': 'no'}
        for attempt in range(2):
            try:
                self._throttle()
                resp = self.session.get(
                    url, params=params, timeout=15,
                    headers={'Referer': 'https://finance.sina.com.cn'},
                )
                if resp.status_code != 200 or not resp.text:
                    continue
                data = json.loads(resp.text)
                if not data:
                    continue
                df = pd.DataFrame(data)
                df = df.rename(columns={'day': 'date'})
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df['date'] = pd.to_datetime(df['date'].astype(str).str[:10])
                df = df.set_index('date').sort_index()
                df = df[~df.index.duplicated(keep='last')]
                df = df.dropna(subset=['close']).tail(days)
                if len(df) >= 10:
                    return df[['open', 'high', 'low', 'close', 'volume']]
            except Exception:
                time.sleep(0.5)
        return None

    def fetch_akshare(self, code: str, days: int = 120) -> Optional[pd.DataFrame]:
        """akshare 备用（网络不稳定时可能失败）"""
        try:
            import akshare as ak
            c = self.normalize_code(code)
            end = datetime.now()
            start = end - timedelta(days=days + 30)
            df = ak.stock_zh_a_hist(
                symbol=c, period='daily',
                start_date=start.strftime('%Y%m%d'),
                end_date=end.strftime('%Y%m%d'),
                adjust='qfq',
            )
            if df is None or df.empty:
                return None
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna(subset=['close']).tail(days)
        except Exception:
            return None

    def get_history(self, code: str, days: int = 120,
                    start_date: str = None, end_date: str = None,
                    use_cache: bool = True, incremental: bool = True) -> Optional[pd.DataFrame]:
        """
        获取历史K线，自动多源回退；支持持久缓存与增量更新。

        Returns:
            DataFrame: index=date, columns=open,high,low,close,volume
        """
        cached = None
        if use_cache:
            cached = self._load_cache(code, days, check_ttl=self.CACHE_TTL_HOURS is not None)
            if cached is None:
                cached = self.load_persistent_cache(code, days)
            if cached is not None and self.is_cache_fresh(cached, days):
                out = cached.tail(days)
                return self._filter_date_range(out, start_date, end_date)

        df = None
        for fetcher in [self.fetch_tencent, self.fetch_sina, self.fetch_akshare]:
            df = fetcher(code, days)
            if df is not None and len(df) >= 15:
                break
            time.sleep(0.2)

        if df is None:
            if cached is not None:
                out = cached.tail(days)
                return self._filter_date_range(out, start_date, end_date)
            return None

        if use_cache and incremental and cached is not None:
            df = self._merge_history(cached, df).tail(max(days, len(cached)))
        elif use_cache:
            pass

        if use_cache:
            self._save_cache(code, days, df)

        return self._filter_date_range(df.tail(days), start_date, end_date)

    @staticmethod
    def _filter_date_range(df: pd.DataFrame, start_date: str = None,
                           end_date: str = None) -> pd.DataFrame:
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
        return df if len(df) >= 10 else df

    def fetch_batch(self, codes: List[str], days: int = 120,
                    start_date: str = None, end_date: str = None,
                    show_progress: bool = True) -> Dict[str, pd.DataFrame]:
        """批量获取，带进度显示"""
        result = {}
        total = len(codes)
        failed = []

        for i, code in enumerate(codes, 1):
            if show_progress and (i % 25 == 0 or i == total):
                print(f"  进度: {i}/{total} (成功{len(result)})", flush=True)

            df = self.get_history(code, days=days, start_date=start_date,
                                  end_date=end_date, use_cache=True)
            if df is not None and len(df) >= 15:
                result[code] = df
            else:
                failed.append(code)

        if show_progress:
            print(f"  完成: 成功 {len(result)}/{total}", flush=True)
            if failed and len(failed) <= 10:
                print(f"  失败: {', '.join(failed)}")
            elif failed:
                print(f"  失败: {len(failed)} 只")

        return result


_default_fetcher: Optional[HistoryFetcher] = None


def get_history_fetcher() -> HistoryFetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = HistoryFetcher()
    return _default_fetcher
