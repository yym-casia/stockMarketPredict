# -*- coding: utf-8 -*-
"""回测数据集本地存储：首次拉取后持久化，后续优先读本地并仅增量更新。"""

import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd

from src.history_fetcher import get_history_fetcher


class BacktestDataStore:
    STORE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'backtest_store',
    )
    MANIFEST_FILE = 'manifest.json'

    def __init__(self, store_dir: str = None):
        self.store_dir = store_dir or self.STORE_DIR
        os.makedirs(self.store_dir, exist_ok=True)

    @staticmethod
    def _pool_key(pool: List[str], days: int) -> str:
        raw = f"{days}|{'|'.join(sorted(pool))}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _snapshot_path(self, pool_key: str, days: int) -> str:
        return os.path.join(self.store_dir, f'raw_{days}d_{pool_key}.pkl')

    def _manifest_path(self) -> str:
        return os.path.join(self.store_dir, self.MANIFEST_FILE)

    def _read_manifest(self) -> dict:
        path = self._manifest_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_manifest(self, manifest: dict):
        with open(self._manifest_path(), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _load_snapshot(self, path: str) -> Dict[str, pd.DataFrame]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_snapshot(self, path: str, raw: Dict[str, pd.DataFrame]):
        try:
            with open(path, 'wb') as f:
                pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f'  快照保存失败: {e}')

    def load_or_fetch(
        self, pool: List[str], days: int,
    ) -> Tuple[Dict[str, pd.DataFrame], dict]:
        """
        优先本地加载；仅对缺失或过期股票联网拉取。
        Returns: (raw_data, stats)
        """
        fetcher = get_history_fetcher()
        pool_key = self._pool_key(pool, days)
        snap_path = self._snapshot_path(pool_key, days)
        manifest = self._read_manifest()

        raw: Dict[str, pd.DataFrame] = {}
        from_cache = 0
        updated = 0
        fetched_new = 0

        use_snapshot = (
            manifest.get('pool_key') == pool_key
            and manifest.get('days') == days
            and os.path.exists(snap_path)
        )
        if use_snapshot:
            raw = self._load_snapshot(snap_path)
            print(f'  读取本地快照: {len(raw)} 只 ({os.path.basename(snap_path)})')

        need_network: List[str] = []
        for code in pool:
            df = raw.get(code)
            if df is not None and fetcher.is_cache_fresh(df, days):
                from_cache += 1
                continue
            cached = fetcher.load_persistent_cache(code, days)
            if cached is not None and fetcher.is_cache_fresh(cached, days):
                raw[code] = cached
                from_cache += 1
                continue
            need_network.append(code)

        if need_network:
            print(f'  本地命中 {from_cache} 只, 需联网 {len(need_network)} 只', flush=True)
            for i, code in enumerate(need_network, 1):
                if i % 25 == 0 or i == len(need_network):
                    print(f'  进度: {i}/{len(need_network)} '
                          f'(缓存{from_cache} 更新{updated} 新增{fetched_new})', flush=True)
                had = code in raw or fetcher.load_persistent_cache(code, days) is not None
                df = fetcher.get_history(code, days=days, use_cache=True, incremental=True)
                if df is not None and len(df) >= 15:
                    raw[code] = df
                    if had:
                        updated += 1
                    else:
                        fetched_new += 1
            print(f'  完成: 共 {len(raw)}/{len(pool)} 只 '
                  f'(缓存{from_cache} 更新{updated} 新增{fetched_new})', flush=True)
        else:
            print(f'  全部本地命中: {from_cache}/{len(pool)} 只', flush=True)

        pool_set = set(pool)
        raw = {c: df for c, df in raw.items() if c in pool_set}

        self._save_snapshot(snap_path, raw)
        self._write_manifest({
            'pool_key': pool_key,
            'days': days,
            'pool_size': len(pool),
            'code_count': len(raw),
            'snapshot_file': os.path.basename(snap_path),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })

        stats = {
            'from_cache': from_cache,
            'updated': updated,
            'fetched_new': fetched_new,
            'total': len(raw),
        }
        return raw, stats
