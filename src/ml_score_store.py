# -*- coding: utf-8 -*-
"""ML 评分磁盘缓存：按股票+日期增量存储，避免每次全量预计算。"""

import json
import os
import pickle
from datetime import datetime
from typing import Dict, Optional

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'stock_predictor.pkl',
)


class MLScoreStore:
    CACHE_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'ml_score_cache',
    )
    STOCK_DIR = os.path.join(CACHE_DIR, 'stocks')
    META_FILE = os.path.join(CACHE_DIR, 'meta.json')

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        os.makedirs(self.STOCK_DIR, exist_ok=True)

    def model_version(self) -> str:
        if os.path.exists(self.model_path):
            return str(int(os.path.getmtime(self.model_path)))
        return 'none'

    def _stock_path(self, code: str, model_ver: str) -> str:
        return os.path.join(self.STOCK_DIR, f'{code}_{model_ver}.pkl')

    def load_stock(self, code: str, model_ver: str = None) -> Dict[str, float]:
        """读取单只股票 date -> score 映射。"""
        model_ver = model_ver or self.model_version()
        path = self._stock_path(code, model_ver)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def save_stock(self, code: str, scores: Dict[str, float], model_ver: str = None):
        model_ver = model_ver or self.model_version()
        path = self._stock_path(code, model_ver)
        try:
            with open(path, 'wb') as f:
                pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    def update_meta(self, stats: dict):
        meta = self.read_meta()
        meta.update({
            'model_version': self.model_version(),
            'model_path': os.path.basename(self.model_path),
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **stats,
        })
        try:
            with open(self.META_FILE, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def read_meta(self) -> dict:
        if not os.path.exists(self.META_FILE):
            return {}
        try:
            with open(self.META_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def stable_key(code: str, date) -> str:
        import pandas as pd
        return f"{code}|{pd.Timestamp(date).strftime('%Y-%m-%d')}"
