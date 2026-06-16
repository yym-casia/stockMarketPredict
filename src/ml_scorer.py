# -*- coding: utf-8 -*-
"""ML 评分模块 — 训练/加载 XGBoost，为选股提供概率分"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.features import FeatureEngineer
from src.model import StockPredictor
from src.ml_score_store import MLScoreStore, DEFAULT_MODEL_PATH

MODEL_DIR = 'models'


class MLScorer:
    def __init__(self, predictor: StockPredictor = None, feature_engineer: FeatureEngineer = None):
        self.predictor = predictor or StockPredictor()
        self.fe = feature_engineer or FeatureEngineer()
        self.feature_names: List[str] = []
        self._cache: Dict[str, float] = {}
        self._score_store = MLScoreStore()

    @property
    def ready(self) -> bool:
        return self.predictor.model is not None and len(self.feature_names) > 0

    def _cache_key(self, df: pd.DataFrame, date, code: str = None) -> str:
        if code:
            return MLScoreStore.stable_key(code, date)
        return f"{id(df)}_{pd.Timestamp(date)}"

    def train_from_pool(self, ml_data: Dict[str, pd.DataFrame], train_end_date: str,
                        config: dict) -> Dict:
        """用回测起点之前的数据训练，避免未来信息泄露"""
        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        predict_days = data_cfg.get('predict_days', 5)
        target_profit = data_cfg.get('target_profit', 5.0)
        max_features = model_cfg.get('max_features', 50)
        train_ratio = model_cfg.get('train_ratio', 0.8)

        cutoff = pd.Timestamp(train_end_date)
        frames = []
        for code, df in ml_data.items():
            if df is None or len(df) < 30:
                continue
            sub = df[df.index < cutoff].copy()
            if len(sub) < 25:
                continue
            sub = self.fe.create_target(sub, predict_days, target_profit)
            sub['stock_code'] = code
            frames.append(sub)

        if not frames:
            raise ValueError('训练数据不足')

        combined = pd.concat(frames, ignore_index=False)
        X, self.feature_names = self.fe.select_features(combined, max_features)
        y = combined['target']

        valid = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[valid], y[valid]
        if len(X) < 200:
            raise ValueError(f'有效样本过少: {len(X)}')

        split = int(len(X) * train_ratio)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        self.predictor = StockPredictor(model_type=model_cfg.get('type', 'xgboost'))
        metrics = self.predictor.train(X_train, y_train, X_val, y_val)
        self.feature_names = self.predictor.feature_names

        pos_rate = y.mean() * 100
        return {
            'samples': len(X),
            'positive_rate': round(pos_rate, 1),
            'features': len(self.feature_names),
            **{k: round(v, 4) for k, v in metrics.items()},
        }

    def save(self, path: str = DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.predictor.save_model(path)

    def load(self, path: str = DEFAULT_MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        self.predictor = StockPredictor()
        self.predictor.load_model(path)
        self.feature_names = self.predictor.feature_names
        return True

    def _compute_score(self, df: pd.DataFrame, date) -> float:
        if not self.ready or df is None or len(df) < 20:
            return 0.0

        dt = pd.Timestamp(date)
        if dt not in df.index:
            return 0.0

        loc = df.index.get_loc(dt)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        if loc < 15:
            return 0.0

        row_df = df.iloc[[loc]]
        missing = [c for c in self.feature_names if c not in row_df.columns]
        if missing:
            return 0.0

        X = row_df[self.feature_names].fillna(0)
        try:
            return float(self.predictor.predict_proba(X)[0])
        except Exception:
            return 0.0

    def score_at(self, df: pd.DataFrame, date, code: str = None) -> float:
        """给定特征 DataFrame 和日期，返回上涨概率"""
        key = self._cache_key(df, date, code)
        if key in self._cache:
            return self._cache[key]

        proba = self._compute_score(df, date)
        self._cache[key] = proba
        return proba

    def score_stock_code(self, code: str, days: int = 90) -> float:
        """实盘用: 拉取历史并评分"""
        from src.history_fetcher import get_history_fetcher
        raw = get_history_fetcher().get_history(code, days=days + 30)
        if raw is None or len(raw) < 25:
            return 0.0
        feat = self.fe.calculate_technical_indicators(raw.copy())
        return self.score_at(feat, feat.index[-1], code=code)

    def precompute(
        self,
        ml_data: Dict[str, pd.DataFrame],
        dates: List[str],
        persist: bool = True,
    ):
        """预计算评分；persist=True 时读写磁盘缓存，仅补算缺失条目。"""
        import sys
        ts_dates = [pd.Timestamp(d) for d in dates]
        date_strs = [d.strftime('%Y-%m-%d') for d in ts_dates]
        total = len(ml_data)
        model_ver = self._score_store.model_version()
        from_disk = 0
        computed = 0

        if persist:
            print(f'   ML磁盘缓存: model_v{model_ver}', flush=True)

        for i, (code, df) in enumerate(ml_data.items(), 1):
            disk_scores = self._score_store.load_stock(code, model_ver) if persist else {}
            changed = False

            for td, ds in zip(ts_dates, date_strs):
                if td not in df.index:
                    continue
                key = MLScoreStore.stable_key(code, td)
                if ds in disk_scores:
                    self._cache[key] = disk_scores[ds]
                    from_disk += 1
                    continue
                proba = self._compute_score(df, td)
                disk_scores[ds] = proba
                self._cache[key] = proba
                computed += 1
                changed = True

            if persist and changed:
                self._score_store.save_stock(code, disk_scores, model_ver)

            if i % 50 == 0 or i == total:
                print(f'   ML预计算: {i}/{total} 股 | 磁盘{from_disk} 新算{computed} '
                      f'| 内存 {len(self._cache)} 条', flush=True)
                sys.stdout.flush()

        if persist:
            self._score_store.update_meta({
                'stock_count': total,
                'date_count': len(date_strs),
                'from_disk': from_disk,
                'computed': computed,
            })

    def clear_cache(self):
        self._cache.clear()


def build_ml_features(pool: List[str] = None, days: int = 280,
                      raw_data: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    """批量构建 ML 特征数据（可复用已拉取的 raw_data）"""
    from src.history_fetcher import get_history_fetcher

    fe = FeatureEngineer()
    if raw_data is None:
        raw_data = get_history_fetcher().fetch_batch(pool or [], days=days)
    ml_data = {}
    for code, df in raw_data.items():
        try:
            feat = fe.calculate_technical_indicators(df.copy())
            feat['pct_change'] = feat['close'].pct_change() * 100
            ml_data[code] = feat
        except Exception:
            pass
    return ml_data


def get_ml_scorer(config: dict = None, auto_train: bool = False,
                  ml_data: Dict = None, train_end_date: str = None) -> Optional[MLScorer]:
    """加载或训练 ML 评分器"""
    scorer = MLScorer()
    if scorer.load():
        return scorer

    if auto_train and ml_data and train_end_date and config:
        try:
            info = scorer.train_from_pool(ml_data, train_end_date, config)
            scorer.save()
            print(f"  ML训练完成: {info['samples']}样本 正样本率{info['positive_rate']}% "
                  f"验证准确率{info.get('val_accuracy', 0):.2%}")
            return scorer
        except Exception as e:
            print(f"  ML训练失败: {e}")
    return None
