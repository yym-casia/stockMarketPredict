# -*- coding: utf-8 -*-
"""训练并保存 ML 选股模型"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import yaml
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_pool_manager import get_screening_pool
from src.ml_scorer import MLScorer, build_ml_features, DEFAULT_MODEL_PATH
from run_backtest_capital import load_config, get_trading_days


def main():
    print('=' * 60)
    print('🤖 训练 ML 选股模型')
    print('=' * 60)

    config = load_config()
    pool = get_screening_pool(config)
    trading_days = get_trading_days(200)
    train_end = trading_days[0] if trading_days else datetime.now().strftime('%Y-%m-%d')

    print(f'\n📥 加载 {len(pool)} 只股票...')
    ml_data = build_ml_features(pool, days=300)
    print(f'   成功: {len(ml_data)} 只')

    scorer = MLScorer()
    info = scorer.train_from_pool(ml_data, train_end, config)
    scorer.save()

    print(f'\n✅ 模型已保存: {DEFAULT_MODEL_PATH}')
    print(f'   样本: {info["samples"]} | 正样本率: {info["positive_rate"]}%')
    print(f'   训练准确率: {info.get("train_accuracy", 0):.2%}')
    print(f'   验证准确率: {info.get("val_accuracy", 0):.2%}')
    print(f'   特征数: {info["features"]}')


if __name__ == '__main__':
    main()
