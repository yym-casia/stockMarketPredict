# -*- coding: utf-8 -*-
"""分析50只翻倍股共性，生成 doubler_patterns.json"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_pool_manager import get_pool_manager
from src.history_fetcher import get_history_fetcher
from src.doubler_analyzer import analyze_pool


def main():
    pool = get_pool_manager().get_all_stocks()[:500]
    print(f'📥 加载 {len(pool)} 只股票历史数据...')
    raw = get_history_fetcher().fetch_batch(pool, days=300)
    print(f'   成功 {len(raw)} 只')

    print('🔍 挖掘翻倍样本并提炼特征...')
    result = analyze_pool(raw, n_samples=50)
    th = result['analysis']['derived_thresholds']
    patterns = result['analysis'].get('common_patterns', [])

    print(f'\n✅ 样本数: {result["sample_count"]}')
    print('📌 共性特征:')
    for p in patterns:
        print(f'   · {p}')
    print('\n📐 推导阈值:')
    for k, v in th.items():
        print(f'   {k}: {v}')
    print(f'\n💾 已保存: data/doubler_patterns.json')


if __name__ == '__main__':
    main()
