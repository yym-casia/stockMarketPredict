# -*- coding: utf-8 -*-
"""测试历史数据获取器"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from src.history_fetcher import get_history_fetcher

fetcher = get_history_fetcher()
codes = ['600036', '000001', '601919']

print('=' * 50)
print('历史数据获取测试（腾讯→新浪→akshare）')
print('=' * 50)

for code in codes:
    df = fetcher.get_history(code, days=120, use_cache=False)
    if df is not None:
        print(f'✅ {code}: {len(df)}天 ({df.index[0].date()} ~ {df.index[-1].date()}) 收盘{df["close"].iloc[-1]:.2f}')
    else:
        print(f'❌ {code}: 获取失败')

print('\n批量测试 10 只...')
batch = fetcher.fetch_batch(codes * 3 + ['600519', '000858'], days=90)
print(f'批量结果: {len(batch)}/10 成功')
