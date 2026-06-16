# -*- coding: utf-8 -*-
"""
测试阶段1优化：技术指标筛选
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 添加项目路径
sys.path.insert(0, r'D:\yym\code\stockMarketPredict')

from src.data_fetcher_multi import MultiSourceDataFetcher
from stock_pool_manager import get_pool_manager, get_stock_sector


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算技术指标"""
    df = df.copy()
    
    # 计算RSI (14日)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 成交量均线
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # 价格均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    
    return df


def get_stock_history(stock_code: str, days: int = 30):
    """获取股票历史数据"""
    try:
        import akshare as ak
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 20)
        
        df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                               start_date=start_date.strftime('%Y%m%d'),
                               end_date=end_date.strftime('%Y%m%d'),
                               adjust="qfq")
        
        if df is not None and not df.empty and len(df) >= 20:
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume',
                         'amount', 'amplitude', 'pct_change', 'change', 'turnover']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = calculate_technical_indicators(df)
            return df
    except Exception as e:
        print(f"  获取 {stock_code} 失败: {e}")
    
    return None


def check_technical_signals(df) -> dict:
    """检查技术指标信号"""
    if df is None or len(df) < 20:
        return {'valid': False, 'reason': '数据不足'}
    
    latest = df.iloc[-1]
    
    # RSI检查
    rsi = latest['rsi']
    if pd.isna(rsi):
        return {'valid': False, 'reason': 'RSI计算失败'}
    
    rsi_ok = 30 <= rsi <= 60
    
    # MACD检查
    macd_hist = latest['macd_hist']
    if pd.isna(macd_hist):
        return {'valid': False, 'reason': 'MACD计算失败'}
    
    macd_ok = macd_hist > 0
    
    # 成交量检查
    volume_ratio = latest['volume_ratio']
    volume_ok = volume_ratio > 0.8 if not pd.isna(volume_ratio) else True
    
    # 均线检查
    ma5 = latest['ma5']
    ma10 = latest['ma10']
    ma_ok = ma5 >= ma10 if not pd.isna(ma5) and not pd.isna(ma10) else True
    
    # 综合评分
    score = 0
    if rsi_ok: score += 25
    if macd_ok: score += 25
    if volume_ok: score += 25
    if ma_ok: score += 25
    
    return {
        'valid': score >= 50,
        'score': score,
        'rsi': rsi,
        'macd_ok': macd_ok,
        'volume_ratio': volume_ratio if not pd.isna(volume_ratio) else 1.0,
        'ma_ok': ma_ok,
        'reason': f"RSI:{rsi:.1f} MACD:{'金叉' if macd_ok else '非金叉'}"
    }


def test_with_sample_stocks():
    """测试几只样本股票"""
    print("=" * 60)
    print("阶段1优化测试：技术指标筛选")
    print("=" * 60)
    
    # 测试股票列表（包含不同特征的股票）
    test_stocks = [
        '000001',  # 平安银行
        '000002',  # 万科A
        '600519',  # 贵州茅台
        '000725',  # 京东方A
        '002594',  # 比亚迪
    ]
    
    print(f"\n测试 {len(test_stocks)} 只股票...\n")
    
    passed = 0
    failed = 0
    
    for code in test_stocks:
        print(f"分析 {code}...", end=' ')
        
        df = get_stock_history(code, days=30)
        signals = check_technical_signals(df)
        
        if signals['valid']:
            print(f"✅ 通过 (评分:{signals['score']}/100, {signals['reason']})")
            passed += 1
        else:
            print(f"❌ 跳过 ({signals['reason']})")
            failed += 1
        
        time.sleep(0.5)  # 避免请求过快
    
    print(f"\n{'='*60}")
    print(f"测试结果: {passed} 只通过, {failed} 只未通过")
    print(f"通过率: {passed/(passed+failed)*100:.1f}%")
    
    if passed > 0:
        print("\n✅ 技术指标筛选功能正常！")
    else:
        print("\n⚠️ 所有股票均未通过筛选，可能需要调整参数")


if __name__ == "__main__":
    test_with_sample_stocks()
