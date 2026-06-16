# 数据获取模块 - 支持 akshare 真实数据

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import time
warnings.filterwarnings('ignore')


class AkshareDataFetcher:
    """基于 akshare 的A股数据获取"""
    
    def __init__(self):
        self.name = "akshare"
    
    def get_stock_list(self, market: str = "A股") -> List[str]:
        """获取股票列表"""
        try:
            import akshare as ak
            
            if market == "沪深300":
                df = ak.index_stock_cons_weight_csindex(symbol="000300")
                return df['股票代码'].tolist()[:50]
            elif market == "中证500":
                df = ak.index_stock_cons_weight_csindex(symbol="000905")
                return df['股票代码'].tolist()[:50]
            else:
                df = ak.stock_zh_a_spot_em()
                df = df[~df['名称'].str.contains('ST|退|N')]
                df = df.sort_values('成交额', ascending=False)
                return df['代码'].tolist()[:100]
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            return []
    
    def get_stock_history(self, stock_code: str, days: int = 60, retries: int = 3) -> pd.DataFrame:
        """获取单只股票历史数据（带重试）"""
        import akshare as ak
        
        for attempt in range(retries):
            try:
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
                
                code = stock_code.replace('.SH', '').replace('.SZ', '')
                
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"
                )
                
                if df is None or len(df) == 0:
                    time.sleep(0.5)
                    continue
                
                df = df.rename(columns={
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '日期': 'date'
                })
                
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.tail(days)
                
                return df
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    return None
        
        return None
    
    def fetch_history_data(self, stock_codes: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """批量获取多只股票历史数据"""
        result = {}
        total = len(stock_codes)
        
        print(f"正在获取 {total} 只股票的历史数据...")
        
        for i, code in enumerate(stock_codes):
            if (i + 1) % 10 == 0:
                print(f"  进度: {i+1}/{total}")
            
            df = self.get_stock_history(code, days)
            if df is not None and len(df) >= days * 0.8:
                result[code] = df
            
            # 控制请求频率
            time.sleep(0.3)
        
        print(f"成功获取 {len(result)} 只股票数据")
        return result


def generate_realistic_mock_data(stock_code: str, days: int = 60, seed: int = None):
    """生成真实的模拟数据（基于A股特征）"""
    if seed is not None:
        np.random.seed(seed)
    
    # A股特征：日波动约2-3%，涨跌停限制10%
    base_price = np.random.uniform(5, 50)
    daily_volatility = np.random.uniform(0.015, 0.03)
    
    # 生成带有趋势的价格
    trend = np.random.choice([-1, 0, 1]) * np.random.uniform(0, 0.001)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    prices = [base_price]
    
    for _ in range(days - 1):
        ret = np.random.randn() * daily_volatility + trend
        ret = np.clip(ret, -0.10, 0.10)  # 涨跌停限制
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    # 生成OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.015, 0.015, days)),
        'high': prices * (1 + np.abs(np.random.randn(days)) * 0.015),
        'low': prices * (1 - np.abs(np.random.randn(days)) * 0.015),
        'close': prices,
        'volume': np.random.randint(1000000, 50000000, days).astype(float)
    }, index=dates)
    
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    return df


class DataFetcher:
    """统一数据获取接口"""
    
    def __init__(self, source: str = "akshare"):
        self.source = source
        if source == "akshare":
            self.fetcher = AkshareDataFetcher()
        else:
            self.fetcher = None
    
    def get_stock_list(self, **kwargs) -> List[str]:
        if self.fetcher:
            result = self.fetcher.get_stock_list(**kwargs)
            if result:
                return result
        
        # 备用列表
        return ['000001', '000002', '600000', '600036', '601318',
                '000651', '000333', '600519', '601166', '000858',
                '002415', '000725', '600276', '601012', '002304',
                '000063', '002236', '600887', '601888', '002475',
                '300750', '600900', '601688', '000568', '600309',
                '002352', '300059', '600809', '000538', '002007']
    
    def get_stock_history(self, stock_code: str, days: int = 60) -> pd.DataFrame:
        if self.fetcher:
            result = self.fetcher.get_stock_history(stock_code, days)
            if result is not None:
                return result
        
        # 使用模拟数据
        return generate_realistic_mock_data(stock_code, days, seed=hash(stock_code) % 10000)
    
    def fetch_history_data(self, stock_codes: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        result = {}
        
        if self.fetcher:
            result = self.fetcher.fetch_history_data(stock_codes, days)
        
        # 如果真实数据不足，用模拟数据补充
        if len(result) < len(stock_codes) * 0.5:
            print("真实数据获取不足，使用模拟数据补充...")
            for i, code in enumerate(stock_codes):
                if code not in result:
                    df = generate_realistic_mock_data(code, days, seed=i*42)
                    result[code] = df
        
        return result
