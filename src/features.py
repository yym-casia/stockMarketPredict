# 特征工程模块

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self):
        self.feature_names = []
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        # 假设df包含以下列: open, high, low, close, volume
        # 如果列名不同，需要先重命名
        
        # ===== 价格特征 =====
        # 收益率
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        
        # 波动率
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # ===== 移动平均线 =====
        for period in [5, 10, 20, 60]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'ma_{period}_ratio'] = df['close'] / df[f'ma_{period}']
        
        # 均线多头排列
        df['ma_bullish'] = (
            (df['ma_5'] > df['ma_10']) & 
            (df['ma_10'] > df['ma_20'])
        ).astype(int)
        
        # ===== MACD =====
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = 2 * (df['macd'] - df['macd_signal'])
        df['macd_cross'] = ((df['macd'] > df['macd_signal']) & 
                           (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        # ===== RSI =====
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        
        # ===== KDJ =====
        low_min = df['low'].rolling(9).min()
        high_max = df['high'].rolling(9).max()
        df['kdj_rsv'] = (df['close'] - low_min) / (high_max - low_min + 1e-8) * 100
        df['kdj_k'] = df['kdj_rsv'].ewm(alpha=1/3, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(alpha=1/3, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        
        # ===== 布林带 =====
        df['boll_mid'] = df['close'].rolling(20).mean()
        df['boll_std'] = df['close'].rolling(20).std()
        df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
        df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
        df['boll_position'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-8)
        
        # ===== 成交量特征 =====
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_5']
        df['volume_price_trend'] = df['volume'] * df['return_1d']
        
        # 放量特征
        df['volume_breakout'] = (df['volume'] > 2 * df['volume_ma_10']).astype(int)
        
        # ===== 形态特征 =====
        # 阳线/阴线
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        
        # 连续上涨/下跌
        df['consecutive_up'] = self._count_consecutive(df['is_bullish'])
        df['consecutive_down'] = self._count_consecutive(1 - df['is_bullish'])
        
        # 突破特征
        df['break_ma20'] = ((df['close'] > df['ma_20']) & 
                           (df['close'].shift(1) <= df['ma_20'].shift(1))).astype(int)
        df['break_ma60'] = ((df['close'] > df['ma_60']) & 
                           (df['close'].shift(1) <= df['ma_60'].shift(1))).astype(int)
        
        # 新高/新低
        df['is_new_high_20'] = (df['close'] == df['close'].rolling(20).max()).astype(int)
        df['is_new_low_20'] = (df['close'] == df['close'].rolling(20).min()).astype(int)
        
        # ===== 涨跌幅特征 =====
        df['amplitude'] = (df['high'] - df['low']) / df['open']  # 振幅
        df['change_pct'] = (df['close'] - df['open']) / df['open']  # 涨跌幅
        
        # ===== 动量特征 =====
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # 填充NaN
        df = df.bfill().ffill()
        
        return df
    
    def _count_consecutive(self, series: pd.Series) -> pd.Series:
        """计算连续出现次数"""
        groups = (series != series.shift(1)).cumsum()
        return series.groupby(groups).cumsum()
    
    def create_target(self, df: pd.DataFrame, predict_days: int = 5, 
                      target_profit: float = 5.0) -> pd.DataFrame:
        """创建目标变量
        
        Args:
            df: 包含close列的DataFrame
            predict_days: 预测天数
            target_profit: 目标涨幅(%)
        
        Returns:
            添加了target列的DataFrame
        """
        df = df.copy()
        
        # 未来N天的最大涨幅
        future_max = df['close'].shift(-predict_days).rolling(predict_days).max()
        df['future_max_return'] = (future_max - df['close']) / df['close'] * 100
        
        # 目标：未来5天涨幅超过5%
        df['target'] = (df['future_max_return'] >= target_profit).astype(int)
        
        # 未来5天的最低价和最高价（用于计算止盈止损）
        df['future_low'] = df['close'].shift(-predict_days).rolling(predict_days).min()
        df['future_high'] = df['close'].shift(-predict_days).rolling(predict_days).max()
        
        return df
    
    def select_features(self, df: pd.DataFrame, max_features: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """选择特征"""
        # 排除非特征列
        exclude_cols = ['target', 'future_max_return', 'future_low', 'future_high',
                       'open', 'high', 'low', 'close', 'volume']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 选择数值型特征
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # 限制特征数量
        if len(numeric_features) > max_features:
            # 根据方差选择
            variances = df[numeric_features].var()
            selected_features = variances.nlargest(max_features).index.tolist()
        else:
            selected_features = numeric_features
        
        self.feature_names = selected_features
        
        return df[selected_features], selected_features
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple:
        """准备训练数据"""
        from sklearn.model_selection import train_test_split
        
        # 选择特征
        X, feature_names = self.select_features(df)
        y = df[target_col]
        
        # 移除包含NaN的行
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # 时间序列不能打乱
        )
        
        return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    # 测试特征工程
    import numpy as np
    
    # 创建测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(100),
        'high': close + np.abs(np.random.randn(100)),
        'low': close - np.abs(np.random.randn(100)),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # 计算特征
    fe = FeatureEngineer()
    df = fe.calculate_technical_indicators(df)
    df = fe.create_target(df)
    
    print(f"特征数量: {len(df.columns)}")
    print(f"目标分布:\n{df['target'].value_counts()}")
