# 模型训练模块

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb


class StockPredictor:
    """股票预测模型"""
    
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.best_params = {}
    
    def build_model(self, params: Optional[Dict] = None):
        """构建模型"""
        if params is None:
            params = self._get_default_params()
        
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(**params)
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(**params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return self.model
    
    def _get_default_params(self) -> Dict:
        """获取默认参数"""
        if self.model_type == "xgboost":
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        elif self.model_type == "lightgbm":
            return {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary',
                'metric': 'auc',
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': -1
            }
        return {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              early_stopping_rounds=50) -> Dict:
        """训练模型"""
        if self.model is None:
            self.build_model()
        
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # 保存特征名
        self.feature_names = list(X_train.columns)
        
        # 训练结果
        train_pred = self.model.predict(X_train)
        train_proba = self.model.predict_proba(X_train)[:, 1]
        
        results = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred, zero_division=0),
            'train_recall': recall_score(y_train, train_pred, zero_division=0),
            'train_f1': f1_score(y_train, train_pred, zero_division=0)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            results.update({
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred, zero_division=0),
                'val_recall': recall_score(y_val, val_pred, zero_division=0),
                'val_f1': f1_score(y_val, val_pred, zero_division=0)
            })
        
        return results
    
    def predict(self, X) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def predict_proba(self, X) -> np.ndarray:
        """预测概率"""
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test) -> Dict:
        """评估模型"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """获取特征重要性"""
        if self.model is None:
            return pd.DataFrame()
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)
    
    def save_model(self, path: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'best_params': self.best_params
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型已保存到 {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data.get('best_params', {})
        print(f"模型已从 {path} 加载")


class Backtester:
    """回测模块"""
    
    def __init__(self, initial_capital: float = 100000, 
                 commission: float = 0.15,
                 stamp_tax: float = 0.1):
        self.initial_capital = initial_capital
        self.commission = commission / 100  # 转为小数
        self.stamp_tax = stamp_tax / 100
    
    def run_backtest(self, predictions: pd.DataFrame, 
                     take_profit: float = 8.0,
                     stop_loss: float = 3.0) -> Dict:
        """运行回测
        
        Args:
            predictions: 包含股票代码、日期、预测概率、实际结果的DataFrame
            take_profit: 止盈比例(%)
            stop_loss: 止损比例(%)
        
        Returns:
            回测结果
        """
        results = []
        capital = self.initial_capital
        position_size = capital * 0.2  # 每只股票仓位20%
        
        for _, row in predictions.iterrows():
            stock_code = row.get('stock_code', 'unknown')
            buy_price = row.get('buy_price', 0)
            predict_proba = row.get('predict_proba', 0)
            
            # 模拟未来5天表现
            future_high = row.get('future_high', buy_price)
            future_low = row.get('future_low', buy_price)
            future_close = row.get('future_close', buy_price)
            
            # 计算止盈止损
            tp_price = buy_price * (1 + take_profit / 100)
            sl_price = buy_price * (1 - stop_loss / 100)
            
            # 判断是否触发止盈或止损
            if future_high >= tp_price:
                # 触发止盈
                sell_price = tp_price
                outcome = 'take_profit'
                return_pct = take_profit
            elif future_low <= sl_price:
                # 触发止损
                sell_price = sl_price
                outcome = 'stop_loss'
                return_pct = -stop_loss
            else:
                # 持有到期
                sell_price = future_close
                outcome = 'hold'
                return_pct = (sell_price - buy_price) / buy_price * 100
            
            # 计算实际收益（扣除手续费）
            buy_cost = position_size * (1 + self.commission)
            sell_revenue = position_size * (1 + return_pct / 100) * (1 - self.commission - self.stamp_tax)
            profit = sell_revenue - buy_cost
            
            results.append({
                'stock_code': stock_code,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'predict_proba': predict_proba,
                'outcome': outcome,
                'return_pct': return_pct,
                'profit': profit
            })
        
        results_df = pd.DataFrame(results)
        
        # 计算统计指标
        total_trades = len(results_df)
        win_trades = len(results_df[results_df['return_pct'] > 0])
        loss_trades = len(results_df[results_df['return_pct'] <= 0])
        
        summary = {
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_trades / total_trades * 100 if total_trades > 0 else 0,
            'avg_return': results_df['return_pct'].mean(),
            'total_profit': results_df['profit'].sum(),
            'avg_win': results_df[results_df['return_pct'] > 0]['return_pct'].mean() if win_trades > 0 else 0,
            'avg_loss': results_df[results_df['return_pct'] <= 0]['return_pct'].mean() if loss_trades > 0 else 0,
            'take_profit_rate': len(results_df[results_df['outcome'] == 'take_profit']) / total_trades * 100,
            'stop_loss_rate': len(results_df[results_df['outcome'] == 'stop_loss']) / total_trades * 100,
            'results': results_df
        }
        
        return summary


if __name__ == "__main__":
    # 测试模型训练
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    y = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    predictor = StockPredictor(model_type="xgboost")
    train_results = predictor.train(X_train, y_train, X_test, y_test)
    print(f"训练结果: {train_results}")
    
    # 评估模型
    eval_results = predictor.evaluate(X_test, y_test)
    print(f"评估结果: accuracy={eval_results['accuracy']:.4f}, f1={eval_results['f1']:.4f}")
    
    # 特征重要性
    importance = predictor.get_feature_importance()
    print(f"特征重要性:\n{importance}")
