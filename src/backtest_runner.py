# 回测运行器 - 使用60天历史数据测试策略

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_fetcher import DataFetcher
from src.features import FeatureEngineer
from src.model import StockPredictor, Backtester
from src.strategy import StockSelector, StopLossProfitCalculator, RecommendationEngine
import yaml
import json


def load_config():
    """加载配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_mock_data(stock_code: str, days: int = 60, seed: int = None):
    """生成模拟数据（用于测试）
    
    实际使用时应该从NeoData获取真实数据
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成价格数据
    base_price = np.random.uniform(10, 100)
    returns = np.random.randn(days) * 0.02  # 日收益率
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 生成OHLCV数据
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.02, 0.02, days)),
        'high': prices * (1 + np.abs(np.random.randn(days)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(days)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days).astype(float)
    }, index=dates)
    
    # 确保high >= open, close 且 low <= open, close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    return df


def run_backtest(config: dict, use_real_data: bool = False):
    """运行回测"""
    print("=" * 60)
    print("🚀 股票预测系统 - 60天历史回测")
    print("=" * 60)
    
    # 1. 准备数据
    print("\n📊 步骤1: 准备数据...")
    
    if use_real_data:
        # 使用真实数据
        fetcher = DataFetcher(source="neodata")
        stock_list = fetcher.get_stock_list()
        market_data = fetcher.fetch_history_data(stock_list, config['data']['history_days'])
    else:
        # 使用模拟数据
        print("⚠️ 使用模拟数据进行测试（NeoData接口暂不支持批量历史数据）")
        stock_list = [f"00000{i}.SZ" for i in range(1, 21)] + [f"60000{i}.SH" for i in range(1, 11)]
        market_data = {}
        for i, code in enumerate(stock_list):
            market_data[code] = generate_mock_data(code, days=60, seed=i*42)
    
    print(f"✅ 已准备 {len(market_data)} 只股票的数据")
    
    # 2. 特征工程
    print("\n🔧 步骤2: 特征工程...")
    fe = FeatureEngineer()
    
    processed_data = {}
    for code, df in market_data.items():
        try:
            df = fe.calculate_technical_indicators(df)
            df = fe.create_target(
                df, 
                predict_days=config['data']['predict_days'],
                target_profit=config['data']['target_profit']
            )
            processed_data[code] = df
        except Exception as e:
            print(f"  处理 {code} 时出错: {e}")
            continue
    
    print(f"✅ 特征工程完成，每只股票 {len(fe.feature_names)} 个特征")
    
    # 3. 合并数据训练模型
    print("\n🤖 步骤3: 训练模型...")
    
    all_data = []
    for code, df in processed_data.items():
        df_copy = df.copy()
        df_copy['stock_code'] = code
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 准备训练数据
    X, feature_names = fe.select_features(combined_df)
    y = combined_df['target']
    
    # 移除NaN
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    # 时间序列分割
    split_idx = int(len(X) * config['model']['train_ratio'])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 训练模型
    predictor = StockPredictor(model_type=config['model']['type'])
    train_results = predictor.train(X_train, y_train, X_test, y_test)
    
    print(f"✅ 模型训练完成")
    print(f"   - 训练准确率: {train_results['train_accuracy']:.4f}")
    print(f"   - 验证准确率: {train_results.get('val_accuracy', 0):.4f}")
    print(f"   - 验证F1: {train_results.get('val_f1', 0):.4f}")
    
    # 4. 模型评估
    print("\n📈 步骤4: 模型评估...")
    eval_results = predictor.evaluate(X_test, y_test)
    
    print(f"✅ 模型评估完成")
    print(f"   - 准确率: {eval_results['accuracy']:.4f}")
    print(f"   - 精确率: {eval_results['precision']:.4f}")
    print(f"   - 召回率: {eval_results['recall']:.4f}")
    print(f"   - F1分数: {eval_results['f1']:.4f}")
    
    # 特征重要性
    importance = predictor.get_feature_importance(top_n=10)
    print(f"\n   Top 10 特征:")
    for _, row in importance.iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    # 5. 回测
    print("\n💰 步骤5: 策略回测...")
    
    # 在测试集上模拟交易
    test_predictions = pd.DataFrame({
        'stock_code': combined_df.loc[X_test.index, 'stock_code'].values,
        'predict_proba': predictor.predict_proba(X_test),
        'predict_label': predictor.predict(X_test),
        'actual_label': y_test.values,
        'buy_price': combined_df.loc[X_test.index, 'close'].values,
        'future_high': combined_df.loc[X_test.index, 'future_high'].values,
        'future_low': combined_df.loc[X_test.index, 'future_low'].values,
        'future_close': combined_df.loc[X_test.index, 'close'].shift(-5).values
    })
    
    # 只选择预测为正的股票
    selected = test_predictions[test_predictions['predict_label'] == 1].copy()
    
    # 计算成功率
    if len(selected) > 0:
        success_rate = selected['actual_label'].mean() * 100
        print(f"✅ 预测成功率: {success_rate:.2f}% ({selected['actual_label'].sum()}/{len(selected)})")
    else:
        success_rate = 0
        print("⚠️ 没有选中任何股票")
    
    # 运行回测
    backtester = Backtester(
        initial_capital=config['backtest']['initial_capital'],
        commission=config['backtest']['commission'],
        stamp_tax=config['backtest']['stamp_tax']
    )
    
    backtest_results = backtester.run_backtest(
        selected,
        take_profit=config['data']['take_profit'],
        stop_loss=config['data']['stop_loss']
    )
    
    print(f"\n📊 回测结果:")
    print(f"   - 总交易次数: {backtest_results['total_trades']}")
    print(f"   - 盈利次数: {backtest_results['win_trades']}")
    print(f"   - 亏损次数: {backtest_results['loss_trades']}")
    print(f"   - 胜率: {backtest_results['win_rate']:.2f}%")
    print(f"   - 平均收益: {backtest_results['avg_return']:.2f}%")
    print(f"   - 止盈比例: {backtest_results['take_profit_rate']:.2f}%")
    print(f"   - 止损比例: {backtest_results['stop_loss_rate']:.2f}%")
    
    # 6. 生成今日推荐
    print("\n⭐ 步骤6: 生成今日推荐...")
    
    selector = StockSelector(config['stock_selection'])
    sl_calc = StopLossProfitCalculator(
        take_profit_pct=config['data']['take_profit'],
        stop_loss_pct=config['data']['stop_loss']
    )
    
    # 使用最新数据生成推荐
    recommendations = []
    for code, df in processed_data.items():
        try:
            latest = df.iloc[-1:].copy()
            X_latest = latest[feature_names]
            proba = predictor.predict_proba(X_latest)[0]
            pred = predictor.predict(X_latest)[0]
            
            if pred == 1:
                sl_tp = sl_calc.calculate(latest['close'].values[0])
                recommendations.append({
                    'code': code,
                    'confidence': proba,
                    'buy_price': latest['close'].values[0],
                    'take_profit': sl_tp['take_profit'],
                    'stop_loss': sl_tp['stop_loss'],
                    'take_profit_pct': sl_tp['take_profit_pct'],
                    'stop_loss_pct': sl_tp['stop_loss_pct']
                })
        except Exception as e:
            continue
    
    # 按置信度排序，选择前5
    recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)[:5]
    
    print(f"\n📈 今日推荐股票 (Top 5):")
    print("-" * 60)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['code']}")
        print(f"   买入价: {rec['buy_price']:.2f}  置信度: {rec['confidence']*100:.1f}%")
        print(f"   止盈: {rec['take_profit']:.2f} (+{rec['take_profit_pct']}%)")
        print(f"   止损: {rec['stop_loss']:.2f} (-{rec['stop_loss_pct']}%)")
        print()
    
    # 7. 总结
    print("=" * 60)
    print("📋 回测总结")
    print("=" * 60)
    
    target_success_rate = config['targets']['success_rate']
    min_success_rate = config['targets']['min_success_rate']
    
    print(f"\n目标成功率: {target_success_rate}%")
    print(f"最低成功率: {min_success_rate}%")
    print(f"实际成功率: {success_rate:.2f}%")
    
    if success_rate >= target_success_rate:
        print("✅ 已达成目标!")
    elif success_rate >= min_success_rate:
        print("⚠️ 未达成目标，但达到最低要求")
    else:
        print("❌ 未达到最低要求，需要优化策略")
    
    print(f"\n💡 优化建议:")
    if success_rate < min_success_rate:
        print("   1. 增加训练数据量")
        print("   2. 优化特征工程")
        print("   3. 调整模型参数")
        print("   4. 增加止盈止损灵活性")
    
    return {
        'success_rate': success_rate,
        'backtest_results': backtest_results,
        'recommendations': recommendations,
        'feature_importance': importance.to_dict('records')
    }


if __name__ == "__main__":
    config = load_config()
    results = run_backtest(config, use_real_data=False)
