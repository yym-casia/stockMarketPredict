# 股票预测系统 - 主入口

import sys
import os

# 设置控制台编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.backtest_runner import run_backtest, load_config
from src.data_fetcher import DataFetcher
from src.features import FeatureEngineer
from src.model import StockPredictor
from src.strategy import RecommendationEngine, StockSelector, StopLossProfitCalculator
import yaml
import json


def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    print("=" * 60)
    print("    Stock Market Predictor - 股票预测系统")
    print("=" * 60)
    print()
    print("目标: 每日推荐5只股票，预测未来5天涨幅>=5%，成功率>=90%")
    print()
    
    # 运行回测
    results = run_backtest(config, use_real_data=False)
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(output_dir, exist_ok=True)
    
    import pickle
    from datetime import datetime
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存结果
    result_path = os.path.join(output_dir, f'backtest_{timestamp}.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n✅ 结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
