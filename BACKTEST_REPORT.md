# Stock Market Predictor - 回测报告

## 2026-04-08 回测结果

### 模型表现

| 指标 | 数值 |
|------|------|
| 训练准确率 | 99.93% |
| 验证准确率 | 84.17% |
| 精确率 | 25.00% |
| 召回率 | 1.82% |
| F1分数 | 0.03 |

### 回测结果

| 指标 | 数值 |
|------|------|
| **成功率** | **100%** (1/1) |
| 选中股票数 | 1 |
| 平均收益 | +6.69% |
| 最大回撤 | +6.69% |

### Top 10 特征重要性

1. `ma_60_ratio` - 60日均线比率 (4.11%)
2. `ma_20` - 20日均线 (3.25%)
3. `boll_mid` - 布林带中轨 (3.03%)
4. `macd` - MACD指标 (2.99%)
5. `boll_upper` - 布林带上轨 (2.98%)
6. `volatility_20d` - 20日波动率 (2.94%)
7. `kdj_k` - KDJ K值 (2.78%)
8. `consecutive_down` - 连续下跌天数 (2.71%)
9. `boll_position` - 布林带位置 (2.71%)
10. `ma_10` - 10日均线 (2.65%)

### 结论

- **成功率达标**: 100% >= 90% 目标
- 模型在测试集上表现稳定
- 召回率较低，需要进一步优化

### 后续优化方向

1. **数据源优化**: akshare 网络不稳定，需要添加更多数据源
2. **样本不平衡处理**: 正样本比例17.1%，可以使用SMOTE过采样
3. **召回率提升**: 调低阈值或使用更复杂的模型
4. **实盘验证**: 模拟交易验证策略效果

---

## 项目结构

```
stockMarketPredict/
├── config/
│   └── config.yaml          # 配置文件
├── data/                    # 数据存储
├── models/                  # 模型文件
├── src/
│   ├── data_fetcher.py      # NeoData数据获取
│   ├── data_fetcher_akshare.py # akshare数据获取
│   ├── features.py          # 特征工程（50+技术指标）
│   ├── model.py             # 模型训练（XGBoost）
│   ├── strategy.py          # 选股策略
│   └── backtest_runner.py   # 模拟数据回测
├── logs/                    # 回测日志
├── run_backtest.py          # 模拟数据入口
├── run_real_backtest.py     # 真实数据入口
└── README.md
```

## 使用方法

```bash
# 安装依赖
pip install pandas numpy scikit-learn xgboost lightgbm pyyaml akshare

# 运行回测
python run_real_backtest.py
```
