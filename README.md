# 📈 股票预测系统 (Stock Market Predictor)

## 项目简介

基于机器学习的股票预测系统，目标：
- **每日推荐**: 5只股票
- **预测周期**: 未来5天
- **目标涨幅**: ≥5%
- **成功率目标**: ≥90%

## 项目结构

```
stockMarketPredict/
├── config/                 # 配置文件
│   └── config.yaml        # 主配置
├── data/                   # 数据存储
├── models/                 # 模型文件
├── notebooks/              # Jupyter notebooks
├── src/                    # 源代码
│   ├── __init__.py
│   ├── data_fetcher.py    # 数据获取模块
│   ├── features.py        # 特征工程模块
│   ├── model.py           # 模型训练模块
│   ├── strategy.py        # 选股策略模块
│   └── backtest_runner.py # 回测运行器
├── tests/                  # 单元测试
├── logs/                   # 日志文件
├── requirements.txt        # Python依赖
├── run_backtest.py        # 主入口
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行回测

```bash
python run_backtest.py
```

### 3. 查看推荐

系统会输出：
- 模型训练结果
- 回测统计
- 今日推荐股票（Top 5）
- 止盈止损点位

## 核心模块

### 数据获取 (data_fetcher.py)
- 支持 NeoData 金融数据源
- 获取股票历史行情
- 获取涨停股票列表
- 支持实时行情查询

### 特征工程 (features.py)
- 技术指标计算（MA, MACD, RSI, KDJ, 布林带等）
- 成交量特征
- 形态特征
- 动量特征
- 目标变量生成

### 模型训练 (model.py)
- XGBoost 分类器
- LightGBM 分类器
- 交叉验证
- 特征重要性分析
- 模型持久化

### 选股策略 (strategy.py)
- 股票筛选
- 止盈止损计算
- 推荐引擎
- 风险管理

## 配置说明

编辑 `config/config.yaml`:

```yaml
data:
  history_days: 60        # 历史数据天数
  predict_days: 5         # 预测天数
  target_profit: 5.0      # 目标涨幅(%)
  take_profit: 8.0        # 止盈比例(%)
  stop_loss: 3.0          # 止损比例(%)

model:
  type: "xgboost"         # 模型类型
  train_ratio: 0.8        # 训练集比例

targets:
  success_rate: 90        # 目标成功率(%)
  min_success_rate: 60    # 最低成功率(%)
```

## 回测说明

系统使用60天历史数据进行回测：
1. 计算技术指标
2. 训练预测模型
3. 在测试集上模拟交易
4. 计算成功率和收益率
5. 生成推荐股票

## 注意事项

⚠️ **风险提示**
- 本系统仅供学习和研究使用
- 股市有风险，投资需谨慎
- 预测结果不构成投资建议
- 过去的表现不代表未来收益

## 后续优化

1. **数据源优化**: 接入更多数据源（资金流向、北向资金、机构调研等）
2. **模型优化**: 尝试 LSTM、Transformer 等深度学习模型
3. **策略优化**: 多因子模型、组合优化
4. **实时推送**: 定时任务每日推送推荐

## License

MIT License
