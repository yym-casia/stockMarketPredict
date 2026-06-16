# -*- coding: utf-8 -*-
"""
真实数据回测运行器 - 使用 Baostock + 新浪财经
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data_fetcher_multi import MultiSourceDataFetcher
from src.features import FeatureEngineer
from src.model import StockPredictor
from src.stock_tracker import StockTracker
import yaml
import pickle


def load_config():
    """加载配置文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_backtest_real(config: dict):
    """使用真实数据运行回测"""
    print("=" * 60)
    print("  Stock Market Predictor - 真实数据回测 (Baostock)")
    print("=" * 60)
    
    # 1. 获取数据
    print("\n[步骤1] 获取股票列表和历史数据...")
    
    fetcher = MultiSourceDataFetcher()
    
    # 获取沪深A股热门股票（从东方财富API获取或使用备用列表）
    print("  获取股票列表...")
    stock_list = fetcher.get_stock_list(top_n=5000)  # 获取前5000只热门股票
    
    print(f"股票列表: {len(stock_list)} 只")
    
    # 批量获取历史数据
    days = config['data']['history_days']
    raw_data = fetcher.fetch_history_data(stock_list, days=days)
    
    if not raw_data:
        print("无法获取历史数据，请检查网络连接")
        fetcher.close()
        return
    
    fetcher.close()
    
    # 2. 特征工程
    print("\n[步骤2] 特征工程...")
    engineer = FeatureEngineer()
    
    processed_data = {}
    for code, df in raw_data.items():
        try:
            df_features = engineer.calculate_technical_indicators(df)
            if df_features is not None and len(df_features) >= 30:
                processed_data[code] = df_features
        except Exception as e:
            continue
    
    print(f"特征工程完成: {len(processed_data)} 只股票")
    
    if not processed_data:
        print("没有足够的有效数据进行回测")
        return
    
    # 3. 合并数据并训练模型
    print("\n[步骤3] 合并数据并训练模型...")
    
    all_data = []
    for code, df in processed_data.items():
        df_copy = df.copy()
        df_copy['code'] = code
        all_data.append(df_copy)
    
    if not all_data:
        print("没有有效数据")
        return
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"合并后数据量: {len(combined)} 行")
    
    # 创建目标变量
    combined = engineer.create_target(combined, 
                                      predict_days=config['data']['predict_days'],
                                      target_profit=config['data']['target_profit'])
    
    # 过滤有效数据
    combined = combined.dropna(subset=['target'])
    print(f"有效样本数: {len(combined)}")
    
    # 特征列 - 排除目标变量和未来数据
    exclude_cols = ['code', 'target', 'buy_price', 'future_return', 
                   'future_max_return', 'future_low', 'future_high']
    feature_names = [col for col in combined.columns if col not in exclude_cols]
    
    # 只保留数值型特征
    numeric_cols = combined[feature_names].select_dtypes(include=[np.number]).columns.tolist()
    feature_names = numeric_cols
    
    X = combined[feature_names].values
    y = combined['target'].values
    
    print(f"特征数量: {len(feature_names)}")
    print(f"正样本比例: {y.mean()*100:.1f}% ({int(y.sum())}/{len(y)})")
    
    # 4. 训练模型
    print("\n[步骤4] 训练模型...")
    
    from sklearn.model_selection import train_test_split
    
    # 使用DataFrame格式
    X_df = pd.DataFrame(X, columns=feature_names)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    predictor = StockPredictor(
        model_type=config['model']['type']
    )
    
    predictor.train(X_train, y_train)
    
    # 评估
    train_result = predictor.evaluate(X_train, y_train)
    val_result = predictor.evaluate(X_val, y_val)
    
    print(f"训练完成:")
    print(f"  - 训练准确率: {train_result['accuracy']:.4f}")
    print(f"  - 验证准确率: {val_result['accuracy']:.4f}")
    
    # F1 Score
    from sklearn.metrics import f1_score
    y_pred = predictor.predict(X_val)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"  - 验证F1: {f1:.4f}")
    
    # 5. 模型评估
    print("\n[步骤5] 模型评估...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    print(f"评估结果:")
    print(f"  - 准确率: {val_result['accuracy']:.4f}")
    print(f"  - 精确率: {val_result['precision']:.4f}")
    print(f"  - 召回率: {val_result['recall']:.4f}")
    print(f"  - F1分数: {val_result['f1']:.4f}")
    
    # 特征重要性
    if hasattr(predictor.model, 'feature_importances_'):
        importances = predictor.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nTop 10 特征:")
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"  - {feature_names[idx]}: {importances[idx]:.4f}")
    
    # 6. 策略回测
    print("\n[步骤6] 策略回测...")
    
    # 模拟历史回测
    trades = []
    for code, df in processed_data.items():
        try:
            df_test = df.iloc[-20:].copy()  # 最近20天作为测试期
            
            X_test = df_test[feature_names].values
            
            if np.isnan(X_test).any():
                continue
            
            predictions = predictor.predict(X_test)
            probabilities = predictor.predict_proba(X_test)
            
            # 找买入信号 - 降低阈值到50%
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if pred == 1 and prob[1] >= 0.50:  # 置信度50%以上
                    buy_price = df_test.iloc[i]['close']
                    
                    # 计算未来5天收益
                    if i + config['data']['predict_days'] < len(df_test):
                        future_price = df_test.iloc[i+config['data']['predict_days']]['close']
                        ret = (future_price - buy_price) / buy_price * 100
                        
                        trades.append({
                            'code': code,
                            'buy_date': df_test.index[i].strftime('%Y-%m-%d'),
                            'buy_price': buy_price,
                            'future_price': future_price,
                            'return': ret,
                            'success': ret >= config['data']['target_profit']
                        })
        except:
            continue
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        success_rate = trades_df['success'].mean() * 100
        avg_return = trades_df['return'].mean()
        
        print(f"预测成功率: {success_rate:.1f}% ({len(trades_df[trades_df['success']])}/{len(trades_df)})")
        print(f"选中股票数: {len(trades_df)}")
        print(f"平均收益: {avg_return:.2f}%")
        print(f"最大回撤: {trades_df['return'].min():.2f}%")
        
        # 达成目标检查
        target_success = config['targets']['success_rate']
        min_success = config['targets']['min_success_rate']
        
        print(f"\n============================================================")
        print(f"回测总结")
        print(f"============================================================")
        print(f"目标成功率: {target_success}%")
        print(f"最低成功率: {min_success}%")
        print(f"实际成功率: {success_rate:.1f}%")
        
        if success_rate >= target_success:
            print(f">>> 已达成目标!")
        elif success_rate >= min_success:
            print(f">>> 接近目标，继续优化...")
        else:
            print(f">>> 未达成目标，需要更多优化")
        print(f"============================================================")
        
        # 保存结果
        result = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'success_rate': success_rate,
            'avg_return': avg_return,
            'total_trades': len(trades_df),
            'trades': trades_df.to_dict('records')
        }
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(log_path, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"\n回测结果已保存到: {log_path}")
    else:
        print("没有产生任何交易信号")
    
    # 7. 生成今日推荐
    print("\n[步骤7] 生成今日推荐...")
    
    # 使用预定义的人气排名 (基于之前获取的数据)
    # 格式: code -> rank
    # 排名越小越热门
    POPULARITY_RANK = {
        # 银行 (热门)
        '600036': 333, '600000': 305, '601166': 1121, '601398': 890,
        '601939': 950, '600015': 680, '600016': 420, '601288': 780,
        '601818': 820, '601328': 1050, '600919': 1150, '601229': 1200,
        '600926': 1300, '601577': 1350, '600958': 1400, '600928': 950,
        '601077': 1100, '600865': 1450, '600816': 1500, '600742': 1550,
        # 券商
        '600030': 1120, '601788': 680, '601688': 520, '600999': 450,
        '000776': 2100, '002736': 1850, '601555': 1600, '600837': 750,
        '601099': 1650, '601375': 900, '600369': 1700, '000712': 2500,
        '002500': 2200, '601198': 1750, '000686': 1800,
        # 保险
        '601318': 1150, '601319': 1250, '601601': 870, '601336': 920,
        '601628': 980, '601899': 1050, '601111': 1550,
        # 消费
        '600519': 684, '000858': 2931, '000568': 2100, '600809': 2300,
        '600887': 990, '600197': 2400, '603589': 2500, '603719': 2600,
        '000596': 2700, '000798': 2800, '002304': 3427, '002507': 3500,
        '002558': 3600, '000729': 3700, '600199': 3800, '600059': 3900,
        # 医药
        '600276': 510, '600518': 400, '000566': 2000, '002044': 1950,
        '300003': 1700, '002007': 1600, '002038': 1800, '000423': 1900,
        '000513': 2050, '600535': 2150, '600529': 2250, '603939': 2350,
        '002262': 2450, '300015': 1650, '002294': 2550,
        # 科技
        '600703': 400, '000063': 2654, '002475': 3585, '002230': 3300,
        '002236': 3363, '300033': 2200, '300676': 1800, '002410': 3400,
        '002049': 3500, '002185': 3600, '002156': 3700, '000100': 3800,
        '000725': 2854, '600183': 3900, '603160': 3400,
        # 新能源
        '600438': 450, '601012': 1070, '600392': 500, '002594': 2800,
        '300750': 1800, '002466': 3200, '002460': 3300, '603799': 450,
        '002812': 3400, '300014': 3500, '002074': 3600, '002129': 3700,
        '600468': 3800, '600405': 3900, '002202': 3400, '600905': 1450,
        # 制造业
        '600104': 980, '600600': 600, '600019': 650, '600050': 890,
        '600009': 750, '600012': 850, '600011': 900, '600100': 950,
        '601766': 1000, '600893': 1050, '600316': 1100, '601989': 1150,
        '600038': 1200, '600031': 1250, '000425': 1300, '601877': 1350,
        # 房地产
        '600048': 500, '000001': 2613, '001979': 3500, '601155': 600,
        '600340': 700, '600383': 800, '600325': 900, '600606': 1000,
        '600639': 1100, '600675': 1200,
        # 基建
        '601668': 550, '601390': 650, '601800': 700, '601186': 800,
        '601618': 850, '600170': 900, '600028': 780, '601669': 950,
    }
    
    print(f"  使用预定义人气排名 (共{len(POPULARITY_RANK)}只)")
    
    # ========== 构建热门股票的人气排名 ==========
    # 对于涨幅热门股，用涨跌幅作为"人气"的代理
    # 涨幅>9.9%的股票 -> rank 1~300 (涨幅越大rank越小)
    # 涨幅2%-5%的股票 -> rank 301~600 (涨幅越大rank越小)
    stock_change_map = fetcher.get_stock_change_map()
    
    hot_rank = {}
    if stock_change_map:
        # 涨幅>9.9%的排序
        above_10 = [(c, ch) for c, ch in stock_change_map.items() if ch > 9.9]
        above_10.sort(key=lambda x: x[1], reverse=True)
        for i, (code, _) in enumerate(above_10):
            hot_rank[code] = i + 1
        
        # 涨幅2%-5%的排序
        range_2_5 = [(c, ch) for c, ch in stock_change_map.items() if 2 <= ch <= 5]
        range_2_5.sort(key=lambda x: x[1], reverse=True)
        for i, (code, _) in enumerate(range_2_5):
            hot_rank[code] = 300 + i + 1
        
        print(f"  热门股票人气排名: {len(hot_rank)} 只 (涨停>{len(above_10)} + 2-5%>{len(range_2_5)})")
    
    # ========== 获取今日实时价格作为买入参考 ==========
    print("\n[步骤7.5] 获取今日实时价格...")
    all_codes = list(processed_data.keys())
    realtime_quotes = fetcher.get_realtime_quotes(all_codes)
    
    # 构建实时价格映射 {code: {'open': x, 'price': y, 'close': z}}
    realtime_map = {}
    if not realtime_quotes.empty:
        for _, row in realtime_quotes.iterrows():
            code = row['code'].replace('sz', '').replace('sh', '').replace('bj', '')
            realtime_map[code] = {
                'open': row['open'],
                'price': row['price'],  # 当前价格
                'last_close': row.get('last_close', row['open'])  # 昨收
            }
    
    print(f"  成功获取 {len(realtime_map)} 只股票实时数据")
    
    # 判断当前时间
    now = datetime.now()
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=0, second=0, microsecond=0)
    
    # 决定买入价逻辑
    if now < market_open:
        # 开盘前运行：用今日开盘价作为买入价（可以实际买入）
        buy_price_source = "今日开盘价"
        use_realtime_open = True
    elif now <= market_close:
        # 盘中运行：用当前价格作为参考买入价
        buy_price_source = "当前价格(盘中)"
        use_realtime_open = False
    else:
        # 收盘后运行：用今日收盘价作为明日买入参考
        buy_price_source = "今日收盘价(预估明日)"
        use_realtime_open = False
    
    print(f"  当前时间: {now.strftime('%H:%M')} | 买入价来源: {buy_price_source}")
    
    # ========== 短线策略参数 ==========
    SHORT_TERM_CONFIDENCE = 0.40   # 置信度 > 40%
    SHORT_TERM_RANK = 1000          # 人气排名 < 1000
    SHORT_TERM_EXPECTED = 30        # 预期收益 > 30%
    SHORT_TERM_TAKE_PROFIT = 5      # 止盈 5%
    SHORT_TERM_STOP_LOSS = 3        # 止损 3%
    
    # ========== 中线策略参数 ==========
    MID_TERM_CONFIDENCE = 0.40     # 置信度 > 40%
    MID_TERM_RANK = 1000            # 人气排名 < 1000
    MID_TERM_EXPECTED = 30          # 预期收益 > 30%
    MID_TERM_TAKE_PROFIT = 10       # 止盈 10%
    MID_TERM_STOP_LOSS = 5          # 止损 5%
    
    recommendations_short = []
    recommendations_mid = []
    debug_info = []
    
    for code, df in processed_data.items():
        try:
            latest = df.iloc[-1:].copy()
            X_latest = latest[feature_names]
            
            if X_latest.isna().any().any():
                continue
            
            proba = predictor.predict_proba(X_latest)
            pred = predictor.predict(X_latest)
            
            # 处理不同形状的输出
            if isinstance(proba, np.ndarray):
                if proba.ndim == 1:
                    # 只有1个值，表示正类概率
                    proba_1d = proba[0]
                else:
                    proba_1d = proba[0][1] if proba.shape[1] > 1 else proba[0][0]
            else:
                proba_1d = float(proba)
            
            # 优先用热门排名，其次用预定义人气排名
            rank = hot_rank.get(code, POPULARITY_RANK.get(code, 9999))
            
            # 计算预期收益
            expected_return = proba_1d * 50
            
            # 技术信号加成
            tech_signal = 0
            try:
                if latest['boll_position'].values[0] > 0.8:
                    tech_signal += 10
                if latest['ma_bullish'].values[0] == 1:
                    tech_signal += 10
                if latest['macd'].values[0] > latest['macd_signal'].values[0]:
                    tech_signal += 10
            except:
                pass
            
            total_expected = expected_return + tech_signal
            
            debug_info.append({
                'code': code,
                'pred': pred,
                'proba': proba_1d,
                'rank': rank,
                'expected': total_expected
            })
            
            # 预测为上涨且置信度>阈值
            if pred == 1 and proba_1d >= SHORT_TERM_CONFIDENCE:
                # 计算主力成本线作为买入参考
                # 主力成本线 = 最低价 + (最高价 - 最低价) / 3
                last_close = latest['close'].values[0]
                
                # 获取近期高低点（用最近20天）
                recent_df = df.tail(20)
                recent_high = recent_df['high'].max()
                recent_low = recent_df['low'].min()
                
                # 计算主力成本线
                main_cost_line = recent_low + (recent_high - recent_low) / 3
                
                # 获取实时价格信息
                rt = realtime_map.get(code, {})
                current_price = rt.get('price', last_close)
                open_price = rt.get('open', current_price)
                
                # 买入价取主力成本线和当前价格的较小值（确保能买到）
                # 如果主力成本线高于当前价，说明已经跌破支撑，用当前价
                # 如果主力成本线低于当前价，可以在成本线附近买入
                buy_price = min(main_cost_line, current_price)
                
                # 计算开盘涨幅（相对于昨日收盘）
                open_change = (open_price - last_close) / last_close * 100 if last_close > 0 else 0
                
                # 计算当前价格相对主力成本线的位置
                cost_position = (current_price - main_cost_line) / main_cost_line * 100 if main_cost_line > 0 else 0
                
                # 如果已经涨停或大幅高开，标记为高风险
                high_risk = open_change >= 9.5 or (use_realtime_open and open_change >= 5)
                
                # 短线推荐条件
                if rank <= SHORT_TERM_RANK and total_expected >= SHORT_TERM_EXPECTED:
                    take_profit = buy_price * (1 + SHORT_TERM_TAKE_PROFIT / 100)
                    stop_loss = buy_price * (1 - SHORT_TERM_STOP_LOSS / 100)
                    
                    recommendations_short.append({
                        'code': code,
                        'confidence': proba_1d,
                        'expected_return': total_expected,
                        'buy_price': buy_price,
                        'main_cost_line': main_cost_line,
                        'current_price': current_price,
                        'cost_position': cost_position,
                        'take_profit': take_profit,
                        'take_profit_rate': SHORT_TERM_TAKE_PROFIT,
                        'stop_loss': stop_loss,
                        'stop_loss_rate': SHORT_TERM_STOP_LOSS,
                        'popularity_rank': rank,
                        'strategy': '短线(5天)',
                        'change': stock_change_map.get(code, 0),
                        'open_change': open_change,
                        'high_risk': high_risk
                    })
                
                # 中线推荐条件
                if rank <= MID_TERM_RANK and total_expected >= MID_TERM_EXPECTED:
                    take_profit = buy_price * (1 + MID_TERM_TAKE_PROFIT / 100)
                    stop_loss = buy_price * (1 - MID_TERM_STOP_LOSS / 100)
                    
                    recommendations_mid.append({
                        'code': code,
                        'confidence': proba_1d,
                        'expected_return': total_expected,
                        'buy_price': buy_price,
                        'main_cost_line': main_cost_line,
                        'current_price': current_price,
                        'cost_position': cost_position,
                        'take_profit': take_profit,
                        'take_profit_rate': MID_TERM_TAKE_PROFIT,
                        'stop_loss': stop_loss,
                        'stop_loss_rate': MID_TERM_STOP_LOSS,
                        'popularity_rank': rank,
                        'strategy': '中线(2-4周)',
                        'change': stock_change_map.get(code, 0),
                        'open_change': open_change,
                        'high_risk': high_risk
                    })
        except Exception as e:
            print(f"  {code}: error - {e}")
            continue
    
    # 打印调试信息
    print(f"\n所有候选股票预测 (共{len(debug_info)}只):")
    debug_info = sorted(debug_info, key=lambda x: x['proba'], reverse=True)
    for d in debug_info[:10]:
        print(f"  {d['code']}: pred={d['pred']}, proba={d['proba']:.2f}, rank={d['rank']}, expected={d['expected']:.1f}%")
    
    print(f"\n短线推荐候选: {len(recommendations_short)}")
    print(f"中线推荐候选: {len(recommendations_mid)}")
    
    # 排序并取Top 5
    recommendations_short = sorted(recommendations_short, key=lambda x: x['expected_return'], reverse=True)[:5]
    recommendations_mid = sorted(recommendations_mid, key=lambda x: x['expected_return'], reverse=True)[:5]
    
    # ========== 股票跟踪记录 ==========
    today = datetime.now().strftime('%Y-%m-%d')
    tracker = StockTracker()
    
    # 保存今日推荐到跟踪记录
    all_recommendations = []
    for rec in recommendations_short:
        rec['name'] = fetcher.get_stock_name(rec['code'])
        rec['strategy'] = '短线(5天)'
        all_recommendations.append(rec)
    for rec in recommendations_mid:
        rec['name'] = fetcher.get_stock_name(rec['code'])
        rec['strategy'] = '中线(2-4周)'
        all_recommendations.append(rec)
    
    if all_recommendations:
        tracker.add_recommendations(all_recommendations, today)
        print(f"\n📝 已将 {len(all_recommendations)} 只股票加入跟踪记录")
    
    # 显示历史跟踪记录
    print(tracker.format_history_display(limit=15))
    
    # ========== 输出结果 ==========
    if recommendations_short or recommendations_mid:
        print(f"\n{'='*70}")
        print(f"  今日股票推荐 (2026-04-08)")
        print(f"{'='*70}")
        
        # 短线推荐
        if recommendations_short:
            print(f"\n【短线策略】持股5天，止盈{SHORT_TERM_TAKE_PROFIT}%，止损{SHORT_TERM_STOP_LOSS}%")
            print("-" * 110)
            print(f"{'排名':<4} {'代码':<10} {'主力成本':<8} {'当前价':<8} {'买入价':<8} {'置信度':<8} {'预期':<8} {'止盈':<8} {'止损':<8} {'风险':<6}")
            print("-" * 110)
            for i, rec in enumerate(recommendations_short, 1):
                risk_mark = "⚠️" if rec.get('high_risk', False) else ""
                cost_line = rec.get('main_cost_line', rec['buy_price'])
                curr_price = rec.get('current_price', rec['buy_price'])
                print(f"{i:<4} {rec['code']:<10} {cost_line:>7.2f}  {curr_price:>7.2f}  {rec['buy_price']:>7.2f}  {rec['confidence']*100:.1f}%{'':<4} {rec['expected_return']:.1f}%{'':<3} {rec['take_profit']:.2f}{'':<5} {rec['stop_loss']:.2f}  {risk_mark}")
        
        # 中线推荐
        if recommendations_mid:
            print(f"\n【中线策略】持股2-4周，止盈{MID_TERM_TAKE_PROFIT}%，止损{MID_TERM_STOP_LOSS}%")
            print("-" * 110)
            print(f"{'排名':<4} {'代码':<10} {'主力成本':<8} {'当前价':<8} {'买入价':<8} {'置信度':<8} {'预期':<8} {'止盈':<8} {'止损':<8} {'风险':<6}")
            print("-" * 110)
            for i, rec in enumerate(recommendations_mid, 1):
                risk_mark = "⚠️" if rec.get('high_risk', False) else ""
                cost_line = rec.get('main_cost_line', rec['buy_price'])
                curr_price = rec.get('current_price', rec['buy_price'])
                print(f"{i:<4} {rec['code']:<10} {cost_line:>7.2f}  {curr_price:>7.2f}  {rec['buy_price']:>7.2f}  {rec['confidence']*100:.1f}%{'':<4} {rec['expected_return']:.1f}%{'':<3} {rec['take_profit']:.2f}{'':<5} {rec['stop_loss']:.2f}  {risk_mark}")
    else:
        print("\n今日无符合条件的推荐股票")


if __name__ == "__main__":
    config = load_config()
    run_backtest_real(config)
