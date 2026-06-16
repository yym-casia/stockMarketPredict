# -*- coding: utf-8 -*-
"""
每日交易流水线 - 统一入口

约束:
  - 最多持有 5 只股票
  - 持股不超过配置的上限（默认 portfolio.max_hold_days）
  - 跳过周末和法定节假日
  - 每日输出买卖操作建议
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.trading_calendar import is_trading_day, refresh_trading_dates_cache
from src.portfolio_manager import PortfolioManager
from src.position_manager import PositionManager
from src.daily_advisor import DailyAdvisor
from src.data_fetcher_multi import MultiSourceDataFetcher

from daily_pipeline_v4 import (
    generate_recommendations,
    load_config,
)
from src.strategy_filters import load_strategy_config
from src.market_regime import build_live_regime
from src.eod_screener import (
    screen_like_backtest, enrich_candidate_names, regime_to_sentiment,
    candidates_to_screened_rows,
)
from src.shakeout_strategy import shakeout_rebuy_candidates, merge_shakeout_cfg


def export_daily_csv(operations: dict, today: str):
    """导出当日操作 CSV"""
    filename = f"data/recommendations_{today}.csv"
    os.makedirs('data', exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '代码', '名称', '操作', '优先级', '买入价', '当前价',
            '收益率%', '持有天数', '止盈价', '止损价', '原因',
        ])
        for op_type, key, price_key in [
            ('卖出', 'sell', 'current_price'),
            ('部分卖出', 'partial_sell', 'current_price'),
            ('买入', 'buy', 'buy_price'),
            ('持有', 'hold', 'current_price'),
        ]:
            for item in operations['operations'].get(key, []):
                writer.writerow([
                    item['code'], item['name'], op_type,
                    item.get('priority', 'normal'),
                    item.get('buy_price', ''),
                    item.get(price_key, item.get('current_price', '')),
                    f"{item.get('profit_pct', 0):.2f}" if 'profit_pct' in item else '',
                    item.get('days_held', ''),
                    item.get('take_profit', ''),
                    item.get('stop_loss', item.get('current_stop_loss', '')),
                    item.get('reason', ''),
                ])
    print(f"\n💾 操作记录已导出: {filename}")


def main():
    today = datetime.now().strftime('%Y-%m-%d')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    config = load_config()
    portfolio_cfg = config.get('portfolio', {})
    max_hold_days = portfolio_cfg.get('max_hold_days', 15)
    strat_cfg_preview = load_strategy_config(config)
    print("=" * 60)
    print("🚀 智能交易系统 - 每日运行")
    print(f"   时间: {now_str}")
    print(f"   策略: 收盘后选股(回测同款 +91.3%) | 涨幅{strat_cfg_preview['min_change']:.0f}-"
          f"{strat_cfg_preview['max_change']:.0f}% | 移动止盈"
          f"{strat_cfg_preview.get('trailing_start', 15):.0f}%/{strat_cfg_preview.get('trailing_pct', 6):.0f}%")
    print(f"   持仓≤{max_hold_days}天 | 全仓止损{strat_cfg_preview.get('stop_loss', 3):.0f}%"
          f"{'(收盘确认)' if strat_cfg_preview.get('stop_loss_on_close') else ''}"
          f" | 弱市≤{strat_cfg_preview.get('weak_market_max_positions', 2)}仓"
          f" | 趋势末期过滤{'开' if strat_cfg_preview.get('late_trend_filter_enabled') else '关'}")
    print("=" * 60)

    refresh_trading_dates_cache()

    if not is_trading_day():
        print(f"\n⏸️  今日({today})非交易日，跳过执行")
        advisor = DailyAdvisor(
            PortfolioManager(max_hold_days=max_hold_days),
            PositionManager(max_hold_days=max_hold_days),
        )
        advisor.save_operations({
            'date': today,
            'generated_at': now_str,
            'is_trading_day': False,
            'summary': '今日非交易日，无操作建议',
            'operations': {'sell': [], 'partial_sell': [], 'buy': [], 'hold': []},
            'urgent_alerts': [],
            'portfolio_summary': {},
        })
        return

    strat_cfg = load_strategy_config(config)
    max_positions = portfolio_cfg.get('max_positions', 5)
    min_confidence = config['data'].get('min_confidence', 0.85)
    tp_pct = config['data']['take_profit']
    sl_pct = strat_cfg.get('stop_loss', config['data']['stop_loss'])

    portfolio = PortfolioManager(
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        trend_stop_loss_partial_ratio=strat_cfg.get('trend_stop_loss_partial_ratio', 0),
    )
    bt_cfg = config.get('backtest', {})
    position_mgr = PositionManager(
        max_hold_days=max_hold_days,
        stop_loss_pct=sl_pct,
        trailing_start=strat_cfg.get('trailing_start', 4.0),
        trailing_pct=strat_cfg.get('trailing_pct', 2.5),
        add_trigger_pct=bt_cfg.get('add_trigger_pct', 10.0),
        add_ratio=bt_cfg.get('add_ratio', 0.0),
        trend_add_amount_pct=bt_cfg.get('trend_add_amount_pct'),
        trend_stop_loss_partial_ratio=strat_cfg.get('trend_stop_loss_partial_ratio', 0),
        trend_dip_refill_pct=strat_cfg.get('trend_dip_refill_pct'),
        trend_half_hold_exit_pct=strat_cfg.get('trend_half_hold_exit_pct'),
        stop_loss_on_close=strat_cfg.get('stop_loss_on_close', True),
    )
    advisor = DailyAdvisor(portfolio, position_mgr)
    fetcher = MultiSourceDataFetcher()

    # 收盘数据更新 + 回测同款市场/选股（一次加载，后续复用）
    print("\n📥 收盘数据更新 & 回测同款分析...")
    held_codes = set(portfolio.get_active_positions().keys())
    shakeout_watches = portfolio.get_shakeout_watches(today)
    eod = screen_like_backtest(
        config, held_codes=held_codes, shakeout_watches=shakeout_watches,
        refresh_data=True, verbose=True,
    )
    sentiment = regime_to_sentiment(eod['regime'])
    sentiment_score = sentiment.get('score', 50)
    live_regime = build_live_regime(sentiment_score, strat_cfg)
    should_enter = eod['can_enter']
    reason = eod['reason']
    max_positions = eod.get('max_positions', max_positions)
    if live_regime.get('defensive'):
        print(f"   ⚠️ 熊市防御模式 | 评分{sentiment_score:.0f}")

    # === 第一步：评估现有持仓（卖出优先）===
    print("\n📊 第一步：评估现有持仓...")
    active = portfolio.get_active_positions()
    print(f"   当前持仓: {len(active)}/{max_positions} 只")

    categories = {'sell': [], 'partial_sell': [], 'hold': []}
    urgent_alerts = []

    if active:
        codes = list(active.keys())
        price_data = fetcher.get_realtime_quotes(codes)
        total_equity = bt_cfg.get('initial_capital', 100000)
        categories, urgent_alerts = advisor.evaluate_positions(
            price_data, today, total_equity=total_equity,
            confirm_stop_close=True,
            regime=live_regime, strat_cfg=strat_cfg)
        print(f"   卖出信号: {len(categories['sell'])} | "
              f"部分止盈: {len(categories['partial_sell'])} | "
              f"加仓: {len(categories.get('add', []))} | "
              f"继续持有: {len(categories['hold'])}")

        advisor.apply_sell_actions(categories, today)

    shakeout_extra_rows = []
    shakeout_watches = portfolio.get_shakeout_watches(today)
    if shakeout_watches and eod.get('all_data') and eod['can_enter']:
        held_after_sell = set(portfolio.get_active_positions().keys())
        shakeout_cands = shakeout_rebuy_candidates(
            shakeout_watches, eod['all_data'], eod['date'], held_after_sell,
            strat_cfg, regime=eod.get('regime'),
        )
        if shakeout_cands:
            shakeout_extra_rows = enrich_candidate_names(
                candidates_to_screened_rows(shakeout_cands, eod['all_data'], eod['date']),
            )
            print(f"\n📌 当日止损后连阳接回: {len(shakeout_extra_rows)} 只")

    # === 第二步：市场环境结论 ===
    mainlines = sentiment.get('mainlines', [])
    print(f"\n🌍 第二步：市场环境（基准日 {eod['date']}）")
    print(f"   入场建议: {'✅ 可以入场' if should_enter else '❌ 建议观望'} | {reason}")
    if mainlines:
        print(f"   热门板块: {', '.join(mainlines)}")
    if should_enter:
        print(f"   今日最大持仓: {max_positions}只")

    # === 第三步：生成买入信号（受仓位限制）===
    buy_signals = []
    active_count = len(portfolio.get_active_positions())
    available_slots = max(0, max_positions - active_count)

    if should_enter and available_slots > 0:
        print(f"\n🔍 第三步：收盘选股（回测同款，可买入 {available_slots} 只）...")
        seen_codes = set()
        screened = []
        for row in shakeout_extra_rows + enrich_candidate_names(eod['screened_rows']):
            code = row.get('code_clean', row.get('code'))
            if code not in seen_codes:
                seen_codes.add(code)
                screened.append(row)

        if screened:
            candidates = generate_recommendations(screened, tp_pct, sl_pct, top_n=10)
            buy_signals = advisor.select_buys(
                candidates, available_slots, min_confidence,
                ml_rank_only=strat_cfg.get('ml_rank_only', True))

            shakeout_ratio = float(merge_shakeout_cfg(strat_cfg).get('shakeout_rebuy_size_ratio', 0.5))
            for rec in buy_signals:
                portfolio.add_position(rec, eod['date'])
                if rec.get('shakeout_rebuy'):
                    portfolio.remove_shakeout_watch(rec['code'])
                    print(f"   ✅ 连阳接回: {rec['code']} {rec['name']} "
                          f"({rec.get('yang_days', 0)}连阳, 半仓{shakeout_ratio:.0%})")
                else:
                    print(f"   ✅ 新增: {rec['code']} {rec['name']} "
                          f"置信度{rec['confidence']*100:.1f}%")
    elif available_slots == 0:
        print(f"\n⚠️  持仓已满({max_positions}只)，今日不新增买入")
    else:
        print(f"\n⚠️  市场环境不佳，今日不新增买入")

    # === 第四步：生成并保存操作建议 ===
    operations = advisor.build_operations(categories, buy_signals, {
        'should_enter': should_enter,
        'reason': reason,
        'sentiment': sentiment,
    }, today)
    operations['screen_mode'] = 'eod_backtest'
    operations['screen_date'] = eod.get('date', today)
    operations['urgent_alerts'] = urgent_alerts

    advisor.save_operations(operations)
    advisor.print_operations(operations)
    export_daily_csv(operations, today)

    # 历史战绩
    stats = portfolio.get_stats()
    if stats.get('total_trades', 0) > 0:
        print(f"\n📈 历史战绩: {stats['total_trades']}笔 | "
              f"胜率{stats['win_rate']}% | 均收益{stats['avg_profit']:+.2f}%")

    print("\n✅ 每日分析完成！")


if __name__ == "__main__":
    main()
