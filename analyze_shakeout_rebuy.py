# -*- coding: utf-8 -*-
"""分析止损接回标的的后期涨幅与收益贡献（支持多方案对比）"""
import json
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

from stock_pool_manager import get_pool_manager
from src.backtest_data_store import BacktestDataStore
from daily_pipeline_v4 import calculate_technical_indicators


def load_variants():
    with open('data/backtest_capital_results.json', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('variants', {}), data.get('period', {})


def match_rebuy_rounds(trades):
    rebuy_buys = [t for t in trades if t.get('action') == 'buy' and '震仓' in t.get('reason', '')]
    rounds = []
    for buy in rebuy_buys:
        code = buy['code']
        buy_date = buy['date']
        sells = [
            t for t in trades
            if t.get('code') == code and t.get('action') == 'sell' and t.get('date') >= buy_date
        ]
        sells.sort(key=lambda x: x['date'])
        sell = sells[0] if sells else None
        if sell:
            outcome = 'win' if sell.get('profit', 0) > 0 else 'loss'
            rounds.append({
                'code': code,
                'buy_date': buy_date,
                'buy_price': buy['price'],
                'amount': buy.get('amount', 0),
                'sell_date': sell['date'],
                'sell_price': sell['price'],
                'profit_pct': sell.get('profit_pct', 0),
                'profit': sell.get('profit', 0),
                'hold_days': sell.get('hold_days', 0),
                'sell_reason': sell.get('reason', ''),
                'outcome': outcome,
            })
        else:
            rounds.append({
                'code': code, 'buy_date': buy_date, 'buy_price': buy['price'],
                'amount': buy.get('amount', 0), 'sell_date': None, 'sell_price': None,
                'profit_pct': 0, 'profit': 0, 'hold_days': 0,
                'sell_reason': '未平仓', 'outcome': 'open',
            })
    return rounds


def enrich_post_buy_path(rounds, forward_days=(5, 10, 20)):
    codes = list({r['code'] for r in rounds})
    if not codes:
        return
    raw, _ = BacktestDataStore().load_or_fetch(codes, days=300)
    all_data = {}
    for code, df in raw.items():
        try:
            all_data[code] = calculate_technical_indicators(df.copy())
        except Exception:
            pass
    sectors = {}
    try:
        sectors = get_pool_manager().stock_sectors
    except Exception:
        pass

    for r in rounds:
        r['name'] = sectors.get(r['code'], r['code'])
        df = all_data.get(r['code'])
        if df is None:
            continue
        buy_dt = r['buy_date']
        idx = df.index[df.index <= buy_dt]
        if idx.empty:
            continue
        pos = len(idx) - 1
        buy_close = float(df.iloc[pos]['close'])
        r['buy_close'] = buy_close
        for n in forward_days:
            if pos + n < len(df):
                fwd_close = float(df.iloc[pos + n]['close'])
                r[f't+{n}d_pct'] = round((fwd_close / buy_close - 1) * 100, 2)
            else:
                r[f't+{n}d_pct'] = None
        end_date = r['sell_date'] or str(df.index[-1])[:10]
        window = df.loc[(df.index > buy_dt) & (df.index <= end_date)]
        if not window.empty:
            max_high = float(window['high'].max())
            r['max_gain_pct'] = round((max_high / buy_close - 1) * 100, 2)
            peak_idx = window['high'].idxmax()
            r['max_gain_days'] = len(df.loc[df.index[pos]:peak_idx]) - 1
        else:
            r['max_gain_pct'] = 0
            r['max_gain_days'] = 0


def rebuy_sell_profit(trades):
    """接回买入对应卖出的利润合计。"""
    rounds = match_rebuy_rounds(trades)
    return sum(r['profit'] for r in rounds if r['outcome'] in ('win', 'loss'))


def print_variant_report(name, rounds, period, summary):
    closed = [r for r in rounds if r['outcome'] in ('win', 'loss')]
    wins = [r for r in closed if r['outcome'] == 'win']
    losses = [r for r in closed if r['outcome'] == 'loss']
    rebuy_profit = sum(r['profit'] for r in closed)
    trend_profit = summary.get('trend_profit', 0)
    rebuy_pct_of_total = (rebuy_profit / trend_profit * 100) if trend_profit else 0

    print('\n' + '=' * 72)
    print(f'【{name}】接回收益分析')
    print('=' * 72)
    print(f'接回笔数: {len(rounds)} | 半仓约50%资金')
    print(f'接回贡献利润: {rebuy_profit:+,.0f} 元 | 占策略总利润 {rebuy_pct_of_total:.1f}%'
          f' (总利润 {trend_profit:+,.0f})')

    if closed:
        print(f'\n整轮平仓: {len(closed)}笔 | 胜率 {len(wins)/len(closed)*100:.1f}% '
              f'({len(wins)}胜/{len(losses)}负)')
        print(f'  均收益率: {sum(r["profit_pct"] for r in closed)/len(closed):+.2f}%')
        print(f'  金额均盈亏: {rebuy_profit/len(closed):+,.0f} 元/笔')
        if wins:
            print(f'  盈利单: 均收益 {sum(r["profit_pct"] for r in wins)/len(wins):+.2f}% | '
                  f'均金额 {sum(r["profit"] for r in wins)/len(wins):+,.0f}元')
        if losses:
            print(f'  亏损单: 均收益 {sum(r["profit_pct"] for r in losses)/len(losses):+.2f}% | '
                  f'均金额 {sum(r["profit"] for r in losses)/len(losses):+,.0f}元')

    for label, key in [('T+5', 't+5d_pct'), ('T+10', 't+10d_pct'), ('T+20', 't+20d_pct')]:
        vals = [r[key] for r in rounds if r.get(key) is not None]
        if vals:
            pos = sum(1 for v in vals if v > 0)
            print(f'  {label}: 均{sum(vals)/len(vals):+.2f}% 上涨{pos/len(vals)*100:.0f}%')

    max_gains = [r.get('max_gain_pct', 0) for r in rounds if 'max_gain_pct' in r]
    if max_gains:
        print(f'  持仓最高: 均{sum(max_gains)/len(max_gains):+.2f}% | '
              f'≥10%:{sum(1 for g in max_gains if g>=10)}只 ≥20%:{sum(1 for g in max_gains if g>=20)}只')

    by_reason = defaultdict(list)
    for r in closed:
        tag = r['sell_reason'].split('(')[0].strip()[:18]
        by_reason[tag].append((r['profit_pct'], r['profit']))
    if by_reason:
        print('\n  卖出原因:')
        for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            pcts = [x[0] for x in items]
            amts = [x[1] for x in items]
            print(f'    {reason}: {len(items)}笔 均收益{sum(pcts)/len(pcts):+.1f}% '
                  f'金额{sum(amts):+,.0f}元')

    if rounds:
        print('\n  逐笔:')
        for i, r in enumerate(rounds, 1):
            t10 = r.get('t+10d_pct')
            t10s = f'{t10:+.1f}%' if t10 is not None else '-'
            mx = r.get('max_gain_pct', 0)
            print(f'    {i}. {r["code"]} {r["buy_date"]} 买{r["buy_price"]:.2f} '
                  f'T+10{t10s} 最高{mx:+.1f}% → {r["profit_pct"]:+.1f}% '
                  f'({r["profit"]:+,.0f}元) {r["sell_reason"][:24]}')

    return {
        'name': name,
        'rebuy_count': len(rounds),
        'rebuy_profit': round(rebuy_profit, 2),
        'rebuy_pct_of_trend': round(rebuy_pct_of_total, 2),
        'win_rate': round(len(wins) / len(closed) * 100, 1) if closed else 0,
        'avg_profit_pct': round(sum(r['profit_pct'] for r in closed) / len(closed), 2) if closed else 0,
        'rounds': rounds,
    }


def main():
    variants, period = load_variants()
    if not variants:
        print('无回测数据')
        return

    print('=' * 72)
    print('止损接回 · 后期收益对比')
    print(f'区间: {period.get("start")} ~ {period.get("end")}')
    print('=' * 72)

    reports = []
    for name, v in variants.items():
        trades = v.get('trades', [])
        summary = v.get('summary', {})
        rounds = match_rebuy_rounds(trades)
        enrich_post_buy_path(rounds)
        reports.append(print_variant_report(name, rounds, period, summary))

    print('\n' + '=' * 72)
    print('【三方案接回对比摘要】')
    print(f'{"方案":<16} {"接回":>4} {"接回利润":>10} {"占总额":>7} {"胜率":>6} {"均收益":>7}')
    print('-' * 72)
    for r in reports:
        print(f'{r["name"]:<16} {r["rebuy_count"]:>4} {r["rebuy_profit"]:>+10,.0f} '
              f'{r["rebuy_pct_of_trend"]:>6.1f}% {r["win_rate"]:>5.1f}% {r["avg_profit_pct"]:>+6.1f}%')

  # vs 普通买入
    if '基准3连阳' in variants:
        base_trades = variants['基准3连阳']['trades']
        rebuy_codes_dates = {(b['code'], b['date']) for b in base_trades
                             if b.get('action') == 'buy' and '震仓' in b.get('reason', '')}
        normal_sells = [t for t in base_trades if t.get('action') == 'sell']
        print('\n【基准：接回 vs 普通趋势单（卖出笔）】')
        print(f'  普通趋势卖出: {len(normal_sells)}笔')
        if normal_sells:
            nw = sum(1 for t in normal_sells if t.get('profit', 0) > 0)
            print(f'  普通胜率: {nw/len(normal_sells)*100:.1f}% | '
                  f'均收益 {sum(t.get("profit_pct",0) for t in normal_sells)/len(normal_sells):+.2f}%')
        rebuy_r = next((x for x in reports if x['name'] == '基准3连阳'), None)
        if rebuy_r and rebuy_r['rebuy_count']:
            print(f'  接回胜率: {rebuy_r["win_rate"]:.1f}% | 均收益 {rebuy_r["avg_profit_pct"]:+.2f}%')

    out = {
        'period': period,
        'variants': {r['name']: {
            'rebuy_count': r['rebuy_count'],
            'rebuy_profit': r['rebuy_profit'],
            'rebuy_pct_of_trend': r['rebuy_pct_of_trend'],
            'win_rate': r['win_rate'],
            'avg_profit_pct': r['avg_profit_pct'],
            'rounds': r['rounds'],
        } for r in reports},
    }
    path = 'data/shakeout_rebuy_analysis.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n💾 {path}')


if __name__ == '__main__':
    main()
