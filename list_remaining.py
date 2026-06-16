import json
with open(r'D:\yym\code\stockMarketPredict\data\stock_tracking.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
items = sorted(data.items(), key=lambda x: x[1]['confidence'], reverse=True)
for code, info in items:
    latest_date = max(info['daily_prices'].keys())
    close = info['daily_prices'][latest_date]['close']
    tp = info['take_profit']
    sl = info['stop_loss']
    conf = info['confidence']
    print(f'{code} {info["name"]} | conf={conf:.4f} | 现价={close} | 止盈={tp:.4f} | 止损={sl:.4f}')
