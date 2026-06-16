import json

path = r"D:\yym\code\stockMarketPredict\data\stock_tracking.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

remove_codes = []
keep_codes = []

for code, info in data.items():
    latest_date = max(info["daily_prices"].keys())
    latest = info["daily_prices"][latest_date]
    close = latest["close"]
    tp = info["take_profit"]
    sl = info["stop_loss"]
    
    if close >= tp:
        remove_codes.append((code, info["name"], "止盈", close, tp))
    elif close <= sl:
        remove_codes.append((code, info["name"], "止损", close, sl))
    else:
        keep_codes.append(code)

print("=== 触及止盈/止损，需出局 ===")
for code, name, reason, close, ref in remove_codes:
    print(f"  {code} {name} | {reason} | 现价 {close} / 参考 {ref:.4f}")

print(f"\n共 {len(remove_codes)} 只出局，{len(keep_codes)} 只保留")

# 保留的写入新文件
new_data = {k: v for k, v in data.items() if k in keep_codes}
with open(path, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print(f"\n已更新 {path}，移除了 {len(remove_codes)} 只股票")
