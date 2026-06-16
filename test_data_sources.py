# -*- coding: utf-8 -*-
"""
数据源稳定性测试 - 无emoji版本
"""
import sys
import os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import pandas as pd
import time
from datetime import datetime, timedelta

print("=" * 60)
print("  数据源稳定性测试")
print("=" * 60)

results = []

# ============ 1. NeoData ============
print("\n[1] NeoData (localhost:19000)...")
try:
    url = "http://localhost:19000/proxy/api"
    headers = {
        "Remote-URL": "https://jprx.m.qq.com/aizone/skillserver/v1/proxy/teamrouter_neodata/query",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": "neodata",
        "sub_channel": "qclaw",
        "query": "贵州茅台最新行情",
        "request_id": f"test_{int(time.time()*1000)}",
        "data_type": "api"
    }
    start = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    elapsed = time.time() - start
    
    if resp.status_code == 200:
        data = resp.json()
        if data.get("code") == "200":
            print(f"  [OK] NeoData 可用 ({elapsed:.2f}s)")
            results.append(("NeoData", "OK", elapsed))
        else:
            print(f"  [ERR] NeoData: {data.get('msg')}")
            results.append(("NeoData", "ERROR", elapsed))
    else:
        print(f"  [ERR] NeoData HTTP {resp.status_code}")
        results.append(("NeoData", "HTTP_ERROR", elapsed))
except Exception as e:
    print(f"  [FAIL] NeoData: {e}")
    results.append(("NeoData", "FAIL", 0))

# ============ 2. 新浪财经 ============
print("\n[2] 新浪财经...")
try:
    url = "https://hq.sinajs.cn/list=sh600519,sz000001"
    headers = {"Referer": "https://finance.sina.com.cn"}
    start = time.time()
    resp = requests.get(url, headers=headers, timeout=10)
    elapsed = time.time() - start
    
    if resp.status_code == 200 and resp.text:
        lines = resp.text.strip().split('\n')
        count = len([l for l in lines if '="' in l])
        print(f"  [OK] 新浪财经可用 ({elapsed:.2f}s, {count}只股票)")
        results.append(("新浪财经", "OK", elapsed))
    else:
        print(f"  [ERR] 新浪财经 HTTP {resp.status_code}")
        results.append(("新浪财经", "HTTP_ERROR", elapsed))
except Exception as e:
    print(f"  [FAIL] 新浪财经: {e}")
    results.append(("新浪财经", "FAIL", 0))

# ============ 3. 腾讯财经 ============
print("\n[3] 腾讯财经...")
try:
    url = "https://web.sqt.gtimg.cn/q=r_sh600519,r_sz000001"
    start = time.time()
    resp = requests.get(url, timeout=10)
    elapsed = time.time() - start
    
    if resp.status_code == 200 and resp.text:
        lines = resp.text.strip().split('\n')
        count = len([l for l in lines if l.strip()])
        print(f"  [OK] 腾讯财经可用 ({elapsed:.2f}s, {count}只股票)")
        results.append(("腾讯财经", "OK", elapsed))
    else:
        print(f"  [ERR] 腾讯财经 HTTP {resp.status_code}")
        results.append(("腾讯财经", "HTTP_ERROR", elapsed))
except Exception as e:
    print(f"  [FAIL] 腾讯财经: {e}")
    results.append(("腾讯财经", "FAIL", 0))

# ============ 4. 东方财富实时行情 ============
print("\n[4] 东方财富实时行情...")
try:
    url = "https://push2.eastmoney.com/api/qt/ulist.np/get"
    params = {
        "fltt": 2, "invt": 2,
        "fields": "f12,f14,f2,f3,f4,f5",
        "secids": "1.600519,0.000001",
        "_": int(time.time() * 1000)
    }
    start = time.time()
    resp = requests.get(url, params=params, timeout=10)
    elapsed = time.time() - start
    
    if resp.status_code == 200:
        data = resp.json()
        stocks = data.get("data", {}).get("diff", [])
        if stocks:
            print(f"  [OK] 东方财富可用 ({elapsed:.2f}s, {len(stocks)}只股票)")
            results.append(("东方财富", "OK", elapsed))
        else:
            print(f"  [ERR] 东方财富 返回为空")
            results.append(("东方财富", "EMPTY", elapsed))
    else:
        print(f"  [ERR] 东方财富 HTTP {resp.status_code}")
        results.append(("东方财富", "HTTP_ERROR", elapsed))
except Exception as e:
    print(f"  [FAIL] 东方财富: {e}")
    results.append(("东方财富", "FAIL", 0))

# ============ 5. 腾讯财经历史K线 ============
print("\n[5] 腾讯财经历史K线...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.history_fetcher import get_history_fetcher
    start = time.time()
    df = get_history_fetcher().fetch_tencent('600036', 120)
    elapsed = time.time() - start
    if df is not None and len(df) >= 60:
        print(f"  [OK] 腾讯K线可用 ({elapsed:.2f}s, {len(df)}天, {df.index[0].date()}~{df.index[-1].date()})")
        results.append(("腾讯K线", "OK", elapsed))
    else:
        print(f"  [ERR] 腾讯K线 返回不足")
        results.append(("腾讯K线", "EMPTY", elapsed))
except Exception as e:
    print(f"  [FAIL] 腾讯K线: {e}")
    results.append(("腾讯K线", "FAIL", 0))

# ============ 6. 东方财富历史K线 ============
print("\n[6] 东方财富历史K线...")
try:
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "cb": "", "loginId": "", "utm_source": "wind", "lmt": "0",
        "klt": "101", "fqt": "1", "secid": "1.600519",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "_": int(time.time() * 1000)
    }
    start = time.time()
    resp = requests.get(url, params=params, timeout=10)
    elapsed = time.time() - start
    
    if resp.status_code == 200:
        data = resp.json()
        klines = data.get("data", {}).get("klines", [])
        if klines:
            print(f"  [OK] 东方财富K线可用 ({elapsed:.2f}s, {len(klines)}天)")
            results.append(("东方财富K线", "OK", elapsed))
        else:
            print(f"  [ERR] 东方财富K线 返回为空")
            results.append(("东方财富K线", "EMPTY", elapsed))
    else:
        print(f"  [ERR] 东方财富K线 HTTP {resp.status_code}")
        results.append(("东方财富K线", "HTTP_ERROR", elapsed))
except Exception as e:
    print(f"  [FAIL] 东方财富K线: {e}")
    results.append(("东方财富K线", "FAIL", 0))

# ============ 7. Baostock ============
print("\n[7] Baostock...")
try:
    import baostock as bs
    
    start = time.time()
    lg = bs.login()
    if lg.error_code == '0':
        rs = bs.query_history_k_data_plus(
            "sh.600519",
            "date,code,open,high,low,close,volume",
            start_date=(datetime.now() - timedelta(days=70)).strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            frequency="d", adjustflag="2"
        )
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
        elapsed = time.time() - start
        
        if data_list:
            print(f"  [OK] Baostock可用 ({elapsed:.2f}s, {len(data_list)}天)")
            results.append(("Baostock", "OK", elapsed))
        else:
            print(f"  [ERR] Baostock 返回为空")
            results.append(("Baostock", "EMPTY", elapsed))
    else:
        print(f"  [ERR] Baostock 登录失败: {lg.error_msg}")
        results.append(("Baostock", "LOGIN_FAIL", 0))
except ImportError:
    print(f"  [WARN] Baostock 未安装")
    results.append(("Baostock", "NOT_INSTALLED", 0))
except Exception as e:
    print(f"  [FAIL] Baostock: {e}")
    results.append(("Baostock", "FAIL", 0))

# ============ 汇总 ============
print("\n" + "=" * 60)
print("  测试结果汇总")
print("=" * 60)

available = []
for name, status, elapsed in results:
    if status == "OK":
        print(f"  [OK] {name}: {elapsed:.2f}s")
        available.append((name, elapsed))
    elif status in ("NOT_INSTALLED", "NEED_TOKEN"):
        print(f"  [--] {name}: {status}")
    else:
        print(f"  [X]  {name}: {status}")

print("\n" + "-" * 60)
if available:
    available.sort(key=lambda x: x[1])
    print(f"推荐数据源: {available[0][0]} (响应最快)")
    if len(available) > 1:
        print(f"备选: {', '.join([x[0] for x in available[1:]])}")
else:
    print("无可用的数据源!")
print("=" * 60)
