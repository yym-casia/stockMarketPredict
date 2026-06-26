# -*- coding: utf-8 -*-
"""股票推荐系统 Dashboard 服务（含定时任务 + 盘中告警）

注意: 本进程不加载 akshare，避免 py_mini_racer 多进程崩溃。
所有数据获取通过独立子进程执行。
"""
import http.server
import json
import os
import sys
import subprocess
import threading
import time
import urllib.parse
from datetime import datetime, time as dt_time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PORT = 8088
DAILY_SCRIPT = 'run_daily.py'
SCREEN_SCRIPT = 'run_screen_now.py'
INTRADAY_SCRIPT = 'intraday_check.py'
BACKTEST_SCRIPT = 'run_backtest_capital.py'
SCHEDULED_HOUR = 15
SCHEDULED_MINUTE = 0
INTRADAY_INTERVAL = 300
CHECK_INTERVAL = 60
TRADING_CACHE = os.path.join('data', 'trading_dates_cache.json')

_script_lock = threading.Lock()
_task_state = {
    'running': False,
    'task': None,
    'label': '',
    'started_at': None,
    'finished_at': None,
    'success': None,
    'message': '就绪',
    'log_tail': '',
}
ACTION_SCRIPTS = {
    'screen': (SCREEN_SCRIPT, '收盘后选股(回测同款)'),
    'daily': (DAILY_SCRIPT, '每日分析'),
    'intraday': (INTRADAY_SCRIPT, '盘中检查'),
    'backtest': (BACKTEST_SCRIPT, '资金回测'),
}
# 收盘选股/每日分析需拉取3000股并算ML特征，默认10分钟易超时
TASK_TIMEOUTS = {
    'screen': 3600,
    'daily': 3600,
    'intraday': 300,
    'backtest': 3600,
}


def _is_weekday(dt: datetime) -> bool:
    return dt.weekday() < 5


def _is_trading_day_local(dt: datetime) -> bool:
    """仅用本地缓存判断，不 import akshare"""
    if not _is_weekday(dt):
        return False
    if not os.path.exists(TRADING_CACHE):
        return True
    try:
        with open(TRADING_CACHE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        dates = set(cache.get(str(dt.year), []))
        if dates:
            return dt.strftime('%Y-%m-%d') in dates
    except Exception:
        pass
    return True


def _is_market_hours_local(dt: datetime) -> bool:
    if not _is_trading_day_local(dt):
        return False
    t = dt.time()
    morning = dt_time(9, 30) <= t <= dt_time(11, 30)
    afternoon = dt_time(13, 0) <= t <= dt_time(15, 0)
    return morning or afternoon


def _already_ran_today(marker_file: str) -> bool:
    today_str = datetime.now().strftime('%Y-%m-%d')
    if os.path.exists(marker_file):
        with open(marker_file, 'r') as f:
            if f.read().strip() == today_str:
                return True
    return False


def _mark_ran_today(marker_file: str):
    os.makedirs(os.path.dirname(marker_file), exist_ok=True)
    with open(marker_file, 'w') as f:
        f.write(datetime.now().strftime('%Y-%m-%d'))


def _run_script(script: str, label: str, timeout: int = 3600) -> dict:
    """串行执行子进程，避免多个 akshare 进程同时运行"""
    if not _script_lock.acquire(blocking=False):
        msg = f'另一个任务正在运行，请稍后再试'
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 跳过 {script}（{msg}）")
        return {'ok': False, 'message': msg, 'log_tail': ''}
    try:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}] 开始执行 {script} ({label})...")
        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NO_WINDOW
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=timeout,
            env=env,
            creationflags=creationflags,
        )
        out = (result.stdout or '').strip()
        err = (result.stderr or '').strip()
        tail = '\n'.join(out.split('\n')[-8:]) if out else err[-500:]
        if result.returncode == 0:
            print(f"[{now}] {script} 执行成功")
            if out:
                for line in out.split('\n')[-5:]:
                    print(f"  {line}")
            return {'ok': True, 'message': f'{label}完成', 'log_tail': tail}
        print(f"[{now}] {script} 执行失败 (code={result.returncode})")
        if err:
            print(f"  错误: {err[-500:]}")
        return {'ok': False, 'message': f'{label}失败 (code={result.returncode})', 'log_tail': tail}
    except subprocess.TimeoutExpired:
        msg = f'{label}执行超时'
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {script} {msg}")
        return {'ok': False, 'message': msg, 'log_tail': ''}
    except Exception as e:
        msg = f'{label}异常: {e}'
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {script} {msg}")
        return {'ok': False, 'message': msg, 'log_tail': ''}
    finally:
        _script_lock.release()


def _start_action(task_key: str) -> dict:
    """后台启动任务，供 API 调用"""
    if task_key not in ACTION_SCRIPTS:
        return {'ok': False, 'message': f'未知任务: {task_key}'}
    if _task_state['running']:
        return {'ok': False, 'message': '已有任务在执行中', 'task': _task_state['task']}

    script, label = ACTION_SCRIPTS[task_key]
    timeout = TASK_TIMEOUTS.get(task_key, 3600)

    def worker():
        _task_state.update({
            'running': True,
            'task': task_key,
            'label': label,
            'started_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'finished_at': None,
            'success': None,
            'message': f'{label}执行中...',
            'log_tail': '',
        })
        result = _run_script(script, label, timeout=timeout)
        _task_state.update({
            'running': False,
            'finished_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'success': result['ok'],
            'message': result['message'],
            'log_tail': result.get('log_tail', ''),
        })

    threading.Thread(target=worker, daemon=True).start()
    return {'ok': True, 'message': f'已启动{label}', 'task': task_key}


def run_daily_pipeline():
    """收盘后执行完整每日分析"""
    while True:
        try:
            now = datetime.now()
            if not _is_trading_day_local(now):
                time.sleep(CHECK_INTERVAL)
                continue

            target = now.replace(hour=SCHEDULED_HOUR, minute=SCHEDULED_MINUTE,
                                 second=0, microsecond=0)
            if now >= target:
                marker = os.path.join('data', '.last_pipeline_run')
                if not _already_ran_today(marker):
                    _run_script(DAILY_SCRIPT, '收盘分析', timeout=TASK_TIMEOUTS['daily'])
                    _mark_ran_today(marker)
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"[定时任务] 异常: {e}")
            time.sleep(CHECK_INTERVAL)


def run_intraday_alerts():
    """交易时段内定期检查卖出信号"""
    while True:
        try:
            now = datetime.now()
            if _is_trading_day_local(now) and _is_market_hours_local(now):
                if now.hour == SCHEDULED_HOUR and now.minute >= SCHEDULED_MINUTE - 5:
                    time.sleep(CHECK_INTERVAL)
                    continue
                _run_script(INTRADAY_SCRIPT, '盘中告警', timeout=TASK_TIMEOUTS['intraday'])
                time.sleep(INTRADAY_INTERVAL)
            else:
                time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"[盘中告警] 异常: {e}")
            time.sleep(CHECK_INTERVAL)


def _parse_subprocess_json(stdout: str):
    raw = (stdout or '').strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for line in reversed(raw.splitlines()):
        s = line.strip()
        if s.startswith('{') and s.endswith('}'):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue
    return None


def _fetch_kline_json(code: str, start_date: str, days: str = '30') -> dict:
    """子进程拉 K 线，避免 Dashboard 主进程加载 akshare。"""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'trade_kline.py')
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
    try:
        result = subprocess.run(
            [sys.executable, script, code, start_date, str(days)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=45,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            creationflags=creationflags,
        )
        parsed = _parse_subprocess_json(result.stdout)
        if parsed is not None:
            return parsed
        err = (result.stderr or '').strip() or (result.stdout or '').strip()
        if not err:
            err = f'exit {result.returncode}'
        return {'ok': False, 'error': err[:500]}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'K线拉取超时'}
    except json.JSONDecodeError as e:
        return {'ok': False, 'error': f'K线数据解析失败: {e}'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


class Handler(http.server.SimpleHTTPRequestHandler):
    API_ROUTES = {
        'api/tracking': os.path.join('data', 'stock_tracking.json'),
        'api/operations': os.path.join('data', 'daily_operations.json'),
        'api/portfolio': os.path.join('data', 'portfolio.json'),
        'api/backtest': os.path.join('data', 'backtest_capital_results.json'),
    }
    STATIC_FILES = {
        'dashboard.html', 'backtest_viz.html',
    }

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        path = self.path.lstrip('/').split('?')[0]
        if path.startswith('api/action/'):
            task_key = path.split('/')[-1]
            result = _start_action(task_key)
            self._send_json(result, 200 if result.get('ok') else 409)
            return
        self._send_json({'ok': False, 'message': 'Not found'}, 404)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.lstrip('/')
        if path == 'api/task/status':
            self._send_json(dict(_task_state))
            return

        if path == 'api/kline':
            q = urllib.parse.parse_qs(parsed.query)
            code = (q.get('code') or [''])[0].strip()
            start = (q.get('start') or [''])[0].strip()
            days = (q.get('days') or ['30'])[0].strip() or '30'
            if not code or not start:
                self._send_json({'ok': False, 'error': '缺少参数 code 或 start'}, 400)
                return
            self._send_json(_fetch_kline_json(code, start, days))
            return

        if path == '' or path == 'dashboard.html':
            path = 'dashboard.html'
        elif path in self.API_ROUTES:
            path = self.API_ROUTES[path]

        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()

            ct = 'application/json; charset=utf-8' if path.endswith('.json') else 'text/html; charset=utf-8'
            self.send_response(200)
            self.send_header('Content-Type', ct)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Content-Length', str(len(content.encode('utf-8'))))
            self.end_headers()
            self.wfile.write(content.encode('utf-8'))
        except FileNotFoundError:
            if path in self.API_ROUTES or path.endswith('.json'):
                self._send_json({
                    'error': 'file_not_found',
                    'path': path,
                    'hint': '请在前端点击「运行回测」或「重新选股」生成数据',
                }, 404)
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                self.wfile.write(b'File not found')

    def log_message(self, fmt, *args):
        pass


if __name__ == '__main__':
    threading.Thread(target=run_daily_pipeline, daemon=True).start()
    threading.Thread(target=run_intraday_alerts, daemon=True).start()

    print(f'定时任务: 交易日 {SCHEDULED_HOUR}:{SCHEDULED_MINUTE:02d} 执行 {DAILY_SCRIPT}')
    print(f'盘中告警: 交易时段每 {INTRADAY_INTERVAL // 60} 分钟检查卖出信号')
    print(f'注意: Dashboard 进程不加载 akshare，数据任务在独立子进程中运行')
    bind_host = '127.0.0.1'
    print(f'Serving on http://{bind_host}:{PORT}/dashboard.html')
    print(f'回测可视化: http://{bind_host}:{PORT}/backtest_viz.html')
    #print(f'外网访问: http://82.157.98.97:{PORT}/dashboard.html')
    print(f'前端操作: 刷新 / 重新选股 / 每日分析 / 盘中检查 / 运行回测')
    print('提示: 须保持本窗口运行，Ctrl+C 会停止服务导致浏览器「拒绝连接」')
    Handler.allow_reuse_address = True
    try:
        httpd = http.server.HTTPServer((bind_host, PORT), Handler)
        httpd.serve_forever()
    except OSError as e:
        print(f'❌ 端口 {PORT} 启动失败: {e}')
        print(f'   可结束占用进程后重试，或修改 serve_dashboard.py 中 PORT')
        sys.exit(1)
