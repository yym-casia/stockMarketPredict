# -*- coding: utf-8 -*-
"""东方财富行业板块 — 回测/实盘统一数据源"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import requests

INDUSTRY_MAP_FILE = 'data/industry_map.json'
EM_BOARDS_FILE = 'data/eastmoney_boards.json'
CACHE_TTL_HOURS = 72
NON_EM_SECTOR = '其他'
EM_HOSTS = (
    'https://17.push2delay.eastmoney.com',
    'https://push2delay.eastmoney.com',
    'https://17.push2.eastmoney.com',
    'https://push2.eastmoney.com',
    'https://29.push2delay.eastmoney.com',
    'https://29.push2.eastmoney.com',
    'https://48.push2.eastmoney.com',
)
EM_UT = 'bd1d9ddb04089700cf9c27f6f7426281'


def _normalize_code(raw) -> str:
    code = str(raw).strip()
    if '.' in code:
        code = code.split('.')[-1]
    return code.zfill(6)


def load_industry_map(path: str = INDUSTRY_MAP_FILE) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'map' in data:
        return data['map']
    return data if isinstance(data, dict) else {}


def save_industry_map(mapping: Dict[str, str], path: str = INDUSTRY_MAP_FILE):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({
            'updated': datetime.now().isoformat(),
            'count': len(mapping),
            'map': mapping,
        }, f, ensure_ascii=False, indent=2)


def _map_is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        updated = meta.get('updated', '')
        if not updated:
            return False
        dt = datetime.fromisoformat(updated)
        return datetime.now() - dt < timedelta(hours=CACHE_TTL_HOURS)
    except Exception:
        return False


def _em_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://quote.eastmoney.com/center/boardlist.html',
    })
    return s


def _em_get_json(path: str, params: dict, retries: int = 8) -> dict:
    last_err = None
    session = _em_session()
    for attempt in range(retries):
        host = EM_HOSTS[attempt % len(EM_HOSTS)]
        try:
            resp = session.get(f'{host}{path}', params=params, timeout=25)
            if resp.status_code in (502, 503, 504):
                raise requests.HTTPError(f'{resp.status_code} gateway', response=resp)
            resp.raise_for_status()
            data = resp.json()
            if data.get('data') is not None:
                return data
        except Exception as e:
            last_err = e
            time.sleep(0.5 + attempt * 0.4)
    raise last_err or RuntimeError('东方财富接口请求失败')


def _em_fetch_paginated(path: str, base_params: dict) -> List[dict]:
    rows: List[dict] = []
    page = 1
    page_size = min(int(base_params.get('pz', 100) or 100), 100)
    while True:
        params = dict(base_params)
        params['pn'] = str(page)
        params['pz'] = str(page_size)
        data = _em_get_json(path, params)
        diff = (data.get('data') or {}).get('diff') or []
        if not diff:
            break
        rows.extend(diff)
        total = int((data.get('data') or {}).get('total') or 0)
        if total and len(rows) >= total:
            break
        if len(diff) < page_size:
            break
        page += 1
        time.sleep(0.15)
    return rows


def _fetch_industry_boards_akshare() -> List[Tuple[str, str]]:
    """akshare 备用：东方财富行业板块列表。"""
    import akshare as ak
    df = ak.stock_board_industry_name_em()
    boards = []
    for _, row in df.iterrows():
        code = str(row.get('板块代码', '')).strip()
        name = str(row.get('板块名称', '')).strip()
        if code and name:
            boards.append((code, name))
    return boards


def _fetch_board_members_akshare(board_name: str) -> List[str]:
    import akshare as ak
    df = ak.stock_board_industry_cons_em(symbol=board_name)
    col = '代码' if '代码' in df.columns else df.columns[1]
    return [_normalize_code(c) for c in df[col].tolist() if c]


def _fetch_industry_boards() -> List[Tuple[str, str]]:
    """返回 [(板块代码, 板块名称), ...]"""
    params = {
        'pz': '200',
        'po': '1',
        'np': '1',
        'ut': EM_UT,
        'fltt': '2',
        'invt': '2',
        'fid': 'f3',
        'fs': 'm:90 t:2 f:!50',
        'fields': 'f12,f14,f3',
    }
    try:
        rows = _em_fetch_paginated('/api/qt/clist/get', params)
    except Exception as e:
        print(f'直连板块列表失败: {e}，尝试 akshare...')
        return _fetch_industry_boards_akshare()
    boards = []
    for r in rows:
        code = str(r.get('f12', '')).strip()
        name = str(r.get('f14', '')).strip()
        if code and name:
            boards.append((code, name))
    return boards


def _fetch_board_members(board_code: str, board_name: str = '') -> List[str]:
    params = {
        'pz': '200',
        'po': '1',
        'np': '1',
        'ut': EM_UT,
        'fltt': '2',
        'invt': '2',
        'fid': 'f3',
        'fs': f'b:{board_code}',
        'fields': 'f12',
    }
    try:
        rows = _em_fetch_paginated('/api/qt/clist/get', params)
        return [_normalize_code(r.get('f12')) for r in rows if r.get('f12')]
    except Exception:
        if board_name:
            return _fetch_board_members_akshare(board_name)
        raise


def _secid(code: str) -> str:
    return f'1.{code}' if code.startswith('6') else f'0.{code}'


def _load_f127_cache() -> Dict[str, str]:
    if os.path.exists(F127_CACHE_FILE):
        with open(F127_CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_f127_cache(cache: Dict[str, str]):
    os.makedirs(os.path.dirname(F127_CACHE_FILE) or '.', exist_ok=True)
    with open(F127_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)


def fetch_stock_f127(code: str, cache: Optional[Dict[str, str]] = None) -> str:
    cache = cache if cache is not None else _load_f127_cache()
    if code in cache:
        return cache[code]
    data = _em_get_json('/api/qt/stock/get', {
        'secid': _secid(code),
        'fields': 'f127',
    }, retries=4)
    val = str((data.get('data') or {}).get('f127') or '').strip()
    cache[code] = val
    return val


F127_BRIDGE_FILE = 'data/f127_board_bridge.json'
F127_CACHE_FILE = 'data/f127_cache.json'


def _load_f127_bridge() -> Dict[str, str]:
    if os.path.exists(F127_BRIDGE_FILE):
        with open(F127_BRIDGE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f).get('map', {})
    return {}


def _save_f127_bridge(bridge: Dict[str, str]):
    os.makedirs(os.path.dirname(F127_BRIDGE_FILE) or '.', exist_ok=True)
    with open(F127_BRIDGE_FILE, 'w', encoding='utf-8') as f:
        json.dump({'updated': datetime.now().isoformat(), 'map': bridge}, f, ensure_ascii=False, indent=2)


def build_f127_board_bridge(mapping: Dict[str, str], sample_limit: int = 0) -> Dict[str, str]:
    """用已映射股票建立 申万二级(f127) -> 东方财富行业板块 桥接表。"""
    from collections import Counter
    bridge_counts: Dict[str, Counter] = {}
    if sample_limit > 0:
        items = list(mapping.items())[:sample_limit]
    else:
        per_board: Dict[str, str] = {}
        for code, board in mapping.items():
            if board not in per_board:
                per_board[board] = code
        items = [(c, b) for b, c in per_board.items()]
    f127_cache = _load_f127_cache()
    for i, (code, board) in enumerate(items):
        try:
            f127 = fetch_stock_f127(code, f127_cache)
            if f127:
                bridge_counts.setdefault(f127, Counter())[board] += 1
        except Exception as e:
            print(f'  桥接样本 {code} 失败: {e}', flush=True)
        time.sleep(0.05)
    _save_f127_cache(f127_cache)
    bridge = {k: v.most_common(1)[0][0] for k, v in bridge_counts.items() if v}
    if bridge:
        _save_f127_bridge(bridge)
    return bridge


def fill_mapping_via_f127(
    codes: List[str],
    mapping: Dict[str, str],
    bridge: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    bridge = bridge or _load_f127_bridge()
    if not bridge and mapping:
        bridge = build_f127_board_bridge(mapping)
    filled = dict(mapping)
    f127_cache = _load_f127_cache()
    for i, code in enumerate(codes):
        if code in filled and filled[code] not in ('', '其他', '扩展池'):
            continue
        try:
            f127 = fetch_stock_f127(code, f127_cache)
            board = bridge.get(f127, '')
            if board:
                filled[code] = board
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            _save_f127_cache(f127_cache)
            print(f'  f127补全进度 {i + 1}/{len(codes)}', flush=True)
        time.sleep(0.03)
    _save_f127_cache(f127_cache)
    return filled


def fetch_eastmoney_industry_map(
    force: bool = False,
    path: str = INDUSTRY_MAP_FILE,
    sleep_sec: float = 0.08,
) -> Dict[str, str]:
    """拉取东方财富行业板块成分股，构建 code -> 板块名称 映射。"""
    if not force and _map_is_fresh(path):
        cached = load_industry_map(path)
        if len(cached) > 500:
            print(f'行业映射缓存命中: {len(cached)} 只')
            return cached

    mapping: Dict[str, str] = load_industry_map(path) if os.path.exists(path) else {}
    print(f'正在拉取东方财富行业板块成分股... (已有 {len(mapping)} 只)')

    try:
        boards = _fetch_industry_boards()
    except Exception as e:
        print(f'行业板块列表获取失败: {e}')
        if mapping:
            print(f'使用本地缓存: {len(mapping)} 只')
            return mapping
        return {}

    total = len(boards)
    print(f'行业板块数: {total}')
    for idx, (board_code, sector) in enumerate(boards):
        try:
            codes = _fetch_board_members(board_code, sector)
            added = 0
            for code in codes:
                if len(code) == 6:
                    mapping[code] = sector
                    added += 1
            print(f'  [{idx + 1}/{total}] {sector}: +{added} 只 (累计 {len(mapping)})', flush=True)
        except Exception as e:
            print(f'  [{idx + 1}/{total}] {sector} 失败: {e}', flush=True)
        if sleep_sec > 0:
            time.sleep(sleep_sec)
        if (idx + 1) % 5 == 0:
            save_industry_map(mapping, path)
            print(f'  >> 检查点 {idx + 1}/{total}, 已映射 {len(mapping)} 只', flush=True)

    if len(mapping) < 2500:
        print('成分股映射不足，尝试 f127 桥接补全...')
        bridge = build_f127_board_bridge(mapping) if mapping else {}
        if bridge:
            print(f'  f127桥接表: {len(bridge)} 条')
    else:
        bridge = _load_f127_bridge()

    if mapping:
        save_industry_map(mapping, path)
        _save_eastmoney_boards(sorted(set(mapping.values())))
        print(f'行业映射已保存: {len(mapping)} 只 -> {path}')
    return mapping


def _save_eastmoney_boards(boards: List[str]):
    os.makedirs(os.path.dirname(EM_BOARDS_FILE) or '.', exist_ok=True)
    with open(EM_BOARDS_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'updated': datetime.now().isoformat(),
            'count': len(boards),
            'boards': boards,
        }, f, ensure_ascii=False, indent=2)


def load_eastmoney_board_names() -> Set[str]:
    """东方财富行业板块标准名称集合。"""
    if os.path.exists(EM_BOARDS_FILE):
        with open(EM_BOARDS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        boards = data.get('boards') or []
        if boards:
            return set(boards)
    return set(load_industry_map().values())


def get_eastmoney_sector(code: str, mapping: Optional[Dict[str, str]] = None) -> str:
    mapping = mapping or load_industry_map()
    return mapping.get(_normalize_code(code), NON_EM_SECTOR)


def is_eastmoney_sector(sector: str, boards: Optional[Set[str]] = None) -> bool:
    if not sector or sector in (NON_EM_SECTOR, '扩展池'):
        return False
    boards = boards or load_eastmoney_board_names()
    return sector in boards


def complete_industry_map_for_pool(
    pool_codes: List[str],
    mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """为股票池补全行业映射（成分股 + f127 桥接）。"""
    mapping = dict(mapping or load_industry_map())
    missing = [c for c in pool_codes if mapping.get(c, '其他') in ('其他', '扩展池', '')]
    if missing:
        print(f'f127 补全缺失行业: {len(missing)} 只')
        mapping = fill_mapping_via_f127(missing, mapping)
        save_industry_map(mapping)
    return mapping


def _base_pool_sector_lookup() -> Dict[str, str]:
    from stock_pool_manager import StockPoolManager
    lookup: Dict[str, str] = {}
    for sector, stocks in StockPoolManager.BASE_POOL.items():
        for code in stocks:
            lookup[code.zfill(6)] = sector
    return lookup


def apply_industry_map_to_pool(
    manager,
    mapping: Optional[Dict[str, str]] = None,
    overwrite_placeholder: bool = True,
    overwrite_all: bool = False,
    eastmoney_only: bool = False,
) -> int:
    """将行业映射写入股票池 stock_sectors。"""
    if mapping is None:
        mapping = load_industry_map()
    if not mapping:
        mapping = fetch_eastmoney_industry_map()

    boards = load_eastmoney_board_names()
    base_lookup = {} if eastmoney_only else _base_pool_sector_lookup()
    placeholders = {'扩展池', NON_EM_SECTOR, '其他', ''}
    updated = 0
    for code in manager.stock_pool:
        old = manager.stock_sectors.get(code, '')
        if overwrite_all or eastmoney_only:
            raw = mapping.get(code)
            new = raw if raw and (not eastmoney_only or raw in boards or not boards) else NON_EM_SECTOR
        elif overwrite_placeholder and old in placeholders:
            new = mapping.get(code) or base_lookup.get(code, NON_EM_SECTOR)
        elif code in base_lookup and old in placeholders:
            new = base_lookup[code]
        else:
            continue
        if eastmoney_only and new not in boards and new != NON_EM_SECTOR:
            new = NON_EM_SECTOR
        if new != old:
            manager.stock_sectors[code] = new
            updated += 1

    manager.sector_stocks = {}
    for code, sector in manager.stock_sectors.items():
        if code in manager.stock_pool:
            manager.sector_stocks.setdefault(sector, []).append(code)

    manager.last_update = datetime.now()
    manager._save_pool()
    return updated


def get_live_sector_boards(top_n: int = 50) -> List[Dict]:
    """从东方财富获取行业板块涨跌幅（按涨幅降序）。"""
    try:
        params = {
            'pz': '200',
            'po': '1',
            'np': '1',
            'ut': EM_UT,
            'fltt': '2',
            'invt': '2',
            'fid': 'f3',
            'fs': 'm:90 t:2 f:!50',
            'fields': 'f14,f3',
        }
        rows = _em_fetch_paginated('/api/qt/clist/get', params)
        sectors = []
        for r in rows:
            name = str(r.get('f14', '')).strip()
            if not name:
                continue
            change = float(r.get('f3', 0) or 0)
            sectors.append({
                'sector': name,
                'change': round(change, 2),
                'momentum_5d': round(change, 2),
                'momentum_score': round(change * 3, 2),
                'count': 0,
            })
        sectors.sort(key=lambda x: x['change'], reverse=True)
        return sectors[:top_n]
    except Exception as e:
        print(f'板块数据获取失败: {e}')
        return []


def build_regime_from_sectors(sectors: List[Dict], cfg: Optional[dict] = None) -> Dict:
    """将板块列表转为 market_regime 兼容结构。"""
    cfg = cfg or {}
    min_sec = cfg.get('mainline_min_sector_change', 0.4)
    hot_sectors: Set[str] = set()
    for m in sectors:
        if m.get('change', 0) >= min_sec and m.get('momentum_5d', 0) > 0:
            hot_sectors.add(m['sector'])

    cold_sectors: Set[str] = set()
    if sectors:
        for m in sectors[-3:]:
            if m.get('change', 0) < -0.5:
                cold_sectors.add(m['sector'])

    return {
        'mainlines': sectors,
        'hot_sectors': hot_sectors,
        'cold_sectors': cold_sectors,
        'regime': 'neutral',
        'can_enter': True,
        'max_positions': 5,
        'score': 50.0,
    }


def get_live_hot_sectors(cfg: Optional[dict] = None) -> Set[str]:
    from src.market_regime import resolve_hot_sectors
    cfg = cfg or {}
    top_n = int(cfg.get('hot_sector_top_n', cfg.get('mainline_top_n', 8)))
    sectors = get_live_sector_boards(top_n=max(top_n, 20))
    regime = build_regime_from_sectors(sectors, cfg)
    return resolve_hot_sectors(regime, cfg)


def get_live_sector_changes() -> Dict[str, float]:
    """板块名 -> 当日涨跌幅。"""
    return {s['sector']: s['change'] for s in get_live_sector_boards(top_n=200)}


def unify_pool_sectors_eastmoney(
    pool_file: str = 'data/stock_pool.json',
    force_fetch: bool = False,
    fill_missing: bool = True,
) -> Dict:
    """统一股票池板块为东方财富行业板块标准。"""
    from stock_pool_manager import get_pool_manager
    manager = get_pool_manager(pool_file)
    pool_codes = list(manager.stock_pool)

    mapping = fetch_eastmoney_industry_map(force=force_fetch)
    if fill_missing:
        missing = [c for c in pool_codes if c not in mapping]
        if missing:
            print(f'东方财富成分股未覆盖 {len(missing)} 只，尝试 f127 桥接补全...')
            try:
                if not _load_f127_bridge():
                    build_f127_board_bridge(mapping)
                mapping = fill_mapping_via_f127(missing, mapping)
                save_industry_map(mapping)
                _save_eastmoney_boards(sorted(set(mapping.values())))
            except Exception as e:
                print(f'f127 补全跳过: {e}')

    updated = apply_industry_map_to_pool(
        manager,
        mapping=mapping,
        overwrite_all=True,
        eastmoney_only=True,
    )
    boards = load_eastmoney_board_names()
    mapped = sum(
        1 for c in pool_codes
        if is_eastmoney_sector(manager.stock_sectors.get(c, ''), boards)
    )
    other = sum(1 for c in pool_codes if manager.stock_sectors.get(c) == NON_EM_SECTOR)
    return {
        'pool_size': len(pool_codes),
        'updated': updated,
        'mapped': mapped,
        'other': other,
        'placeholder': 0,
        'industry_map_size': len(mapping),
        'board_count': len(boards),
    }


def enrich_pool_sectors(
    pool_file: str = 'data/stock_pool.json',
    force_fetch: bool = False,
    overwrite_all: bool = False,
    eastmoney_only: bool = False,
) -> Dict:
    if eastmoney_only or overwrite_all:
        return unify_pool_sectors_eastmoney(pool_file, force_fetch=force_fetch)
    from stock_pool_manager import get_pool_manager
    manager = get_pool_manager(pool_file)
    mapping = fetch_eastmoney_industry_map(force=force_fetch)
    updated = apply_industry_map_to_pool(
        manager,
        mapping=mapping,
        overwrite_placeholder=True,
        overwrite_all=overwrite_all,
    )
    pool_codes = list(manager.stock_pool)
    boards = load_eastmoney_board_names()
    mapped = sum(
        1 for c in pool_codes
        if is_eastmoney_sector(manager.stock_sectors.get(c, ''), boards)
    )
    placeholder = sum(1 for c in pool_codes if manager.stock_sectors.get(c) == '扩展池')
    return {
        'pool_size': len(pool_codes),
        'updated': updated,
        'mapped': mapped,
        'other': sum(1 for c in pool_codes if manager.stock_sectors.get(c) == NON_EM_SECTOR),
        'placeholder': placeholder,
        'industry_map_size': len(mapping),
        'board_count': len(boards),
    }
