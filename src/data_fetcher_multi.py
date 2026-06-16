# -*- coding: utf-8 -*-
"""
多数据源股票数据获取模块
主要使用: 新浪财经 (历史K线)
备用: Baostock
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

from src.history_fetcher import get_history_fetcher


class SinaHistoryFetcher:
    """新浪财经 - 历史K线数据"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Referer': 'https://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_stock_history(self, stock_code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取股票历史K线（腾讯→新浪→akshare 多源回退）"""
        return get_history_fetcher().get_history(stock_code, days=days)

    def fetch_history_data(self, stock_codes: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """批量获取历史数据（带缓存，顺序请求防封）"""
        print(f"正在获取 {len(stock_codes)} 只股票的历史数据...")
        result = get_history_fetcher().fetch_batch(stock_codes, days=days)
        print(f"成功获取 {len(result)} 只股票数据")
        return result


class SinaFinanceFetcher:
    """新浪财经 - 实时行情"""
    
    def __init__(self):
        self.base_url = "https://hq.sinajs.cn/list="
        self.headers = {
            "Referer": "https://finance.sina.com.cn",
            "User-Agent": "Mozilla/5.0"
        }
    
    def get_realtime_quotes(self, stock_codes: List[str]) -> pd.DataFrame:
        """获取实时行情"""
        # 转换代码格式
        codes = []
        for code in stock_codes:
            code = code.replace('.SH', '').replace('.SZ', '')
            if code.startswith('6'):
                codes.append(f"sh{code}")
            else:
                codes.append(f"sz{code}")
        
        url = self.base_url + ','.join(codes)
        
        try:
            resp = requests.get(url, headers=self.headers, timeout=10)
            if resp.status_code != 200 or not resp.text:
                return pd.DataFrame()
            
            lines = resp.text.strip().split('\n')
            stocks = []
            
            for line in lines:
                if '="' not in line:
                    continue
                
                code = line.split('=')[0].split('_')[-1]
                data = line.split('"')[1]
                
                if not data:
                    continue
                
                parts = data.split(',')
                if len(parts) < 6:
                    continue
                
                try:
                    price = float(parts[3]) if parts[3] else 0
                    last_close = float(parts[2]) if parts[2] else 0
                    change = round((price - last_close) / last_close * 100, 2) if last_close > 0 else 0
                    stocks.append({
                        'code': code,
                        'name': parts[0],
                        'open': float(parts[1]) if parts[1] else 0,
                        'last_close': last_close,
                        'price': price,
                        'change': change,
                        'high': float(parts[4]) if parts[4] else 0,
                        'low': float(parts[5]) if parts[5] else 0,
                        'volume': float(parts[8]) if len(parts) > 8 and parts[8] else 0,
                    })
                except:
                    continue
            
            return pd.DataFrame(stocks)
            
        except Exception as e:
            print(f"获取实时行情失败: {e}")
            return pd.DataFrame()
    
    def get_popularity_rank(self) -> Dict[str, int]:
        """获取股票人气排名
        
        Returns:
            Dict[股票代码, 排名名次] - 排名越小越热门
        """
        try:
            # 新浪财经资金流向排行
            url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getNameList"
            params = {
                "page": 1,
                "num": 1000,  # 获取前1000只
                "node": "hs_a"
            }
            
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                return {}
            
            import json
            data = json.loads(resp.text)
            
            if not isinstance(data, list):
                return {}
            
            # 返回 code -> rank 的映射 (注意返回的是symbol)
            rank_dict = {}
            for i, item in enumerate(data):
                symbol = item.get('symbol', '')  # 格式: sh600519
                if symbol:
                    # 转换为标准代码格式
                    code = symbol.replace('sh', '').replace('sz', '')
                    rank_dict[code] = i + 1
            
            return rank_dict
            
        except Exception as e:
            print(f"获取人气排名失败: {e}")
            return {}


class MultiSourceDataFetcher:
    """多数据源统一接口 - 主要使用新浪财经"""
    
    def __init__(self):
        self.sina_history = SinaHistoryFetcher()
        self.sina_realtime = SinaFinanceFetcher()
        self._popularity_rank = None
    
    def get_popularity_rank(self) -> Dict[str, int]:
        """获取股票人气排名"""
        if self._popularity_rank is None:
            self._popularity_rank = self.sina_realtime.get_popularity_rank()
        return self._popularity_rank
    
    def get_stock_list(self, top_n: int = 500) -> List[str]:
        """获取股票列表 - 涨幅筛选策略
        
        策略：涨幅2%-5%的100只 + 涨幅>9.9%的全部
        Returns:
            List[str]: 股票代码列表
        """
        # 获取涨幅筛选的股票池
        print("  获取涨幅筛选股票池...")
        
        # 先获取涨幅筛选的股票
        gain_data = self._get_gaining_stocks_with_change()
        
        if gain_data:
            stock_codes = list(gain_data.keys())
            print(f"  涨幅筛选获取到 {len(stock_codes)} 只股票")
            # 保存到实例变量，供外部获取涨跌幅数据
            self._stock_change_map = gain_data
            return stock_codes
        
        # 如果失败，使用资金流向策略
        print("  涨幅筛选失败，使用资金流向策略")
        money_flow_stocks = self._get_money_flow_stocks(top_n)
        if money_flow_stocks:
            self._stock_change_map = {}
            return money_flow_stocks
        
        # 最终备用
        print("  使用备用列表")
        self._stock_change_map = {}
        return self._get_stock_list_fallback(top_n)
    
    def get_stock_change_map(self) -> Dict[str, float]:
        """获取股票涨跌幅映射表（涨幅>9.9%的优先，人气更高）"""
        return getattr(self, '_stock_change_map', {})
    
    def _get_extended_stock_list(self, count: int) -> List[str]:
        """获取扩展股票列表 - 涨幅筛选策略
        
        策略：涨幅2%-5%的100只 + 涨幅>9.9%的全部（不足50只则全部选中）
        """
        print("  获取涨幅筛选股票池...")
        
        # 获取涨幅2%-5%和>9.9%的股票
        gain_stocks = self._get_gaining_stocks()
        
        if gain_stocks:
            print(f"  涨幅筛选获取到 {len(gain_stocks)} 只股票")
            return gain_stocks
        
        # 如果失败，使用备用列表
        print("  涨幅筛选失败，使用资金流向策略")
        return self._get_money_flow_stocks(count)
    
    def _get_gaining_stocks_with_change(self) -> Dict[str, float]:
        """获取涨幅2%-5%和>9.9%的股票池（使用新浪财经）
        
        - 涨幅2%-5%: 选100只
        - 涨幅>9.9%: 全部选中（不足50只则全部）
        
        Returns:
            Dict[code, change]: 股票代码 -> 涨跌幅
        """
        session = requests.Session()
        session.headers.update({
            'Referer': 'https://finance.sina.com.cn',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        all_stocks = []
        
        # 分页获取涨幅排行，每页100只，获取足够多的数据
        for page in range(1, 20):  # 最多20页（2000只）
            url = f'https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page={page}&num=100&node=hs_a&sort=changepercent&asc=0'
            
            try:
                resp = session.get(url, timeout=10)
                data = resp.json()
                if not isinstance(data, list) or not data:
                    break
                all_stocks.extend(data)
                
                # 如果这一页涨幅都<2%，可以提前结束
                last_change = data[-1].get('changepercent', 0) if data else 0
                if last_change < 2:
                    break
                    
            except Exception as e:
                print(f"    获取第{page}页失败")
                break
        
        if not all_stocks:
            print("    获取涨幅数据失败")
            return {}
        
        # 转换代码格式，并建立涨跌幅映射
        def get_code_and_change(s):
            symbol = s.get('symbol', '')
            change = s.get('changepercent', 0) or 0
            # 去掉前缀 sh/sz/bj
            if symbol.startswith('sh') or symbol.startswith('sz'):
                return symbol[2:], change
            return None, None
        
        # 筛选涨幅2%-5%
        range_2_5 = []
        # 筛选涨幅>9.9%
        above_10 = []
        
        for s in all_stocks:
            code, change = get_code_and_change(s)
            if not code:
                continue
            # 排除北交所、新股、创业板(300/301)、科创板(688)
            if code.startswith('bj') or code.startswith('N') or code.startswith('300') or code.startswith('301') or code.startswith('688'):
                continue
            
            if 2 <= change <= 5:
                range_2_5.append((code, change))
            elif change > 9.9:
                above_10.append((code, change))
        
        print(f"    涨幅2%-5%: {len(range_2_5)} 只")
        print(f"    涨幅>9.9%: {len(above_10)} 只")
        
        # 构建股票池: 涨幅2%-5%的100只 + 涨幅>9.9%的全部
        pool_2_5 = range_2_5[:100]
        pool_above_10 = above_10 if len(above_10) >= 50 else above_10
        
        # 构建结果字典
        result = {}
        for code, change in pool_2_5:
            result[code] = change
        for code, change in pool_above_10:
            result[code] = change
        
        print(f"    最终股票池: {len(result)} 只 ({len(pool_2_5)}只2%-5% + {len(pool_above_10)}只>9.9%)")
        
        return result
    
    def _get_money_flow_stocks(self, top_n: int) -> List[str]:
        """根据资金净流入获取热门股票
        
        热门板块策略：获取资金净流入排名靠前的行业板块，然后获取板块内资金净流入排名靠前的个股
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://data.eastmoney.com'
        }
        
        all_stocks = []
        
        # 1. 获取行业板块资金净流入排名
        sector_url = 'https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=50&po=1&np=1&fltt=2&invt=2&fid=f62&fs=m:90+t:2&fields=f12,f14,f62,f184'
        
        try:
            resp = requests.get(sector_url, headers=headers, timeout=10)
            data = resp.json()
            if data and 'data' in data and data['data']:
                sectors = data['data']['diff']
                # 按主力净流入排序，取前10个热门板块
                sorted_sectors = sorted(sectors, key=lambda x: x.get('f62', 0) or 0, reverse=True)
                hot_sectors = sorted_sectors[:10]
                print(f"    热门板块: {[s.get('f14','') for s in hot_sectors]}")
        except Exception as e:
            print(f"    获取行业板块失败: {e}")
            hot_sectors = []
        
        # 2. 获取上海A股和深圳A股的资金净流入排名
        market_params = ['m:0+t:6', 'm:1+t:6']
        
        for market in market_params:
            url = f'https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f62&fs={market}&fields=f12,f14,f2,f3,f62,f184'
            
            try:
                resp = requests.get(url, headers=headers, timeout=10)
                data = resp.json()
                if data and 'data' in data and data['data']:
                    stocks = data['data']['diff']
                    # 按主力净流入排序
                    sorted_stocks = sorted(stocks, key=lambda x: x.get('f62', 0) or 0, reverse=True)
                    # 取前50只资金净流入最多的股票
                    for s in sorted_stocks[:50]:
                        code = s.get('f12', '')
                        if code.startswith('6') or code.startswith('0') or code.startswith('3'):
                            all_stocks.append(code)
            except Exception as e:
                print(f"    获取{market}资金流向失败: {e}")
        
        # 去重
        all_stocks = list(dict.fromkeys(all_stocks))
        print(f"    从资金净流入获取到 {len(all_stocks)} 只热门股票")
        return all_stocks[:top_n]
    
    def _get_popular_stocks_by_performance(self, stock_codes: List[str]) -> List[str]:
        """根据近期涨跌幅获取热门股票（涨幅>30%）
        
        热门板块定义：板块内股票上涨超过30%，并且存在连扳股票
        """
        popular = []
        
        # 使用Baostock获取少量样本的涨跌幅
        import baostock as bs
        lg = bs.login()
        if lg.error_code != '0':
            return []
        
        # 抽样获取30只股票的5日涨幅
        sample_size = min(30, len(stock_codes))
        sample_codes = stock_codes[:sample_size]
        
        for code in sample_codes:
            try:
                # 格式转换
                bs_code = code
                if code.startswith('6'):
                    bs_code = 'sh.' + code
                elif code.startswith('0'):
                    bs_code = 'sz.' + code
                
                rs = bs.query_history_k_data_plus(
                    bs_code,
                    "date,code,close,turn",
                    start_date=bs.format_date(5),  # 5天前
                    end_date=bs.format_date(0),
                    frequency="d",
                    adjustflag="3"
                )
                
                if rs.error_code == '0' and rs.next():
                    data_list = []
                    while rs.next():
                        data_list.append(rs.get_row_data())
                    
                    if len(data_list) >= 2:
                        first_close = float(data_list[0][2])
                        last_close = float(data_list[-1][2])
                        change_pct = (last_close - first_close) / first_close * 100
                        
                        if change_pct > 30:
                            popular.append(code)
            except:
                pass
        
        bs.logout()
        
        return popular
    
    def _get_stock_list_by_performance(self, popular_codes: List[str], top_n: int) -> List[str]:
        """根据热门程度分配股票列表"""
        n_popular = top_n // 2
        n_other = top_n - n_popular
        
        # 如果热门股票不够，从全部股票中补充
        all_stocks = self._get_stock_list_fallback(top_n * 2)
        
        # 热门股票
        popular = popular_codes[:n_popular] if popular_codes else all_stocks[:n_popular]
        
        # 其他股票（不热门的）
        other_pool = [s for s in all_stocks if s not in popular]
        other = other_pool[:n_other]
        
        result = popular + other
        
        print(f"  股票池分配: 热门股票 {len(popular)} 只, 其他股票 {len(other)} 只")
        
        return result
    
    def _get_stock_list_fallback(self, top_n: int = 100) -> List[str]:
        """备用股票列表 - 按行业分布"""
        stocks = [
            # 银行
            '600036', '600000', '601166', '601398', '601939', '600015', '600016', '601288', '601818', '601328',
            '600919', '601229', '600926', '601577', '600958', '600928', '601077', '600865', '600816', '600742',
            # 券商
            '600030', '601788', '601688', '600999', '000776', '002736', '601555', '600837', '601099', '601375',
            '600369', '000712', '002500', '601198', '000686',
            # 保险
            '601318', '601319', '601601', '601336', '601628', '601899', '000069', '601111',
            # 消费
            '600519', '000858', '000568', '600809', '600887', '600197', '603589', '603719', '000596', '000798',
            '002304', '002507', '002558', '000729', '600199', '600059', '000559', '603288',
            # 医药
            '600276', '600518', '000566', '002044', '300003', '002007', '002038', '000423', '000513', '600535',
            '600529', '603939', '002262', '300015', '002294', '000403', '002424', '002421', '000028', '000538',
            # 科技
            '600703', '000063', '002475', '002230', '002236', '300033', '300676', '002410', '002049', '002185',
            '002156', '000100', '000725', '600183', '603160', '002463', '002371', '300750',
            # 新能源
            '600438', '601012', '600392', '002594', '300750', '002466', '002460', '603799', '002812', '300014',
            '002074', '002129', '600468', '600405', '002202', '600905', '601865', '600111',
            # 基建
            '601668', '601390', '601800', '601186', '601618', '600170', '600028', '601669',
            # 电力
            '600900', '600795', '600021', '600027', '600011', '600863', '600508', '600101', '601991',
            # 化工
            '600096', '600141', '600486', '600309', '600273', '600409', '600352', '600426', '002601', '002709',
            # 地产
            '600048', '000002', '001979', '601155', '600340', '600383', '600325', '000069',
            # 半导体
            '688981', '688256', '688396', '688536', '688317', '688008', '688012', '688126', '688200', '688300',
            # 军工
            '600893', '600760', '600038', '600316', '601989', '600372', '000547', '002013', '600855', '600184',
            # 传媒
            '600637', '600880', '002027', '300033', '002292', '300251', '002624', '300459',
            # 农业
            '600108', '600195', '600251', '600354', '600359', '600127', '000713', '002124',
            # 环保
            '600388', '600526', '601827', '600187', '600217', '002310', '300070', '300187',
            # 半导体/芯片
            '688981', '688256', '688396', '688536', '688317', '688008', '688012', '688126', '688200', '688300',
            '002371', '002185', '002409', '002049', '002475',
        ]
        
        # 去重
        stocks = list(dict.fromkeys(stocks))
        return stocks[:top_n]
    
    def get_stock_history(self, stock_code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取历史数据（多源回退 + 缓存）"""
        return get_history_fetcher().get_history(stock_code, days=days)
    
    @staticmethod
    def calc_change(price: float, last_close: float) -> float:
        """计算今日涨跌幅(%)"""
        if last_close > 0 and price > 0:
            return round((price - last_close) / last_close * 100, 2)
        return 0.0

    def get_realtime_quotes(self, stock_codes: List[str]) -> pd.DataFrame:
        """获取实时行情（含今日涨跌幅）"""
        df = self.sina_realtime.get_realtime_quotes(stock_codes)
        if df is not None and not df.empty and 'change' not in df.columns:
            df['change'] = df.apply(
                lambda r: self.calc_change(r.get('price', 0), r.get('last_close', 0)), axis=1
            )
        return df
    
    def fetch_history_data(self, stock_codes: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """批量获取历史数据"""
        return self.sina_history.fetch_history_data(stock_codes, days)
    
    def get_stock_name(self, stock_code: str) -> str:
        """获取股票名称
        
        Args:
            stock_code: 股票代码
        
        Returns:
            股票名称，如果获取失败则返回股票代码
        """
        try:
            # 转换代码格式
            code = stock_code.replace('.SH', '').replace('.SZ', '')
            if code.startswith('6'):
                sina_code = f"sh{code}"
            else:
                sina_code = f"sz{code}"
            
            url = f"https://hq.sinajs.cn/list={sina_code}"
            headers = {
                'Referer': 'https://finance.sina.com.cn',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            resp = requests.get(url, headers=headers, timeout=5)
            # Sina API返回GBK编码
            resp.encoding = 'gbk'
            
            text = resp.text
            # 格式: var hq_str_sh600519="名称,当前价,...";
            if '"' in text:
                data = text.split('"')[1]
                name = data.split(',')[0]
                return name if name else stock_code
        except:
            pass
        
        return stock_code
    
    def close(self):
        """关闭连接"""
        pass


if __name__ == "__main__":
    # 测试
    fetcher = MultiSourceDataFetcher()
    
    print("=" * 60)
    print("  数据获取测试")
    print("=" * 60)
    
    # 测试获取历史数据
    print("\n[1] 获取历史数据...")
    codes = ['600519', '000001', '600036', '601318']
    for code in codes:
        df = fetcher.get_stock_history(code, 60)
        if df is not None:
            print(f"  {code}: {len(df)}天数据, 最新收盘价: {df['close'].iloc[-1]:.2f}")
        else:
            print(f"  {code}: 获取失败")
    
    # 测试批量获取
    print("\n[2] 批量获取...")
    data = fetcher.fetch_history_data(codes[:5], 60)
    print(f"  成功获取 {len(data)} 只股票")
    
    print("\n测试完成!")
