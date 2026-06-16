# 数据获取模块

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time


class NeoDataFetcher:
    """NeoData 金融数据获取类"""
    
    def __init__(self, proxy_port: int = 19000):
        self.base_url = f"http://localhost:{proxy_port}/proxy/api"
        self.remote_url = "https://jprx.m.qq.com/aizone/skillserver/v1/proxy/teamrouter_neodata/query"
        self.headers = {
            "Remote-URL": self.remote_url,
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
    
    def query(self, query_text: str, data_type: str = "all") -> Dict:
        """执行自然语言查询"""
        payload = {
            "channel": "neodata",
            "sub_channel": "qclaw",
            "query": query_text,
            "request_id": f"req_{int(time.time() * 1000)}",
            "data_type": data_type
        }
        
        try:
            response = self.session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.encoding = 'utf-8'
            result = response.json()
            return result
        except Exception as e:
            print(f"查询失败: {e}")
            return {"code": "error", "msg": str(e)}
    
    def get_stock_history(self, stock_code: str, days: int = 60) -> Optional[pd.DataFrame]:
        """获取股票历史行情数据"""
        query = f"{stock_code}最近{days}天行情数据"
        result = self.query(query, data_type="api")
        
        if result.get("code") != "200":
            print(f"获取{stock_code}数据失败: {result.get('msg')}")
            return None
        
        # 解析API返回数据
        api_data = result.get("data", {}).get("apiData", {})
        api_recall = api_data.get("apiRecall", [])
        
        if not api_recall:
            print(f"{stock_code}暂无历史数据")
            return None
        
        # 提取行情数据
        for item in api_recall:
            if item.get("type") == "basic_info":
                content = item.get("content", {})
                # 解析行情数据
                if isinstance(content, dict):
                    # 尝试提取K线数据
                    kline_data = content.get("kline", content.get("行情", None))
                    if kline_data:
                        df = pd.DataFrame(kline_data)
                        return df
        
        return None
    
    def get_realtime_quote(self, stock_code: str) -> Optional[Dict]:
        """获取股票实时行情"""
        query = f"{stock_code}最新行情"
        result = self.query(query, data_type="api")
        
        if result.get("code") != "200":
            return None
        
        api_data = result.get("data", {}).get("apiData", {})
        api_recall = api_data.get("apiRecall", [])
        
        for item in api_recall:
            if item.get("type") == "basic_info":
                return item.get("content", {})
        
        return None
    
    def get_limit_up_stocks(self) -> List[Dict]:
        """获取今日涨停股票列表"""
        query = "今日涨停股票列表"
        result = self.query(query, data_type="all")
        
        if result.get("code") != "200":
            return []
        
        stocks = []
        doc_data = result.get("data", {}).get("docData", {})
        doc_recall = doc_data.get("docRecall", [])
        
        for group in doc_recall:
            for doc in group.get("docList", []):
                content = doc.get("content", "")
                # 解析涨停股票信息
                # 这里需要根据实际返回格式解析
                stocks.append({
                    "title": doc.get("title", ""),
                    "content": content,
                    "time": doc.get("publishTime", 0)
                })
        
        return stocks
    
    def get_hot_stocks(self, limit: int = 50) -> List[Dict]:
        """获取热门股票（涨幅榜）"""
        query = f"A股涨幅榜前{limit}名"
        result = self.query(query, data_type="api")
        
        if result.get("code") != "200":
            return []
        
        api_data = result.get("data", {}).get("apiData", {})
        entity_list = api_data.get("entity", [])
        
        return entity_list


class DataFetcher:
    """数据获取主类"""
    
    def __init__(self, source: str = "neodata"):
        self.source = source
        if source == "neodata":
            self.fetcher = NeoDataFetcher()
        else:
            raise ValueError(f"不支持的数据源: {source}")
    
    def get_stock_list(self) -> List[str]:
        """获取股票列表"""
        # 简化版本：返回一些常见股票代码
        # TODO: 从数据源获取完整股票列表
        return [
            "000001.SZ",  # 平安银行
            "000002.SZ",  # 万科A
            "000063.SZ",  # 中兴通讯
            "000333.SZ",  # 美的集团
            "000651.SZ",  # 格力电器
            "000858.SZ",  # 五粮液
            "002415.SZ",  # 海康威视
            "002594.SZ",  # 比亚迪
            "300059.SZ",  # 东方财富
            "300750.SZ",  # 宁德时代
            "600000.SH",  # 浦发银行
            "600036.SH",  # 招商银行
            "600519.SH",  # 贵州茅台
            "600887.SH",  # 伊利股份
            "601318.SH",  # 中国平安
        ]
    
    def fetch_history_data(self, stock_list: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """批量获取历史数据"""
        data = {}
        total = len(stock_list)
        
        for i, stock_code in enumerate(stock_list):
            print(f"正在获取 {stock_code} 数据 ({i+1}/{total})...")
            df = self.fetcher.get_stock_history(stock_code, days)
            if df is not None and not df.empty:
                data[stock_code] = df
            time.sleep(0.5)  # 避免请求过快
        
        print(f"成功获取 {len(data)} 只股票的数据")
        return data
    
    def save_data(self, data: Dict[str, pd.DataFrame], path: str):
        """保存数据到文件"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到 {path}")
    
    def load_data(self, path: str) -> Dict[str, pd.DataFrame]:
        """从文件加载数据"""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


if __name__ == "__main__":
    # 测试数据获取
    fetcher = DataFetcher(source="neodata")
    
    # 获取股票列表
    stock_list = fetcher.get_stock_list()
    print(f"股票列表: {stock_list[:5]}...")
    
    # 获取涨停股票
    limit_up = fetcher.fetcher.get_limit_up_stocks()
    print(f"涨停股票数量: {len(limit_up)}")
