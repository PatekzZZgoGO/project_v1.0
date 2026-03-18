# src/data/data_get.py
"""
A 股数据获取模块
支持：批量获取、本地缓存、增量更新
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
from typing import List, Dict, Optional, Tuple
import requests
from tqdm import tqdm

# 项目配置
from configs.config import DATA_DIR, RESULTS_DIR, ensure_directories


class DataFetcher:
    """
    A 股数据获取器
    支持批量获取、本地缓存、增量更新
    """
    
    def __init__(self, symbol: str = None, market: str = 'CN'):
        """
        初始化数据获取器
        
        Args:
            symbol: 股票代码，如 '000001.SZ'，None 表示获取全部
            market: 市场，'CN'=A 股，'US'=美股
        """
        self.symbol = symbol
        self.market = market
        self.base_url = "https://api.example.com"  # 替换为实际 API
        
        # 数据目录
        self.stocks_dir = DATA_DIR / 'raw' / 'stocks'
        self.indices_dir = DATA_DIR / 'raw' / 'indices'
        self.stock_list_path = DATA_DIR / 'raw' / 'stock_list.csv'
        
        # 确保目录存在
        ensure_directories()
        self.stocks_dir.mkdir(parents=True, exist_ok=True)
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        
        # API 限流配置
        self.rate_limit_delay = 0.5  # 请求间隔（秒）
        self.max_retries = 3
        self.retry_delay = 5
        
        print(f"✅ 数据获取器初始化完成")
        print(f"   数据目录：{DATA_DIR}")
        print(f"   个股目录：{self.stocks_dir}")
    
    # ==================== 股票列表获取 ====================
    
    def get_stock_list(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        获取 A 股股票列表
        
        Args:
            force_refresh: 是否强制刷新
            
        Returns:
            股票列表 DataFrame
        """
        # 检查缓存
        if self.stock_list_path.exists() and not force_refresh:
            df = pd.read_csv(self.stock_list_path)
            print(f"📂 从缓存加载股票列表：{len(df)} 只股票")
            return df
        
        print("📊 正在获取最新股票列表...")
        
        # 模拟获取股票列表（替换为实际 API）
        stock_list = self._fetch_stock_list_from_api()
        
        if stock_list is not None and len(stock_list) > 0:
            stock_list.to_csv(self.stock_list_path, index=False, encoding='utf-8-sig')
            print(f"✅ 股票列表已保存：{len(stock_list)} 只")
        
        return stock_list
    
    def _fetch_stock_list_from_api(self) -> Optional[pd.DataFrame]:
        """
        从 API 获取股票列表
        请根据实际使用的数据源修改此方法
        """
        # 示例：使用 AkShare 获取（需要安装：pip install akshare）
        try:
            import akshare as ak
            df = ak.stock_info_a_code_name()
            df.columns = ['code', 'name']
            df['market'] = df['code'].apply(lambda x: 'SH' if x.startswith('6') else 'SZ')
            df['symbol'] = df['code'] + '.' + df['market']
            return df
        except Exception as e:
            print(f"⚠️ AkShare 获取失败：{e}")
            return self._get_mock_stock_list()
    
    def _get_mock_stock_list(self) -> pd.DataFrame:
        """
        获取模拟股票列表（用于测试）
        """
        # 生成模拟的 A 股代码
        codes = []
        # 沪市
        for i in range(1, 100):
            codes.append(f"600{str(i).zfill(3)}.SH")
        # 深市
        for i in range(1, 100):
            codes.append(f"000{str(i).zfill(3)}.SZ")
            codes.append(f"300{str(i).zfill(3)}.SZ")
        
        df = pd.DataFrame({
            'code': [c.split('.')[0] for c in codes],
            'name': [f'股票{i}' for i in range(len(codes))],
            'market': [c.split('.')[1] for c in codes],
            'symbol': codes
        })
        
        return df
    
    # ==================== 单只股票数据获取 ====================
    
    def fetch_historical_data(
        self,
        start_date: str = '2020-01-01',
        end_date: str = None,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        获取单只股票历史数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期，None 表示今天
            use_cache: 是否使用缓存
            force_refresh: 是否强制刷新
            
        Returns:
            历史数据 DataFrame
        """
        if self.symbol is None:
            print("❌ 请先指定股票代码")
            return None
        
        # 构建缓存文件路径
        cache_path = self.stocks_dir / f"{self.symbol}.csv"
        
        # 检查缓存
        if use_cache and cache_path.exists() and not force_refresh:
            df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
            
            # 检查是否需要更新
            if len(df) > 0:
                last_date = df.index[-1]
                today = datetime.now()
                
                # 如果缓存数据是最近的，直接返回
                if (today - last_date).days < 1:
                    print(f"📂 从缓存加载：{self.symbol} ({len(df)} 条)")
                    return df
                
                # 否则增量更新
                print(f"🔄 检测到旧数据，准备增量更新...")
                df_new = self._fetch_from_api(
                    start_date=last_date.strftime('%Y-%m-%d'),
                    end_date=end_date
                )
                
                if df_new is not None and len(df_new) > 0:
                    df = pd.concat([df, df_new]).drop_duplicates()
                    df.to_csv(cache_path, encoding='utf-8-sig')
                    print(f"✅ 已更新：{self.symbol} (新增 {len(df_new)} 条)")
                    return df
            
            return df
        
        # 从 API 获取
        print(f"📡 正在获取：{self.symbol}")
        df = self._fetch_from_api(start_date, end_date)
        
        if df is not None and len(df) > 0:
            # 保存到缓存
            df.to_csv(cache_path, encoding='utf-8-sig')
            print(f"💾 已保存：{self.symbol} ({len(df)} 条)")
        
        return df
    
    def _fetch_from_api(
        self,
        start_date: str,
        end_date: str = None
    ) -> Optional[pd.DataFrame]:
        """
        从 API 获取股票数据
        请根据实际使用的数据源修改此方法
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 示例：使用 AkShare 获取
        try:
            import akshare as ak
            code = self.symbol.split('.')[0]
            market = self.symbol.split('.')[1].lower()
            
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # 前复权
            )
            
            if len(df) > 0:
                df.columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume', 
                             'Turnover', 'Amplitude', 'PctChange', 'Change', 'TurnoverRate']
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
        except Exception as e:
            print(f"⚠️ API 获取失败：{e}")
        
        # 返回模拟数据（用于测试）
        return self._generate_mock_data(start_date, end_date)
    
    def _generate_mock_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟数据"""
        dates = pd.date_range(start_date, end_date, freq='B')  # 工作日
        df = pd.DataFrame({
            'Open': np.random.randn(len(dates)).cumsum() + 100,
            'High': np.random.randn(len(dates)).cumsum() + 102,
            'Low': np.random.randn(len(dates)).cumsum() + 98,
            'Close': np.random.randn(len(dates)).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        return df
    
    # ==================== 批量获取所有股票 ====================
    
    def fetch_all_stocks(
        self,
        start_date: str = '2020-01-01',
        end_date: str = None,
        force_refresh: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, bool]:
        """
        批量获取所有 A 股历史数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            force_refresh: 是否强制刷新
            skip_existing: 是否跳过已存在的股票
            
        Returns:
            获取结果字典 {symbol: success}
        """
        print("\n" + "=" * 70)
        print("🚀 开始批量获取 A 股数据")
        print("=" * 70)
        print(f"日期范围：{start_date} 至 {end_date or '今天'}")
        print(f"跳过已有数据：{skip_existing}")
        print("=" * 70 + "\n")
        
        # 获取股票列表
        stock_list = self.get_stock_list(force_refresh=force_refresh)
        symbols = stock_list['symbol'].tolist()
        
        print(f"📋 共 {len(symbols)} 只股票需要获取\n")
        
        # 记录结果
        results = {}
        success_count = 0
        fail_count = 0
        skip_count = 0
        
        # 进度条
        pbar = tqdm(symbols, desc="获取进度", unit="只")
        
        for i, symbol in enumerate(pbar):
            # 构建缓存路径
            cache_path = self.stocks_dir / f"{symbol}.csv"
            
            # 检查是否跳过
            if skip_existing and cache_path.exists() and not force_refresh:
                skip_count += 1
                pbar.set_postfix({'成功': success_count, '失败': fail_count, '跳过': skip_count})
                continue
            
            # 获取数据
            self.symbol = symbol
            df = self.fetch_historical_data(
                start_date=start_date,
                end_date=end_date,
                use_cache=False,
                force_refresh=force_refresh
            )
            
            if df is not None and len(df) > 0:
                results[symbol] = True
                success_count += 1
            else:
                results[symbol] = False
                fail_count += 1
            
            pbar.set_postfix({'成功': success_count, '失败': fail_count, '跳过': skip_count})
            
            # API 限流
            time.sleep(self.rate_limit_delay)
        
        # 打印总结
        print("\n" + "=" * 70)
        print("📊 批量获取完成")
        print("=" * 70)
        print(f"总股票数：{len(symbols)}")
        print(f"✅ 成功：{success_count}")
        print(f"❌ 失败：{fail_count}")
        print(f"⏭️  跳过：{skip_count}")
        print(f"💾 数据目录：{self.stocks_dir}")
        print("=" * 70 + "\n")
        
        # 保存获取记录
        record_path = DATA_DIR / 'raw' / 'fetch_record.json'
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total': len(symbols),
                'success': success_count,
                'fail': fail_count,
                'skip': skip_count,
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        return results
    
    # ==================== 增量更新 ====================
    
    def update_all_stocks(self) -> Dict[str, bool]:
        """
        增量更新所有股票数据（只获取最新数据）
        
        Returns:
            更新结果字典
        """
        print("\n" + "=" * 70)
        print("🔄 开始增量更新 A 股数据")
        print("=" * 70 + "\n")
        
        # 获取股票列表
        stock_list = self.get_stock_list()
        symbols = stock_list['symbol'].tolist()
        
        results = {}
        updated_count = 0
        no_update_count = 0
        fail_count = 0
        
        pbar = tqdm(symbols, desc="更新进度", unit="只")
        
        for symbol in pbar:
            self.symbol = symbol
            cache_path = self.stocks_dir / f"{symbol}.csv"
            
            if not cache_path.exists():
                no_update_count += 1
                continue
            
            # 读取现有数据
            df_existing = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
            last_date = df_existing.index[-1]
            
            # 计算需要更新的日期范围
            today = datetime.now()
            if (today - last_date).days < 1:
                no_update_count += 1
                continue
            
            # 获取新数据
            df_new = self._fetch_from_api(
                start_date=last_date.strftime('%Y-%m-%d'),
                end_date=today.strftime('%Y-%m-%d')
            )
            
            if df_new is not None and len(df_new) > 0:
                # 合并数据
                df = pd.concat([df_existing, df_new]).drop_duplicates()
                df.to_csv(cache_path, encoding='utf-8-sig')
                results[symbol] = True
                updated_count += 1
            else:
                fail_count += 1
            
            pbar.set_postfix({'更新': updated_count, '无需更新': no_update_count, '失败': fail_count})
            
            # API 限流
            time.sleep(self.rate_limit_delay)
        
        print("\n" + "=" * 70)
        print("📊 增量更新完成")
        print("=" * 70)
        print(f"✅ 已更新：{updated_count} 只")
        print(f"⏭️  无需更新：{no_update_count} 只")
        print(f"❌ 失败：{fail_count} 只")
        print("=" * 70 + "\n")
        
        return results
    
    # ==================== 数据加载工具 ====================
    
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        从本地加载单只股票数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票数据 DataFrame
        """
        cache_path = self.stocks_dir / f"{symbol}.csv"
        
        if not cache_path.exists():
            print(f"❌ 未找到数据：{symbol}")
            return None
        
        df = pd.read_csv(cache_path, index_col='Date', parse_dates=True)
        return df
    
    def load_multiple_stocks(
        self,
        symbols: List[str],
        columns: List[str] = ['Close']
    ) -> pd.DataFrame:
        """
        加载多只股票数据并合并
        
        Args:
            symbols: 股票代码列表
            columns: 需要的列
            
        Returns:
            合并后的 DataFrame（列名为股票代码）
        """
        data_dict = {}
        
        for symbol in tqdm(symbols, desc="加载数据"):
            df = self.load_stock_data(symbol)
            if df is not None:
                for col in columns:
                    if col in df.columns:
                        data_dict[f"{symbol}_{col}"] = df[col]
        
        return pd.DataFrame(data_dict)
    
    def get_available_stocks(self) -> List[str]:
        """
        获取本地已有数据的股票列表
        
        Returns:
            股票代码列表
        """
        stocks = []
        for file in self.stocks_dir.glob("*.csv"):
            symbol = file.stem
            stocks.append(symbol)
        return sorted(stocks)


# ==================== 命令行工具 ====================

if __name__ == '__main__':
    import sys
    
    fetcher = DataFetcher()
    
    if len(sys.argv) < 2:
        print("""
A 股数据获取工具

用法:
    python data_get.py fetch_all          # 获取所有 A 股数据
    python data_get.py update             # 增量更新所有股票
    python data_get.py fetch 000001.SZ    # 获取单只股票
    python data_get.py list               # 列出本地已有数据的股票
    python data_get.py stock_list         # 获取最新股票列表
        """)
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == 'fetch_all':
        fetcher.fetch_all_stocks(
            start_date='2020-01-01',
            skip_existing=True
        )
    
    elif command == 'update':
        fetcher.update_all_stocks()
    
    elif command == 'fetch' and len(sys.argv) > 2:
        symbol = sys.argv[2]
        fetcher.symbol = symbol
        fetcher.fetch_historical_data(start_date='2020-01-01')
    
    elif command == 'list':
        stocks = fetcher.get_available_stocks()
        print(f"本地已有 {len(stocks)} 只股票数据:")
        for s in stocks[:20]:
            print(f"  {s}")
        if len(stocks) > 20:
            print(f"  ... 还有 {len(stocks) - 20} 只")
    
    elif command == 'stock_list':
        fetcher.get_stock_list(force_refresh=True)
    
    else:
        print(f"❌ 未知命令：{command}")