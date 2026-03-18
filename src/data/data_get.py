# src/data/data_get.py
"""
A 股数据获取模块（Tushare Pro 版）
基于 CSDN 架构设计：智能限流 + 多层级错误恢复 + 智能缓存 + 监控告警
需要 Tushare Token 和 120 积分
"""
import sys
from pathlib import Path
import os
import time
import random
import logging
import pickle
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Optional, Tuple
from threading import Lock

import pandas as pd
from tqdm import tqdm

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import (
    DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR,
    DATA_STOCKS_DIR, DATA_INDICES_DIR, STOCK_LIST_PATH,
    DataFetchConfig, ensure_directories,
)

# ============================================================================
# 第一部分：智能限流系统（Smart Rate Limiter）
# ============================================================================

class SmartRateLimiter:
    """
    智能请求调控系统
    基于反馈的动态请求调节机制，避免触发 Tushare 限流
    """
    
    def __init__(self, window_size: int = 10, min_interval: float = 1.0):
        """
        初始化限流器
        
        Args:
            window_size: 请求时间窗口大小（记录最近 N 次请求）
            min_interval: 最小请求间隔（秒）
        """
        self.request_timestamps = deque(maxlen=window_size)
        self.min_interval = min_interval
        self._lock = Lock()
        
    def get_delay(self) -> float:
        """动态计算下一次请求的延迟时间"""
        with self._lock:
            if len(self.request_timestamps) < 2:
                return random.uniform(0.5, 1.0)
            
            # 计算最近请求的平均间隔
            timestamps = list(self.request_timestamps)
            recent_intervals = [
                timestamps[i] - timestamps[i-1] 
                for i in range(1, len(timestamps))
            ]
            
            avg_interval = sum(recent_intervals) / len(recent_intervals)
            
            # 如果平均间隔小于最小值，则延长延迟
            if avg_interval < self.min_interval:
                return self.min_interval + random.uniform(0.5, 1.5)
            
            return random.uniform(0.8, 1.2)
    
    def wait(self):
        """根据动态计算结果等待"""
        delay = self.get_delay()
        time.sleep(delay)
        with self._lock:
            self.request_timestamps.append(time.time())
    
    def reset(self):
        """重置限流器"""
        with self._lock:
            self.request_timestamps.clear()


# ============================================================================
# 第二部分：多层级错误恢复架构（Resilient Data Fetcher）
# ============================================================================

class ResilientTushareFetcher:
    """
    多层级错误恢复架构
    包含网络层、应用层和数据层的三级恢复机制
    """
    
    def __init__(self, pro_api, max_attempts: int = 5, base_delay: float = 1.0):
        """
        初始化错误恢复获取器
        
        Args:
            pro_api: Tushare Pro API 实例
            max_attempts: 最大重试次数
            base_delay: 基础延迟时间（秒）
        """
        self.pro = pro_api
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.rate_limiter = SmartRateLimiter()
        
    def _validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        数据层验证：验证数据完整性
        
        Args:
            df: 数据 DataFrame
            required_columns: 必需的列名列表
            
        Returns:
            bool: 数据是否有效
        """
        if df is None or len(df) == 0:
            return False
        
        return all(col in df.columns for col in required_columns)
    
    def fetch_daily_with_retry(self, ts_code: str, start_date: str, 
                                end_date: str, max_attempts: int = None) -> Optional[pd.DataFrame]:
        """
        带应用层重试的日线数据获取
        
        Args:
            ts_code: 股票代码（如 000001.SZ）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            max_attempts: 最大重试次数
            
        Returns:
            pd.DataFrame 或 None
        """
        max_attempts = max_attempts or self.max_attempts
        required_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol']
        
        for attempt in range(max_attempts):
            try:
                # 限流等待
                self.rate_limiter.wait()
                
                # 网络层请求
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                # 数据层验证
                if not self._validate_data(df, required_columns):
                    raise ValueError("返回数据不完整或格式错误")
                
                return df
                
            except Exception as e:
                error_msg = str(e)
                
                # 权限错误不重试
                if '权限' in error_msg or '积分' in error_msg:
                    logging.error(f"权限错误，停止重试：{error_msg}")
                    return None
                
                if attempt < max_attempts - 1:
                    # 指数退避策略
                    wait_time = (self.base_delay ** (attempt + 1)) + random.uniform(0, 1)
                    logging.warning(
                        f"第{attempt + 1}次尝试失败，等待{wait_time:.1f}秒后重试... "
                        f"错误：{error_msg[:60]}"
                    )
                    time.sleep(wait_time)
                else:
                    logging.error(f"所有{max_attempts}次尝试失败：{error_msg}")
                    return None
        
        return None
    
    def fetch_stock_basic_with_retry(self, max_attempts: int = 3) -> Optional[pd.DataFrame]:
        """
        获取股票基本信息（带重试）
        
        Args:
            max_attempts: 最大重试次数
            
        Returns:
            pd.DataFrame 或 None
        """
        for attempt in range(max_attempts):
            try:
                self.rate_limiter.wait()
                df = self.pro.stock_basic(
                    exchange='',
                    list_status='L',
                    fields='ts_code,symbol,name,area,industry,list_date'
                )
                
                if df is not None and len(df) > 0:
                    return df
                    
            except Exception as e:
                if attempt < max_attempts - 1:
                    wait_time = (self.base_delay ** (attempt + 1)) + random.uniform(0, 1)
                    logging.warning(f"获取股票列表失败，等待{wait_time:.1f}秒后重试...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"获取股票列表失败：{str(e)}")
                    return None
        
        return None


# ============================================================================
# 第三部分：智能缓存与增量更新（Data Cache Manager）
# ============================================================================

class DataCacheManager:
    """
    智能缓存与增量更新系统
    构建多级缓存系统，减少重复请求
    """
    
    def __init__(self, cache_dir: str = None, expiry_days: int = 1):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            expiry_days: 缓存过期天数
        """
        self.cache_dir = cache_dir or str(DATA_STOCKS_DIR / 'cache')
        self.expiry_days = expiry_days
        os.makedirs(self.cache_dir, exist_ok=True)
        self._lock = Lock()
        
    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> str:
        """生成缓存文件路径"""
        safe_symbol = symbol.replace('.', '_')
        filename = f"{safe_symbol}_{start_date}_{end_date}.pkl"
        return os.path.join(self.cache_dir, filename)
    
    def _get_meta_path(self, symbol: str) -> str:
        """获取元数据文件路径（记录最后更新日期）"""
        safe_symbol = symbol.replace('.', '_')
        return os.path.join(self.cache_dir, f"{safe_symbol}_meta.pkl")
    
    def is_cache_valid(self, cache_path: str) -> bool:
        """
        检查缓存是否有效
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            bool: 缓存是否有效
        """
        if not os.path.exists(cache_path):
            return False
        
        # 检查缓存是否过期
        modified_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return (datetime.now() - modified_time) < timedelta(days=self.expiry_days)
    
    def load_cache(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        加载缓存数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pd.DataFrame 或 None
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logging.warning(f"加载缓存失败：{e}")
                return None
        
        return None
    
    def save_cache(self, symbol: str, start_date: str, end_date: str, 
                   data: pd.DataFrame) -> bool:
        """
        保存数据到缓存
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            data: 数据 DataFrame
            
        Returns:
            bool: 是否保存成功
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date)
        
        try:
            with self._lock:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            return True
        except Exception as e:
            logging.error(f"保存缓存失败：{e}")
            return False
    
    def get_last_trade_date(self, symbol: str) -> Optional[str]:
        """
        获取某只股票最后更新的交易日期
        
        Args:
            symbol: 股票代码
            
        Returns:
            str 或 None: 最后交易日期（YYYYMMDD 格式）
        """
        meta_path = self._get_meta_path(symbol)
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    return meta.get('last_trade_date')
            except Exception:
                pass
        
        return None
    
    def update_meta(self, symbol: str, last_trade_date: str, record_count: int):
        """
        更新元数据
        
        Args:
            symbol: 股票代码
            last_trade_date: 最后交易日期
            record_count: 记录数量
        """
        meta_path = self._get_meta_path(symbol)
        
        try:
            with self._lock:
                meta = {
                    'last_trade_date': last_trade_date,
                    'last_update_time': datetime.now().isoformat(),
                    'record_count': record_count
                }
                with open(meta_path, 'wb') as f:
                    pickle.dump(meta, f)
        except Exception as e:
            logging.warning(f"更新元数据失败：{e}")
    
    def get_incremental_range(self, symbol: str, end_date: str) -> Tuple[Optional[str], str]:
        """
        计算增量更新的日期范围
        
        Args:
            symbol: 股票代码
            end_date: 结束日期（YYYYMMDD）
            
        Returns:
            Tuple: (start_date, end_date)，如果需要全量更新则 start_date 为 None
        """
        last_date = self.get_last_trade_date(symbol)
        
        if last_date and last_date < end_date:
            # 计算下一交易日（简单 +1 天，实际应考虑交易日）
            next_date = (datetime.strptime(last_date, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
            return next_date, end_date
        
        return None, end_date


# ============================================================================
# 第四部分：监控与日志系统（Data Monitor）
# ============================================================================

class DataMonitor:
    """
    数据采集监控与告警系统
    实时监控采集状态，记录日志和统计信息
    """
    
    def __init__(self, log_file: str = None, log_level: int = logging.INFO):
        """
        初始化监控器
        
        Args:
            log_file: 日志文件路径
            log_level: 日志级别
        """
        self.logger = logging.getLogger("TushareDataMonitor")
        self.logger.setLevel(log_level)
        
        # 避免重复添加 handler
        if not self.logger.handlers:
            # 文件 handler
            if log_file:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(log_level)
            else:
                file_handler = logging.StreamHandler()
                file_handler.setLevel(log_level)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # 错误统计
        self.error_stats = {
            "total_requests": 0,
            "success_count": 0,
            "failure_count": 0,
            "error_types": {},
            "failing_symbols": set(),
            "start_time": datetime.now()
        }
        self._lock = Lock()
    
    def log_success(self, symbol: str, record_count: int, duration: float = 0):
        """记录成功事件"""
        with self._lock:
            self.error_stats["total_requests"] += 1
            self.error_stats["success_count"] += 1
        
        self.logger.info(
            f"✅ 成功获取 {symbol} 数据，记录数：{record_count}, 耗时：{duration:.2f}s"
        )
    
    def log_error(self, symbol: str, error_msg: str):
        """记录错误事件"""
        with self._lock:
            self.error_stats["total_requests"] += 1
            self.error_stats["failure_count"] += 1
            error_type = error_msg.split(":")[0] if ":" in error_msg else error_msg[:30]
            self.error_stats["error_types"][error_type] = \
                self.error_stats["error_types"].get(error_type, 0) + 1
            self.error_stats["failing_symbols"].add(symbol)
        
        self.logger.error(f"❌ 获取 {symbol} 失败：{error_msg}")
    
    def log_warning(self, symbol: str, message: str):
        """记录警告事件"""
        self.logger.warning(f"⚠️ {symbol}: {message}")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        with self._lock:
            stats = self.error_stats.copy()
            stats["success_rate"] = (
                stats["success_count"] / stats["total_requests"] * 100 
                if stats["total_requests"] > 0 else 0
            )
            stats["duration"] = (datetime.now() - stats["start_time"]).total_seconds()
        return stats
    
    def print_summary(self):
        """打印采集摘要"""
        stats = self.get_stats()
        
        print("\n" + "=" * 70)
        print("📊 数据采集摘要")
        print("=" * 70)
        print(f"   总请求数：{stats['total_requests']}")
        print(f"   成功：{stats['success_count']} ({stats['success_rate']:.1f}%)")
        print(f"   失败：{stats['failure_count']}")
        print(f"   总耗时：{stats['duration']:.1f} 秒")
        print(f"   平均速度：{stats['total_requests']/stats['duration']:.2f} 请求/秒")
        
        if stats['error_types']:
            print("\n   错误类型统计:")
            for error_type, count in stats['error_types'].items():
                print(f"      - {error_type}: {count} 次")
        
        if stats['failing_symbols']:
            print(f"\n   失败股票列表（前 10 个）:")
            for symbol in list(stats['failing_symbols'])[:10]:
                print(f"      - {symbol}")
        
        print("=" * 70)
    
    def check_health(self, failure_threshold: float = 0.3) -> bool:
        """
        检查系统健康状态
        
        Args:
            failure_threshold: 失败率阈值
            
        Returns:
            bool: 系统是否健康
        """
        stats = self.get_stats()
        failure_rate = stats["failure_count"] / stats["total_requests"] if stats["total_requests"] > 0 else 0
        
        if failure_rate > failure_threshold:
            self.logger.warning(
                f"系统健康度警告：失败率 {failure_rate:.1%} 超过阈值 {failure_threshold:.1%}"
            )
            return False
        
        return True


# ============================================================================
# 第五部分：主数据获取器（Data Fetcher）
# ============================================================================

class DataFetcher:
    """
    A 股数据获取器（Tushare Pro 版）
    整合限流、错误恢复、缓存、监控功能
    """
    
    _initialized = False
    
    def __init__(self, symbol: str = None, market: str = 'CN'):
        """
        初始化数据获取器
        
        Args:
            symbol: 股票代码
            market: 市场标识
        """
        self.symbol = symbol
        self.market = market or DataFetchConfig.MARKET
        
        ensure_directories()
        
        # 初始化 Tushare
        self.pro = None
        self.tushare_available = False
        self.tushare_points = 0
        self._init_tushare()
        
        # 初始化组件
        self.fetcher = ResilientTushareFetcher(self.pro) if self.pro else None
        self.cache_manager = DataCacheManager(cache_dir=str(DATA_STOCKS_DIR / 'cache'))
        self.monitor = DataMonitor(log_file=str(DATA_DIR / 'data_fetch.log'))
        
        # 配置
        self.stocks_dir = DATA_STOCKS_DIR
        self.stock_list_path = STOCK_LIST_PATH
        self.request_delay = 1.0
        
        if not DataFetcher._initialized:
            DataFetcher._initialized = True
            print(f"✅ 数据获取器初始化完成（Tushare Pro 版）")
            print(f"   市场：{self.market}")
            print(f"   数据目录：{DATA_DIR}")
            print(f"   Tushare: {'✅' if self.tushare_available else '❌'}")
            print(f"   当前积分：{self.tushare_points}")
    
    def _init_tushare(self):
        """初始化 Tushare 连接"""
        try:
            import tushare as ts
            from configs.config import TUSHARE_TOKEN
            
            ts.set_token(TUSHARE_TOKEN)
            self.pro = ts.pro_api()
            
            # 检查积分
            try:
                df = self.pro.user()
                self.tushare_points = df['points'].values[0] if 'points' in df.columns else 120
                
                if self.tushare_points >= 120:
                    self.tushare_available = True
                    print(f"✅ Tushare 连接成功（积分：{self.tushare_points}）")
                else:
                    print(f"⚠️ Tushare 积分不足：{self.tushare_points}/120")
                    self.tushare_available = True  # 仍尝试使用
            except Exception as e:
                print(f"⚠️ 无法查询积分：{e}")
                self.tushare_points = 120
                self.tushare_available = True
                
        except Exception as e:
            print(f"❌ Tushare 不可用：{str(e)[:60]}")
            self.tushare_available = False
    
    def fetch_historical_data(self, start_date: str, end_date: str = None,
                              use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        """
        获取历史数据
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            use_cache: 是否使用缓存
            force_refresh: 是否强制刷新
            
        Returns:
            pd.DataFrame: 历史数据
        """
        if not self.symbol:
            raise ValueError("请先设置股票代码")
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # 转换日期格式
        start_fmt = start_date.replace('-', '')
        end_fmt = end_date.replace('-', '')
        
        # 缓存路径
        cache_file = self._get_stock_cache_path(self.symbol)
        
        # 从缓存加载
        if use_cache and not force_refresh and cache_file.exists():
            print(f"📂 从缓存加载：{self.symbol}")
            try:
                df = pd.read_csv(cache_file, index_col='Date', parse_dates=True)
                
                if not df.empty:
                    last_date = df.index.max().strftime('%Y%m%d')
                    
                    if last_date < end_fmt:
                        print(f"   缓存最后日期：{last_date}，需要增量更新")
                        df_new = self._fetch_from_api(self.symbol, last_date, end_fmt)
                        
                        if not df_new.empty:
                            df = pd.concat([df, df_new]).drop_duplicates()
                            self._save_to_cache(df, cache_file)
                            self.cache_manager.update_meta(
                                self.symbol, 
                                df.index.max().strftime('%Y%m%d'),
                                len(df)
                            )
                
                print(f"✅ 加载成功：{len(df)} 条记录")
                return df
                
            except Exception as e:
                self.monitor.log_error(self.symbol, f"缓存读取失败：{e}")
                print(f"⚠️ 缓存读取失败：{e}")
        
        # 从 API 获取
        print(f"📡 从 Tushare 获取：{self.symbol}")
        start_time = time.time()
        df = self._fetch_from_api(self.symbol, start_fmt, end_fmt)
        duration = time.time() - start_time
        
        if df is None or df.empty:
            self.monitor.log_error(self.symbol, "API 获取失败或返回空数据")
            print("⚠️ API 获取失败")
            return pd.DataFrame()
        
        if use_cache:
            self._save_to_cache(df, cache_file)
            self.cache_manager.update_meta(self.symbol, end_fmt, len(df))
        
        self.monitor.log_success(self.symbol, len(df), duration)
        return df
    
    def _fetch_from_api(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从 Tushare API 获取数据"""
        if not self.tushare_available or self.fetcher is None:
            return pd.DataFrame()
        
        try:
            df = self.fetcher.fetch_daily_with_retry(symbol, start_date, end_date)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            df = self._clean_data(df, symbol)
            print(f"✅ 成功获取 {symbol} 共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.monitor.log_error(symbol, str(e))
            print(f"❌ 获取失败：{str(e)[:80]}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """清洗数据"""
        column_mapping = {
            'trade_date': 'Date', 'open': 'Open', 'close': 'Close',
            'high': 'High', 'low': 'Low', 'vol': 'Volume', 'amount': 'Turnover'
        }
        df = df.rename(columns=column_mapping)
        
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in required_cols if col in df.columns]
        df = df[available_cols]
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
            df.set_index('Date', inplace=True)
        
        df = df.dropna()
        df['Symbol'] = symbol
        return df
    
    def _get_stock_cache_path(self, symbol: str) -> Path:
        """获取股票缓存文件路径"""
        safe_symbol = symbol.replace('.', '_')
        return self.stocks_dir / f"{safe_symbol}.csv"
    
    def _save_to_cache(self, df: pd.DataFrame, cache_file: Path):
        """保存数据到缓存"""
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file)
        except Exception as e:
            self.monitor.log_warning(self.symbol, f"缓存保存失败：{e}")
    
    def fetch_all_stocks(self, start_date: str = None, end_date: str = None,
                         skip_existing: bool = True, force_refresh: bool = False,
                         stock_list: List[str] = None, resume: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量获取所有股票数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            skip_existing: 是否跳过已存在的股票
            force_refresh: 是否强制刷新
            stock_list: 股票列表
            resume: 是否断点续传
            
        Returns:
            Dict[str, pd.DataFrame]: 股票数据字典
        """
        if start_date is None:
            start_date = DataFetchConfig.BATCH_START_DATE
        if end_date is None:
            end_date = DataFetchConfig.BATCH_END_DATE
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if stock_list is None:
            stock_list = self.get_stock_list()
        
        if not stock_list:
            print("❌ 未获取到股票列表")
            return {}
        
        # 断点续传
        if resume and skip_existing:
            original_count = len(stock_list)
            stock_list = [
                s for s in stock_list 
                if not self._get_stock_cache_path(s).exists()
            ]
            skipped = original_count - len(stock_list)
            print(f"📂 跳过 {skipped} 只已缓存股票，剩余 {len(stock_list)} 只需要获取")
        
        if not stock_list:
            print("✅ 所有股票已缓存，无需获取")
            return {}
        
        print(f"\n📊 开始批量获取股票数据（Tushare Pro 版）")
        print(f"   股票数量：{len(stock_list)}")
        print(f"   日期范围：{start_date} 至 {end_date}")
        print(f"   当前积分：{self.tushare_points}")
        print("-" * 70)
        
        data_dict = {}
        success_count = 0
        fail_count = 0
        
        for i, symbol in enumerate(tqdm(stock_list, desc="获取股票数据")):
            try:
                self.symbol = symbol
                df = self.fetch_historical_data(
                    start_date=start_date,
                    end_date=end_date,
                    use_cache=True,
                    force_refresh=force_refresh
                )
                
                if not df.empty:
                    data_dict[symbol] = df
                    success_count += 1
                else:
                    fail_count += 1
                
                # 每 100 只休息 10 秒
                if (i + 1) % 100 == 0:
                    print(f"\n⏳ 已获取 {i+1} 只，等待 10 秒...")
                    time.sleep(10)
                    
            except KeyboardInterrupt:
                print(f"\n\n⚠️ 用户中断，已获取 {success_count} 只股票")
                print(f"💡 再次运行将自动断点续传")
                break
            except Exception as e:
                self.monitor.log_error(symbol, str(e))
                print(f"\n❌ 获取 {symbol} 失败：{str(e)[:80]}")
                fail_count += 1
                time.sleep(5)
        
        # 打印摘要
        self.monitor.print_summary()
        
        return data_dict
    
    def get_stock_list(self, refresh: bool = False) -> List[str]:
        """获取 A 股股票列表"""
        if not refresh and self.stock_list_path.exists():
            try:
                df = pd.read_csv(self.stock_list_path, dtype={'code': str, 'symbol': str})
                
                if 'symbol' in df.columns:
                    stock_list = df['symbol'].astype(str).tolist()
                elif 'code' in df.columns:
                    stock_list = []
                    for _, row in df.iterrows():
                        code = str(row['code'])
                        market = row.get('market', '')
                        market_code = 'SZ' if market == '1' else 'SH'
                        stock_list.append(f"{code}.{market_code}")
                
                stock_list = [s for s in stock_list if s and len(s) > 5]
                print(f"📂 从缓存加载股票列表：{len(stock_list)} 只")
                return stock_list
                
            except Exception as e:
                print(f"⚠️ 股票列表读取失败：{e}")
        
        if self.pro is None:
            return []
        
        print("📡 从 Tushare 获取股票列表...")
        try:
            df = self.fetcher.fetch_stock_basic_with_retry()
            
            if df is None or df.empty:
                return []
            
            stock_list = df['ts_code'].astype(str).tolist()
            print(f"✅ 获取股票列表：{len(stock_list)} 只")
            
            if stock_list:
                self._save_stock_list(stock_list, df)
            
            return stock_list
            
        except Exception as e:
            print(f"⚠️ 获取股票列表失败：{e}")
            return []
    
    def _save_stock_list(self, stock_list: List[str], df: pd.DataFrame):
        """保存股票列表"""
        try:
            save_df = pd.DataFrame({
                'code': [s.split('.')[0] for s in stock_list],
                'symbol': stock_list,
                'market': ['SZ' if s.endswith('.SZ') else 'SH' for s in stock_list]
            })
            self.stock_list_path.parent.mkdir(parents=True, exist_ok=True)
            save_df.to_csv(self.stock_list_path, index=False)
            print(f"💾 股票列表已保存：{self.stock_list_path}")
        except Exception as e:
            print(f"⚠️ 保存股票列表失败：{e}")
    
    def check_cache_status(self) -> Dict:
        """检查缓存状态"""
        if not self.stocks_dir.exists():
            return {'total': 0, 'cached': 0, 'missing': 0}
        
        stock_list = self.get_stock_list()
        cached = sum(
            1 for s in stock_list 
            if self._get_stock_cache_path(s).exists()
        )
        missing = len(stock_list) - cached
        
        return {
            'total': len(stock_list),
            'cached': cached,
            'missing': missing,
            'coverage': f"{cached/len(stock_list)*100:.1f}%" if stock_list else "0%"
        }


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='A 股数据获取工具（Tushare Pro 版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python data_get.py fetch -s 000001.SZ --start 2023-01-01
  python data_get.py fetch_all --start 2023-01-01 --end 2023-12-31
  python data_get.py status
        """
    )
    
    parser.add_argument(
        'command', 
        choices=['fetch', 'fetch_all', 'status'],
        help='命令：fetch=单只股票，fetch_all=批量获取，status=缓存状态'
    )
    parser.add_argument(
        '--symbol', '-s', 
        type=str, 
        help='股票代码（如 000001.SZ）'
    )
    parser.add_argument(
        '--start', 
        type=str, 
        default='2020-01-01', 
        help='开始日期（YYYY-MM-DD）'
    )
    parser.add_argument(
        '--end', 
        type=str, 
        default=None, 
        help='结束日期（YYYY-MM-DD）'
    )
    parser.add_argument(
        '--force', '-f', 
        action='store_true', 
        help='强制刷新'
    )
    parser.add_argument(
        '--resume', '-r', 
        action='store_true', 
        default=True, 
        help='断点续传'
    )
    
    args = parser.parse_args()
    
    fetcher = DataFetcher(symbol=args.symbol)
    
    if args.command == 'fetch':
        if not args.symbol:
            print("❌ 请指定股票代码：-s 000001.SZ")
            sys.exit(1)
        
        df = fetcher.fetch_historical_data(
            args.start, 
            args.end, 
            force_refresh=args.force
        )
        
        if not df.empty:
            print(f"\n📊 数据预览:")
            print(df.head())
            print(f"\n总计：{len(df)} 条记录")
        else:
            print("❌ 未获取到数据")
    
    elif args.command == 'fetch_all':
        fetcher.fetch_all_stocks(
            start_date=args.start,
            end_date=args.end,
            skip_existing=not args.force,
            force_refresh=args.force,
            resume=args.resume
        )
    
    elif args.command == 'status':
        status = fetcher.check_cache_status()
        print("\n📊 缓存状态:")
        print(f"   股票总数：{status['total']}")
        print(f"   已缓存：{status['cached']}")
        print(f"   未缓存：{status['missing']}")
        print(f"   覆盖率：{status['coverage']}")