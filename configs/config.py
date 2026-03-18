# configs/config.py
"""
项目配置文件 - 统一管理所有路径和参数
支持 YAML 配置覆盖 + Python 默认配置
"""
import os
import yaml
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

# ==================== 路径配置 ====================

# 项目根目录（自动获取，无论在哪里运行）
PROJECT_ROOT = Path(__file__).parent.parent

# 源代码目录
SRC_DIR = PROJECT_ROOT / 'src'

# 数据目录
DATA_DIR = PROJECT_ROOT / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
DATA_STOCKS_DIR = DATA_RAW_DIR / 'stocks'      # 个股数据
DATA_INDICES_DIR = DATA_RAW_DIR / 'indices'    # 指数数据
DATA_CACHE_DIR = DATA_RAW_DIR / 'cache'        # 临时缓存

# 测试数据路径
TEST_DATA_PATH = DATA_DIR / 'test_stock_data.csv'
STOCK_LIST_PATH = DATA_RAW_DIR / 'stock_list.csv'  # 股票列表

# 模型目录
MODELS_DIR = PROJECT_ROOT / 'models'
LSTM_MODEL_PATH = MODELS_DIR / 'best_lstm_model.pth'

# 日志目录
LOGS_DIR = PROJECT_ROOT / 'logs'
LOG_FILE = LOGS_DIR / f'stock_predict_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# 结果输出目录
RESULTS_DIR = PROJECT_ROOT / 'results'
PREDICTION_PNG_PATH = RESULTS_DIR / 'prediction_results.png'
BACKTEST_RESULTS_DIR = RESULTS_DIR / 'backtest'

# 配置文件目录
CONFIGS_DIR = PROJECT_ROOT / 'configs'
SETTINGS_YAML_PATH = CONFIGS_DIR / 'settings.yaml'

# ==================== YAML 配置加载 ====================

class ConfigLoader:
    """YAML 配置加载器"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = self._load_yaml_config()
    
    def _load_yaml_config(self) -> Optional[Dict]:
        """从 YAML 文件加载配置"""
        if not SETTINGS_YAML_PATH.exists():
            print(f"⚠️ 配置文件不存在：{SETTINGS_YAML_PATH}")
            print("   将使用默认配置")
            return None
        
        try:
            with open(SETTINGS_YAML_PATH, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"📂 已加载配置文件：{SETTINGS_YAML_PATH}")
            return config
        except Exception as e:
            print(f"⚠️ 配置文件加载失败：{e}")
            print("   将使用默认配置")
            return None
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值（支持嵌套键）
        
        Args:
            key: 配置键，如 'data.cache_enabled'
            default: 默认值
            
        Returns:
            配置值
        """
        if self._config is None:
            return default
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> Optional[Dict]:
        """获取全部配置"""
        return self._config


# 创建全局配置加载器实例
config_loader = ConfigLoader()


def get_config(key: str, default: Any = None) -> Any:
    """快捷获取配置值"""
    return config_loader.get(key, default)


# ==================== Tushare 配置 ====================

# Tushare Pro Token（用于获取 A 股数据）
# 获取地址：https://tushare.pro/user/token
TUSHARE_TOKEN = '93a18fa278301f88b254a2b8106931b10d1d26832eefa385fe150d7f'

# Tushare 积分要求（daily 接口需要 120 积分）
TUSHARE_MIN_POINTS = 120


# ==================== API 配置 ====================

class APIConfig:
    """API 相关配置"""
    # 从 YAML 加载，否则使用默认值
    AKSHARE_ENABLED = get_config('data.source', 'akshare') == 'akshare'
    API_RETRY_COUNT = get_config('api.retry_count', 5)
    API_RETRY_DELAYS = get_config('api.retry_delays', [5, 10, 30])
    API_TIMEOUT = get_config('api.timeout', 30)
    
    # 限流配置
    RATE_LIMIT_DELAY = get_config('data.batch_fetch.rate_limit_delay', 0.5)
    MAX_WORKERS = get_config('data.batch_fetch.max_workers', 1)
    
    # 股票列表
    DEFAULT_STOCK_CODES = get_config('data.default_stocks.codes', ['000001', '600519', '300750'])
    DEFAULT_STOCK_NAMES = get_config('data.default_stocks.names', ['平安银行', '贵州茅台', '宁德时代'])
    
    # AKShare 具体配置
    AKSHARE_ADJUST = get_config('api.akshare.adjust', 'qfq')  # qfq=前复权
    AKSHARE_PERIOD = get_config('api.akshare.period', 'daily')


# ==================== 模型配置 ====================

class LSTMConfig:
    """LSTM 模型超参数"""
    INPUT_SIZE = get_config('model.lstm.input_size', 5)
    HIDDEN_SIZE = get_config('model.lstm.hidden_size', 64)
    NUM_LAYERS = get_config('model.lstm.num_layers', 2)
    DROPOUT = get_config('model.lstm.dropout', 0.2)
    LEARNING_RATE = get_config('model.lstm.learning_rate', 0.001)
    BATCH_SIZE = get_config('model.lstm.batch_size', 32)
    EPOCHS = get_config('model.lstm.epochs', 100)
    SEQUENCE_LENGTH = get_config('model.lstm.sequence_length', 60)
    VALIDATION_SPLIT = get_config('model.lstm.validation_split', 0.2)
    
    # 早停配置
    EARLY_STOPPING_ENABLED = get_config('model.lstm.early_stopping.enabled', True)
    EARLY_STOPPING_PATIENCE = get_config('model.lstm.early_stopping.patience', 10)
    
    # 模型保存
    SAVE_BEST_ONLY = get_config('model.lstm.save_best_only', True)
    MODEL_PATH = get_config('model.lstm.model_path', str(LSTM_MODEL_PATH))


class TraditionalMLConfig:
    """传统机器学习模型配置"""
    TEST_SIZE = get_config('model.traditional.test_size', 0.2)
    RANDOM_STATE = get_config('run.random_seed', 42)
    MODELS = get_config('model.traditional.models', ['rf', 'gbt', 'xgb'])


# ==================== 特征工程配置 ====================

class FeatureConfig:
    """特征工程配置"""
    # 技术指标
    TECHNICAL_INDICATORS = get_config('model.features.technical_indicators', [
        'MA5', 'MA10', 'MA20',
        'MACD', 'MACD_Signal',
        'RSI',
        'KDJ_K', 'KDJ_D', 'KDJ_J',
        'BOLL_UPPER', 'BOLL_MIDDLE', 'BOLL_LOWER',
        'VOLUME_MA5', 'VOLUME_MA10'
    ])
    
    # 需要归一化的特征
    NORMALIZE_FEATURES = get_config('model.features.normalize', True)
    
    # 滞后特征
    LAG_DAYS = get_config('model.features.lag_features', [1, 2, 3, 5, 10])
    
    # 滚动窗口
    ROLLING_WINDOWS = get_config('model.features.rolling_windows', [5, 10, 20, 60])
    
    # 预测 horizon
    TARGET_HORIZON = get_config('model.features.target_horizon', 5)


# ==================== 回测配置 ====================

class BacktestConfig:
    """回测系统配置"""
    INITIAL_CAPITAL = get_config('backtest.initial_capital', 1000000)
    COMMISSION_RATE = get_config('backtest.commission_rate', 0.0003)
    SLIPPAGE = get_config('backtest.slippage', 0.001)
    STAMP_DUTY = get_config('backtest.stamp_duty', 0.001)  # 印花税
    POSITION_LIMIT = get_config('backtest.position_limit', 0.1)
    MIN_POSITION = get_config('backtest.min_position', 100)
    MAX_POSITIONS = get_config('backtest.max_positions', 10)
    
    # 交易信号
    SIGNAL_THRESHOLD = get_config('backtest.signal.threshold', 0.01)
    LONG_THRESHOLD = get_config('backtest.signal.long_threshold', 0.01)
    SHORT_THRESHOLD = get_config('backtest.signal.short_threshold', -0.01)
    HOLD_DAYS = get_config('backtest.signal.hold_days', 5)
    
    # 风险控制
    STOP_LOSS = get_config('backtest.risk.stop_loss', 0.05)
    TAKE_PROFIT = get_config('backtest.risk.take_profit', 0.10)
    MAX_DRAWDOWN = get_config('backtest.risk.max_drawdown', 0.20)
    
    # 结果保存
    SAVE_RESULTS = get_config('backtest.save_results', True)
    SAVE_TRADES = get_config('backtest.save_trades', True)


# ==================== 数据获取配置 ====================

class DataFetchConfig:
    """数据获取配置（新增）"""
    # 数据源
    SOURCE = get_config('data.source', 'akshare')
    MARKET = get_config('data.market', 'CN')
    
    # 缓存
    CACHE_ENABLED = get_config('data.cache_enabled', True)
    CACHE_FORMAT = get_config('data.cache_format', 'csv')
    CACHE_COMPRESSION = get_config('data.cache_compression', False)
    
    # 批量获取
    BATCH_START_DATE = get_config('data.batch_fetch.start_date', '2020-01-01')
    BATCH_END_DATE = get_config('data.batch_fetch.end_date', None)
    BATCH_SKIP_EXISTING = get_config('data.batch_fetch.skip_existing', True)
    BATCH_FORCE_REFRESH = get_config('data.batch_fetch.force_refresh', False)
    
    # 股票筛选
    EXCLUDE_ST = get_config('data.filter.exclude_st', True)
    EXCLUDE_NEW = get_config('data.filter.exclude_new', True)
    MIN_LISTING_DAYS = get_config('data.filter.min_listing_days', 365)
    MIN_MARKET_CAP = get_config('data.filter.min_market_cap', 0)
    EXCLUDE_SUSPENDED = get_config('data.filter.exclude_suspended', True)
    
    # 数据列
    REQUIRED_COLUMNS = get_config('data.columns.required', ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # 自动更新
    AUTO_UPDATE_ENABLED = get_config('data.auto_update.enabled', True)
    AUTO_UPDATE_TIME = get_config('data.auto_update.update_time', '18:00')
    AUTO_UPDATE_DAYS = get_config('data.auto_update.update_days', [1, 2, 3, 4, 5])


# ==================== 日志配置 ====================

class LogConfig:
    """日志配置"""
    LEVEL = get_config('logging.level', 'INFO')
    FORMAT = get_config('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    DATE_FORMAT = get_config('logging.date_format', '%Y-%m-%d %H:%M:%S')
    FILE_ENABLED = get_config('logging.file.enabled', True)
    CONSOLE_ENABLED = get_config('logging.console.enabled', True)


# ==================== 可视化配置 ====================

class VisualConfig:
    """可视化配置"""
    FIG_DPI = get_config('visualization.figure.dpi', 300)
    FIG_SIZE = get_config('visualization.figure.figsize', [16, 12])
    FONT_SANS_SERIF = get_config('visualization.font.sans_serif', ['SimHei', 'Microsoft YaHei'])
    UNICODE_MINUS = get_config('visualization.font.unicode_minus', False)


# ==================== 运行配置 ====================

class RunConfig:
    """运行配置"""
    MODE = get_config('run.mode', 'train')  # train / predict / backtest / all
    DEVICE = get_config('run.device', 'cpu')
    RANDOM_SEED = get_config('run.random_seed', 42)


# ==================== 工具函数 ====================

def ensure_directories():
    """确保所有必要目录存在"""
    directories = [
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_STOCKS_DIR,
        DATA_INDICES_DIR,
        DATA_CACHE_DIR,
        MODELS_DIR,
        LOGS_DIR,
        RESULTS_DIR,
        BACKTEST_RESULTS_DIR,
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
    print("✅ 目录结构检查完成")


def get_config_summary():
    """打印配置摘要"""
    print("\n" + "=" * 70)
    print("📋 项目配置摘要")
    print("=" * 70)
    print(f"项目根目录：{PROJECT_ROOT}")
    print(f"数据目录：{DATA_DIR}")
    print(f"股票数据目录：{DATA_STOCKS_DIR}")
    print(f"模型目录：{MODELS_DIR}")
    print(f"日志目录：{LOGS_DIR}")
    print(f"结果目录：{RESULTS_DIR}")
    print("-" * 70)
    print(f"数据源：{DataFetchConfig.SOURCE}")
    print(f"市场：{DataFetchConfig.MARKET}")
    print(f"缓存启用：{DataFetchConfig.CACHE_ENABLED}")
    print("-" * 70)
    print(f"LSTM 隐藏层：{LSTMConfig.HIDDEN_SIZE}")
    print(f"LSTM 层数：{LSTMConfig.NUM_LAYERS}")
    print(f"训练轮数：{LSTMConfig.EPOCHS}")
    print(f"序列长度：{LSTMConfig.SEQUENCE_LENGTH}")
    print("-" * 70)
    print(f"初始资金：{BacktestConfig.INITIAL_CAPITAL:,} 元")
    print(f"手续费率：{BacktestConfig.COMMISSION_RATE:.4f}")
    print(f"交易阈值：{BacktestConfig.SIGNAL_THRESHOLD:.2%}")
    print("-" * 70)
    print(f"运行模式：{RunConfig.MODE}")
    print(f"设备：{RunConfig.DEVICE}")
    print(f"随机种子：{RunConfig.RANDOM_SEED}")
    print("=" * 70 + "\n")


def load_yaml_config() -> Optional[Dict]:
    """手动重新加载 YAML 配置"""
    config_loader._config = None  # 重置缓存
    config_loader._load_yaml_config()
    return config_loader._config


# ==================== matplotlib 字体配置 ====================
# 必须在 import matplotlib.pyplot 之前设置

import matplotlib
matplotlib.rcParams['font.sans-serif'] = VisualConfig.FONT_SANS_SERIF
matplotlib.rcParams['axes.unicode_minus'] = VisualConfig.UNICODE_MINUS


# ==================== 导出所有配置 ====================

__all__ = [
    # 路径
    'PROJECT_ROOT', 'SRC_DIR', 'DATA_DIR', 'DATA_RAW_DIR', 'DATA_PROCESSED_DIR',
    'DATA_STOCKS_DIR', 'DATA_INDICES_DIR', 'DATA_CACHE_DIR',
    'TEST_DATA_PATH', 'STOCK_LIST_PATH',
    'MODELS_DIR', 'LSTM_MODEL_PATH',
    'LOGS_DIR', 'LOG_FILE', 'RESULTS_DIR', 'PREDICTION_PNG_PATH',
    'BACKTEST_RESULTS_DIR', 'CONFIGS_DIR', 'SETTINGS_YAML_PATH',
    # Tushare 配置
    'TUSHARE_TOKEN', 'TUSHARE_MIN_POINTS',
    # 配置加载器
    'config_loader', 'get_config', 'load_yaml_config',
    # 配置类
    'APIConfig', 'LSTMConfig', 'TraditionalMLConfig', 'FeatureConfig',
    'BacktestConfig', 'DataFetchConfig', 'LogConfig', 'VisualConfig', 'RunConfig',
    # 工具函数
    'ensure_directories', 'get_config_summary',
]