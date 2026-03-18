import talib
import numpy as np
import pandas as pd  # 确保导入 pandas
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    """特征工程处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征"""
        # 🔥【关键修复】强制转换为 float64，防止 TA-Lib 报错 "input array type is not double"
        high = df['High'].values.astype(np.float64)
        low = df['Low'].values.astype(np.float64)
        close = df['Close'].values.astype(np.float64)
        volume = df['Volume'].values.astype(np.float64)
        
        # 趋势指标
        df['SMA_5'] = talib.SMA(close, timeperiod=5)
        df['SMA_10'] = talib.SMA(close, timeperiod=10)
        df['SMA_20'] = talib.SMA(close, timeperiod=20)
        df['EMA_12'] = talib.EMA(close, timeperiod=12)
        df['EMA_26'] = talib.EMA(close, timeperiod=26)
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # 动量指标
        df['RSI_14'] = talib.RSI(close, timeperiod=14)
        df['ROC'] = talib.ROC(close, timeperiod=10)
        
        # 波动率指标
        df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
        df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2
        )
        
        # 成交量指标
        df['OBV'] = talib.OBV(close, volume)
        df['Volume_SMA'] = talib.SMA(volume, timeperiod=20)
        
        # 价格位置特征
        # 防止除以零或 NaN
        band_width = df['Upper_Band'] - df['Lower_Band']
        df['Price_Position'] = np.where(band_width != 0, (close - df['Lower_Band']) / band_width, 0.5)
        
        return df
    
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加统计特征"""
        close = df['Close'].values.astype(np.float64) # 同样确保这里是 float64
        
        # 收益率
        df['Return_1d'] = np.log(close / np.roll(close, 1))
        df['Return_5d'] = np.log(close / np.roll(close, 5))
        df['Return_10d'] = np.log(close / np.roll(close, 10))
        
        # 波动率
        df['Volatility_5d'] = df['Return_1d'].rolling(window=5).std()
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        
        # 价格动量
        df['Momentum_5d'] = close - np.roll(close, 5)
        df['Momentum_20d'] = close - np.roll(close, 20)
        
        # 相对强度
        df['Relative_Strength'] = close / df['SMA_20']
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        # 确保 Date 列是 datetime 类型，如果不是则转换
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            else:
                raise ValueError("DataFrame index must be DatetimeIndex or have a 'Date' column")

        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Day_of_Month'] = df.index.day
        
        # 周期性编码
        df['Day_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        return df
    
    def create_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
        """创建预测目标（未来N日收益率）"""
        df['Target'] = np.log(df['Close'].shift(-horizon) / df['Close'])
        return df
    
    def prepare_features(self, df: pd.DataFrame, horizon: int = 5) -> tuple:
        """准备最终特征矩阵"""
        df = self.add_technical_indicators(df)
        df = self.add_statistical_features(df)
        df = self.add_time_features(df)
        df = self.create_target(df, horizon)
        
        # 删除NaN值 (由于滚动计算和 shift，头部和尾部会有 NaN)
        df = df.dropna()
        
        if df.empty:
            raise ValueError("处理后数据为空，请检查输入数据长度是否足够计算所有指标。")

        # 特征列（排除目标列和原始价格列）
        exclude_cols = ['Close', 'Target', 'Open', 'High', 'Low', 'Volume']
        # 如果 Date 在 columns 里也排除（虽然通常它是 index）
        if 'Date' in df.columns:
            exclude_cols.append('Date')
            
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols, df