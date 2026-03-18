# src/train/train_local.py
"""
本地数据训练脚本
使用已缓存的股票数据训练 LSTM 模型
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 导入项目配置和模型
sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import (
    DATA_STOCKS_DIR, MODELS_DIR, RESULTS_DIR,
    LSTMConfig, FeatureConfig, RunConfig,
    ensure_directories
)
from src.models.lstm import StockPredictionTrainer


# ==================== 1. 数据加载 ====================

def load_local_stock_data(stock_code: str) -> pd.DataFrame:
    """
    加载本地缓存的股票数据
    
    Args:
        stock_code: 股票代码，如 '000001.SZ'
        
    Returns:
        pd.DataFrame: 股票数据
    """
    # 转换文件名格式：000001.SZ -> 000001_SZ.csv
    file_name = stock_code.replace('.', '_') + '.csv'
    file_path = DATA_STOCKS_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"❌ 未找到数据文件：{file_path}\n"
            f"   请先运行 data_get.py 获取数据"
        )
    
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    print(f"✅ 已加载：{stock_code} | 行数：{len(df)} | 列：{list(df.columns)}")
    return df


# ==================== 2. 特征工程 ====================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建技术指标特征
    
    Args:
        df: 原始股票数据
        
    Returns:
        pd.DataFrame: 添加特征后的数据
    """
    df = df.copy()
    
    # 1. 移动平均线
    for window in [5, 10, 20, 60]:
        df[f'MA{window}'] = df['Close'].rolling(window=window).mean()
    
    # 2. MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 3. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. 布林带
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    
    # 5. 成交量均线
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    
    # 6. 滞后特征
    for lag in [1, 2, 3, 5]:
        df[f'Close_Lag{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag{lag}'] = df['Volume'].shift(lag)
    
    # 7. 收益率特征
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    # 8. 目标变量 (预测未来 N 天收盘价)
    target_horizon = FeatureConfig.TARGET_HORIZON
    df['Target'] = df['Close'].shift(-target_horizon)
    
    # 删除 NaN
    df.dropna(inplace=True)
    
    print(f"✅ 特征工程完成 | 有效行数：{len(df)} | 特征数：{len(df.columns)}")
    return df


# ==================== 3. 数据预处理 ====================

def prepare_data(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    准备训练数据（归一化）
    
    Args:
        df: 特征数据
        feature_cols: 特征列名
        target_col: 目标列名
        
    Returns:
        tuple: (X, y, scaler, target_scaler)
    """
    # 提取特征和目标
    features = df[feature_cols].values
    target = df[[target_col]].values
    
    # 分别归一化（推荐做法）
    scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    X_scaled = scaler.fit_transform(features)
    y_scaled = target_scaler.fit_transform(target)
    
    print(f"✅ 数据归一化完成 | X 形状：{X_scaled.shape} | y 形状：{y_scaled.shape}")
    
    return X_scaled, y_scaled.flatten(), scaler, target_scaler


# ==================== 4. 训练主函数 ====================

def train_stock(stock_code: str, 
                feature_cols: list = None,
                epochs: int = None,
                batch_size: int = None):
    """
    训练单只股票模型
    
    Args:
        stock_code: 股票代码
        feature_cols: 特征列（None 则使用默认）
        epochs: 训练轮数
        batch_size: 批次大小
    """
    print("\n" + "=" * 70)
    print(f"🚀 开始训练：{stock_code}")
    print("=" * 70)
    
    # 确保目录存在
    ensure_directories()
    
    # 1. 加载数据
    df = load_local_stock_data(stock_code)
    
    # 2. 特征工程
    df = create_features(df)
    
    # 3. 选择特征列
    if feature_cols is None:
        feature_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA5', 'MA10', 'MA20', 'MA60',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_upper', 'BB_middle', 'BB_lower',
            'Volume_MA5', 'Volume_MA10',
            'Close_Lag1', 'Close_Lag2', 'Close_Lag3',
            'Return_1d', 'Return_5d'
        ]
        # 过滤掉不存在的列
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    target_col = 'Target'
    
    if target_col not in df.columns:
        print("❌ 目标列不存在，请检查数据")
        return
    
    print(f"📊 使用特征 ({len(feature_cols)}): {feature_cols[:5]}...")
    
    # 4. 数据预处理
    X, y, scaler, target_scaler = prepare_data(df, feature_cols, target_col)
    
    # 5. 初始化训练器
    trainer = StockPredictionTrainer(
        input_size=len(feature_cols),
        sequence_length=LSTMConfig.SEQUENCE_LENGTH,
        learning_rate=LSTMConfig.LEARNING_RATE,
        batch_size=batch_size or LSTMConfig.BATCH_SIZE,
        epochs=epochs or LSTMConfig.EPOCHS
    )
    
    # 6. 开始训练
    history = trainer.train(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=LSTMConfig.VALIDATION_SPLIT,
        save_model=True
    )
    
    # 7. 保存缩放器（用于后续预测）
    scaler_path = MODELS_DIR / f'{stock_code.replace(".", "_")}_scaler.npy'
    target_scaler_path = MODELS_DIR / f'{stock_code.replace(".", "_")}_target_scaler.npy'
    np.save(scaler_path, scaler)
    np.save(target_scaler_path, target_scaler)
    print(f"💾 缩放器已保存：{scaler_path}")
    
    # 8. 重命名模型文件（包含股票代码）
    model_path = MODELS_DIR / f'{stock_code.replace(".", "_")}_best.pth'
    if trainer.model_path.exists():
        import shutil
        shutil.copy(trainer.model_path, model_path)
        print(f"💾 模型已保存：{model_path}")
    
    # 9. 绘制训练历史
    trainer.plot_training_history(
        history, 
        save_path=RESULTS_DIR / f'train_history_{stock_code.replace(".", "_")}.png'
    )
    
    # 10. 可视化预测结果（在训练集上）
    plot_predictions(trainer, X, y, target_scaler, stock_code)
    
    print("\n" + "=" * 70)
    print(f"✅ {stock_code} 训练完成！")
    print("=" * 70)
    
    return trainer, scaler, target_scaler


# ==================== 5. 预测可视化 ====================

def plot_predictions(trainer, X, y, target_scaler, stock_code):
    """绘制预测 vs 实际对比图"""
    import torch
    
    # 使用最后 200 个样本做预测展示
    n_samples = min(200, len(X))
    X_test = X[-n_samples:]
    y_test = y[-n_samples:]
    
    # 创建序列
    seq_len = trainer.sequence_length
    X_seq, y_seq = [], []
    for i in range(len(X_test) - seq_len):
        X_seq.append(X_test[i:i+seq_len])
        y_seq.append(y_test[i+seq_len])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # 预测
    trainer.model.eval()
    X_t = torch.FloatTensor(X_seq).to(trainer.device)
    with torch.no_grad():
        pred_scaled = trainer.model(X_t).cpu().numpy()
    
    # 反归一化
    pred = target_scaler.inverse_transform(pred_scaled).flatten()
    actual = target_scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()
    
    # 绘图
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(actual[:100], label='Actual', linewidth=2, alpha=0.8)
    ax.plot(pred[:100], label='Predicted', linewidth=2, alpha=0.8, linestyle='--')
    ax.set_title(f'Prediction vs Actual - {stock_code}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    save_path = RESULTS_DIR / f'prediction_{stock_code.replace(".", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 预测对比图已保存：{save_path}")


# ==================== 6. 命令行入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='训练股票预测模型')
    parser.add_argument('-s', '--stock', type=str, default='000001.SZ',
                        help='股票代码 (默认：000001.SZ)')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='训练轮数 (默认：从配置读取)')
    parser.add_argument('-b', '--batch', type=int, default=None,
                        help='批次大小 (默认：从配置读取)')
    
    args = parser.parse_args()
    
    try:
        train_stock(
            stock_code=args.stock,
            epochs=args.epochs,
            batch_size=args.batch
        )
    except Exception as e:
        print(f"\n❌ 训练失败：{e}")
        import traceback
        traceback.print_exc()