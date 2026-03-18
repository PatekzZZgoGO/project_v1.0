# src/train/train_all_stocks_streaming.py
"""
多股票流式训练脚本
边读取磁盘数据边训练，内存占用恒定
"""
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import gc
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))
from configs.config import (
    DATA_STOCKS_DIR, MODELS_DIR, RESULTS_DIR,
    LSTMConfig, FeatureConfig, ensure_directories
)
from src.models.lstm import LSTMStockPredictor
from src.train.train_local import (
    load_local_stock_data,
    create_features,
    prepare_data
)


# ==================== EarlyStopping 类 ====================

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
    
    def get_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


# ==================== 单股票数据加载器 ====================

class StockDataLoader:
    """单股票数据加载器"""
    def __init__(self, stock_code, feature_cols, sequence_length):
        self.stock_code = stock_code
        self.feature_cols = feature_cols
        self.sequence_length = sequence_length
        self.X = None
        self.y = None
        self.sequences = None
        self.targets = None
    
    def load(self):
        """加载单只股票数据"""
        try:
            df = load_local_stock_data(self.stock_code)
            df = create_features(df)
            
            if len(df) < self.sequence_length + 10:
                return False
            
            X, y, _, _ = prepare_data(df, self.feature_cols, 'Target')
            
            # 转换为 float32 节省内存
            self.X = X.astype(np.float32)
            self.y = y.astype(np.float32)
            
            # 创建序列
            self.sequences = []
            self.targets = []
            for i in range(len(self.X) - self.sequence_length):
                self.sequences.append(self.X[i:i + self.sequence_length])
                self.targets.append(self.y[i + self.sequence_length])
            
            self.sequences = np.array(self.sequences, dtype=np.float32)
            self.targets = np.array(self.targets, dtype=np.float32)
            
            return True
            
        except Exception as e:
            print(f"❌ {self.stock_code} 加载失败：{e}")
            return False
    
    def get_batch(self, batch_size, shuffle=True):
        """获取一个 batch 的数据"""
        if self.sequences is None:
            return None, None
        
        indices = np.arange(len(self.sequences))
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            X_batch = self.sequences[batch_indices]
            y_batch = self.targets[batch_indices]
            
            yield X_batch, y_batch
    
    def get_validation_data(self, val_split=0.2):
        """获取验证数据"""
        if self.sequences is None:
            return None, None
        
        split_idx = int(len(self.sequences) * (1 - val_split))
        
        X_val = self.sequences[split_idx:]
        y_val = self.targets[split_idx:]
        
        return X_val, y_val
    
    def clear(self):
        """释放内存"""
        self.X = None
        self.y = None
        self.sequences = None
        self.targets = None
        gc.collect()
    
    @property
    def n_samples(self):
        return len(self.sequences) if self.sequences is not None else 0


# ==================== 获取股票代码 ====================

def get_all_stock_codes():
    """获取所有本地股票代码"""
    stock_files = list(DATA_STOCKS_DIR.glob("*.csv"))
    stock_codes = []
    
    for f in stock_files:
        code = f.stem.replace("_", ".")
        stock_codes.append(code)
    
    print(f"📂 发现 {len(stock_codes)} 只本地股票")
    return sorted(stock_codes)


# ==================== 流式训练主函数 ====================

def train_streaming(
    stock_codes: list = None,
    feature_cols: list = None,
    epochs: int = None,
    batch_size: int = None,
    validation_split: float = 0.2,
    save_model: bool = True
):
    """
    流式训练：边读取磁盘边训练
    
    Args:
        stock_codes: 股票代码列表，None 表示所有本地股票
        feature_cols: 特征列
        epochs: 训练轮数
        batch_size: 批次大小
        validation_split: 验证集比例
        save_model: 是否保存模型
    """
    ensure_directories()
    
    # 配置参数
    epochs = epochs or LSTMConfig.EPOCHS
    batch_size = batch_size or LSTMConfig.BATCH_SIZE
    sequence_length = LSTMConfig.SEQUENCE_LENGTH
    
    # 获取股票代码
    if stock_codes is None:
        stock_codes = get_all_stock_codes()
    
    # 特征列
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
    
    print("\n" + "=" * 70)
    print("🚀 多股票流式训练（内存优化）")
    print("=" * 70)
    print(f"📈 股票数量：{len(stock_codes)}")
    print(f"📊 特征数量：{len(feature_cols)}")
    print(f"🔄 序列长度：{sequence_length}")
    print(f"📦 批次大小：{batch_size}")
    print(f"🔁 训练轮数：{epochs}")
    print(f"📉 验证集比例：{validation_split}")
    print("=" * 70)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 使用设备：{device}")
    
    input_size = len(feature_cols)
    model = LSTMStockPredictor(
        input_size=input_size,
        hidden_size=LSTMConfig.HIDDEN_SIZE,
        num_layers=LSTMConfig.NUM_LAYERS,
        dropout=LSTMConfig.DROPOUT
    ).to(device)
    
    # 训练配置
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTMConfig.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    early_stopping = EarlyStopping(patience=LSTMConfig.EARLY_STOPPING_PATIENCE)
    
    # 训练历史
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'stocks_processed': []
    }
    
    # ==================== 流式训练循环 ====================
    print("\n" + "=" * 70)
    print("🏋️ 开始流式训练...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = datetime.now()
        
        # 打乱股票顺序（每个 epoch 不同顺序）
        np.random.shuffle(stock_codes)
        
        model.train()
        epoch_train_losses = []
        total_samples = 0
        valid_stocks = 0
        
        # ---- 遍历所有股票 ----
        for stock_idx, code in enumerate(stock_codes):
            # 1. 加载单只股票数据
            loader = StockDataLoader(code, feature_cols, sequence_length)
            
            if not loader.load():
                loader.clear()
                continue
            
            if loader.n_samples < batch_size:
                loader.clear()
                continue
            
            valid_stocks += 1
            total_samples += loader.n_samples
            
            # 2. 用这只股票的数据训练多个 batch
            stock_losses = []
            for X_batch, y_batch in loader.get_batch(batch_size, shuffle=True):
                X_t = torch.FloatTensor(X_batch).to(device)
                y_t = torch.FloatTensor(y_batch).reshape(-1, 1).to(device)
                
                optimizer.zero_grad()
                output = model(X_t)
                loss = criterion(output, y_t)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                stock_losses.append(loss.item())
            
            avg_stock_loss = np.mean(stock_losses)
            epoch_train_losses.extend(stock_losses)
            
            # 3. 释放这只股票的内存
            loader.clear()
            
            # 4. 打印进度
            if (stock_idx + 1) % 50 == 0 or stock_idx == len(stock_codes) - 1:
                elapsed = (datetime.now() - epoch_start).total_seconds()
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"股票 {stock_idx+1}/{len(stock_codes)} | "
                      f"已处理 {code} | "
                      f"当前 Loss: {avg_stock_loss:.6f} | "
                      f"耗时：{elapsed:.1f}s")
        
        # ---- Epoch 结束 ----
        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else 0
        current_lr = optimizer.param_groups[0]['lr']
        
        # ---- 验证（使用最后几只股票） ----
        model.eval()
        val_losses = []
        
        # 取最后 10 只股票做验证
        val_codes = stock_codes[-min(10, len(stock_codes)):]
        
        for code in val_codes:
            loader = StockDataLoader(code, feature_cols, sequence_length)
            
            if not loader.load():
                loader.clear()
                continue
            
            X_val, y_val = loader.get_validation_data(validation_split)
            
            if X_val is None or len(X_val) < batch_size:
                loader.clear()
                continue
            
            # 采样验证数据（避免太多）
            max_val_samples = min(1000, len(X_val))
            val_indices = np.random.choice(len(X_val), max_val_samples, replace=False)
            X_val = X_val[val_indices]
            y_val = y_val[val_indices]
            
            with torch.no_grad():
                X_t = torch.FloatTensor(X_val).to(device)
                y_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
                
                output = model(X_t)
                loss = criterion(output, y_t)
                val_losses.append(loss.item())
            
            loader.clear()
        
        avg_val_loss = np.mean(val_losses) if val_losses else avg_train_loss
        
        # 记录历史
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        history['stocks_processed'].append(valid_stocks)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 打印 Epoch 总结
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        print(f"\n" + "-" * 70)
        print(f"Epoch {epoch+1:03d}/{epochs} 完成 | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.6f} | "
              f"耗时：{epoch_time:.1f}s")
        print("-" * 70)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            if save_model:
                save_path = MODELS_DIR / 'multi_stock_streaming_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': {
                        'input_size': input_size,
                        'hidden_size': LSTMConfig.HIDDEN_SIZE,
                        'num_layers': LSTMConfig.NUM_LAYERS,
                        'dropout': LSTMConfig.DROPOUT,
                        'sequence_length': sequence_length,
                        'feature_cols': feature_cols,
                        'stock_count': len(stock_codes),
                        'training_type': 'streaming'
                    },
                    'history': history,
                    'val_loss': best_val_loss
                }, save_path)
                print(f"  💾 已保存最佳模型：{save_path.name}")
        
        # 早停检查
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"\n⏹️ 早停触发于 Epoch {epoch+1}")
            break
        
        # 每个 epoch 后垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ==================== 训练完成 ====================
    # 恢复最佳模型
    model = early_stopping.get_best_model(model)
    
    # 绘制训练历史
    print("\n📊 正在绘制训练历史...")
    plot_training_history(history, save_path=RESULTS_DIR / 'streaming_training_history.png')
    
    print("\n" + "=" * 70)
    print("✅ 流式训练完成！")
    print(f"📈 训练股票数：{len(stock_codes)}")
    print(f"🏆 最佳验证 Loss: {best_val_loss:.6f}")
    print("=" * 70)
    
    return model, history


def plot_training_history(history: dict, save_path: Path):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1].plot(history['learning_rate'], label='Learning Rate', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 处理股票数
    axes[2].plot(history['stocks_processed'], label='Stocks Processed', color='green', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Number of Stocks')
    axes[2].set_title('Stocks Processed per Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 训练历史图已保存：{save_path}")


# ==================== 命令行入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='多股票流式训练')
    parser.add_argument('-s', '--stocks', type=str, nargs='+', default=None,
                        help='指定股票代码列表')
    parser.add_argument('-e', '--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存模型')
    
    args = parser.parse_args()
    
    train_streaming(
        stock_codes=args.stocks,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_model=not args.no_save
    )