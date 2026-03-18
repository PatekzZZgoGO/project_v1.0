# src/models/lstm.py
"""
LSTM 股票预测模型 (PyTorch)
支持配置管理 + 模型保存/加载 + 训练历史可视化
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
from pathlib import Path
from datetime import datetime

# 导入项目配置
from configs.config import (
    MODELS_DIR,
    RESULTS_DIR,
    LSTMConfig,
    ensure_directories
)


class LSTMStockPredictor(nn.Module):
    """LSTM 股票预测模型 (PyTorch)"""
    
    def __init__(self, input_size: int, 
                 hidden_size: int = None,
                 num_layers: int = None,
                 dropout: float = None):
        """
        初始化 LSTM 模型
        
        Args:
            input_size: 输入特征数
            hidden_size: 隐藏层大小（从配置读取默认值）
            num_layers: LSTM 层数（从配置读取默认值）
            dropout: Dropout 比例（从配置读取默认值）
        """
        super(LSTMStockPredictor, self).__init__()
        
        # 从配置读取默认参数（允许手动覆盖）
        self.hidden_size = hidden_size or LSTMConfig.HIDDEN_SIZE
        self.num_layers = num_layers or LSTMConfig.NUM_LAYERS
        self.dropout = dropout or LSTMConfig.DROPOUT
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=4,
            dropout=self.dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(self.hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, x):
        """前向传播"""
        # LSTM 层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 取最后一个时间步
        out = attn_out[:, -1, :]
        
        # 全连接层
        out = self.dropout_layer(self.relu(self.fc1(out)))
        out = self.fc2(out)
        
        return out


class StockPredictionTrainer:
    """模型训练器 (PyTorch)"""
    
    def __init__(self, input_size: int, 
                 sequence_length: int = None,
                 learning_rate: float = None,
                 batch_size: int = None,
                 epochs: int = None):
        """
        初始化训练器
        
        Args:
            input_size: 输入特征数
            sequence_length: 时间序列长度（从配置读取默认值）
            learning_rate: 学习率（从配置读取默认值）
            batch_size: 批次大小（从配置读取默认值）
            epochs: 训练轮数（从配置读取默认值）
        """
        # 确保目录存在
        ensure_directories()
        
        # 从配置读取默认参数（允许手动覆盖）
        self.sequence_length = sequence_length or LSTMConfig.SEQUENCE_LENGTH
        self.learning_rate = learning_rate or LSTMConfig.LEARNING_RATE
        self.batch_size = batch_size or LSTMConfig.BATCH_SIZE
        self.epochs = epochs or LSTMConfig.EPOCHS
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备：{self.device}")
        
        # 初始化模型
        self.model = LSTMStockPredictor(input_size).to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 模型保存路径
        self.model_path = MODELS_DIR / 'best_lstm_model.pth'
        self.history_path = RESULTS_DIR / 'training_history.npy'
        
        # 早停参数
        self.patience = 15
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        创建时间序列样本
        
        Args:
            X: 特征数组 (n_samples, n_features)
            y: 目标数组 (n_samples,)
            
        Returns:
            tuple: (X_seq, y_seq) 序列样本
        """
        X_seq, y_seq = [], []
        
        # 确保数据足够长
        if len(X) <= self.sequence_length:
            raise ValueError(
                f"❌ 数据量 ({len(X)}) 不足以创建长度为 {self.sequence_length} 的序列"
            )
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = None,
              batch_size: int = None,
              validation_split: float = 0.2,
              save_model: bool = True) -> dict:
        """
        训练模型
        
        Args:
            X: 特征数组
            y: 目标数组
            epochs: 训练轮数（覆盖默认值）
            batch_size: 批次大小（覆盖默认值）
            validation_split: 验证集比例
            save_model: 是否保存模型
            
        Returns:
            dict: 训练历史 {'train_loss': [], 'val_loss': []}
        """
        # 覆盖默认参数
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        
        print("\n" + "=" * 60)
        print("🔥 LSTM 模型训练配置")
        print("=" * 60)
        print(f"序列长度：{self.sequence_length}")
        print(f"隐藏层大小：{self.model.hidden_size}")
        print(f"LSTM 层数：{self.model.num_layers}")
        print(f"Dropout: {self.model.dropout}")
        print(f"学习率：{self.learning_rate}")
        print(f"批次大小：{batch_size}")
        print(f"训练轮数：{epochs}")
        print(f"验证集比例：{validation_split * 100:.1f}%")
        print("=" * 60 + "\n")
        
        # 创建时间序列样本
        print("⏳ 正在创建时间序列样本...")
        X_seq, y_seq = self.create_sequences(X, y)
        print(f"✅ 序列样本形状：X={X_seq.shape}, y={y_seq.shape}")
        
        # 划分训练验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_seq, y_seq, test_size=validation_split, shuffle=False
        )
        print(f"📊 训练集：{len(X_train)} 样本，验证集：{len(X_val)} 样本")
        
        # 转换为 Tensor
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # 数据加载器
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 训练循环
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        print(f"\n🚀 开始训练 (Epochs: {epochs})...\n")
        
        for epoch in range(epochs):
            # ========== 训练阶段 ==========
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # ========== 验证阶段 ==========
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_t)
                val_loss = self.criterion(val_output, y_val_t).item()
            
            # 记录历史
            current_lr = self.optimizer.param_groups[0]['lr']
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['lr'].append(current_lr)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_model:
                    self.save_model()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} - "
                      f"Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, "
                      f"LR: {current_lr:.6f}")
            
            # 早停检查
            if self.patience_counter >= self.patience:
                print(f"\n⚠️ 早停触发 (Early Stopping) @ Epoch {epoch+1}")
                print(f"   最佳验证损失：{self.best_val_loss:.6f}")
                break
        
        # 加载最佳模型
        if save_model and self.model_path.exists():
            self.load_model()
            print("\n✅ 已加载最佳模型权重")
        
        # 保存训练历史
        self._save_history(history)
        
        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print(f"📈 最佳验证损失：{self.best_val_loss:.6f}")
        print(f"💾 模型保存路径：{self.model_path}")
        print("=" * 60 + "\n")
        
        return history
    
    def save_model(self, path: Path = None):
        """
        保存模型权重
        
        Args:
            path: 保存路径（默认使用配置路径）
        """
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'sequence_length': self.sequence_length,
                'input_size': self.model.lstm.input_size,
            }
        }, save_path)
        
        print(f"💾 模型已保存：{save_path}")
    
    def load_model(self, path: Path = None, input_size: int = None):
        """
        加载模型权重
        
        Args:
            path: 模型路径（默认使用配置路径）
            input_size: 输入特征数（用于初始化模型）
            
        Returns:
            bool: 是否加载成功
        """
        load_path = path or self.model_path
        
        if not load_path.exists():
            print(f"⚠️ 模型文件不存在：{load_path}")
            return False
        
        try:
            checkpoint = torch.load(load_path, map_location=self.device)
            
            # 如果需要，重新初始化模型
            if input_size is not None:
                self.model = LSTMStockPredictor(input_size).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            print(f"📂 模型已加载：{load_path}")
            print(f"   最佳验证损失：{self.best_val_loss:.6f}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
            return False
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数组
            
        Returns:
            np.ndarray: 预测结果
        """
        self.model.eval()
        
        # 创建序列
        X_seq, _ = self.create_sequences(X, np.zeros(len(X)))
        X_t = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy()
        
        return predictions.flatten()
    
    def predict_single(self, X_seq: np.ndarray) -> float:
        """
        预测单个序列
        
        Args:
            X_seq: 单个序列样本 (sequence_length, n_features)
            
        Returns:
            float: 预测值
        """
        self.model.eval()
        
        # 转换为 Tensor
        X_t = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(X_t).cpu().numpy()
        
        return float(prediction[0, 0])
    
    def _save_history(self, history: dict):
        """保存训练历史"""
        np.save(self.history_path, history)
        print(f"💾 训练历史已保存：{self.history_path}")
    
    def load_history(self) -> dict:
        """加载训练历史"""
        if self.history_path.exists():
            history = np.load(self.history_path, allow_pickle=True).item()
            print(f"📂 训练历史已加载：{self.history_path}")
            return history
        else:
            print(f"⚠️ 训练历史文件不存在：{self.history_path}")
            return None
    
    def plot_training_history(self, history: dict = None, save_path: Path = None):
        """
        绘制训练历史曲线
        
        Args:
            history: 训练历史（如果为 None，尝试加载）
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        if history is None:
            history = self.load_history()
        
        if history is None:
            print("❌ 无法绘制：没有训练历史数据")
            return
        
        save_path = save_path or (RESULTS_DIR / 'training_history.png')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 学习率曲线
        if 'lr' in history:
            axes[1].plot(history['lr'], label='Learning Rate', color='red', linewidth=2)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 训练历史图已保存：{save_path}")