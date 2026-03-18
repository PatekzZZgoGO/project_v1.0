# src/analysis/analyze_features.py
"""
特征重要性分析脚本
对已训练好的模型进行特征重要性分析
"""
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split

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


def load_model_and_data(stock_code: str, device: str = 'cpu'):
    """加载模型和数据"""
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，自动切换到 CPU")
        device = 'cpu'
    
    device = torch.device(device)
    print(f"🚀 使用设备：{device}")
    
    model_path = MODELS_DIR / f'{stock_code.replace(".", "_")}_best.pth'
    if not model_path.exists():
        model_path = MODELS_DIR / 'best_lstm_model.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    input_size = checkpoint['config']['input_size']
    model = LSTMStockPredictor(
        input_size=input_size,
        hidden_size=checkpoint['config']['hidden_size'],
        num_layers=checkpoint['config']['num_layers'],
        dropout=checkpoint['config']['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    df = load_local_stock_data(stock_code)
    df = create_features(df)
    
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
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X, y, scaler, target_scaler = prepare_data(df, feature_cols, 'Target')
    
    sequence_length = checkpoint['config']['sequence_length']
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    _, X_val, _, y_val = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    
    print(f"✅ 模型加载完成")
    print(f"✅ 验证集样本数：{len(X_val)}")
    
    return model, X_val, y_val, feature_cols, device


def analyze_feature_importance(model, X_val, y_val, feature_names, device, stock_code: str):
    """分析特征重要性"""
    print("\n" + "=" * 70)
    print("🔍 正在分析特征重要性...")
    print("=" * 70)
    
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
        base_output = model(X_val_t)
        base_loss = criterion(base_output, y_val_t).item()
    
    print(f"基准验证损失：{base_loss:.6f}\n")
    
    importance = []
    n_features = X_val.shape[2]
    
    for i in range(n_features):
        X_permuted = X_val.copy()
        np.random.shuffle(X_permuted[:, :, i])
        
        with torch.no_grad():
            X_perm_t = torch.FloatTensor(X_permuted).to(device)
            perm_output = model(X_perm_t)
            perm_loss = criterion(perm_output, y_val_t).item()
        
        importance_score = perm_loss - base_loss
        importance.append((feature_names[i], importance_score))
        
        if (i + 1) % 5 == 0:
            print(f"  进度：{i+1}/{n_features}")
    
    importance_sorted = sorted(importance, key=lambda x: x[1], reverse=True)
    
    print("\n📊 特征重要性排名")
    print("=" * 70)
    print(f"{'排名':<6} {'特征名':<25} {'重要性得分':>15} {'影响程度'}")
    print("-" * 70)
    
    for rank, (name, score) in enumerate(importance_sorted, 1):
        if score > 0.001:
            level = "🔴 高"
        elif score > 0.0001:
            level = "🟡 中"
        else:
            level = "🟢 低"
        print(f"{rank:<6} {name:<25} {score:>15.6f}  {level}")
    
    print("=" * 70)
    
    # 🔧 修复：stock_code 现在已定义
    save_path = RESULTS_DIR / f'feature_importance_{stock_code.replace(".", "_")}.npy'
    np.save(save_path, importance_sorted, allow_pickle=True)
    
    plot_feature_importance(importance_sorted, stock_code)
    
    return importance_sorted


def plot_feature_importance(importance_sorted: list, stock_code: str, top_n: int = 15):
    """绘制特征重要性图"""
    import matplotlib.pyplot as plt
    
    top_features = importance_sorted[:top_n][::-1]
    names = [f[0] for f in top_features]
    scores = [f[1] for f in top_features]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    
    bars = ax.barh(names, scores, color=colors)
    ax.set_xlabel('Importance Score (Loss Increase)', fontsize=12)
    ax.set_title(f'Feature Importance - {stock_code}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + max(scores)*0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = RESULTS_DIR / f'feature_importance_{stock_code.replace(".", "_")}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 特征重要性图已保存：{save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='特征重要性分析')
    parser.add_argument('-s', '--stock', type=str, default='000001.SZ',
                        help='股票代码')
    parser.add_argument('--cuda', action='store_true', help='使用 GPU')
    
    args = parser.parse_args()
    
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    
    try:
        model, X_val, y_val, feature_cols, device = load_model_and_data(args.stock, device)
        # 🔧 修复：传递 stock_code 参数
        analyze_feature_importance(model, X_val, y_val, feature_cols, device, args.stock)
    except Exception as e:
        print(f"❌ 分析失败：{e}")
        import traceback
        traceback.print_exc()