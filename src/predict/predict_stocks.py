"""
股票预测脚本 - 使用训练好的 LSTM 模型
自动计算技术指标特征
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / 'models' / 'multi_stock_streaming_best.pth'

# ⚠️ 数据目录：stocks 子目录
DATA_DIR = PROJECT_ROOT / 'data' / 'raw' / 'stocks'
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

RESULTS_DIR = PROJECT_ROOT / 'results' / 'predictions'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================== 模型定义 ====================
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMStockPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
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
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        out = self.dropout_layer(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out


# ==================== 特征列 ====================
FEATURE_COLS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'MA5', 'MA10', 'MA20', 'MA60',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'RSI',
    'BB_upper', 'BB_middle', 'BB_lower',
    'Volume_MA5', 'Volume_MA10',
    'Close_Lag1', 'Close_Lag2', 'Close_Lag3',
    'Return_1d', 'Return_5d'
]


# ==================== 自动计算特征 ====================
def calculate_features(df):
    """从原始 OHLCV 数据计算技术指标"""
    df = df.copy()
    
    # 标准化列名（支持多种格式）
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['open', '开盘']:
            col_mapping[col] = 'Open'
        elif col_lower in ['high', '最高']:
            col_mapping[col] = 'High'
        elif col_lower in ['low', '最低']:
            col_mapping[col] = 'Low'
        elif col_lower in ['close', '收盘']:
            col_mapping[col] = 'Close'
        elif col_lower in ['volume', '成交量']:
            col_mapping[col] = 'Volume'
        elif col_lower in ['date', '日期', 'time', '时间']:
            col_mapping[col] = 'Date'
    
    df = df.rename(columns=col_mapping)
    
    # 检查必要列
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None
    
    # 确保数值类型
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=required)
    
    if len(df) < 70:  # 需要足够数据计算特征
        return None
    
    # 移动平均线
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 布林带
    df['BB_middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    
    # 成交量均线
    df['Volume_MA5'] = df['Volume'].rolling(5).mean()
    df['Volume_MA10'] = df['Volume'].rolling(10).mean()
    
    # 滞后特征
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    df['Close_Lag3'] = df['Close'].shift(3)
    
    # 收益率
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    
    # 删除空值
    df = df.dropna().reset_index(drop=True)
    
    return df


# ==================== 加载模型 ====================
def load_trained_model(model_path, device='cpu'):
    print(f"📥 正在加载模型：{model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"❌ 模型文件不存在：{model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    input_size = config.get('input_size', 23)
    hidden_size = config.get('hidden_size', 64)
    num_layers = config.get('num_layers', 2)
    dropout = config.get('dropout', 0.2)
    sequence_length = config.get('sequence_length', 60)
    
    print(f"   输入特征数：{input_size}")
    print(f"   隐藏层大小：{hidden_size}")
    print(f"   LSTM 层数：{num_layers}")
    print(f"   Dropout: {dropout}")
    print(f"   序列长度：{sequence_length}")
    
    model = LSTMStockPredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型加载成功！")
    print(f"   训练股票数：{config.get('stock_count', 'N/A')}")
    
    return model, config


# ==================== 股票代码格式转换 ====================
def code_to_filename(stock_code):
    """将股票代码转换为文件名格式"""
    # 603339.SH -> 603339_SH
    # 000001.SZ -> 000001_SZ
    return stock_code.replace('.', '_')


def filename_to_code(filename):
    """将文件名转换为股票代码"""
    # 603339_SH.csv -> 603339.SH
    name = filename.replace('.csv', '')
    parts = name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return name


# ==================== 加载股票数据 ====================
def load_stock_data(stock_code, sequence_length=60):
    """加载单只股票数据并计算特征"""
    filename = code_to_filename(stock_code)
    data_path = DATA_DIR / f'{filename}.csv'
    
    if not data_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(data_path)
        
        # 检查是否是股票信息表（非行情数据）
        if 'market' in df.columns and 'Close' not in df.columns:
            return None, None
        
        # 计算特征
        df = calculate_features(df)
        
        if df is None or len(df) < sequence_length:
            return None, None
        
        # 提取特征
        data = df[FEATURE_COLS].iloc[-sequence_length:].values
        X_seq = torch.FloatTensor(data).unsqueeze(0)
        
        return X_seq, df
        
    except Exception as e:
        print(f"⚠️ {stock_code} 加载失败：{e}")
        return None, None


# ==================== 预测单只股票 ====================
def predict_stock(model, stock_code, device='cpu', sequence_length=60):
    X_seq, df = load_stock_data(stock_code, sequence_length)
    
    if X_seq is None:
        return None, None, None
    
    X_seq = X_seq.to(device)
    
    with torch.no_grad():
        prediction = model(X_seq)
    
    pred_value = prediction.cpu().numpy()[0, 0]
    latest_close = df['Close'].iloc[-1] if 'Close' in df.columns else None
    
    return pred_value, latest_close, df


# ==================== 批量预测 ====================
def predict_multiple_stocks(model, stock_codes, device='cpu', sequence_length=60):
    results = []
    success_count = 0
    fail_count = 0
    
    print(f"\n🚀 开始预测 {len(stock_codes)} 只股票...")
    print("=" * 60)
    
    for i, code in enumerate(stock_codes):
        if (i + 1) % 100 == 0:
            print(f"  进度：{i + 1}/{len(stock_codes)} (成功：{success_count}, 失败：{fail_count})")
        
        pred, close, df = predict_stock(model, code, device, sequence_length)
        
        if pred is not None:
            results.append({
                'stock_code': code,
                'prediction': pred,
                'latest_close': close,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            success_count += 1
        else:
            fail_count += 1
    
    if len(results) == 0:
        print("\n⚠️ 没有成功的预测结果！")
        return None
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('prediction', ascending=False)
    
    output_path = RESULTS_DIR / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"✅ 预测完成！")
    print(f"   成功：{success_count}")
    print(f"   失败：{fail_count}")
    print(f"   结果已保存：{output_path}")
    
    return results_df


# ==================== 获取股票代码 ====================
def get_all_stock_codes():
    """获取所有股票代码"""
    if not DATA_DIR.exists():
        return []
    
    codes = []
    for f in DATA_DIR.glob('*.csv'):
        # 跳过股票信息表
        if f.name in ['stocks.csv', 'stock_info.csv', 'index.csv']:
            continue
        code = filename_to_code(f.name)
        codes.append(code)
    
    return sorted(codes)


# ==================== 主函数 ====================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备：{device}")
    print("=" * 60)
    
    # 数据目录信息
    print(f"📂 数据目录：{DATA_DIR}")
    if DATA_DIR.exists():
        csv_count = len(list(DATA_DIR.glob('*.csv')))
        print(f"   CSV 文件数：{csv_count}")
    print("=" * 60)
    
    # 加载模型
    model, config = load_trained_model(MODEL_PATH, device)
    
    sequence_length = config.get('sequence_length', 60)
    
    # 获取股票代码
    all_codes = get_all_stock_codes()
    print(f"📈 可用股票数：{len(all_codes)}")
    
    if len(all_codes) == 0:
        print("❌ 没有可用的股票数据！")
        print(f"   请检查目录：{DATA_DIR}")
        return
    
    # 测试前 100 只
    test_codes = all_codes[:min(100, len(all_codes))]
    print(f"\n🧪 测试模式：预测前 {len(test_codes)} 只股票")
    
    results = predict_multiple_stocks(model, test_codes, device, sequence_length)
    
    if results is not None:
        print("\n📊 预测结果 Top 10：")
        print(results.head(10).to_string(index=False))
    
    # ========== 全量预测（取消注释使用） ==========
    # print(f"\n🚀 全量预测：{len(all_codes)} 只股票")
    # results = predict_multiple_stocks(model, all_codes, device, sequence_length)


if __name__ == '__main__':
    main()