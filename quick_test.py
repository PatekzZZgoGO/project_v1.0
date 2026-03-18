"""
快速测试模型是否能加载
"""
import torch
from pathlib import Path

MODEL_PATH = Path('models/multi_stock_streaming_best.pth')

print("=" * 60)
print("🔍 模型文件检查")
print("=" * 60)

# 检查文件
if not MODEL_PATH.exists():
    print(f"❌ 模型文件不存在：{MODEL_PATH}")
    exit(1)

print(f"✅ 模型文件存在：{MODEL_PATH}")
print(f"   文件大小：{MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")

# 加载 checkpoint
print("\n📥 加载模型信息...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

print("\n📊 模型信息：")
print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"   最佳验证 Loss: {checkpoint.get('best_val_loss', 'N/A')}")

config = checkpoint.get('config', {})
print(f"\n⚙️  模型配置：")
for key, value in config.items():
    print(f"   {key}: {value}")

print(f"\n📦 模型参数量：")
state_dict = checkpoint['model_state_dict']
total_params = sum(p.numel() for p in state_dict.values())
print(f"   总参数：{total_params:,}")

print("\n✅ 模型文件正常！")
print("=" * 60)