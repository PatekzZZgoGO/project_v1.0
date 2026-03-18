# restructure_project.py
import os
import shutil
from pathlib import Path

# 定义新目录结构
NEW_STRUCTURE = {
    'src': ['__init__.py'],
    'src/data': ['__init__.py'],
    'src/features': ['__init__.py'],
    'src/models': ['__init__.py'],
    'src/backtest': ['__init__.py'],
    'src/utils': ['__init__.py'],
    'models': [],
    'data': [],
    'data/raw': [],
    'data/processed': [],
    'configs': [],
    'logs': [],
    'notebooks': [],
    'tests': [],
}

# 文件映射：原文件名 → 新路径
FILE_MAPPING = {
    'DataGet.py': 'src/data/data_get.py',
    'FeatureEngineer.py': 'src/features/feature_engineer.py',
    'LSTMStockPredictor.py': 'src/models/lstm.py',
    'TraditionalMLModels.py': 'src/models/traditional.py',
    'BacktestSystem.py': 'src/backtest/backtest_system.py',
    'best_lstm_model.pth': 'models/best_lstm_model.pth',
    'test_stock_data.csv': 'data/test_stock_data.csv',
}

# 需要删除的临时文件
TEMP_FILES = {'show.py', 'fix_imports.py', 'generate_test_data.py', 'show_tree.py'}

# 需要保留在根目录的文件
ROOT_FILES = {'main.py', 'README.txt', 'lib.txt', 'requirements.txt', '.gitignore'}

def create_directories():
    """创建新目录结构"""
    print("📁 创建目录结构...")
    for dir_path, init_files in NEW_STRUCTURE.items():
        Path(dir_path).mkdir(exist_ok=True)
        for init_file in init_files:
            init_path = Path(dir_path) / init_file
            if not init_path.exists():
                init_path.touch()
                print(f"   ✅ 创建 {init_path}")
    print("   ✅ 目录结构创建完成\n")

def move_files():
    """移动文件到新位置"""
    print("📦 移动文件...")
    for old_name, new_path in FILE_MAPPING.items():
        old_path = Path(old_name)
        if old_path.exists():
            new_path_obj = Path(new_path)
            new_path_obj.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path_obj))
            print(f"   ✅ {old_name} → {new_path}")
        else:
            print(f"   ⚠️  {old_name} 不存在，跳过")
    print("   ✅ 文件移动完成\n")

def clean_temp_files():
    """清理临时文件"""
    print("🧹 清理临时文件...")
    for temp_file in TEMP_FILES:
        temp_path = Path(temp_file)
        if temp_path.exists():
            temp_path.unlink()
            print(f"   ✅ 删除 {temp_file}")
    print("   ✅ 清理完成\n")

def update_imports():
    """提示用户需要更新导入语句"""
    print("⚠️  需要更新导入语句！")
    print("""
    在 main.py 和其他文件中，将导入语句从：
        from LSTMStockPredictor import LSTMStockPredictor
        from DataGet import DataGet
        from FeatureEngineer import FeatureEngineer
        from BacktestSystem import BacktestSystem
    
    改为：
        from src.models.lstm import LSTMStockPredictor
        from src.data.data_get import DataGet
        from src.features.feature_engineer import FeatureEngineer
        from src.backtest.backtest_system import BacktestSystem
    
    或者添加 src 到 Python 路径：
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    """)

def create_requirements():
    """创建 requirements.txt"""
    req_path = Path('requirements.txt')
    if not req_path.exists():
        requirements = """
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
torch>=2.0.0
matplotlib>=3.7.0
akshare>=1.10.0
tqdm>=4.65.0
"""
        req_path.write_text(requirements.strip())
        print("   ✅ 创建 requirements.txt")

def create_gitignore():
    """创建 .gitignore"""
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
.venv/
env/
*.egg-info/
dist/
build/

# 模型文件
models/*.pth
models/*.pt
models/*.h5

# 数据文件
data/*.csv
data/*.parquet

# 日志
logs/*.log

# IDE
.idea/
.vscode/
*.swp
*.swo

# 系统
.DS_Store
Thumbs.db
"""
        gitignore_path.write_text(gitignore_content.strip())
        print("   ✅ 创建 .gitignore")

def main():
    print("=" * 60)
    print("🚀 项目结构优化开始")
    print("=" * 60 + "\n")
    
    create_directories()
    move_files()
    clean_temp_files()
    create_requirements()
    create_gitignore()
    update_imports()
    
    print("\n" + "=" * 60)
    print("✅ 项目结构优化完成！")
    print("=" * 60)
    print("\n📋 下一步操作：")
    print("1. 运行 python restructure_project.py 的备份（已自动完成）")
    print("2. 更新 main.py 中的导入语句")
    print("3. 运行 python main.py 测试是否正常工作")
    print("4. 如有问题，检查导入路径是否正确")

if __name__ == "__main__":
    main()