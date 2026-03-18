# main.py
"""
股票预测系统 - 机器学习版
整合数据获取、特征工程、模型训练、回测分析
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 核心模块导入
from src.data.data_get import DataFetcher
from src.features.feature_engineer import FeatureEngineer
from src.models.lstm import StockPredictionTrainer
from src.models.traditional import TraditionalMLModels
from src.backtest.backtest_system import BacktestSystem

# 配置导入
from configs.config import (
    PROJECT_ROOT,
    DATA_DIR,
    DATA_STOCKS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    LSTMConfig,
    BacktestConfig,
    DataFetchConfig,
    ensure_directories,
    get_config_summary,
)

# 第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def main():
    """主程序"""
    # 确保目录存在
    ensure_directories()
    
    # 打印配置摘要（可选）
    get_config_summary()
    
    print("\n" + "=" * 70)
    print("🚀 股票预测系统 - 机器学习版")
    print("=" * 70)
    print(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"项目根目录：{PROJECT_ROOT}")
    print("=" * 70 + "\n")
    
    # 确保必要目录存在
    ensure_directories()
    
    # ========== 1. 获取数据 ==========
    print("📊 步骤 1/7: 获取股票数据")
    print("-" * 70)
    
    fetcher = DataFetcher(symbol='AAPL', market='US')
    df = fetcher.fetch_historical_data(
        start_date='2020-01-01',
        end_date=None,
        use_cache=True
    )
    
    # 检查数据
    if df is None or df.empty:
        print("⚠️ API 获取失败，尝试加载本地测试数据...")
        test_data_paths = [
            DATA_DIR / 'test_stock_data.csv',
            PROJECT_ROOT / 'test_stock_data.csv',
        ]
        
        for test_path in test_data_paths:
            if test_path.exists():
                df = pd.read_csv(test_path, index_col='Date', parse_dates=True)
                print(f"✅ 已加载本地测试数据：{len(df)} 条记录 (路径：{test_path})")
                break
        else:
            print("❌ 未找到任何数据源，程序终止")
            return
    
    print(f"✅ 数据获取完成：{len(df)} 条记录")
    print(f"   日期范围：{df.index[0]} 至 {df.index[-1]}")
    print(f"   列名：{list(df.columns)}")
    
    # ========== 2. 特征工程 ==========
    print("\n📊 步骤 2/7: 特征工程")
    print("-" * 70)
    
    engineer = FeatureEngineer()
    X, y, feature_names, df_processed = engineer.prepare_features(df, horizon=5)
    
    print(f"✅ 特征工程完成")
    print(f"   特征数量：{len(feature_names)}")
    print(f"   样本数量：{len(X)}")
    print(f"   特征名：{feature_names[:5]}... (共{len(feature_names)}个)")
    
    # 检查数据有效性
    if len(X) < 100:
        print("⚠️ 样本数量过少，可能影响模型训练效果")
    
    # ========== 3. 划分数据集 ==========
    print("\n📊 步骤 3/7: 划分数据集")
    print("-" * 70)
    
    train_ratio = 0.8
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"✅ 数据集划分完成")
    print(f"   训练集：{len(X_train)} 样本 ({train_ratio*100:.0f}%)")
    print(f"   测试集：{len(X_test)} 样本 ({(1-train_ratio)*100:.0f}%)")
    
    # ========== 4. 训练传统 ML 模型 ==========
    print("\n📊 步骤 4/7: 训练传统机器学习模型")
    print("-" * 70)
    
    try:
        ml_models = TraditionalMLModels()
        ml_results = ml_models.train_and_evaluate(X_train, X_test, y_train, y_test)
        print("✅ 传统 ML 模型训练完成")
    except Exception as e:
        print(f"⚠️ 传统 ML 模型训练失败：{e}")
        ml_results = None
    
    # ========== 5. 训练 LSTM 模型 ==========
    print("\n📊 步骤 5/7: 训练 LSTM 深度学习模型")
    print("-" * 70)
    
    try:
        trainer = StockPredictionTrainer(
            input_size=X.shape[1],
            sequence_length=30,
            epochs=100,
            batch_size=32
        )
        
        history = trainer.train(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            save_model=True
        )
        
        print("✅ LSTM 模型训练完成")
        
        # 绘制训练历史
        trainer.plot_training_history(history)
        
    except Exception as e:
        print(f"⚠️ LSTM 模型训练失败：{e}")
        history = None
        trainer = None
    
    # ========== 6. 预测 ==========
    print("\n📊 步骤 6/7: 生成预测")
    print("-" * 70)
    
    if trainer is not None:
        try:
            predictions = trainer.predict(X_test)
            print(f"✅ 预测完成：{len(predictions)} 条预测结果")
        except Exception as e:
            print(f"⚠️ 预测失败：{e}")
            predictions = np.zeros(len(X_test))
    else:
        predictions = np.zeros(len(X_test))
    
    # ========== 7. 回测 ==========
    print("\n📊 步骤 7/7: 策略回测")
    print("-" * 70)
    
    # 获取测试集对应的日期数据
    df_test = df_processed.iloc[split_idx:].copy()
    
    # 确保预测值和测试数据长度一致
    min_len = min(len(predictions), len(df_test))
    predictions = predictions[:min_len]
    df_test = df_test.iloc[:min_len]
    
    backtest = BacktestSystem(
        initial_capital=100000,
        commission_rate=0.0003,
        slippage=0.001
    )
    
    metrics = backtest.run_backtest(
        df=df_test,
        predictions=predictions,
        threshold=0.01,
        save_results=True
    )
    
    print("✅ 回测完成")
    
    # ========== 8. 生成报告 ==========
    print("\n📊 生成最终报告")
    print("-" * 70)
    
    report = backtest.generate_report(metrics)
    print(report)
    
    # ========== 9. 可视化 ==========
    print("\n📊 生成可视化图表")
    print("-" * 70)
    
    visualize_results(
        df=df_processed,
        predictions=predictions,
        actual=y_test[:min_len],
        history=history,
        metrics=metrics
    )
    
    print("\n" + "=" * 70)
    print("✅ 所有步骤完成！")
    print("=" * 70)
    print(f"📁 结果保存目录：{backtest.results_dir}")
    print("=" * 70 + "\n")


def visualize_results(df: pd.DataFrame, predictions: np.ndarray, 
                      actual: np.ndarray, history: dict, metrics: dict):
    """可视化结果"""
    # 🔥 添加中文字体支持
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    import matplotlib.pyplot as plt
    
    # 强制对齐长度
    min_len = min(len(predictions), len(actual))
    predictions = predictions[:min_len]
    actual = actual[:min_len]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # ========== 图 1: 预测 vs 实际 ==========
    ax1 = axes[0, 0]
    x_axis = range(len(actual))
    ax1.plot(x_axis, actual, label='Actual', alpha=0.7, linewidth=2, color='blue')
    ax1.plot(x_axis, predictions, label='Predicted', alpha=0.7, linewidth=2, 
             linestyle='--', color='orange')
    ax1.legend()
    ax1.set_title('Predicted vs Actual Returns', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Return')
    ax1.grid(True, alpha=0.3)
    
    # ========== 图 2: 训练损失 ==========
    ax2 = axes[0, 1]
    if history is not None:
        ax2.plot(history['train_loss'], label='Train Loss', linewidth=2, color='green')
        ax2.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red')
        ax2.legend()
        ax2.set_title('Training History', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Training History', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14)
    
    # ========== 图 3: 收益率分布 ==========
    ax3 = axes[0, 2]
    ax3.hist(actual, bins=30, alpha=0.6, label='Actual', color='blue', 
             edgecolor='black')
    ax3.hist(predictions, bins=30, alpha=0.6, label='Predicted', color='orange', 
             edgecolor='black')
    ax3.legend()
    ax3.set_title('Return Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== 图 4: 累积收益 ==========
    ax4 = axes[1, 0]
    cumulative_actual = np.cumsum(actual)
    cumulative_pred = np.cumsum(predictions)
    ax4.plot(x_axis, cumulative_actual, label='Actual Cumulative', linewidth=2, 
             color='blue')
    ax4.plot(x_axis, cumulative_pred, label='Predicted Cumulative', linewidth=2, 
             linestyle='--', color='orange')
    ax4.legend()
    ax4.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Cumulative Return')
    ax4.grid(True, alpha=0.3)
    
    # ========== 图 5: 组合价值曲线 ==========
    ax5 = axes[1, 1]
    if 'portfolio_values' in metrics:
        portfolio_vals = metrics['portfolio_values']
        ax5.plot(range(len(portfolio_vals)), portfolio_vals, linewidth=2, 
                 color='green', label='Portfolio Value')
        ax5.axhline(y=metrics.get('initial_capital', 100000), color='gray', 
                   linestyle='--', label='Initial Capital', alpha=0.5)
        ax5.legend()
        ax5.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Trading Day')
        ax5.set_ylabel('Value (元)')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'No Portfolio Data', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=14)
    
    # ========== 图 6: 回测指标摘要 ==========
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    📊 回测绩效摘要
    
    总收益率：{metrics.get('total_return_pct', 0):+.2f}%
    年化收益：{metrics.get('annual_return_pct', 0):+.2f}%
    夏普比率：{metrics.get('sharpe_ratio', 0):.3f}
    最大回撤：{metrics.get('max_drawdown_pct', 0):.2f}%
    胜率：{metrics.get('win_rate_pct', 0):.1f}%
    交易次数：{metrics.get('num_trades', 0)}
    盈亏比：{metrics.get('profit_factor', 0):.2f}
    """
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('Performance Summary', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    save_path = RESULTS_DIR / f'prediction_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存：{save_path}")
    plt.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
        import traceback
        traceback.print_exc()