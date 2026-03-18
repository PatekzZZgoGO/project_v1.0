┌─────────────────────────────────────────────────────────────┐
│                    股市预测系统架构                          │
├─────────────────────────────────────────────────────────────┤
│  数据层  │  yfinance/Tushare → 原始行情数据                 │
│  特征层  │  TA-Lib技术指标 + 统计特征 + 市场情绪            │
│  模型层  │  LSTM/Transformer/XGBoost 多模型融合             │
│  评估层  │  回测系统 + 风险评估 + 性能指标                  │
│  应用层  │  预测可视化 + 交易信号生成                       │
└─────────────────────────────────────────────────────────────┘

D:\project_v1.0\
├── data/
│   ├── raw/
│   │   └── stocks/          # 5011只股票原始数据 (603339_SH.csv 格式)
│   └── processed/           # 处理后的特征数据
├── models/
│   └── multi_stock_streaming_best.pth   # 训练好的 LSTM 模型
├── results/
│   └── predictions/         # 预测结果 CSV
├── src/
│   ├── predict/
│   │   └── predict_stocks.py    # 预测脚本
│   ├── models/
│   │   └── lstm.py              # 模型定义
│   ├── data/
│   │   └── process_all_stocks.py # 数据处理
│   ├── train/
│   │   └── train_model.py       # 训练脚本
│   └── backtest/
│       └── simple_backtest.py   # 回测脚本
└── requirements.txt


# ========== 日常预测流程 ==========

# 1. 确认数据存在
Get-ChildItem data\raw\stocks\*.csv | Measure-Object

# 2. 运行预测
python src\predict\predict_stocks.py

# 3. 查看结果
Get-Content results\predictions\predictions_*.csv -Head 11

# 4. Excel 打开详细结果
start excel results\predictions\

# ========== 重新训练流程 ==========

# 1. 重新处理数据
python src\data\process_all_stocks.py

# 2. 重新训练模型
python src\train\train_model.py

# 3. 验证预测
python src\predict\predict_stocks.py