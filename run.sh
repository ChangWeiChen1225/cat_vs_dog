#!/bin/bash

# 遇到錯誤就停止執行 (確保流程安全)
set -e

# --- 1. 環境安裝 ---
echo "正在檢查並安裝必要的套件..."
pip install -r requirements.txt

# --- 2. 資料準備 ---
# 執行資料下載與分割 
echo "正在下載並準備資料集 (Train/Val/Test Split)..."
python data_setup.py --src_train_dir train --base_dir data --split_ratio 0.1 --seed 42

# --- 3. 開始訓練 ---
echo "開始模型訓練..."
python train.py --epochs 15 --batch_size 128 --lr 0.001 --save_path ./checkpoints/best_model.pth

# --- 4. 模型評估 ---
echo "訓練完成，開始執行測試集評估..."
python eval.py --model_path ./checkpoints/best_model.pth