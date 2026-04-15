import os
import zipfile
import random
import shutil
from pathlib import Path
import argparse

def setup_data(args):
  # 下載數據 (假設已上傳 kaggle.json)
  if not os.path.exists('dogs-vs-cats.zip'):
    os.system('kaggle competitions download -c dogs-vs-cats')

  # 解壓縮
  with zipfile.ZipFile('dogs-vs-cats.zip', 'r') as zip_ref:
      zip_ref.extractall('.')
  with zipfile.ZipFile('train.zip', 'r') as zip_ref:
      zip_ref.extractall('.')

  organize_and_split_data(args.src_train_dir, args.base_dir, args.split_ratio, args.seed)
  
# 從train之中隨機抽樣，以train:val:test=8:1:1的比例移動到train/val/test對應資料夾中
def split_and_move(src_train_dir, base_path, files, class_name, split_ratio=0.1):
  # 隨機打亂列表
  random.shuffle(files)

  # 計算切割點
  num_val = int(len(files) * split_ratio)
  num_test = int(len(files) * split_ratio)
  val_files = files[:num_test]
  test_files = files[num_test:num_test * 2]
  train_files = files[num_test * 2:]

  # 移動到val
  for f in val_files:
    shutil.move(os.path.join(src_train_dir, f), base_path / 'val' / class_name / f)

  # 移動到test
  for f in test_files:
    shutil.move(os.path.join(src_train_dir, f), base_path / 'test' / class_name / f)

  # 移動到train
  for f in train_files:
    shutil.move(os.path.join(src_train_dir, f), base_path / 'train' / class_name / f)

  print(f"完成 {class_name} 分類: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")


def organize_and_split_data(src_train_dir, base_dir, split_ratio=0.1, seed=42):
  # 設定random seed，確保結果可復現
  random.seed(seed)

  # 定義路徑，並建立符合ImageFolder的結構 (data/train/cats, data/train/dogs)
  base_path = Path(base_dir)
  train_path = base_path / 'train'
  val_path = base_path / 'val'
  test_path = base_path / 'test'
  for cls in ['cats', 'dogs']:
    (train_path / cls).mkdir(parents=True, exist_ok=True)
    (test_path / cls).mkdir(parents=True, exist_ok=True)
    (val_path / cls).mkdir(parents=True, exist_ok=True)

  # 讀取所有圖片並按類別分類
  all_files = os.listdir(src_train_dir)
  cat_files = [f for f in all_files if f.startswith('cat')]
  dog_files = [f for f in all_files if f.startswith('dog')]


  # 執行分配
  split_and_move(src_train_dir, base_path, cat_files, 'cats', split_ratio)
  split_and_move(src_train_dir, base_path, dog_files, 'dogs', split_ratio)

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--src_train_dir', type=str, default='train')
  parser.add_argument('--base_dir', type=str, default='data')
  parser.add_argument('--split_ratio', type=float, default=0.1)
  parser.add_argument('--seed', type=int, default=42)
  setup_data(parser.parse_args())
  print("Data Setup Complete!")
