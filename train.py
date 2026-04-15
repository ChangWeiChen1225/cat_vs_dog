import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dataset import get_loaders
from model import get_model
from utils import plot_curves, save_checkpoint
import argparse

def train(args):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)

  # 設定transform將圖片resize成固定大小，並做Data Augmentation，接著進行normalize
  # 取得dataloader並利用其設定batch, shuffle參數
  train_loader = get_loaders('data/train', 'train', batch_size=args.batch_size, num_workers=2)
  val_loader = get_loaders('data/val', 'val', batch_size=args.batch_size, num_workers=2)

  model = get_model().to(device)
  # 設定損失函數以及優化器
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.AdamW(model.parameters(), lr=args.lr)

  # 宣告 Scheduler
  # mode='min': 因為監控的是Loss，Loss越小越好
  # factor: 縮小倍率，0.1代表變成原本的 10%
  # patience: 容忍次數，如果連續2個Epoch驗證Loss沒降，就動手調整lr
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)


  # 建立紀錄容器，供後續繪製learning curve
  history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
  }

  best_val_acc = 0.0

  # 檢查是否有舊的 Checkpoint 存在，並載入
  start_epoch = 0
  checkpoint_path = './checkpoints/last_model.pth'

  if os.path.exists(checkpoint_path):
    print(f"找到舊有的 Checkpoint: {checkpoint_path}，準備恢復訓練...")
    checkpoint = torch.load(checkpoint_path)
    
    # 載入權重與狀態
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    best_val_acc = checkpoint['best_acc']
    
    print(f"成功從第 {start_epoch} Epoch 恢復訓練。")

  # 訓練迴圈起始點
  for epoch in range(start_epoch, args.epochs):
    # ==================== 訓練階段 ====================
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    # 套用進度條
    train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Train")

    for images, labels in train_pbar:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # 累計訓練指標
      train_loss += loss.item() * images.size(0)
      _, predicted = outputs.max(1)
      train_total += labels.size(0)
      train_correct += predicted.eq(labels).sum().item()

      # 更新進度條右側的資訊
      train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*train_correct/train_total:.2f}%")

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_correct / train_total

    # ==================== 驗證階段 ====================
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    all_val_labels = []
    all_val_preds = []

    # 不須更新權重，因此關閉梯度計算
    with torch.no_grad():
      val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] Valid")
      for images, labels in val_pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        val_total += labels.size(0)
        val_correct += predicted.eq(labels).sum().item()

        # 收集資料用於計算 Precision / Recall
        all_val_labels.extend(labels.cpu().numpy())
        all_val_preds.extend(predicted.cpu().numpy())

        val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*val_correct/val_total:.2f}%")

    avg_val_loss = val_loss / len(val_loader.dataset)
    avg_val_acc = val_correct / val_total

    # ==================== 更新 Scheduler ====================
    # 傳入當前的驗證指標
    scheduler.step(avg_val_loss)
    # 獲取當前學習率以便印出
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: Current Learning Rate: {current_lr}")

    # 紀錄每個epoch的acc跟loss，供後續繪製learning curve
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(avg_val_acc)

    # 使用 sklearn 計算進階指標
    val_precision = precision_score(all_val_labels, all_val_preds, average='binary')
    val_recall = recall_score(all_val_labels, all_val_preds, average='binary')

    # ==================== 總結與儲存 ====================
    print(f"\nSummary Epoch {epoch+1}:")
    print(f"Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}")
    print(f"Valid - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

    # 儲存最佳模型 (Best Model Saving)
    is_best = avg_val_acc > best_val_acc
    if is_best:
      best_val_acc = avg_val_acc
      torch.save(model.state_dict(), args.save_path)
      print(f"--> Best model saved with Acc: {best_val_acc:.4f}")

    # 準備要儲存的資訊
    checkpoint_state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(), # 如果有使用 scheduler
        'best_acc': best_val_acc,
    }

    # 執行儲存
    save_checkpoint(checkpoint_state, is_best)


  # 訓練結束，繪製learning curve
  plot_curves(history)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--save_path', type=str, default='best_model.pth')
  train(parser.parse_args())
