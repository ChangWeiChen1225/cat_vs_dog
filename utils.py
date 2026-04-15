import os
import torch
import shutil
import matplotlib.pyplot as plt

# 萬一訓練中斷，可以從最後一格接下去
def save_checkpoint(state, is_best, checkpoint_dir='./checkpoints'):
  """
  state: 包含 epoch, model_state, optimizer_state, best_acc 等資訊的字典
  is_best: 是否為目前為止表現最好的模型
  """
  os.makedirs(checkpoint_dir, exist_ok=True)
  
  # 儲存最後一個 Epoch 的狀態 (last_model.pth)
  last_path = os.path.join(checkpoint_dir, 'last_model.pth')
  torch.save(state, last_path)
  
  # 如果是最佳模型，額外存一份 (best_model.pth)
  if is_best:
    best_path = os.path.join(checkpoint_dir, 'best_model.pth')
    shutil.copyfile(last_path, best_path) # 直接複製 last_model 即可，效率較高


# 定義繪製learning curve的函式
def plot_curves(history):
  epochs = range(1, len(history['train_loss']) + 1)

  plt.figure(figsize=(12, 5))

  # 畫Loss曲線
  plt.subplot(1, 2, 1)
  plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
  plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
  plt.title('Training and Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()

  # 畫Accuracy曲線
  plt.subplot(1, 2, 2)
  plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
  plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
  plt.title('Training and Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.tight_layout()
  
  # 儲存learning curve
  plt.savefig('learning_curves.png')
  plt.show()
