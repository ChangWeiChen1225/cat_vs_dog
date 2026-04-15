import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, dropout_rate=0.5):
  # 載入ResNet50的預訓練權重
  model = models.resnet50(weights='IMAGENET1K_V2')

  # 取得ResNet50原本最後一層fc的輸入特徵維度 (in_features)
  # 跳過原本ResNet50的最後一層(輸出通常是 2048) 之後會換成我們要的目標輸出層 (num_classes=2)
  in_features = model.fc.in_features

  model.fc = nn.Sequential(
    # fully connection 層：從 2048 維降到 512
    nn.Linear(in_features, 512),
    # 加上激活函數 (否則兩層Linear合併起來只是另一層Linear)
    nn.ReLU(),
    # Dropout層，防止過擬合
    nn.Dropout(p=dropout_rate),
    # 輸出層
    nn.Linear(512, num_classes)
  )

  return model
