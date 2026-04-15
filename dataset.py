from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms():
  """
  定義影像預處理流程。
  訓練集加入 Augmentation，驗證/測試集則維持原始狀態（僅 Resize）。
  """
  # ImageNet 標準標配的 Mean 與 Std
  norm_mean = [0.485, 0.456, 0.406]
  norm_std = [0.229, 0.224, 0.225]

  data_transforms = {
    'train': transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(15),
      transforms.ToTensor(),
      transforms.Normalize(norm_mean, norm_std)
    ]),
    'val': transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(norm_mean, norm_std)
    ]),
    'test': transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(norm_mean, norm_std)
    ]),
  }
  return data_transforms


def get_loaders(data_dir, train_val_test_set, batch_size=64, num_workers=2):
  transform = get_transforms()

  # 讀取資料夾
  dataset = datasets.ImageFolder(data_dir, transform=transform[train_val_test_set])

  # 建立 Loaders， 考量到資料傳輸瓶頸，num_workers=2將cpu的兩個核心都分配上，pin_memory=True加速CPU到GPU的傳輸，prefetch_factor=2預先加載
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2)

  return data_loader



