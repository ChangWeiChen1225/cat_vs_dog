import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from torch.utils.data import DataLoader
from model import get_model
from dataset import get_loaders
import numpy as np

def evaluate(model_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(device)
  model = get_model().to(device)
  # 導入訓練好的權重
  model.load_state_dict(torch.load(model_path))
  model.eval()

  test_loader = get_loaders('data/test', 'test')

  all_preds = []
  all_labels = []
  all_probs = []

  with torch.no_grad():
    for images, labels in test_loader:
      images = images.to(device)
      outputs = model(images)
      probs = torch.softmax(outputs, dim=1)
      _, preds = torch.max(outputs, 1)

      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())
      all_probs.extend(probs[:, 1].cpu().numpy())

  # performance指標
  print(classification_report(all_labels, all_preds, target_names=['Cat', 'Dog']))

  # Confusion Matrix
  cm = confusion_matrix(all_labels, all_preds)
  plt.figure(figsize=(6,4))
  sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.savefig('confusion_matrix.png')

  # 繪製ROC Curve
  fpr, tpr, _ = roc_curve(all_labels, all_probs)
  roc_auc = auc(fpr, tpr)
  plt.figure()
  plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
  plt.plot([0, 1], [0, 1], 'k--')
  plt.legend()
  plt.savefig('roc_curve.png')

if __name__ == "__main__":
  evaluate('best_model.pth')
