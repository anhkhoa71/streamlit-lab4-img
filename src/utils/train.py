import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MixupCutmix:
    def __init__(self, mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob

    def _rand_bbox(self, W, H, lam, device):
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()

        cx = torch.randint(0, W, (1,), device=device)
        cy = torch.randint(0, H, (1,), device=device)

        x1 = torch.clamp(cx - cut_w // 2, 0, W)
        x2 = torch.clamp(cx + cut_w // 2, 0, W)
        y1 = torch.clamp(cy - cut_h // 2, 0, H)
        y2 = torch.clamp(cy + cut_h // 2, 0, H)

        return x1.item(), y1.item(), x2.item(), y2.item()

    def __call__(self, images, labels):
        if torch.rand(1).item() > self.prob:
            return images, labels

        device = images.device
        batch_size = images.size(0)

        indices = torch.randperm(batch_size, device=device)

        x1 = images
        x2 = images[indices]
        y1 = labels
        y2 = labels[indices]

        # ---------------- MIXUP ----------------
        if torch.rand(1).item() < self.switch_prob:
            lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().to(device)
            images = lam * x1 + (1 - lam) * x2

        # ---------------- CUTMIX ----------------
        else:
            lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().to(device)

            W = images.size(3)
            H = images.size(2)
            x1b, y1b, x2b, y2b = self._rand_bbox(W, H, lam, device)

            images[:, :, y1b:y2b, x1b:x2b] = x2[:, :, y1b:y2b, x1b:x2b]

            lam = 1 - ((x2b - x1b) * (y2b - y1b) / (W * H))

        labels = lam * y1 + (1 - lam) * y2
        return images, labels

def train_model(model, dataloader_dict, dataset_dict, criterion, optimizer, num_epochs, device, save_path='resnet34_best_model', mixup=None):
    Loss_train, Accuracy_train = [], []
    Loss_val, Accuracy_val = [], []

    best_val_acc = 0.0  # để theo dõi mô hình tốt nhất

    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total_samples = len(dataset_dict[phase])

            loop = tqdm(dataloader_dict[phase], desc=f"{phase.capitalize()} phase", leave=False)

            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                if mixup is not None:
                    inputs, labels_softs = mixup(inputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if mixup is not None:
                        loss = -(labels_softs * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()
                    else:
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Cập nhật loss và acc
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

                loop.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(running_corrects / ((loop.n + 1) * inputs.size(0))):.4f}'
                })

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples

            if phase == 'train':
                Loss_train.append(epoch_loss)
                Accuracy_train.append(epoch_acc)
            else:
                Loss_val.append(epoch_loss)
                Accuracy_val.append(epoch_acc)

                # ✅ Nếu val acc tốt hơn trước thì lưu model
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), save_path)
                    print(f"💾 Saved best model (val acc: {best_val_acc*100:.2f}%) to '{save_path}'")

        # ✅ In kết quả sau mỗi epoch
        print(f"✅ Epoch {epoch+1}/{num_epochs} -- "
              f"Train Loss: {Loss_train[-1]:.4f}, Train Acc: {Accuracy_train[-1]*100:.2f}% | "
              f"Val Loss: {Loss_val[-1]:.4f}, Val Acc: {Accuracy_val[-1]*100:.2f}%")

    print(f"\n🎯 Training complete. Best Val Acc: {best_val_acc*100:.2f}%")
    return Loss_train, Accuracy_train, Loss_val, Accuracy_val

def test_model(model_path, model, dataloader, dataset_dict, criterion, device, class_names=None):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    test_loss = 0.0
    correct = 0
    total = 0
    total_samples = len(dataset_dict)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / total_samples
    accuracy = 100 * correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(set(all_labels)))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
