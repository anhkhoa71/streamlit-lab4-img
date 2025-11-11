import torch
from src.utils.dataset import ImageTranforms
from src.models.convnextv1 import ConvNeXt
from src.models.vit import VisionTransformer
from src.models.resnet34 import ResNet34
from src.models.shufflenetv1 import ShuffleNet
from src.models.vgg19 import VGG19Net
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, log_loss
)
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

def load_model(model, path, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model

def inference(model, img, labels, device):
    model.eval()
    transform = ImageTranforms()
    inputs = transform(img, phase='test')
    inputs = inputs.unsqueeze(0).to(device)

    with torch.no_grad():
        start_time = time()
        outputs = model(inputs)
        elapsed_time = time() - start_time
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_label = labels[pred_idx]
        pred_acc = probs[pred_idx].item() * 100
        return pred_label, pred_acc, elapsed_time
    
def load_model_dict(path, device):
    model_dict = {
        'VGG19': VGG19Net(num_classes=6),
        'ResNet34': ResNet34(num_classes=6),
        'ShuffleNet': ShuffleNet(num_classes=6),
        'ConvNeXt': ConvNeXt(num_classes=6),
        'ViT': VisionTransformer(
            patch_size=16,
            num_classes=6,
            emb_dim=384,
            mlp_size=384*4,
            num_heads=6,
            num_transformer_layers=10
        )
    }
    
    loaded_models = {}
    for model_name, model in model_dict.items():
        model_name_lower = model_name.lower()
        model_path = os.path.join(path, f'{model_name_lower}.pth')
        loaded_models[model_name] = load_model(model, model_path, device)
    
    return loaded_models 

def inference_model_dict(model_dict, img, labels, device):
    results = {}
    for model_name, model in model_dict.items():
        pred_label, pred_acc, elapsed_time = inference(model, img, labels, device)
        results[model_name] = {
            'pred_label': pred_label,
            'pred_acc': pred_acc,
            'elapsed_time': elapsed_time
        }
    return results

def evaluate_model_dict(model_dict, loader, class_names, device):
    results_dict = {}
    confusion_metrics_dict = {}

    for model_name, model in model_dict.items():
        
        all_preds = []
        all_labels = []
        all_probs = []

        model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc=f'Evaluating {model_name}'):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                batch_preds = probs.argmax(dim=1)

                all_probs.append(probs.cpu())
                all_preds.append(batch_preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)

        cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))  # ✅ FIX

        results_dict[model_name] = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'mcc': matthews_corrcoef(all_labels, all_preds),
            'log_loss': log_loss(all_labels, all_probs),
        }
        confusion_metrics_dict[model_name] = cm

    return results_dict, confusion_metrics_dict

def plot_results(results_dict, confusion_metrics_dict, class_names, path):  # ✅ Thêm class_names
    os.makedirs(os.path.join(path, 'confusion_matrix'), exist_ok=True)
    os.makedirs(os.path.join(path, 'metrics'), exist_ok=True)
    
    # Plot confusion matrices
    for model_name, cm in confusion_metrics_dict.items():
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)  # ✅ FIX
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix of {model_name}')
        plt.savefig(os.path.join(path, 'confusion_matrix', f'{model_name}_confusion_matrix.png'))
        plt.close()
    
    # Plot metrics comparison
    metrics = list(next(iter(results_dict.values())).keys())
    models = list(results_dict.keys())
    
    for metric in metrics:
        metric_values = [results_dict[model][metric] for model in models]
        plt.figure(figsize=(8,6))
        plt.bar(models, metric_values, color=sns.color_palette("pastel", len(models)))
        for i, val in enumerate(metric_values):
            plt.text(i, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        plt.ylim(0, 1.1)  # ✅ Tăng để text không bị cắt
        plt.ylabel('Value')
        plt.title(f'Comparison of {metric} across models')
        plt.savefig(os.path.join(path, 'metrics', f'{metric}_comparison.png'))
        plt.close()  # ✅ Thêm close để tránh memory leak

def pipeline_evaluation(model_dict, loader, class_names, device, model_path, result_path):
    model_dict = load_model_dict(model_path, device)
    results_dict, confusion_metrics_dict = evaluate_model_dict(model_dict, loader, class_names, device)
    plot_results(results_dict, confusion_metrics_dict, class_names, result_path)  # ✅ Truyền class_names