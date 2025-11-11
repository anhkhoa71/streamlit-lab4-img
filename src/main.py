from utils.inference import pipeline_evaluation, load_model_dict
from utils.dataset import loader_dict, dataset_dict
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = loader_dict['test']
class_names = dataset_dict['test'].class_names

model_path = '../models'
result_path = '../results'
model_dict = load_model_dict(model_path, device)
pipeline_evaluation(model_dict, loader, class_names, device, model_path, result_path)

