import os
import random
import shutil
from glob import glob

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


def train_valid_split(root, valid_ratio=0.1, seed=42):
    random.seed(seed)
    train_path = os.path.join(root, 'train')
    valid_path = os.path.join(root, 'val')
    os.makedirs(valid_path, exist_ok=True)

    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path, class_name)
        valid_class_path = os.path.join(valid_path, class_name)
        os.makedirs(valid_class_path, exist_ok=True)

        files = os.listdir(class_path)
        num_valid = int(len(files) * valid_ratio)
        valid_files = random.sample(files, num_valid)
        for img in valid_files:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(valid_class_path, img)
            shutil.move(src_path, dst_path)

class ImageTranforms:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize=224):
        self.data_transform = {'train':
                               transforms.Compose([
                                   transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)
                               ]),
                               'val':
                               transforms.Compose([
                                   transforms.Resize((resize, resize)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)
                               ]),
                               'test':
                               transforms.Compose([
                                   transforms.Resize((resize, resize)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)
                               ])}
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

class ImageDataset(Dataset):
    def __init__(self, root, transform=None, phase='train'):
        self.transform = transform
        self.phase = phase
        self.file_path = glob(os.path.join(root, phase, '*', '*.jpg'))

        self.class_names = [os.path.basename(path) for path in glob(os.path.join(root, phase, '*'))]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        img_path = self.file_path[idx]
        img = Image.open(img_path)
        class_name = os.path.split(os.path.dirname(img_path))[-1]
        label = self.class_to_idx[class_name]
        img_transformed = self.transform(img, self.phase)
        return img_transformed, label


# valid_ratio = 0.2
# data_path = '../data'

# if not os.path.exists(os.path.join(data_path, 'val')):
#     train_valid_split(data_path, valid_ratio)

# phases = ['train', 'val', 'test']
# dataset_dict = {phase: ImageDataset(root=data_path, transform=ImageTranforms(), phase=phase)
#                 for phase in phases}

# loader_dict = {phase: DataLoader(
#                     dataset_dict[phase],
#                     batch_size=16,
#                     shuffle=(phase == 'train'),
#                     num_workers=0
#                 ) for phase in phases}
