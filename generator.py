import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class CustomImageDatasetSegMultiGPU(Dataset):
    def __init__(self, input_file, transform=None):
        self.in_file = input_file
        self.data = self.load_data()
        self.transform = transform

    def load_data(self):
        with open(self.in_file, "r") as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        train_folder = './dataset/train_tile_aug/'
        seg_image_folder = './dataset/train_label_tile_aug/'

        line = self.data[idx]
        try:
            image = np.array(Image.open(train_folder + line).resize((320, 320)), dtype=np.uint8)[..., 0]
            image = np.expand_dims(image, -1)
            label = np.load(seg_image_folder + line.replace('jpg', 'png') + '.npy')
            label = np.asarray(label, dtype=np.float32)
        except Exception as ex:
            print('failed at idx: ', idx)
            print(ex)
            print('file image: ' + train_folder + line)
            print('file label: ' + seg_image_folder + line.replace('jpg', 'png') + '.npy')
            return None

        if self.transform:
            image = self.transform(image)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        label = torch.from_numpy(label).float()

        return image, label

def file_len(name):
    num_lines = 0
    with open(name) as f:
        for i, l in enumerate(f):
            if len(l) > 1:
                num_lines += 1
    return num_lines

class MeanIoU(torch.nn.Module):
    def __init__(self, num_classes):
        super(MeanIoU, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_true, y_pred):
        y_true = torch.argmax(y_true, dim=-1)
        y_pred = torch.argmax(y_pred, dim=-1)
        iou = []
        for cls in range(self.num_classes):
            true_class = y_true == cls
            pred_class = y_pred == cls
            intersection = torch.logical_and(true_class, pred_class).sum().item()
            union = torch.logical_or(true_class, pred_class).sum().item()
            if union != 0:
                iou.append(intersection / union)
        return sum(iou) / len(iou)

def categorical_focal_loss(alpha=0.75, gamma=2.0):
    def focal_loss(y_true, y_pred):
        print("Shape of y_true:", y_true.shape)
        print("Shape of y_pred:", y_pred.shape)
        
        epsilon = 1e-9
        y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        weight = alpha * y_true * torch.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return loss.mean()

    return focal_loss

