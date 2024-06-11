import numpy as np
import torch

def output_to_rgb(output_image, mask): 
    if torch.is_tensor(output_image):
        output_image = output_image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    rgb_image = np.zeros((output_image.shape[0], output_image.shape[1], 3))
    
    diff = output_image - mask
    rgb_image[np.squeeze(diff<0, -1), ...] = (0,1,0)
    rgb_image[np.squeeze(diff>0, -1), ...] = (1,0,0)
    rgb_image[np.squeeze((output_image==1) * (mask==1), -1), ...] = (1,1,1)
    
    return rgb_image

def get_confusion_matrix(output_image, mask):
    if torch.is_tensor(output_image):
        output_image = output_image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()

    assert output_image.shape == mask.shape
    assert output_image.dtype == mask.dtype
    assert (np.sum(output_image==1) + np.sum(output_image==0)) == np.prod(output_image.shape)
    assert (np.sum(mask==1) + np.sum(mask==0)) == np.prod(mask.shape)

    TP = np.sum((output_image==1) * (mask==1)) 
    FP = np.sum((output_image==1) * (mask==0)) 
    FN = np.sum((output_image==0) * (mask==1)) 
    TN = np.sum((output_image==0) * (mask==0)) 
    
    return TP, FP, FN, TN

def compute_metrics(out, mask):
    tp, fp, fn, tn = get_confusion_matrix(out, mask)
    if tp+fp+fn != 0:
        iou = tp/(tp+fp+fn)
        dice = 2*tp / (2*tp+fp+fn)
    else:
        iou = -1
        dice = -1
    
    acc = (tp+tn)/(tp+fp+fn+tn)
    
    return iou, acc, dice, tp, fp, fn, tn
