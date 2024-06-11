import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import generator as CG
import oil_spill_model as my_model

BATCH = 8
BATCH_VAL = 8
lrate = 0.002
EPOCHS = 5
PATIENCE = 10
num_classes = 5

HISTORY_FOLDER = './history_trained/'
PLOT_HISTORY_FOLDER = './plot_history/'
NN_WEIGHTS = './nn_weights/'
NN_CHECKPOINTS = './nn_checkpoints/'

def create_folders():
    try:
        os.makedirs(HISTORY_FOLDER, exist_ok=True)
        os.makedirs(PLOT_HISTORY_FOLDER, exist_ok=True)
        os.makedirs(NN_WEIGHTS, exist_ok=True)
        os.makedirs(NN_CHECKPOINTS, exist_ok=True)
    except Exception as ex:
        print(ex)
        exit(-1)

def l_rate_decay(epoch):
    return lrate * (0.5 ** (epoch // 10))
  
def trainSegMulticlass(model, train_file, validation_file, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    nb_train_samples = CG.file_len(train_file)
    nb_validation_samples = CG.file_len(validation_file)
    
    criterion = CG.categorical_focal_loss()
    optimizer = optim.Adam(model.parameters(), lr=lrate, betas=(0.9, 0.999))
    
    x_train = CG.CustomImageDatasetSegMultiGPU(train_file)
    x_val = CG.CustomImageDatasetSegMultiGPU(validation_file)

    best_loss = float('inf')
    patience_counter = 0

    train_loader = DataLoader(x_train, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(x_val, batch_size=BATCH_VAL, shuffle=False)
    
    history = {'loss': [], 'val_loss': [], 'categorical_accuracy': [], 'val_categorical_accuracy': []}

    labels_one_hot = None
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            outputs = F.softmax(outputs, dim=1)
            labels = labels.long()

            if labels_one_hot is not None:
                labels_one_hot = F.one_hot(labels, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()

                loss = criterion(labels_one_hot, outputs)
                loss = loss.mean()
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item() 
            
        if total != 0:
            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
        else:
            train_loss = 0
            train_acc = 0
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                
                val_outputs = F.softmax(val_outputs, dim=1)
                
                val_loss += criterion(val_labels, val_outputs).item()


                _, val_predicted = torch.max(val_outputs, dim=1)
                val_predicted = val_predicted.unsqueeze(1)
                val_labels = val_labels.int() 
                
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        if val_total != 0: 
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
        else:
            val_loss = 0
            val_acc = 0

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['categorical_accuracy'].append(train_acc)
        history['val_categorical_accuracy'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), NN_CHECKPOINTS + f'weight_seg_{model_name}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = l_rate_decay(epoch)
    
    return history

def plot_history(h, net_name, save_plot=False):
    list_keys = list(h.keys())
    list_keys = [s for s in list_keys if 'val_' not in s and 'lr' != s]
    print('Ready to', 'plot' if save_plot == False else 'save', ':', list_keys)
    for key in list_keys:
        plt.figure(key)
        plt.plot(h[key], 'o-')
        plt.plot(h['val_' + key], '^-')
        plt.title('model ' + key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        if save_plot == True:
            plt.savefig(PLOT_HISTORY_FOLDER + net_name + '__' + key + '.png', format='png')
        else:
            plt.show()
    if save_plot == True:
        print('Plots saved in: ' + PLOT_HISTORY_FOLDER)

if __name__ == '__main__':
    create_folders()

    model = my_model.OilSpillNet(input_shape=(1, 320, 320))
    net_name = "OilSpillNet"

    print(model)
    
    if torch.cuda.is_available():
        print('Num GPUs available: ', torch.cuda.device_count())
    else:
        print('GPU is not available!')
    print('Available devices:')
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    print('\nPyTorch version: ', torch.__version__)

    path_input_images_train = './train_set.txt'
    path_output_images_validation = './validation_set.txt'
    
    print('Start training...')
    namesession = time.strftime("%Y-%m-%d_%H-%M") + '_' + net_name
    h = trainSegMulticlass(model, path_input_images_train, path_output_images_validation, namesession)
    torch.save(model.state_dict(), NN_WEIGHTS + namesession + '.pth')
    np.save(HISTORY_FOLDER + 'h_' + namesession, h)
    print('Net saved as: ' + namesession + '.pth')
    plot_history(h, net_name, save_plot=True)
