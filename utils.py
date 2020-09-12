import torch
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import torch.nn as nn
import matplotlib.pyplot as plt


def get_stats(dataset):

    if type(dataset) == np.ndarray:
        dset = torch.tensor(dataset, dtype=torch.float)
    else:
        dset = dataset

    if dset.max().item() == 255:
        dset.div_(255.0)

    return dset.mean(), dset.std()

def get_model_summary(model, input_size):
    print(summary(model, input_size))

def loss_fn():
    return nn.CrossEntropyLoss()

def get_optimizer(model, lr=0, momentum=0):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def get_scheduler(optimizer, step_size=6, gamma=0.1):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)

def plot_metrics(*metric_list, plot_type="Loss"):
    fig, ax = plt.subplots(figsize = (6, 6))
    for metric in metric_list:
        ax.plot(metric['metric'], label=metric['label'])

    ax.set_title(f'Variation of {plot_type.lower()} with epochs', fontsize=14)
    ax.set_ylabel(plot_type, fontsize=10)
    ax.set_xlabel('Number of Epochs', fontsize=10)
    fig.tight_layout()

def plot_incorrect_images(img_list):
    n_cols = int(np.sqrt(img_list))
    n_rows = int(np.ceil(np.sqrt(img_list)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize = (15, 15))
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        title = f'Target : {img_list[i][2]} \n  Pred : {img_list[i][1]}'
        ax.imshow(img_list[i][0].cpu().numpy().squeeze(), cmap = 'Greys')
        ax.set_title(title)
    fig.tight_layout()


