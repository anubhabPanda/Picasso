import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, OneCycleLR

def StepLR_scheduler(optimizer, step_size=6, gamma=0.1):
    return StepLR(optimizer, step_size=step_size, gamma=gamma)

def LR_on_pleateau_scheduler(optimizer, patience=10, threshold=0.0001, threshold_mode='rel'):
    return ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                             patience=patience, threshold=threshold, threshold_mode=threshold_mode, 
                             cooldown=0, min_lr=0, eps=1e-08, verbose=False)

def OneCylePolicy(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, 
                        pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, 
                        max_momentum=0.95, div_factor=25., final_div_factor=1e4, last_epoch=-1):
    
    return OneCycleLR(optimizer, max_lr, total_steps, epochs, steps_per_epoch, 
                        pct_start, anneal_strategy, cycle_momentum, base_momentum, 
                        max_momentum, div_factor, final_div_factor, last_epoch)


def plot_lr(lr_min, lr_max, total_iterations, step_size):
    iterations = np.arange(0, total_iterations, 1)
    cycle = np.floor(1 + iterations/(2*step_size))
    xt = np.abs(iterations/step_size - 2*cycle + 1)
    lrt = lr_min + (lr_max - lr_min)*(1-xt)

    cycle_width = 2
    if max(cycle) > 1:
        cycle_width = cycle_width*max(cycle)
    
    figsize = (cycle_width, 3)
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(iterations, lrt)
    ax.axhline(y = lr_min, xmax = total_iterations, color='red')
    ax.axhline(y = lr_max, xmax = total_iterations, color='red')
    ax.set_xlabel("Iterations", fontsize = 12)
    ax.set_ylabel("Learning Rate", fontsize = 12)
    ax.set_title("Variation of Learning Rate with iterations", fontsize = 14)
    fig.tight_layout()
    ax.text(lr_min - lr_max/10, "min_lr")
    ax.text(lr_max + lr_max/10, "max_lr")
    ax.margins(x=0.1, y=0.3)
    plt.show()




