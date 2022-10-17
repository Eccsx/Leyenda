import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.image import psnr, ssim
from tensorflow.keras.callbacks import History

import numpy as np

def is_model_already_trained(model):
    is_weigths = os.path.exists(f'models/weigths/{model.name}.h5')
    is_history = os.path.exists(f'models/history/{model.name}.json')
    
    if is_weigths and is_history:
        return True
    else:
        if is_weigths:
            os.remove(f'models/weigths/{model.name}.h5')
        if is_history:
            os.remove(f'models/history/{model.name}.json')
        return False


def save_model_training(model):
    model.save_weights(f'models/weigths/{model.name}.h5')
    with open(f'models/history/{model.name}.json', 'w') as f:
        json.dump(model.history.history, f)


def load_model_training(model):
    model.load_weights(f'models/weigths/{model.name}.h5')
    model.history = History()
    with open(f'models/history/{model.name}.json', 'r') as f:
        model.history.history = json.load(f)
    model.history.epoch = list(range(len(model.history.history['loss'])))
        

def plot_model_history(model):
    history = model.history.history
    epoch = model.history.epoch
    keys = list(history.keys())
    palette = iter(sns.color_palette())
    
    fig, axs = plt.subplots(1, 2,
                            figsize=(15, 5),
                            sharex=True)
    
    sns.lineplot(x=epoch, y=history[keys[0]],
                 color=next(palette),
                 ax=fig.axes[0],
                 label=keys[0])
    sns.lineplot(x=epoch, y=history[keys[2]],
                 color=next(palette),
                 ax=fig.axes[0],
                 label=keys[2])

    sns.lineplot(x=epoch, y=history[keys[1]],
                 color=next(palette),
                 ax=fig.axes[1],
                 label=keys[1])
    sns.lineplot(x=epoch, y=history[keys[3]],
                 color=next(palette),
                 ax=fig.axes[1],
                 label=keys[3])

    fig.suptitle(f'{model.name} training history')
    fig.supxlabel('epoch')
    
    axs[0].set_title('loss')
    axs[1].set_title('accuracy')
    
    plt.tight_layout()


def plot_models_history(models):
    palette = iter(sns.color_palette())
    
    fig, axs = plt.subplots(1, 4,
                            figsize=(15, 5),
                            sharex=True)
    
    for m in models:
        history = m.history.history
        keys = list(history.keys())
        epoch = m.history.epoch
        color = next(palette)
    
        sns.lineplot(x=epoch, y=history[keys[0]],
                     color=color,
                     ax=fig.axes[0],
                     label=m.name)
        
        sns.lineplot(x=epoch, y=history[keys[2]],
                     color=color,
                     ax=fig.axes[1],
                     label=m.name)

        sns.lineplot(x=epoch, y=history[keys[1]],
                     color=color,
                     ax=fig.axes[2],
                     label=m.name)
        
        sns.lineplot(x=epoch, y=history[keys[3]],
                     color=color,
                     ax=fig.axes[3],
                     label=m.name)
        
    axs[0].set_title(keys[0])
    axs[1].set_title(keys[2])
    axs[2].set_title(keys[1])
    axs[3].set_title(keys[3])

    fig.suptitle('training history comparision')
    fig.supxlabel('epoch')
    
    plt.tight_layout()
    

def PSNR(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=1.0)
    
def SSIM(y_true, y_pred):
    return ssim(y_true, y_pred, max_val=1.0)
    
    
def compare_image_sets(image_sets, number, labels, size=(15, 7)):
    rand = np.random.choice(range(len(image_sets[0])), size=number, replace=False)
    
    fig, axs = plt.subplots(len(image_sets), number,
                            figsize=size, sharey=True)
    
    for ax, s, l in zip(axs, image_sets, labels):
        for i in range(number):
            ax[i].imshow(s[rand[i]])
            
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(False)
            
            ax[i].set_title(l)
    
    fig.suptitle('image sets comparision')
    plt.tight_layout()