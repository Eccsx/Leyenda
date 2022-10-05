import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def is_model_already_trained(model):
    is_weigths = os.path.exists(f'models/weigths/{model.name}.h5')
    is_history = os.path.exists(f'models/history/{model.name}.pkl')
    
    if is_weigths and is_history:
        return True
    else:
        if is_weigths:
            os.remove(f'models/weigths/{model.name}.h5')
        if is_history:
            os.remove(f'models/history/{model.name}.pkl')
        return False


def save_model_training(model):
    model.save_weights(f'models/weigths/{model.name}.h5')
    joblib.dump(f'models/history/{model.name}.pkl', model.history)


def load_model_training(model, NUM_EPOCH):
    model.load_weights(f'models/weigths/{model.name}.h5')
    model.history = joblib.load(f'models/history/{model.name}.pkl')
        

def plot_model_history(model):
    history = model.history
    palette = iter(sns.color_palette())
    
    fig, axs = plt.subplots(1, 2,
                            figsize=(15, 5),
                            sharex=True)
    
    sns.lineplot(x=history.epoch, y=history.history['loss'],
                 color=next(palette),
                 ax=fig.axes[0],
                 label='loss')
    sns.lineplot(x=history.epoch, y=history.history['val_loss'],
                 color=next(palette),
                 ax=fig.axes[0],
                 label='val_loss')

    sns.lineplot(x=history.epoch, y=history.history['sparse_categorical_accuracy'],
                 color=next(palette),
                 ax=fig.axes[1],
                 label='sparse_categorical_accuracy')
    sns.lineplot(x=history.epoch, y=history.history['val_sparse_categorical_accuracy'],
                 color=next(palette),
                 ax=fig.axes[1],
                 label='val_sparse_categorical_accuracy')

    fig.suptitle(f'{model.name} training history', fontsize=16)
    fig.supxlabel('epoch')
    
    plt.tight_layout()


def plot_models_history(models):
    palette = iter(sns.color_palette())
    
    fig, axs = plt.subplots(1, 4,
                            figsize=(15, 5),
                            sharex=True)
    
    for m in models:
        history = m.history
    
        sns.lineplot(x=history.epoch, y=history.history['loss'],
                     color=next(palette),
                     ax=fig.axes[0],
                     label=m.name)
        
        sns.lineplot(x=history.epoch, y=history.history['val_loss'],
                     color=next(palette),
                     ax=fig.axes[1],
                     label=m.name)

        sns.lineplot(x=history.epoch, y=history.history['sparse_categorical_accuracy'],
                     color=next(palette),
                     ax=fig.axes[2],
                     label=m.name)
        
        sns.lineplot(x=history.epoch, y=history.history['val_sparse_categorical_accuracy'],
                     color=next(palette),
                     ax=fig.axes[3],
                     label=m.name)
        
    axs[0].set_title('loss')
    axs[1].set_title('val_loss')
    axs[2].set_title('sparse_categorical_accuracy')
    axs[3].set_title('val_sparse_categorical_accuracy')

    fig.suptitle('Training history comparision', fontsize=16)
    fig.supxlabel('epoch')
    
    plt.tight_layout()