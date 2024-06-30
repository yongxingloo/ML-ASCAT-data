"""

@author: Jing Sun

"""
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torch.utils.data

class pack_dataset(torch.utils.data.Dataset):
    """
    Takes in the input and target data arrays along with an optional transform function, 
    and returns a PyTorch dataset that packs input and target data into tuples. 

    Args:
    - input_data (np.ndarray): The input data.
    - target_data (np.ndarray): The target data.
    - transform (callable, optional): A function to apply to the data before returning. Defaults to None.
    """
    def __init__(self, input_data, target_data, transform=None):
        self.input_data  = input_data
        self.target_data = target_data
        self.transform   = transform
    
    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_data  = torch.Tensor(self.input_data[index]).float()
        target_data = torch.Tensor(self.target_data[index]).float()

        if self.transform:
            input_data, target_data = self.transform(input_data, target_data)

        return input_data, target_data


def prepare_data(noisy_input, target_signal):
    """
    Prepares the input and target signal arrays for training by reshaping them into 
    the expected format, and converting the resulting arrays to PyTorch tensors.
    
    Args:
    - noisy_input (np.ndarray): The noisy input array with shape (batch_size, *input_shape).
    - target_signal (np.ndarray): The target signal array with shape (batch_size, *target_shape).
    
    Returns:
    - input_bag (pack_dataset): The prepared input data in the form of an input bag for training.
    """
    noisy_input   = noisy_input.reshape(noisy_input.shape[0], 1, *noisy_input.shape[1:])
    target_signal = target_signal.reshape(target_signal.shape[0], 1, *target_signal.shape[1:])
    
    input_bag     = pack_dataset(noisy_input, target_signal, transform=None)
    
    return input_bag


def split_dataset(data, train_percentage=0.85):
    """
    Splits a dataset into a training set and a validation set.

    Args:
    - data: The dataset to be split.
    - train_percentage: The percentage of the data to use for training. Defaults to 0.85.

    Returns:
    - The training set and validation set.
    """
    len_train = int(round(len(data) * train_percentage))
    len_valid = len(data) - len_train
    
    train_data, valid_data = torch.utils.data.random_split(data, [len_train, len_valid])
    
    return train_data, valid_data

def create_folder_struct():
    """
    Create a folder structure for storing output.
    
    Returns:
    - dirs (str): Directory for storing output.

    """
    date = datetime.now().strftime('%Y-%m-%d')
    hr_min = datetime.now().strftime('%H-%M')
    dirs = os.path.join('..', 'output', f'{date}_{hr_min}')
    os.makedirs(dirs, exist_ok=True)

    print('Folder structure created!')

    return dirs

def write_summary(model, output_dir):
    """
    Write model summary to a text file.

    Args:
    - model (torch.nn.Module): The PyTorch model.
    - output_dir (str): The path to the output directory.
    
    Returns:
    - None
    """
    filepath = os.path.join(output_dir, 'model_print.txt')
    with open(filepath, 'a') as f:
        f.write(f'Model architecture: {model}\n')
        for param_name, param_tensor in model.state_dict().items():
            # Move tensor to CPU and convert to NumPy array
            param_array = param_tensor.cpu().numpy()  
            f.write(f'{param_name}\n{np.array(param_array)}\n')
            
    print('Model summary written to file!')
    
    return

def plot_loss(train_losses, valid_losses, dirs):
    """
    Plots the training and validation losses and saves the plot to a file.

    Args:
    - train_losses (List[float]): The list of training losses for each epoch.
    - valid_losses (List[float]): The list of validation losses for each epoch.
    - dirs (str): The directory where the loss plot should be saved.

    Returns:
    - None.
    """
    epoch = np.arange(1, len(train_losses)+1)
    plt.plot(epoch, train_losses, 'b--', epoch, valid_losses, 'r--')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper right')

    # Join the directory and filename using os.path.join()
    loss_path = os.path.join(dirs, 'loss_plot.png')

    # Add error handling in case the directory does not exist
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print(f"Created directory {dirs}")

    plt.savefig(loss_path, dpi=300)
    plt.close()

    print('Loss written and plotted')
    
    return

def plot_example(target_signal, noisy_input, output_signal, signal_diff, dirs, v_min, v_max, example_num):
    """
    Plot example seismic traces.

    Args:
    - noisy_input (numpy.ndarray): The noisy input seismic trace.
    - output_signal (numpy.ndarray): The predicted output signal.
    - target_signal (numpy.ndarray): The true target signal.
    - signal_diff (numpy.ndarray): The difference between the predicted and true signals.
    - dirs (str): The path to the output directory.
    - vmin (float): The minimum value for the color map.
    - vmax (float): The maximum value for the color map.
    - example_num (int): The index of the example to plot.
    
    Returns:
    - None
    """
    titles = ['Ground Truth', 'Test Input', 'DNN Output', 'Difference']
    sets   = [target_signal, noisy_input, output_signal, signal_diff]
    plt.subplots(figsize=(15,17))
    for i in range (1, 5):
        ax=plt.subplot(2, 2, i )
        plt.imshow(sets[i-1][example_num], vmin=v_min, vmax=v_max, cmap='rainbow', aspect='auto', origin='upper')
        plt.colorbar()
        plt.title(titles[i-1], size=17);
        plt.ylabel('y',fontsize=17);
        plt.xlabel('x'   ,fontsize=17);
        ax.tick_params(axis='x', which='major', labelsize=17,pad=2)
        ax.tick_params(axis='y', which='major', labelsize=17,pad=2)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')     
    
    plt.savefig(dirs + '/example_plot.png', dpi=300)    
    plt.close()
    
    print('Example plotted!')
    
    return