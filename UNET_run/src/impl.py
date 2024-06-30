"""
Notes :

Normalize data as always (for each image have factor, save factor / image) only for divergence
Modify package of tensorloader in func.py :

    divergence_normalized (with 1 hole), u_raw (with 1 hole),v_raw (with 1 hole), 
    divergence_normalized_without_hole (ground truth), factors for divergence to denormalize

    When you compute physics loss, you need 
@author: Jing Sun

"""
import numpy as np
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt

def physics_loss(DNN_output_div,u,v,x,y):

    """
    calculate the abs error between theoretical D and networkàs D
    Experiment 1. 1 channel in and 1 channel out, only use divergence data. Only compute dataloss, same as before
    Experiment 2. 3 different datasets and 1 channel output. -> 3 : D,U,V, 1: D 
    """
    return physics_loss

def train_model(num_epochs, train_pairs, valid_pairs, device, model, criterion_type, batch_size,
                                        learning_rate, weight_decay, dirs):
    """
    Train a PyTorch model using the given training and validation pairs.
    
    Args:
    - num_epochs (int): number of epochs to train the model
    - train_pairs (torch.utils.data.Dataset): training dataset
    - valid_pairs (torch.utils.data.Dataset): validation dataset
    - device (str): device to use for training and inference ('cpu' or 'cuda')
    - model (torch.nn.Module): PyTorch model to train
    - criterion_type (str): type of loss function to use ('L1' or 'MSE')
    - batch_size (int): size of mini-batches for training and validation
    - learning_rate (float): learning rate for the optimizer
    - weight_decay (float): L2 penalty (regularization) parameter for the optimizer
    - dirs (str): directory to save log file
    
    Returns:
    - train_losses (list): list of training losses for each epoch
    - valid_losses (list): list of validation losses for each epoch
    - model (torch.nn.Module): trained PyTorch model
    - logging (logging.Logger): logger object for recording training and validation losses
    """
    # Move model to the device
    model.to(device)
    
    # Create data loaders for training and validation
    train_loader  = torch.utils.data.DataLoader(train_pairs, shuffle=False, batch_size=batch_size)
    valid_loader  = torch.utils.data.DataLoader(valid_pairs, shuffle=False, batch_size=batch_size)
    

    # Create loss function based on criterion_type
    if criterion_type == 'MAE':
        criterion = nn.L1Loss()
    elif criterion_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid criterion type. Must be 'L1' or 'MSE'.")

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create logger object
    logging.basicConfig(level=logging.INFO, filename=dirs+'/loss_record.log', filemode='w',
                        format='%(asctime)s   %(levelname)s   %(message)s')
    logging.info('Training starts!')
    
    # Initialize lists to record losses
    train_losses = []
    valid_losses = []
    
    # Loop over epochs
    for epoch in range(num_epochs):
       # Train the model
        model.train()
        running_train_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):  
            # Move tensors to the configured device #Inputs and targets stay for divergence, and add for u and v. Duplicate this function
            inputs  = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            btch_train_loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            btch_train_loss.backward()
            optimizer.step()

            running_train_loss += btch_train_loss.item() 
            
        epoch_train_loss = running_train_loss/len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Evaluate the model on validation data
        model.eval()
        running_valid_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valid_loader):  
                # Move tensors to the configured device
                inputs  = inputs.to(device)
                targets = targets.to(device)
                    
                # Forward pass
                outputs = model(inputs)
                btch_valid_loss = criterion(outputs, targets)

                # Plotting the first output of the batch
                if i == 0:  # Adjust according to when and how often you want to plot
                    # Move the tensor to CPU and convert to numpy for plotting
                    output_to_plot = outputs[0].cpu().detach().numpy()
                    
                    # Assuming output is a single-channel image (e.g., segmentation mask)
                    if output_to_plot.shape[0] == 1:  # Single channel image, remove the channel dimension
                        output_to_plot = output_to_plot.squeeze(0)

                    plt.figure()
                    plt.imshow(output_to_plot, cmap='viridis',vmin=0,vmax=1)
                    plt.title(f'Epoch {epoch+1}, Batch {i+1}')
                    plt.colorbar()
                    plt.savefig(f'./visualisation/output_epoch_{epoch+1}_batch_{i+1}.png')
                                    
                
                running_valid_loss += btch_valid_loss.item() 
        
        epoch_valid_loss = running_valid_loss/len(valid_loader)
        valid_losses.append(epoch_valid_loss)
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], train loss: {epoch_train_loss:.5f}, validation Loss: {epoch_valid_loss:.5f}")

        logging.info(
            f"Epoch:[{epoch+1}/{num_epochs}]\t train loss={epoch_train_loss:.5f}\t validation loss={epoch_valid_loss:.5f}\t"
        )

    logging.info("Training is done!")
    
    return train_losses, valid_losses, model, logging

def inference(test_input, device, model, batch_size):
    """
    Predicts the output signal and output noise for a given test input using the specified PyTorch model.
    
    Args:
    - test_input (np.ndarray): Test input of shape (num_samples, num_channels, time_steps).
    - device (str): Device to run the model on (e.g. 'cpu', 'cuda').
    - model (torch.nn.Module): PyTorch model to make predictions with.
    - batch_size (int): Batch size to use for making predictions.
    
    Returns:
    - output_signal (np.ndarray): Predicted output signal of shape (num_samples, time_steps).
    """
    reshaped_test_input = test_input.reshape(test_input.shape[0], 1, *test_input.shape[1:])

    num_samples = reshaped_test_input.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    test_in = torch.from_numpy(reshaped_test_input).float()
    test_in = test_in.to(device)

    model = model.to(device)
    model.eval();
    
    output_signal = []
   
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_in = test_in[start_idx:end_idx]
            batch_out = model(batch_in).detach().cpu().numpy().astype(np.float64)
            output_signal.append(batch_out[:, 0, :, :])
            
    # Concatenate the predicted output signal for all batches   
    output_signal = np.concatenate(output_signal, axis=0)
        
    return output_signal

def save_model(model, dirs):
    """
    Saves PyTorch model's state dictionary to the specified directory.
    
    """
    torch.save({
                'model_state_dict': model.state_dict(),
                }, dirs + "/model.pt")
    
    print('Model saved!')
    return

def load_model(model, dirs):
    """
    Loads PyTorch model's state dictionary from the specified directory.
   
    Args:
    - model (torch.nn.Module): PyTorch model to load the state dictionary into.
    - dirs (str): Directory to load the model from.
   
    Returns:
    - model (torch.nn.Module): PyTorch model with the loaded state dictionary.
    """
    checkpoint = torch.load(dirs + '/model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('Model loaded!')
    
    return model
    