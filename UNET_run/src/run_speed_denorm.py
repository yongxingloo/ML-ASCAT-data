"""

@author: Jing Sun

"""
from unet import *
from func import *
from impl import *
import numpy as np
import os

# Allow the program to use duplicated libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

""" Define DNN settings """
device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels    = 1
out_channels   = 1
features       = [32,64,128,256,512]
filter_size    = 3
criterion_type = 'MAE'
batch_size     = 256
learning_rate  = 0.001
weight_decay   = 0#1e-5
num_epochs     = 50
model          = UNET(in_channels, out_channels, filter_size, features)

"""  Training  """
# Load data  
train_input  = np.load("../../Wind_speed_Preprocessing_Denorm/Output/Final_for_run/train_input.npy")
train_target = np.load("../../Wind_speed_Preprocessing_Denorm/Output/Final_for_run/train_target.npy")

train_input_bag     = prepare_data(train_input, train_target)

# Split the data into training and validation sets  '''
train_pairs, valid_pairs = split_dataset(train_input_bag, train_percentage=0.8)

# Train the model
dirs = create_folder_struct()
train_losses, valid_losses, model, log = train_model(num_epochs, train_pairs, valid_pairs, device, model, 
                                                     criterion_type, batch_size, learning_rate, weight_decay, dirs)

# Save the trained model and loss plot
write_summary(model, dirs)
plot_loss(train_losses, valid_losses, dirs)
save_model(model, dirs)

"""  Inference  """  
# Load data for inference
test_input   = np.load("../../Wind_speed_Preprocessing_Denorm/Output/Final_for_run/test_input.npy")
test_target = np.load("../../Wind_speed_Preprocessing_Denorm/Output/Final_for_run/test_target.npy")

## Load model if pre-trained
# dirs  = "../output/2023-03-25_13-56/L1_bs2_lr0.001/" 
# model = UNET(in_channels, out_channels, filter_size, features)
# model = load_model(model, dirs)

# Execute inference and save files
test_output = inference(test_input, device, model, batch_size)
np.save(dirs + "/DNN_output_speed.npy", test_output)

v_min = np.nanmin(test_target)
print(v_min,' = vmin')
v_max = np.nanmax(test_target)
print(v_max,' = vmax')
plot_example(test_target, test_input, test_output, 
            (test_target-test_output), dirs, v_min, v_max, example_num=1)
