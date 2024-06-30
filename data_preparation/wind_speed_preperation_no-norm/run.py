# This code is the run.py file for the METOP-A Wind Data Preprocessing program. From a initial base file (1 asc and 1 des), outputs all
# the necessary files for the training of the neural network. The program is divided in 8 main steps:

# 1. Import the data and request metadata
# 2. Plot the variables
# 3. Splice, filter and pad the data
# 4. Plot 100 random images
# 5. Plot 100 random images without nan values and create files
# 6. Create dataset with holes
# 7. Merge the datasets
# 8. Split the dataset into training and testing sets

################################################################

import os
import traceback
import numpy as np

try:
    print('METOP-A Wind Data Preprocessing program')
    print('start of preprocessing, running batch files')

    file_asc = '../data/cmems_obs-wind_glo_phy_my_l3-metopa-ascat-asc-0.125deg_P1D-i-2007_2021.nc'
    file_des = '../data/cmems_obs-wind_glo_phy_my_l3-metopa-ascat-des-0.125deg_P1D-i-2007_2021.nc'

    print('\n-------------------------------------')

    print('file to import (1):', file_asc)
    print('file to import (2):', file_des)

    ############################################
    from Functions.metadata import request_metadata

    print('\n-------------------------------------')

    print('metadata ASC')
    request_metadata(file_asc)

    print('\n-------------------------------------')

    print('metadata DES')
    request_metadata(file_des)
    print('done:)')
    print('\n-------------------------------------')

    ############################################
    from Functions.plotting_data import plot_variables
    print('plotting ASC and DES')

    # Ensure Plots directory exists
    if not os.path.exists('./Plots'):
        os.makedirs('./Plots')

    # Call the plot_variables function with verbose logging
    print(f"Calling plot_variables for {file_asc}")
    plot_variables(file_asc, 0)
    print('1/2 :)')

    print(f"Calling plot_variables for {file_des}")
    plot_variables(file_des, 0)
    print('2/2 :)')
    print('done :)')
    print('\n-------------------------------------')
    ############################################

    from Functions.splice_filter_pad import splice_filter_pad_asc, splice_filter_pad_des

    print('splicing, filtering and padding ASC and DES')
    splice_filter_pad_asc(file_asc,5,30)
    print('1/2')

    splice_filter_pad_des(file_des,5,30)

    print('2/2')


    print('\n-------------------------------------')
    ############################################
    from Functions.random100_plot import plotting_post_processing_asc, plotting_post_processing_des

    print('plotting 100 random images ASC and DES')

    spliced_file_asc = './Output/Full_dataset_spliced_filtered_asc.npy'
    spliced_file_des = './Output/Full_dataset_spliced_filtered_des.npy'
    
    plotting_post_processing_asc(spliced_file_asc)
    print('1/2')

    plotting_post_processing_des(spliced_file_des)
    print('2/2')
    print('\n-------------------------------------')

    ############################################
    from Functions.random100_plot_nonnan import plotting_post_processing_nonnan_asc, plotting_post_processing_nonnan_des
    print('plotting 100 random images ASC and DES without nan values and creating files')
    plotting_post_processing_nonnan_asc(spliced_file_asc)
    print('1/2')

    plotting_post_processing_nonnan_des(spliced_file_des)
    print('2/2')
    print('\n-------------------------------------')

    ############################################
    from Functions.holes_creation import create_localized_random_shapes, save_holes_dataset_asc,save_holes_dataset_des
    print('creating dataset with holes ASC and DES')
    save_holes_dataset_asc(spliced_file_asc)
    print('1/2')
    print('run done :)')

    save_holes_dataset_des(spliced_file_des)
    print('2/2')
    print('run done :)')

    print('\n-------------------------------------')

    ############################################
    from Functions.random100_plot import plotting_post_processing_asc, plotting_post_processing_des

    print('plotting 100 random images ASC and DES')

    hole_file_asc = './Output/Full_dataset_spliced_filtered_nonnan_holes_asc.npy'
    hole_file_des = './Output/Full_dataset_spliced_filtered_nonnan_holes_des.npy'
    
    plotting_post_processing_asc(hole_file_asc)
    print('1/2')

    plotting_post_processing_des(hole_file_des)
    print('2/2')
    print('\n-------------------------------------')

    ############################################
    import pandas as pd

    print('merging the ASC and DES datasets.')

    normal_asc = np.load('./Output/Full_dataset_spliced_filtered_nonnan_asc.npy')
    print(np.shape(normal_asc),'= shape of normal_asc')
    normal_des = np.load('./Output/Full_dataset_spliced_filtered_nonnan_des.npy')
    print(np.shape(normal_des),'= shape of normal_des')

    print('merging the normal datasets.')
    normal_merged = np.concatenate((normal_asc, normal_des), axis=0)
    print(np.shape(normal_merged),'= shape of normal merged dataset')

    np.save('./Output/Full_dataset_spliced_filtered_nonnan_merged_normal.npy', normal_merged)

    holes_asc = np.load('./Output/Full_dataset_spliced_filtered_nonnan_holes_asc.npy')
    print(np.shape(holes_asc),'= shape of holes_asc')

    holes_des = np.load('./Output/Full_dataset_spliced_filtered_nonnan_holes_des.npy')
    print(np.shape(holes_des),'= shape of holes_des')

    holes_merged = np.concatenate((holes_asc, holes_des), axis=0)
    np.save('./Output/Full_dataset_spliced_filtered_nonnan_merged_holes.npy', holes_merged)

    print('merging the merged holes dataset.')

    holes_merged = np.concatenate((holes_asc, holes_des), axis=0)
    print(np.shape(holes_merged),'= shape of holes merged dataset')

    print('metadata of the merged dataset:')

    metadata_asc = pd.read_csv('./Output/Metadata/metadata_asc.csv')
    print(np.shape(metadata_asc),'= shape of metadata_asc')

    metadata_des = pd.read_csv('./Output/Metadata/metadata_des.csv')
    print(np.shape(metadata_des),'= shape of metadata_des')

    metadata_merged = pd.concat([metadata_asc, metadata_des], axis=0)

    metadata_merged.to_csv('./Output/Metadata/metadata_merged.csv', index=False)

    print('run done :)')

    ############################################
    print('\n-------------------------------------')
    print('splitting the dataset into training and testing sets')
    from Functions.data_split import split

    features = './Output/Full_dataset_spliced_filtered_nonnan_merged_normal.npy'
    targets = './Output/Full_dataset_spliced_filtered_nonnan_merged_holes.npy'
    split(features, targets,0.2)

    print('run done :)')

    ############################################
    print('\n-------------------------------------')
    print('END OF PROCESSING :)))')



except Exception as e:
    print(f"An error occurred: {e}")
    print(traceback.format_exc())


