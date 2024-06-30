# Select 100 random indices from df2


def plotting_post_processing_nonnan_asc(spliced_file):
    import numpy as np
    import matplotlib.pyplot as plt

    df = np.load(spliced_file)
    dataset_merged_nonnan = np.nan_to_num(df)

    random_indices = np.random.choice(dataset_merged_nonnan.shape[0], 100, replace=False)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    fig.suptitle('100 Random Images')  # Set the title for the entire figure

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(dataset_merged_nonnan[random_indices[i]])
        ax.axis('on')  # to hide the axis

    np.save('./Output/Full_dataset_spliced_filtered_nonnan_asc.npy', dataset_merged_nonnan)

    plt.savefig('./Plots/100_random_images_nonnan_asc.png')
    print('run done')

def plotting_post_processing_nonnan_des(spliced_file):
    import numpy as np
    import matplotlib.pyplot as plt

    df = np.load(spliced_file)
    dataset_merged_nonnan = np.nan_to_num(df)

    random_indices = np.random.choice(dataset_merged_nonnan.shape[0], 100, replace=False)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    fig.suptitle('100 Random Images')  # Set the title for the entire figure

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(dataset_merged_nonnan[random_indices[i]])
        ax.axis('on')  # to hide the axis

    np.save('./Output/Full_dataset_spliced_filtered_nonnan_des.npy', dataset_merged_nonnan)

    plt.savefig('./Plots/100_random_images_nonnan_des.png')
    print('run done')


        