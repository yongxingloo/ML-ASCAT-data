# Select 100 random indices from df2
#LOCAL

def plotting_post_processing_asc(spliced_file):
    import numpy as np
    import matplotlib.pyplot as plt

    df = np.load(spliced_file)
    random_indices = np.random.choice(df.shape[0], 100, replace=False)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    fig.suptitle('100 Random Images')  # Set the title for the entire figure

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(df[random_indices[i]])
        ax.axis('on')  # to hide the axis

    plt.savefig('./Plots/100_random_images_asc.png')
    print('run done')

def plotting_post_processing_des(spliced_file):
    import numpy as np
    import matplotlib.pyplot as plt

    df = np.load(spliced_file)
    random_indices = np.random.choice(df.shape[0], 100, replace=False)

    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    fig.suptitle('100 Random Images')  # Set the title for the entire figure

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(df[random_indices[i]])
        ax.axis('on')  # to hide the axis

    plt.savefig('./Plots/100_random_images_des.png')
    print('run done')

        