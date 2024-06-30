import numpy as np
import matplotlib.pyplot as plt

def create_localized_random_shapes(data, num_points, max_radius):
    """
    Creates artificial localized holes with random elliptical shapes in a 2D NumPy dataset,
    tailored for smaller datasets.

    Parameters:
    data (numpy.ndarray): The 2D array in which to create holes.
    num_points (int): The number of points around which to create clusters of missing values.
    max_radius (int): The maximum radius for major and minor axes of ellipses.

    Returns:
    numpy.ndarray: The array with artificial localized holes.
    """
    data_with_holes = data.copy()
    rows, cols = data.shape

    for _ in range(num_points):
        # Randomly choose a center point for the cluster
        center_row = np.random.randint(max_radius, rows - max_radius)
        center_col = np.random.randint(max_radius, cols - max_radius)

        # Randomly determine the lengths of the major and minor axes (ensure minor <= major)
        major_axis = np.random.randint(1, min(max_radius, center_row, rows-center_row,
                                              center_col, cols-center_col) + 1)
        minor_axis = np.random.randint(1, major_axis + 1)

        # Randomly determine the orientation angle (in radians)
        angle = np.random.uniform(0, np.pi)

        # Calculate the rotated ellipse points and create holes
        for i in range(-major_axis, major_axis + 1):
            for j in range(-major_axis, major_axis + 1):
                x = i * np.cos(angle) - j * np.sin(angle)
                y = i * np.sin(angle) + j * np.cos(angle)
                if 0 <= center_row + i < rows and 0 <= center_col + j < cols:
                    if (x**2 / major_axis**2) + (y**2 / minor_axis**2) <= 1:
                        data_with_holes[center_row + i, center_col + j] = np.nan

    return data_with_holes

def display_data_with_holes(data, num_points=5, max_radius=3):
    # Create data with localized random shapes as holes
    data_with_holes = create_localized_random_shapes(data, num_points, max_radius)

    # Plot the data
    plt.imshow(data_with_holes, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Data with Localized Random Shaped Holes')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

def save_holes_dataset_asc(df):
    import numpy as np
    import matplotlib.pyplot as plt
    
    df = np.load(df)

    # Example usage
    data = df[120]  # Example 2D dataset
    num_points = np.random.randint(5,10)  # Number of points around which to create clusters of missing values
    cluster_radius_max = 3  # Radius of each cluster
    data_with_localized_holes = create_localized_random_shapes(data, num_points, cluster_radius_max)

    # Display the resulting dataset with artificial localized holes
    plt.imshow(data_with_localized_holes, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Data with Localized Holes')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    dataset_with_holes_final = np.zeros_like(df)

    for i in range(0,df.shape[0]):
        dataset_with_holes_final[i] = create_localized_random_shapes(df[i],num_points,cluster_radius_max)

    dataset_with_holes_final_nonnan = np.nan_to_num(dataset_with_holes_final)
    np.save('./Output/Full_dataset_spliced_filtered_nonnan_holes_asc.npy', dataset_with_holes_final_nonnan)

def save_holes_dataset_des(df):
    import numpy as np
    import matplotlib.pyplot as plt
    
    df = np.load(df)

    # Example usage
    data = df[120]  # Example 2D dataset
    num_points = np.random.randint(5,10)  # Number of points around which to create clusters of missing values
    cluster_radius_max = 3  # Radius of each cluster
    data_with_localized_holes = create_localized_random_shapes(data, num_points, cluster_radius_max)

    # Display the resulting dataset with artificial localized holes
    plt.imshow(data_with_localized_holes, interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Value')
    plt.title('Data with Localized Holes')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')

    dataset_with_holes_final = np.zeros_like(df)

    for i in range(0,df.shape[0]):
        dataset_with_holes_final[i] = create_localized_random_shapes(df[i],num_points,cluster_radius_max)

    dataset_with_holes_final_nonnan = np.nan_to_num(dataset_with_holes_final)
    np.save('./Output/Full_dataset_spliced_filtered_nonnan_holes_des.npy', dataset_with_holes_final_nonnan)

