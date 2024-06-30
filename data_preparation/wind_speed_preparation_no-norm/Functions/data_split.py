def split(features,targets,ratio):

    # Import the necessary libraries
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Load the feature and target data
    targets = np.load(targets)
    features = np.load(features)

    # Split the data into training and testing sets with an 80-20 ratio
    train_input, test_input, train_target, test_target = train_test_split(targets, features, test_size=ratio, random_state=42)

    # Save the split datasets
    np.save("./Output/Final_for_run/train_input.npy", train_input)
    np.save("./Output/Final_for_run/train_target.npy", train_target)
    np.save("./Output/Final_for_run/test_input.npy", test_input)
    np.save("./Output/Final_for_run/test_target.npy", test_target)