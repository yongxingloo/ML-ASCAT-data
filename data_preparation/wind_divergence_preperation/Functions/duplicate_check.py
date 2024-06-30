
def has_duplicates(array):
    import numpy as np
    import matplotlib.pyplot as plt

    # This will hold the hashes
    seen_hashes = set()

    # Iterate through each 42x42 image
    for image in array:
        # Compute a hash for the current image
        # We use a tuple of the flattened array for hashing
        image_hash = hash(image.tobytes())
        
        # Check if hash already exists
        if image_hash in seen_hashes:
            print('true')
            return True  # Found a duplicate

        # Add the new hash to the set
        seen_hashes.add(image_hash)
    
    print('false')
    return False  # No duplicates found

