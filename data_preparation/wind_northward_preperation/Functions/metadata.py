

# Open your netCDF file

def request_metadata(file):
    import numpy as np
    import xarray as xr

    try:
        ds = xr.open_dataset(file)
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        ds = None

    if ds is not None:
        # Print a summary of the dataset
        print("\nSummary of the dataset:")
        print(ds)

    else:
        print("Dataset is not available for inspection.")