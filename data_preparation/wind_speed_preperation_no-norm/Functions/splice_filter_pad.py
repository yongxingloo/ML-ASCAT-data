def hash_array(array):
    import hashlib
    # Convert the array to a byte representation
    array_bytes = array.tobytes()
    # Use a hash function like SHA-256
    return hashlib.md5(array_bytes).hexdigest()


def count_nan_clusters(array):
    import numpy as np
    from scipy.ndimage import label

    # Create a boolean array: True where the original array is NaN
    nan_mask = np.isnan(array)
    
    # Label connected components of NaNs
    labeled_array, num_features = label(nan_mask)
    
    # num_features is the number of connected NaN clusters
    return num_features

def splice_filter_pad_asc(file,angle,cutoff_percentage):

    import numpy as np
    import pandas as pd
    import xarray as xr
    import os
    
    ds = xr.open_dataset(file)

    # Isolating the array of longitude
    longitude = ds.longitude
    latitude = ds.latitude

    range_angle = angle # in degrees

    # Creating a function that takes every 5 degrees of longitude
    min_longtitude = np.min(longitude)
    max_longtitude = np.max(longitude)
    range_longitude = np.arange(min_longtitude,max_longtitude,range_angle)

    min_latitude = np.min(latitude)
    max_latitude = np.max(latitude)
    range_latitude = np.arange(min_latitude,max_latitude,range_angle)

    time_max = np.max(np.shape(ds.time))
    range_time = np.arange(0,time_max,1)

    time = np.arange(0,time_max,1)
    latitude_range = np.arange(min_latitude,max_latitude,range_angle)
    longitude_range = np.arange(min_longtitude,max_longtitude,range_angle)
    
    metadata_entries = []    
    dataset_separated = []

    max_rows = 0
    max_cols = 0
    k = 0

    for lat in latitude_range:
        for lon in longitude_range:
            all_windspeed_selection = np.array(ds.isel(time=time).wind_speed.sel(latitude=slice(lat, lat+range_angle), longitude=slice(lon, lon+range_angle)))
            for k in range(len(all_windspeed_selection)):
                missing_values = np.isnan(all_windspeed_selection[k])
                percentage_missing = np.mean(missing_values) * 100
                if percentage_missing < cutoff_percentage and count_nan_clusters(all_windspeed_selection[k])<2:
                    dataset_separated.append(all_windspeed_selection[k])
                    rows, cols = all_windspeed_selection[k].shape
                    max_rows = max(max_rows, rows)
                    max_cols = max(max_cols, cols)
                    #Creating hash unique ID
                    hash = hash_array(all_windspeed_selection[k])

                    #Normalizing data REMOVED FROM THIS STEP
                    #min_data = np.nanmin(all_windspeed_selection[k])
                    #max_data = np.nanmax(all_windspeed_selection[k])
                    #all_windspeed_selection[k] = (all_windspeed_selection[k] - min_data) / (max_data - min_data)

                    k += 1
                    metadata_entries.append({
                        'latitude start': lat,
                        'latitude end': lat + range_angle,
                        'longitude start': lon,
                        'longitude end': lon + range_angle,
                        'time': ds.time[k].values,
                        'source': 'asc',  # Adjust 'SomeSource' as needed
                        #'min': min_data,
                        #'max': max_data, 
                        'hash': hash
                    })

    
    print(k,' = total number considered')
    # Pad arrays with np.nan to make them all the same size
    dataset_separated = [np.pad(arr, ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), 'constant', constant_values=np.nan) for arr in dataset_separated]

    #Convert metadata list to dataframe for export
    metadata = pd.DataFrame(metadata_entries)
    metadata.to_csv('./Output/Metadata/metadata_asc.csv', index=False)

    np.save('./Output/Full_dataset_spliced_filtered_asc.npy', dataset_separated)

    print('run done')

def splice_filter_pad_des(file,angle,cutoff_percentage):

    import numpy as np
    import xarray as xr
    import pandas as pd
    import os
    
    ds = xr.open_dataset(file)

    # Isolating the array of longitude
    longitude = ds.longitude
    latitude = ds.latitude

    range_angle = angle # in degrees

    # Creating a function that takes every 5 degrees of longitude
    min_longtitude = np.min(longitude)
    max_longtitude = np.max(longitude)
    range_longitude = np.arange(min_longtitude,max_longtitude,range_angle)

    min_latitude = np.min(latitude)
    max_latitude = np.max(latitude)
    range_latitude = np.arange(min_latitude,max_latitude,range_angle)

    time_max = np.max(np.shape(ds.time))
    range_time = np.arange(0,time_max,1)

    time = np.arange(0,time_max,1)
    latitude_range = np.arange(min_latitude,max_latitude,range_angle)
    longitude_range = np.arange(min_longtitude,max_longtitude,range_angle)

    dataset_separated = []
    metadata_entries = []    

    max_rows = 0
    max_cols = 0


    for lat in latitude_range:
        for lon in longitude_range:
            all_windspeed_selection = np.array(ds.isel(time=time).wind_speed.sel(latitude=slice(lat, lat+range_angle), longitude=slice(lon, lon+range_angle)))
            for k in range(len(all_windspeed_selection)):
                missing_values = np.isnan(all_windspeed_selection[k])
                percentage_missing = np.mean(missing_values) * 100
                if percentage_missing < cutoff_percentage and count_nan_clusters(all_windspeed_selection[k])<2:
                    dataset_separated.append(all_windspeed_selection[k])
                    rows, cols = all_windspeed_selection[k].shape
                    max_rows = max(max_rows, rows)
                    max_cols = max(max_cols, cols)

                    #Creating hash unique ID
                    hash = hash_array(all_windspeed_selection[k])
                    #Normalizing data 
                    #min_data = np.nanmin(all_windspeed_selection[k])
                    #max_data = np.nanmax(all_windspeed_selection[k])
                    #all_windspeed_selection[k] = (all_windspeed_selection[k] - min_data) / (max_data - min_data)

                    metadata_entries.append({
                        'latitude start': lat,
                        'latitude end': lat + range_angle,
                        'longitude start': lon,
                        'longitude end': lon + range_angle,
                        'time': ds.time[k].values,
                        'source': 'des',  # Adjust 'SomeSource' as needed
                        #'min': min_data,
                        #'max': max_data,
                        'hash': hash
                    })

    # Pad arrays with np.nan to make them all the same size
    dataset_separated = [np.pad(arr, ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), 'constant', constant_values=np.nan) for arr in dataset_separated]


    #Convert metadata list to dataframe for export
    metadata = pd.DataFrame(metadata_entries)
    metadata.to_csv('./Output/Metadata/metadata_des.csv', index=False)

    np.save('./Output/Full_dataset_spliced_filtered_des.npy', dataset_separated)

    print('run done')