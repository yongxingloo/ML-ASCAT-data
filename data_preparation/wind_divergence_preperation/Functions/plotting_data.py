def plot_variables(file,time): # Plotting the dataset.

    import xarray as xr
    import matplotlib.pyplot as plt
    import os

    ds = xr.open_dataset(file)

    fig, ax = plt.subplots(2, 2, figsize=(10, 5))

    ds.isel(time=time).wind_speed.plot(ax=ax[0, 0])
    ax[0, 0].set_title('Wind Speed')

    ds.isel(time=time).wind_divergence.plot(ax=ax[0, 1])
    ax[0, 1].set_title('Wind Divergence')

    ds.isel(time=time).eastward_wind.plot(ax=ax[1, 0])
    ax[1, 0].set_title('Eastward Wind')

    ds.isel(time=time).northward_wind.plot(ax=ax[1, 1])
    ax[1, 1].set_title('Northward Wind')

    plt.tight_layout()

    filename = os.path.basename(file)

    filename_without_ext = os.path.splitext(filename)[0]

    fig.savefig(f'./Plots/{filename_without_ext}_parameters_plot.png')
