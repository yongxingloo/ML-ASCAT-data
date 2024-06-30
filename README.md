# ML-ASCAT-data

This is the repository for the code used for the thesis : 
Machine learning application for pixel infilling on METOP-A satellite surface wind data 

## Abstract

Large scale meteorological measurements from satel-
lite data are the basis of climate models and weather
forecasting and are crucial to the understanding of atmo-
spheric physics. However these datasets often contain
missing values. This research proposes to use convo-
lutional neural networks (CNN) to infill missing pixel
values in wind speed data collected by the METOP-
A satellite and assess if this method is better than the
krigging algorithm, which is a non-machine learning
infilling algorithms used today. In testing two model
architectures, the ResNET and the UNET, it is found
that the ResNET produces a mean absolute error that is
half of what is found using krigging (MAE = 0.0316 m/s
> MAE = 0.0163 m/s) whilst outperforming it in terms
of processing time, it is also found that the U-NET can
achieve the same results as the conventional method
whilst only requiring 1/4 of the processing time. A key
finding is that the primary limitation of the neural net-
works isn’t the accuracy of the prediction but the quality
of the assumptions and their implementation.


## Instructions for running the files
All outputs are in repository and can be read as is. If one wants the run the model again, here are instructions

Necessary packages

For preprocessing :

1. xarray
2. netcdf4


Running preprocessing :

    ->Enter folder ./data_preperation/
    ->Enter folder of desired variable preprocessing 
    ->run the run.py file (this will generate all Outputs including everything needed to run the models in the correct paths and also will generate plots for checking)


