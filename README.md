# ML-ASCAT-data
This is the repository for the code used for the project : Machine learning application for pixel infilling on METOP-A satellite surface wind data 

Folders present :

1. data_preparation (all preperation steps including : Slicing, Filtering, Nan-values removal, Mask dataset creation, data split, metadata creation)
2. UNET_run (run command using preprocessed data for UNET architecture)
3. RESNET_run (run command using preprocessd data for RESNET architecture)
4. Krigging_run (run command for the krigging function)
5. data_post-procssing (Metadata read and denormalisation of data)
6. BCO (Retrieval of data from Barbados cloud observatory)
7. NTAS (Retrival of data and correction for ASCAT instrument)
8. analysis (Comparaison metrics for UNET, RESNET, KRIGGING output)

