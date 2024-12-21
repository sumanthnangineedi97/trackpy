
# Overview 

This repository contains the code for tracking cell positions using the Trackpy Brownian algorithm. It enables tracking of cell movement and provides scripts to:

* Convert annotation data from TrackMate to the necessary CSV files like utids, spots_velocity and mito csvs.

* Downsample (sparse) datasets based on frames for linking at low -frame rates.

* Generate visualizations of the Trackpy-predicted linking which includs videos displaying track information of cells.


## Environment Setup

```bash
pip install .
```
    

## Generate preprocessed files

To generate the preprocessed files from the trackmate annotation .xml files, run the gen_preprocess_data.py file.

```bash
python -m gen.gen_preprocess_data
```

## Sparse Data generation

After generating the preprocessed files with non-sparsed data, you can create sparsified datasets by running the gen_sparse_data.py script. Adjust the gap parameter as needed to define the interval for sparsing the data. 

```bash
python -m gen.gen_sparse_data
```

## Generate visualizations 

To visualize the cell movements, use the gen_visualization.py script. This will create a video showcasing the tracked cell movements along with track IDs and linking information.

```bash
python -m gen.gen_visualization
```

## Trackpy linking

The trackpy_linking.py script implements the Trackpy Brownian algorithm for linking tracks in the preprocessed data. It generates a CSV file containing the predicted linked tracks and evaluates performance metrics.

To execute the Trackpy linking process, run the following command:

```bash
python -m gen.trackpy_linking
```

