# Average Volatility Dimensioning AVD for Multivariate Time Series  Experimental Code Repository

This repository implements the experimental workflow for the Average Volatility Dimensioning AVD method introduced in the repository description. AVD is a dimension reduction approach for multivariate time series that transforms multiple sensor dimensions into a single derived signal intended to preserve relevant inner dynamics and inter feature relationships. The code is part of a publication in International Journal for Data Science and should be cited as followed:
Mallinger, K., Marica, E., Schrittwieser, S. et al. Average Volatility Dimensioning (AVD): dimension reduction technique for multivariate time series. Int J Data Sci Anal 22, 30 (2026). https://doi.org/10.1007/s41060-025-00894-w

## Overview

The project focuses on reducing multivariate time series into a univariate representation through Average Volatility Dimensioning. The repository evaluates whether this reduced representation remains useful for downstream classification tasks.

The current code compares AVD based representations on two datasets:

Movement  
Hydraulic

The method is evaluated through a classification workflow in which the generated AVD feature is used as input for supervised models. 

## Conceptual Pipeline

### Step 1  Load and prepare the multivariate time series

The main script loads an aggregated dataset from the `data/` directory. The expected input file format is

`data/<Dataset>_aggregated_data.csv`

where `<Dataset>` is one of the supported dataset names.

For the currently implemented workflow, the script expects the data to contain

`cycle`  
`time`  
multiple sensor feature columns  
`class`

In this repository, a `cycle` denotes one complete multivariate time series sample or recording instance. All rows with the same cycle value belong to the same temporal sequence, while time specifies the order of observations within that sequence. The AVD feature is computed separately for each cycle, and each cycle is later treated as one classification sample. `time` indicates the ordered time step within a given cycle, while `class` is the label assigned to the entire cycle and is assumed to remain the same for all rows belonging to that cycle.

Examples of what a cycle could represent:

one machine run
one movement trial
one recorded activity segment
one hydraulic system sequence

The script then adjusts the time index for some datasets and sets `cycle` and `time` as a multi index. 


### Step 2  Compute the AVD feature

The core method is implemented in `AVD_function.py` through the function

`calculate_AVD_feature(df, metric='MAD', w_size=10, w_incre=1)`

For each sliding window, the code first computes a volatility related statistic for every feature dimension

MAD  
or  
SD

It then calculates pairwise squared differences between the window level feature statistics across all dimensions and averages the upper triangular part of that matrix. The resulting scalar becomes the AVD value for that window.

In this way, AVD produces a single time dependent signal from a multivariate input sequence. :contentReference[oaicite:5]{index=5}

### Step 3  Save the reduced representation

The computed AVD values are written to

`output/Data_Processing_<Dataset>/avd_results_<Metric>_<Dataset>.csv`

This file stores the generated AVD sequence together with the corresponding class labels mapped back from the original cycles. :contentReference[oaicite:6]{index=6}

### Step 4  Reshape for classification

After saving the AVD output, the script reloads the CSV file and reshapes it into a wide cycle based table. Each cycle becomes one sample and the AVD values across time become the input features for classification. The class label is then attached to each cycle instance. 

### Step 5  Classification benchmark

The classification stage uses

`train_test_split` with a 70 30 split  
`LabelEncoder` for class encoding  
`LazyClassifier` from LazyPredict for broad model comparison

The model comparison results are saved to

`output/Results_<Dataset>/Classification_Result_<Metric>.csv`

This provides a simple benchmark of how informative the AVD based representation is for label prediction. :contentReference[oaicite:8]{index=8}

## Repository Structure

### Main files

`main.py`  
Main execution script for loading the data, normalizing cycles, generating the AVD signal, reshaping the output, and running the classification comparison. :contentReference[oaicite:9]{index=9}

`AVD_function.py`  
Core implementation of the AVD computation based on sliding window MAD or SD statistics and averaged pairwise squared differences between dimensions. :contentReference[oaicite:10]{index=10}

`data/`  
Input folder expected to contain aggregated dataset CSV files such as the Movement, Sports, and Hydraulic datasets. :contentReference[oaicite:11]{index=11}

`output/`  
Output folder used for storing processed AVD features and classification results. :contentReference[oaicite:12]{index=12}

`LICENSE`  
Repository license file. The repository currently uses the MIT license. :contentReference[oaicite:13]{index=13}


## AVD method summary

Average Volatility Dimensioning reduces a multivariate time series into a univariate sequence by

1. Taking a sliding window across the multivariate signal  
2. Computing a volatility measure per feature dimension within that window  
3. Measuring how different these per dimension volatility values are from one another  
4. Averaging those pairwise squared differences into one scalar output

The repository currently supports two volatility measures:

MAD  
Mean Absolute Deviation

SD  
Standard Deviation


## Input data requirements

To run the workflow, each dataset should be prepared as an aggregated CSV file with at least the following structure

`cycle` column  
`time` column  
multiple sensor feature columns  
`class` column

The script assumes that each cycle corresponds to one full multivariate sequence and that classification labels are constant within each cycle. :contentReference[oaicite:16]{index=16}

## How to run

A typical workflow is

1. Place the dataset CSV files in the `data/` directory  
2. Open `main.py`  
3. Select the dataset name, metric, window size, and window increment  
4. Run the script  
5. Check the generated files in the `output/` directory

Example function call

```python
Classification_with_AVD_Feature(dataset="Hydraulic", metric="MAD", w_size=10, w_incre=1)
