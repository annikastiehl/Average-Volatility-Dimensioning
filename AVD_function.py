import numpy as np
import pandas as pd
import time

# This function performs AVD dimension reduction to one feature on a given DataFrame, considering either Standard Deviation (SD) or Mean Absolute Deviation (MAD) across a sliding window.
# The user can specify the metric to use (default is MAD), the window size and increment.

def calculate_AVD_feature(df, metric='MAD', w_size=10, w_incre=1):
    # Dictionary to store the results of the AVD calculation
    results = {
        f'{metric}': [],  # Initialize an empty list for the specified metric
    }

    # Loop through the DataFrame with a sliding window
    for i in range(0, len(df) - w_size + 1, w_incre):
        # Dictionary to store the metric values for the current window
        current_window_metrics = {}

        # Iterate over each column in the DataFrame
        for column in df.columns:
            # Extract the data within the current window for the column
            window = df[column].iloc[i:i + w_size].values

            # Calculate the metric based on the user's choice
            if metric == 'SD':
                # Calculate Standard Deviation with a correction for degrees of freedom
                current_window_metrics[column] = np.std(window, ddof=1)
            elif metric == 'MAD':
                # Calculate Mean Absolute Deviation
                mean = np.mean(window)
                current_window_metrics[column] = np.mean(np.abs(window - mean))


        values_for_metric = []

        # Iterate over all pairs of features to compare their values
        for feature1_idx, feature1 in enumerate(df.columns):
            for feature2_idx in range(feature1_idx + 1, len(df.columns)):
                feature2 = df.columns[feature2_idx]
                # Compute the squared difference between the values of the two features
                diff = current_window_metrics[feature1] - current_window_metrics[feature2]
                values_for_metric.append(diff ** 2)

        # Calculate the average of the squared differences for the current window
        avg_value = np.mean(values_for_metric) if values_for_metric else 0 # This is the new AVD feature value

        # Append the value to the results dictionary
        results[f'{metric}'].append(avg_value)

    # Create a DataFrame from the results dictionary
    metrics_df = pd.DataFrame(results)
    
    # Generate the time points
    time_points = np.arange(w_size - 1, w_size - 1 + len(metrics_df) * w_incre, w_incre)
    # Add the time points as an index to the DataFrame
    metrics_df['Time Point'] = time_points
    metrics_df.set_index('Time Point', inplace=True)


    # Return the DataFrame containing the calculated complexity metric values
    return metrics_df