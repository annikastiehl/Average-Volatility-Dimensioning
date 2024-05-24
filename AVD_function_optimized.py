import numpy as np
import pandas as pd
import time

def calculate_AVD_feature_optimized(df, metric='MAD', w_size=10, w_incre=1):
    # Initialize the results list
    results = []

    # Precompute means for each column in the DataFrame
    if metric == 'MAD':
        col_means = df.rolling(window=w_size).mean().values[w_size - 1:]
    elif metric == 'SD':
        col_stds = df.rolling(window=w_size).std(ddof=1).values[w_size - 1:]

    # Iterate over the windows
    for start_idx in range(0, len(df) - w_size + 1, w_incre):
        end_idx = start_idx + w_size

        # Extract the window data
        window_data = df.iloc[start_idx:end_idx].values

        if metric == 'MAD':
            # Calculate the Mean Absolute Deviation for each column in the window
            mean_window = col_means[start_idx]
            mad_values = np.mean(np.abs(window_data - mean_window), axis=0)
            current_window_metrics = mad_values
        elif metric == 'SD':
            # Calculate the Standard Deviation for each column in the window
            sd_values = col_stds[start_idx]
            current_window_metrics = sd_values

        # Compute the pairwise squared differences
        diff_matrix = current_window_metrics[:, None] - current_window_metrics
        squared_diff_matrix = diff_matrix ** 2

        # Calculate the average of the upper triangle of the squared differences matrix
        upper_triangle_indices = np.triu_indices_from(squared_diff_matrix, k=1)
        avg_squared_diff = np.mean(squared_diff_matrix[upper_triangle_indices])

        # Append the result
        results.append(avg_squared_diff)

    # Create the resulting DataFrame
    metrics_df = pd.DataFrame({metric: results})
    time_points = np.arange(w_size - 1, w_size - 1 + len(metrics_df) * w_incre, w_incre)
    metrics_df['Time Point'] = time_points
    metrics_df.set_index('Time Point', inplace=True)


    return metrics_df
