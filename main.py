import numpy as np
import pandas as pd
import os
from AVD import calculate_AVD_feature
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# This is the main function that performs the AVD dimension-reduction and the classification on this feature
# The user can specify the dataset ("Movement", "Sports" or "Hydraulic") and the metric ("MAD" OR "SD"), the window size and increment for the calculate_AVD_feature function

def Classification_with_AVD_Feature(dataset = "Movement", metric = "MAD", w_size=10, w_incre=1):
    # Imorting the data
    path_data = f'data/{dataset}_aggregated_data.csv'
    data = pd.read_csv(path_data)

    # Set "cycle" and "time" as MultiIndex
    data.set_index(['cycle', 'time'], inplace=True)

    # Make a list with feature columns
    sensor_columns = [col for col in data.columns if col != 'class']
    print(sensor_columns)

    # Mean normalization
    mean_per_cycle = data.groupby(level='cycle')[sensor_columns].transform('mean')
    normalized_cycles_data = data.copy()
    normalized_cycles_data[sensor_columns] = data[sensor_columns] - mean_per_cycle

    # Get the 'class' values
    class_info_series = normalized_cycles_data.reset_index().drop_duplicates(subset='cycle').set_index('cycle')['class']

    # Drop the 'class' values from the DataFrame
    normalized_cycles_data.drop(columns=["class"], inplace=True)

    # Initialize empty list to store new AVD feature
    results_list = []

    # Get the cycles
    cycles = normalized_cycles_data.index.get_level_values('cycle').unique()

    # Calculate AVD feature per cycle
    for cyc in cycles:
        print(f"Processing cycle: {cyc}")
        cycle_data = normalized_cycles_data.loc[cyc]

        # Calculate AVD feature for the cycle
        avd_for_cycle = calculate_AVD_feature(pd.DataFrame(cycle_data), metric = metric, w_size=w_size, w_incre=w_incre)

        # Append the AVD value for each time point to the results list
        for time_point, row in avd_for_cycle.iterrows():
            results_list.append((cyc, time_point) + tuple(row.values))

    # Adjust the columns list by removing 'class'
    columns = ['cycle', 'time'] + [f'{metric}']

    # Create DataFrame from the results list
    avd_results = pd.DataFrame(results_list, columns=columns)
    avd_results.set_index(['cycle', 'time'], inplace=True)


    # Map 'class' information back to avd_results DataFrame using class_info_series for all cycles
    avd_results['class'] = avd_results.index.get_level_values('cycle').map(class_info_series)

    print("This is the result:")
    print(avd_results)


    # Save the results
    avd_results.to_csv(f"output/Data_Processing_{dataset}/avd_results_{metric}_{dataset}.csv")

    # Read the csv file with the AVD feature
    avd_results = pd.read_csv(f"output/Data_Processing_{dataset}/avd_results_{metric}_{dataset}.csv")

    # Copy avd_results for later use
    temp = avd_results.copy()

    # Pivot the DataFrame using cycle as index and time as columns
    pivot_df = temp.pivot_table(index='cycle', columns='time')

    # Reorder the columns so that they are sorted
    pivot_df = pivot_df.sort_index(axis=1, level=1)

    # Stack the DataFrame to convert it from wide to long format
    result_df = pivot_df.stack(level=0)

    # Convert series with 'class' value to list
    class_list = class_info_series.tolist()

    # Iterate over unique cycles
    for cycle in result_df.index.levels[0]:
        # Drop 'class' rows for each cycle
        result_df = result_df.drop(index=(cycle, 'class'))
        
    # Add the 'class' column to the original DataFrame
    result_df['class'] = class_list

    # Reset index to have regular columns
    result_df.reset_index(inplace=True)

    # Rename column 'level_1' to 'Metric'
    result_df.rename(columns={'level_1': 'Metric'}, inplace=True)

    # Specify directory to save the results
    save_dir_comparison = f'output/Results_{dataset}/'
    os.makedirs(save_dir_comparison, exist_ok=True)



    # Function to process, classify, and save results for each complexity metric
    def classify_for_each_component(component_df, metric, save_dir):
        # Prepare the dataset for classification
        X = component_df.drop(['class', 'cycle', 'Metric'], axis=1)
        y = component_df['class']

        # Encoding class labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Splitting dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

        # Converting to NumPy arrays
        X_train = np.array(X_train, dtype=float)
        X_test = np.array(X_test, dtype=float)

        # Classification
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        # Save the classification results
        save_path = os.path.join(save_dir, f'Classification_Result_{metric}.csv')
        models.to_csv(save_path)

        # Optionally, print out some information
        print(f"Finished processing for {metric}. Classification results saved to {save_path}")


    # Run the function for the comparison dataset
    classify_for_each_component(result_df, metric, save_dir_comparison)


# Call function with desired parameters
Classification_with_AVD_Feature(dataset = "Movement", metric = "MAD", w_size = 10, w_incre = 1)


# # Our choice of parameters:
# # Movement dataset
# Classification_with_AVD_Feature(dataset = "Movement", metric = "MAD", w_size = 10, w_incre = 1)
# Classification_with_AVD_Feature(dataset = "Movement", metric = "SD", w_size = 10, w_incre = 1)
# # Sports dataset
# Classification_with_AVD_Feature(dataset = "Sports", metric = "MAD", w_size = 6, w_incre = 1)
# Classification_with_AVD_Feature(dataset = "Sports", metric = "SD", w_size = 6, w_incre = 1)
# # Hydraulic dataset
# Classification_with_AVD_Feature(dataset = "Hydraulic", metric = "MAD", w_size = 10, w_incre = 1)
# Classification_with_AVD_Feature(dataset = "Hydraulic", metric = "SD", w_size = 10, w_incre = 1)

