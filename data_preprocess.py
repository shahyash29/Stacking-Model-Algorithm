import pandas as pd

# Load the dataset
file_path = './TrafficDataset2024_0626.csv'
traffic_data = pd.read_csv(file_path)

# Clean column names
traffic_data.columns = traffic_data.columns.str.strip()

# Remove the FRC5 column
traffic_data = traffic_data.drop(columns=['frc'])

# Define target location coordinates
target_latitude = 34.0278022
target_longitude = -118.4741866
tolerance = 0.001

# Filter data for the target location on Highway Two
target_location_data = traffic_data[
    (traffic_data['lat'].between(target_latitude - tolerance, target_latitude + tolerance)) & 
    (traffic_data['lng'].between(target_longitude - tolerance, target_longitude + tolerance)) & 
    (traffic_data['street'].str.contains('Highway Two', case=False, na=False))
]

# Rename the target speed column
target_location_data = target_location_data.rename(columns={'currentSpeed': 'target_speed'})

# Generate newdate and timestamp from the date column in target_location_data
target_location_data['newdate'] = pd.to_datetime(target_location_data['date']).dt.date
target_location_data['timestamp'] = pd.to_datetime(target_location_data['date']).dt.time

# Pivot the original data to have speeds from various streets as features
pivoted_data = traffic_data.pivot_table(index='date', columns=['lat', 'lng', 'street'], values='currentSpeed', aggfunc='first').reset_index()

# Rename columns to include lat and lng
pivoted_data.columns = ['date'] + [f"({lat},{lng})_{street.strip()}" for lat, lng, street in pivoted_data.columns[1:]]

# Generate newdate and timestamp from the date column in pivoted_data
pivoted_data['newdate'] = pd.to_datetime(pivoted_data['date']).dt.date
pivoted_data['timestamp'] = pd.to_datetime(pivoted_data['date']).dt.time

# Merge the pivoted data with the target speed data
final_dataset = pd.merge(pivoted_data, target_location_data[['date', 'target_speed']], on='date', how='left')

# Ensure column names are unique and sorted in ascending order
unique_columns = list(set(final_dataset.columns))
unique_columns.sort()

# Reorder columns for clarity and remove target_location_speed
columns_order = ['date', 'newdate', 'timestamp', 'target_speed'] + \
                [col for col in unique_columns if col not in ['date', 'newdate', 'timestamp', 'target_speed', f"({target_latitude},{target_longitude})_Highway Two"]]

final_dataset = final_dataset[columns_order]

# Save the final dataset with the reordered columns
output_file_path = './Middle_Dataset_final.csv'
final_dataset.to_csv(output_file_path, index=False)

output_file_path
