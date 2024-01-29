import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('Attendance.csv')

# Convert the 'time' column to a timedelta format
df['time'] = pd.to_timedelta(df['Time'])

# # Convert 'Activeness' column to integers
# df['Activeness'] = df['Activeness'].apply(lambda x: int(x))

result_df = df.groupby('Name').agg({
    'time': lambda x: (x.max() - x.min()).total_seconds() / 60,
    'Activeness': lambda x: (x.sum() / len(x)) * 100  # Calculate activeness percentage
}).reset_index()

result_df.rename(columns={'time': 'Total_Detection_Time_Minutes', 'Activeness': 'Activeness_Percentage'}, inplace=True)

# Save the result to a new CSV file
result_df.to_csv('NewAttendance.csv', index=False)
