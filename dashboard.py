import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# Define the path to the dataset directory
data_dir = r'./dataset'  # Update this path to where your files are located

# Load datasets (sesuaikan path file Anda)
@st.cache_data
def load_data():
    dfs = [
        'PRSA_Data_Aotizhongxin_20130301-20170228.csv',
        'PRSA_Data_Changping_20130301-20170228.csv',
        'PRSA_Data_Dingling_20130301-20170228.csv',
        'PRSA_Data_Dongsi_20130301-20170228.csv',
        'PRSA_Data_Guanyuan_20130301-20170228.csv',
        'PRSA_Data_Gucheng_20130301-20170228.csv',
        'PRSA_Data_Huairou_20130301-20170228.csv',
        'PRSA_Data_Nongzhanguan_20130301-20170228.csv',
        'PRSA_Data_Shunyi_20130301-20170228.csv',
        'PRSA_Data_Tiantan_20130301-20170228.csv',
        'PRSA_Data_Wanliu_20130301-20170228.csv',
        'PRSA_Data_Wanshouxigong_20130301-20170228.csv'
    ]
    
    # Read and concatenate all datasets
    dataframes = []
    for file_name in dfs:
        file_path = os.path.join(data_dir, file_name)  # Combine directory and file name
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path)
            
            # Drop 'No' column if it exists
            if 'No' in df.columns:
                df = df.drop(['No'], axis=1)
                
            # Convert 'year', 'month', 'day', 'hour' to datetime
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
            dataframes.append(df)
        else:
            st.warning(f"File {file_name} tidak ditemukan di path: {file_path}")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

data = load_data()

# Set up the Streamlit app
st.title('Air Quality Data Analytics Dashboard')

# Sidebar for interactive filters
st.sidebar.header('Filter Data')
station_filter = st.sidebar.selectbox('Select Station', data['station'].unique())
year_filter = st.sidebar.slider('Select Year', int(data['year'].min()), int(data['year'].max()), (2013, 2017))

# Filter data based on user input
filtered_data = data[(data['station'] == station_filter) & (data['year'] >= year_filter[0]) & (data['year'] <= year_filter[1])]

# Display filtered data
st.subheader(f'Filtered Data for Station: {station_filter} (Year {year_filter[0]} - {year_filter[1]})')
st.write(filtered_data.head())

# Visualizations
st.subheader('PM2.5 Concentration Over Time')
plt.figure(figsize=(10, 6))
sns.lineplot(data=filtered_data, x='datetime', y='PM2.5', color='blue')
plt.title(f'Trend of PM2.5 Concentration in {station_filter}')
plt.xlabel('Time')
plt.ylabel('PM2.5 (µg/m³)')
st.pyplot(plt)

# Correlation heatmap
st.subheader('Correlation Between Environmental Factors')
# Select only numeric columns for correlation analysis
numeric_data = filtered_data.select_dtypes(include=['number'])
correlation = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Air Quality and Environmental Factors')
st.pyplot(plt)

# Analysis 1: Distribution of PM2.5
st.subheader(f'Distribution of PM2.5 in {station_filter}')
plt.figure(figsize=(10, 6))
sns.histplot(filtered_data['PM2.5'], bins=30, kde=True)
plt.title(f'Distribution of PM2.5 Levels in {station_filter}')
st.pyplot(plt)

# Analysis 2: Average Concentration by Year
st.subheader(f'Average Concentration of Pollutants by Year at {station_filter}')
# Select only numeric columns for mean calculation
numeric_data = filtered_data.select_dtypes(include=['number'])
yearly_data = numeric_data.groupby(filtered_data['year']).mean()
st.line_chart(yearly_data[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']])

# Analysis 3: Wind Speed and PM2.5 Relationship
st.subheader(f'Relationship Between Wind Speed and PM2.5 in {station_filter}')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='WSPM', y='PM2.5')
plt.title('Wind Speed vs PM2.5')
st.pyplot(plt)


# Analysis 6: Rainfall and Air Quality
st.subheader(f'Impact of Rainfall on PM2.5 Levels in {station_filter}')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='RAIN', y='PM2.5', color='green')
plt.title('Rainfall vs PM2.5')
st.pyplot(plt)


# Analysis 7: Relationship Between Wind Speed and Air Quality
st.subheader('Relationship Between Wind Speed and Air Quality')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='WSPM', y='PM2.5', color='blue')
plt.title('Wind Speed vs PM2.5')
plt.xlabel('Wind Speed (WSPM)')
plt.ylabel('PM2.5 (µg/m³)')
st.pyplot(plt)

# Calculate correlation between Wind Speed and various pollutants
correlation_ws = filtered_data[['WSPM', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].corr()
st.subheader('Correlation Between Wind Speed and Pollutants')
# Konversi kolom yang bisa dikonversi ke tipe numerik
for col in filtered_data.columns:
    if filtered_data[col].dtype == 'object':
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

# Analysis 8: Daily Fluctuations in Air Quality
st.subheader('Daily Fluctuations in Air Quality')

# Konversi kolom 'datetime' jika perlu
if 'datetime' in filtered_data.columns:
    try:
        # Konversi 'datetime' menjadi tipe datetime
        filtered_data['datetime'] = pd.to_datetime(filtered_data['datetime'], errors='coerce')
        # Set 'datetime' sebagai index dan resample data harian
        filtered_data.set_index('datetime', inplace=True)
        
        # Hanya ambil kolom numerik untuk analisis
        numeric_data = filtered_data.select_dtypes(include=['float64', 'int64'])
        
        # Resample dan hitung rata-rata
        daily_data = numeric_data.resample('D').mean()

        # Plot tren harian untuk masing-masing polutan
        plt.figure(figsize=(14, 7))
        if 'PM2.5' in daily_data.columns:
            sns.lineplot(data=daily_data, x=daily_data.index, y='PM2.5', label='PM2.5')
        if 'PM10' in daily_data.columns:
            sns.lineplot(data=daily_data, x=daily_data.index, y='PM10', label='PM10')
        plt.title('Daily Trend of Air Quality')
        plt.xlabel('Date')
        plt.ylabel('Concentration')
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"Error during analysis: {e}")
else:
    st.error("'datetime' column is missing or incorrectly formatted.")


# Analysis 9: Long-Term Trends in Air Quality
st.subheader('Long-Term Trends in Air Quality')
annual_data = filtered_data.groupby('year').mean()
st.line_chart(annual_data[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']])
plt.title('Annual Trends in Air Quality')
plt.xlabel('Year')
plt.ylabel('Average Concentration')
st.pyplot(plt)

# Hapus baris dengan nilai kosong di kolom 'wd' atau 'PM2.5'
filtered_data = filtered_data.dropna(subset=['wd', 'PM2.5'])

# Konversi kolom 'wd' menjadi kategori jika perlu
filtered_data['wd'] = filtered_data['wd'].astype('category')

# More visualizations or analysis can be added based on the previous questions
st.write('This dashboard allows you to explore air quality data interactively.')
