import numpy as np
import pandas as pd
from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures, ExpandingWindowFeatures
from sklearn.tree import DecisionTreeRegressor
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureElimination
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def getData() -> pd.DataFrame:
    try:
        df = pd.read_csv(r'../data/2022.csv')
        return df
    except Exception as e:
        print(f"An error occurred: {e}")

def dataCleaning() -> pd.DataFrame:
    try:
        df_clean = pd.read_parquet(r'../data/2022/V2_Clean_Data.parquet')
        return df_clean
    except Exception as e:
        print(f'Error: {e}')
        return None

# featureEngineering start here
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        features_to_extract = [
            "month", "quarter", "semester", "year", "week", "day_of_week", "day_of_month",
            "day_of_year", "weekend", "month_start", "month_end", "quarter_start",
            "quarter_end", "year_start", "year_end", "leap_year", "days_in_month", "hour", "minute", "second"
        ]

        temporal = DatetimeFeatures(features_to_extract=features_to_extract).fit_transform(df[['timestamp']])
        for col in temporal.columns:
            df.loc[:, col] = temporal[col].values
        return df
    except Exception as e:
        print(f'add_temporal_features ERROR: {e}')
        return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        lagfeatures = LagFeatures(variables=None, periods=[1, 2, 4, 8, 16, 24], freq=None, sort_index=True,
                                missing_values='raise', drop_original=False)
        lagfeatures.fit(df[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = lagfeatures.transform(df[['timestamp', 'passenger_demand', 'taxi_demand']])

        for col in list(features.columns)[3:]:
            df[col] = features[col].values
        return df
    except Exception as e:
        print(f'add_lag_features ERROR: {e}')
        return df

def add_window_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        window = WindowFeatures(
            variables=None, window=7, min_periods=1,
            functions=['mean', 'std', 'median'], periods=1, freq=None, sort_index=True,
            missing_values='raise', drop_original=False
        )
        window.fit(df[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = window.fit_transform(df[['timestamp', 'passenger_demand', 'taxi_demand']])
        
        for col in list(features.columns)[3:]:
            df[col] = features[col].values
        return df
    except Exception as e:
        print(f'add_window_features ERROR: {e}')
        return df

def add_exp_window_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        expwindow = ExpandingWindowFeatures(
            variables=None, min_periods=None, functions='std',
            periods=1, freq=None, sort_index=True,
            missing_values='raise', drop_original=False
        )
        expwindow.fit(df[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = expwindow.fit_transform(df[['timestamp', 'passenger_demand', 'taxi_demand']])

        for col in list(features.columns)[3:]:
            df[col] = features[col].values
        return df
    except Exception as e:
        print(f'add_exp_window_features ERROR: {e}')
        return df

# Feture Selection Start here

def select_best_features(output_file_path):
    try:
        df = pd.read_parquet(output_file_path)
        
        X = df.drop(columns=['timestamp','passenger_demand', 'taxi_demand'])
        y = df['taxi_demand']
        
        scs = SmartCorrelatedSelection(
            variables=None, method='pearson', threshold=0.5,
            missing_values='ignore', selection_method='variance',
            confirm_variables=False
        )
        scs_columns = set(scs.fit_transform(X).columns)
        
        rfe = RecursiveFeatureElimination(
            DecisionTreeRegressor(max_depth=3), scoring='r2', cv=3, threshold=0.01,
            variables=None, confirm_variables=False
        )
        
        rfe_columns = rfe.fit_transform(X, y)
        scs_columns.update(rfe_columns)
        
        return scs_columns
    except Exception as e:
        print(f'select_best_features Error: {e}')

# Data Scaling here
def normalizeScaling(df: pd.DataFrame) -> pd.DataFrame:
    try:
        scaler = StandardScaler()
        scaler.fit(df.drop(columns='taxi_demand'))
        df.loc[:, df.columns[:-1]] = scaler.transform(df.drop(columns='taxi_demand'))
        return df
    except Exception as e:
        print(f'normalizeScaling Error: {e}')
        return df
    
# DimensonalRedaction start from here
def PCA(df: pd.DataFrame) -> pd.DataFrame:
    try:
        features = df.drop(columns=['taxi_demand'])
        target = df['taxi_demand']
        pca = PCA(n_components=19)
        features_reduced = pca.fit_transform(features)
        reduced_df = pd.DataFrame(features_reduced, columns=[f'PC{i}' for i in range(1, 20)])
        reduced_df['taxi_demand'] = target
        return reduced_df
    except Exception as e:
        print(f'PCA ERROR: {e}')
        return df
    

# Now Time to call the all function and save it 

def finalResult(df_clean):
    try:
        df_with_temporal = add_temporal_features(df_clean.copy())
        df_with_lag = add_lag_features(df_with_temporal.copy())
        df_with_window = add_window_features(df_with_lag.copy())
        df_with_EWF = add_exp_window_features(df_with_window.copy())
        df_final = df_with_EWF.dropna()
        
        if df_final.empty:
            print("DataFrame is empty after processing features.")
        
        return df_final
    except Exception as e:
        print(f'finalResult Error: {e}')
        return None 

# Get the cleaned data
df_clean = dataCleaning()

# Get processed data with feature selection
processed_data = finalResult(df_clean)

if processed_data is not None:
    # Save the processed data
    output_file_path = r"C:/Users/SRA/Desktop/backup/C/MLgrit/time_series_project/uber-taxi-demand/data/featurePipelineFinalData"
    processed_data.to_csv(output_file_path, index=False)
else:
    print("No valid processed data to save.")