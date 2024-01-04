import pandas as pd
import logging

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureElimination
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures, ExpandingWindowFeatures

### Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt= '%d-%b(%m)-%Y %I:%M:%S',
)
logger = logging.getLogger(__name__)

#Lodding Data
def getData(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"in getData(): {e}")

def dataCleaning() -> None:
    global df
    try:
        df['timestamp'] = pd.to_datetime(df.tpep_pickup_datetime)
        df.drop(columns=['tpep_pickup_datetime'], inplace= True)
        df.drop_duplicates(subset= ['timestamp'], inplace=True)
        df = df[~(df.timestamp > pd.Timestamp('2022-12-31 00:00:00'))]
    except Exception as e:
        logger.error(f'in dataCleaning(): {e}')
        
# featureEngineering start here
def add_temporal_features() -> None:
    global df
    try:
        features_to_extract = [
            "month", "quarter", "semester", "year", "week", "day_of_week", "day_of_month",
            "day_of_year", "weekend", "month_start", "month_end", "quarter_start",
            "quarter_end", "year_start", "year_end", "leap_year", "days_in_month", "hour", "minute", "second"
        ]
        temporal = DatetimeFeatures(features_to_extract=features_to_extract).fit_transform(df[['timestamp']])
        for col in temporal.columns:
            df.loc[:, col] = temporal[col].values
    except Exception as e:
        logger.error(f'in add_temporal_features(): {e}')

def add_lag_features() -> None:
    global df
    try:
        lagfeatures = LagFeatures(variables=None, periods=[1, 2, 4, 8, 16, 24], freq=None, sort_index=True,
                                missing_values='raise', drop_original=False)
        lagfeatures.fit(df[['timestamp', 'passenger_demand', 'taxi_demand']])
        features = lagfeatures.transform(df[['timestamp', 'passenger_demand', 'taxi_demand']])
        for col in list(features.columns)[3:]:
            df[col] = features[col].values
    except Exception as e:
        logger.error(f'in The add_lag_features(): {e}')
        
def add_window_features() -> None:
    global df
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
    except Exception as e:
        logger.error(f'in add_window_features(): {e}')
        
def add_exp_window_features() -> None:
    global df
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
    except Exception as e:
        logger.error(f'in add_exp_window_features(): {e}')
        
# Feture Selection Start here
def select_best_features():
    global df
    try:
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
        df = df[list(scs_columns)]
        df['taxi_demand'] = y
    except Exception as e:
        logger.error(f'in select_best_features(): {e}')

# Data Scaling here
def normalizeScaling() -> None:
    global df
    try:
        scaler = StandardScaler()
        scaler.fit(df.drop(columns=['taxi_demand',]))
        df.loc[:, df.columns[:-1]] = scaler.transform(df.drop(columns=['taxi_demand',])) 
    except Exception as e:
        logger.error(f"in normalizeScaling(): {e}")
        
# DimensonalRedaction start from here
def reduceDimensionality() -> None:
    global df
    try:
        features = df.drop(columns=['taxi_demand'])
        target = df['taxi_demand']
        pca = PCA(n_components=19)
        features_reduced = pca.fit_transform(features)
        df = pd.DataFrame(features_reduced, columns=[f'PC{i}' for i in range(1, 20)])
        df['taxi_demand'] = target
    except Exception as e:
        logger.error(f"in reduceDimensionality(): {e}")
        
# Now Time to call the all function and save it 
def preprocessFeatures():
    global df
    try:
        add_temporal_features()
        add_lag_features()
        add_window_features()
        add_exp_window_features()
        df.dropna(inplace=True)
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty after dropping missing values.")
        ### call other steps
        select_best_features()
        normalizeScaling()
        reduceDimensionality()
    except Exception as e:
        logger.error(f'in preprocessFeatures(): {e}')


if __name__ == '__main__':
    processed_data = {}  # Dictionary to store processed data
    
    # Loading data
    for path in [r'../data/2022.csv', r'../data/2023.csv']:
        df = getData(path)
        # Get the cleaned data
        dataCleaning()
        # Get processed data with feature selection
        preprocessFeatures()
        if df is not None:
            file_name = path.split('/')[-1].split('.')[0]  # Extracting file name without extension
            processed_data[file_name] = df.copy()  # Store processed data in the dictionary
            processed_data[file_name].dropna(axis=0, inplace=True)
            output_file_path = f'../data/feature-{file_name}.parquet'
            processed_data[file_name].to_parquet(output_file_path, index=False)
            print(f"Data from {file_name} has been saved successfully!")
        else:
            print(f"No valid processed data in {file_name} to save.")
