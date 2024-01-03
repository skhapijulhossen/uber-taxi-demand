import pandas as pd
import logging

from feature_engine.datetime import DatetimeFeatures
from feature_engine.timeseries.forecasting import LagFeatures, WindowFeatures, ExpandingWindowFeatures
from sklearn.tree import DecisionTreeRegressor
from feature_engine.selection import SmartCorrelatedSelection, RecursiveFeatureElimination
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

### Logging Configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%d-%b(%m)-%Y %I:%M:%S',
)

logger = logging.getLogger(__name__)

# Loading Data
def getData(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"Error in getData(): {e}")
        return None

# Data Cleaning
def dataCleaning(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['timestamp'] = pd.to_datetime(df.tpep_pickup_datetime)
        df.drop(columns=['tpep_pickup_datetime'], inplace= True)
        df.drop_duplicates(subset= ['timestamp'], inplace=True)
        df = df[~(df.timestamp > pd.Timestamp('2022-12-31 00:00:00'))]
        return df
    except Exception as e:
        logger.error(f"Error in dataCleaning(): {e}")
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
        logger.error(f'Error in add_temporal_features(): {e}')
        return None


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
        logger.error(f'Error in add_lag_features(): {e}')
        return None
        

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
        logger.error(f'Error in add_window_features(): {e}')
        return None
        

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
        logger.error(f'Error in add_exp_window_features(): {e}')
        return None
        

# Feture Selection Start here

def select_best_features(df: pd.DataFrame) -> pd.DataFrame:
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
        return df
    except Exception as e:
        logger.error(f'Error in select_best_features(): {e}')
        return None

# Data Scaling here
def normalizeScaling(df: pd.DataFrame) -> pd.DataFrame:
    try:
        scaler = StandardScaler()
        scaler.fit(df.drop(columns='taxi_demand'))
        df.loc[:, df.columns[:-1]] = scaler.transform(df.drop(columns='taxi_demand'))
        return df
    except Exception as e:
        logger.error(f"Error in normalizeScaling(): {e}")
        return None
        
    
# DimensonalRedaction start from here
def reduceDimensionality(df: pd.DataFrame) -> pd.DataFrame:
    try:
        features = df.drop(columns=['taxi_demand'])
        target = df['taxi_demand']
        pca = PCA(n_components=19)
        features_reduced = pca.fit_transform(features)
        df = pd.DataFrame(features_reduced, columns=[f'PC{i}' for i in range(1, 20)])
        df['taxi_demand'] = target
        return df
    except Exception as e:
        logger.error(f"Error in reduceDimensionality(): {e}")
        return None

# Now Time to call the all function and save it 

def preprocessFeatures(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = add_temporal_features(df)
        df = add_lag_features(df)
        df = add_window_features(df)
        df = add_exp_window_features(df)
         # Drop rows with missing values
        df.dropna(inplace=True)
        if df is None or df.empty:
            raise ValueError("DataFrame is None or empty after dropping missing values.")
        ### call other steps
        df = select_best_features(df)
        df = normalizeScaling(df)
        df = reduceDimensionality(df)
        return df
    except Exception as e:
        logger.error(f'Error in preprocessFeatures(): {e}')
        return None




if __name__ == '__main__':
    df = getData(r'../data/2022.csv')
    
    if df is not None:
        df = dataCleaning(df)
        
        if df is not None:
            df = preprocessFeatures(df)
            
            if df is not None:
                output_file_path = r"C:/Users/SRA/Desktop/backup/C/MLgrit/time_series_project/uber-taxi-demand/data/featurePipelineFinalData.csv"
                try:
                    df.to_csv(output_file_path, index=False)
                    logger.info("Processed data saved successfully.")
                except Exception as e:
                    logger.error(f"Error in saving processed data: {e}")
                    logger.warning("Failed to save processed data.")
            else:
                logger.warning("No valid processed data after feature preprocessing.")
        else:
            logger.error("Data cleaning failed.")
    else:
        logger.error("Data loading failed.")