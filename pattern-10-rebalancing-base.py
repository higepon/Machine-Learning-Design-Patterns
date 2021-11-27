# Based on https://github.com/GoogleCloudPlatform/ml-design-patterns/blob/master/03_problem_representation/rebalancing.ipynb.
#   !gsutil cp gs://ml-design-patterns/fraud_data_kaggle.csv .

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

SEED = 42
NROWS = 100000 # None

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'], inplace=True)
    df = pd.get_dummies(df)
    return df

def fraud_rate_df(df: pd.DataFrame) -> float:
    fraud_df = df[df['isFraud'] == 1]
    return len(fraud_df)/len(df)

def fraud_rate(y) -> float:
    fraud_indices = np.where(y == 1)[0]
    return len(fraud_indices) / len(y) # type: ignore

def train_split(df: pd.DataFrame):
    y = df.isFraud.values
    X_train, X_test, _, y_test = train_test_split(df, y, stratify=y, test_size=0.2, random_state=SEED)
    train_df = pd.DataFrame(data=X_train, columns=df.columns)
    X_test.drop(columns=['isFraud'], inplace=True) # type: ignore
    return train_df, X_test, y_test


def down_sample(df: pd.DataFrame):
    fraud = df[df['isFraud'] == 1]
    not_fraud = df[df['isFraud'] == 0]

    not_fraud_sample = not_fraud.sample(random_state=SEED, frac=.005)    
    downsampled_df = pd.concat([not_fraud_sample, fraud])
    return shuffle(downsampled_df, random_state=SEED)    

def train(df: pd.DataFrame):
    train_df, X_test, y_test = train_split(df)
    X_train = train_df.drop(columns=['isFraud']).values
    y_train = train_df.isFraud.values

    print(f'\nFraud Rate in (all, train, test)=({fraud_rate_df(df):.5f}, {fraud_rate(y_train):.5f}, {fraud_rate(y_test):.5f})')

    model = xgb.XGBRegressor(objective='reg:squarederror', seed=SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_pred_formatted = []

    for i in y_pred.tolist():
        y_pred_formatted.append(int(round(i)))

    cm = confusion_matrix(y_test, y_pred_formatted)
    print("\n==== Confusion matrix ====")
    print(cm)


df = pd.read_csv('/kaggle/input/fraud_data_kaggle.csv', nrows=NROWS)
df = process_data(df)

train(df)
