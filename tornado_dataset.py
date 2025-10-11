"""
tornado_dataset.py
------------------
Utility to load the Tornado dataset, similar in style to sklearn.datasets.load_iris().

Expected files (must be in the same directory as this file, or provide a data_dir):
    2025-Quantathon-Tornado-Q-training_data-640-examples.xlsx
    2025-Quantathon-Tornado-Q-test_data-200-examples.xlsx
    2025-Quantathon-Tornado-Q-validation_data-160-examples.xlsx

Returns a dictionary with:
    'train', 'test', 'val' : pandas.DataFrames
"""

import os
import pandas as pd
from imblearn.over_sampling import SMOTE 
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

def load_tornado(data_dir: str = ".", balanced: bool = True, multiclass: bool = False, train_file=None, test_file=None, val_file=None):
    """
    Load the Tornado dataset from Excel files.

    Parameters
    ----------
    data_dir : str, optional
        Path to directory containing the three Excel files.
    return_X_y : bool, optional
        If True, returns (X_train, y_train), (X_test, y_test), (X_val, y_val)
        assuming the last column is the target.
    balanced : bool, optional
        If True, balances the training set before returning using ibmlearn SMOTE

    Returns
    -------
    data : dict or tuple
        If return_X_y is False:
            {
                'train': DataFrame,
                'test': DataFrame,
                'val': DataFrame
            }
        If return_X_y is True:
            ((X_train, y_train), (X_test, y_test), (X_val, y_val))
    """
    filenames = {
        "train": "2025-Quantathon-Tornado-Q-training_data-640-examples.xlsx",
        "test": "2025-Quantathon-Tornado-Q-test_data-200-examples.xlsx",
        "val": "2025-Quantathon-Tornado-Q-validation_data-160-examples.xlsx",
    }
    
    if train_file is not None:
        filenames['train'] = train_file
    if test_file is not None:
        filenames['test'] = test_file
    if val_file is not None:
        filenames['val'] = val_file
    

    data = {}
    for key, fname in filenames.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected file not found: {path}")
        df = pd.read_excel(path)
        df = df.map(lambda x: str(x).strip() if isinstance(x, str) else x)
        df = df.apply(pd.to_numeric, errors='coerce')
        data[key] = df.copy()
        
    X_train = data['train'].copy().drop(columns=['ef_binary', 'ef_class'])

    #Fill Missing Values
    imputer = IterativeImputer(add_indicator=True)
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    
    # Normalize data (fit on train only, apply to all)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)

    def split_X_y(df):
        X = df.copy().drop(columns=['ef_binary', 'ef_class'])
        if multiclass is not True:
            return pd.DataFrame(scaler.transform(imputer.transform(X))), df['ef_binary']
        else:
            return pd.DataFrame(scaler.transform(imputer.transform(X))), df['ef_class']

    X_train, y_train = split_X_y(data["train"])
    X_test, y_test = split_X_y(data["test"])
    X_val, y_val = split_X_y(data["val"])
    
    if balanced is True: 
        K = 5 if multiclass else 25
        smote = SMOTE(random_state=42, k_neighbors=K)
        X_resample, y_resample = smote.fit_resample(X_train.to_numpy(), y_train.to_numpy())
        X_train = pd.DataFrame(X_resample)
        y_train = pd.Series(y_resample)

      
    return (X_train, y_train), (X_test, y_test), (X_val, y_val)


if __name__ == "__main__":
    # Example usage
    dataset = load_tornado()
    print("Tornado dataset loaded successfully:")
    print(f'Training set shape: {dataset["train"].shape}')
    print(f'Validation set shape: {dataset["val"].shape}')
    print(f'Testing set shape: {dataset["test"].shape}')