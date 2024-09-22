from src.logger import logging
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import os,pickle
from zenml import step

@step

def ohe_step(X_train: pd.DataFrame, X_test: pd.DataFrame,
            y_train: pd.Series, y_test: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    cat_col = X_train.select_dtypes(include=['object']).columns
    num_col = X_train.select_dtypes(include=['float64','int64','datetime64[ns]']).columns
    logging.info(cat_col)

    ohe = OneHotEncoder()

    # For X_train

    X_train[cat_col] = X_train[cat_col].apply(lambda x: x.str.strip())
    ohe.fit(X_train[cat_col])
    X_train_cat_encoded = ohe.transform(X_train[cat_col])

    X_train_encoded_df = pd.DataFrame(X_train_cat_encoded.toarray(), columns=ohe.get_feature_names_out(cat_col))

    X_train_encoded_df = X_train_encoded_df.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)

    X_train.drop(columns=cat_col, axis=1, inplace=True)

    X_train_final = pd.concat([X_train, X_train_encoded_df], axis=1)

    # For X_test

    X_test[cat_col] = X_test[cat_col].apply(lambda x: x.str.strip())
    X_test_cat_encoded = ohe.transform(X_test[cat_col])
    X_test_encoded_df = pd.DataFrame(X_test_cat_encoded.toarray(), columns=ohe.get_feature_names_out(cat_col))

    X_test_encoded_df = X_test_encoded_df.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    X_test.drop(columns=cat_col, axis=1, inplace=True)

    X_test_final = pd.concat([X_test, X_test_encoded_df], axis=1)

    encoder_file_path = os.path.join("artifacts","encoder.pkl")
    with open(encoder_file_path,'wb') as f:
        pickle.dump(ohe,f)

    return X_train_final,X_test_final,y_train, y_test