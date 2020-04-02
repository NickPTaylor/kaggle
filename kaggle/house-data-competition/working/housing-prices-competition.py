#%%

import pathlib
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

#%%

# Setup directories.
input_dir = pathlib.Path.cwd().parent / 'input' / 'home-data-for-ml-course'
output_dir = pathlib.Path().cwd() / 'output'

output_dir.mkdir(exist_ok=True)

train_data_path = input_dir / 'train.csv'
test_data_path = input_dir / 'test.csv'

#%%

# Load data sets
train_df = pd.read_csv(train_data_path)
X_train = train_df.drop('SalePrice', axis=1)
y_train = train_df['SalePrice']

X_test = pd.read_csv(test_data_path)

#%%

# Select only numeric data.
X_train = X_train.select_dtypes(np.number)

#%%

# Create modelling pipeline
imp = SimpleImputer(strategy='median')
rf = RandomForestRegressor(random_state=42)

pipeline = Pipeline([
    ('imputation', imp),
    ('random_forest', rf)
])

#%%

# Fit model.
pipeline.fit(X_train, y_train)

# Use cross-validation to evaluate model performance.
cvs = cross_val_score(pipeline, X_train, y_train,
                      scoring='neg_mean_absolute_error')

print("Mean CV score (MAE): {:2.2f}".format(-cvs.mean()))

#%%

# Predict on test set (only use columns that are retained in  training data).
X_test = X_test[X_train.columns]
y_test_pred = pipeline.predict(X_test)

# Create submission file.
output = pd.DataFrame(dict(id=X_test.Id, SalePrice=y_test_pred))
output.to_csv(output_dir / 'submission.csv', index=False)
