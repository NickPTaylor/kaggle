#%%

import pathlib
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

#%%

MAX_CAT_LEVELS = 5

#%%

# Setup directories.
input_dir = pathlib.Path.cwd().parent / 'input' / 'home-data-for-ml-course'
output_dir = pathlib.Path().cwd() / 'output'

output_dir.mkdir(exist_ok=True)

train_data_path = input_dir / 'train.csv'
test_data_path = input_dir / 'test.csv'

#%%

# Load data sets
train_df = pd.read_csv(train_data_path, index_col='Id')

X_train = train_df.drop('SalePrice', axis=1)
y_train = train_df['SalePrice']

X_test = pd.read_csv(test_data_path, index_col='Id')

#%%

# Names of category and numerical columns
cat_cols = list(X_train.select_dtypes(include='object').columns)
num_cols = list(X_train.select_dtypes(exclude='object').columns)

print("Total number of features: {:d}".format(X_train.shape[1]))
print("Total numeric features: {:d}".format(len(num_cols)))
print("Total categorical features: {:d}".format(len(cat_cols)))

# Remove categories with more than 5 categories.
cat_counts = ((col, train_df[col].nunique()) for col in cat_cols)
cat_remove = [col for col, levels in cat_counts if levels > MAX_CAT_LEVELS]
msg = "Remove {:d} categorical features - more than max levels ({})"
print(msg.format(len(cat_remove), MAX_CAT_LEVELS))
cat_cols = list(set(cat_cols) - set(cat_remove))

# Exclude categories that feature in training data but not in
# validation data or test data.
cat_remove = [col for col in cat_cols
              if len(set(X_test[col]) - set(X_train[col])) > 0]
msg = "Remove {:d} categorical features - levels in test but not in train"
print(msg.format(len(cat_remove), len(cat_remove)))
cat_cols = list(set(cat_cols) - set(cat_remove))

# NB: 'set' operations are not ordered. MUST align cat_cols with
# original order of data frame for reproducibility!
cat_cols = [col for col in X_train.columns if col in cat_cols]

print("Total features used: {}".format(len(cat_cols) + len(num_cols)))

#%%

# Pipeline.
pl_preprocess_num = Pipeline([
    ('imp', SimpleImputer(strategy='median'))
])

pl_preprocess_cat = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

pre_processing_pl = ColumnTransformer([
    ('preprocess_num', pl_preprocess_num, num_cols),
    ('preprocess_cat', pl_preprocess_cat, cat_cols)
])

pl = Pipeline([
    ('preprocess', pre_processing_pl),
    ('random_forest', RandomForestRegressor(random_state=0))
])

# %%

# Use cross-validation to evaluate model performance.
cvs = cross_val_score(pl, X_train, y_train, scoring='neg_mean_absolute_error')
print("Mean CV score (MAE): {:2.2f}".format(-cvs.mean()))

#%%

# Fit model.
pl.fit(X_train[num_cols + cat_cols], y_train)

# Predict on test set (only use columns that are retained in  training data).
X_test = X_test[num_cols + cat_cols]
y_test_pred = pl.predict(X_test)

# Create submission file.
output = pd.DataFrame(dict(id=X_test.index, SalePrice=y_test_pred))
output.to_csv(output_dir / 'submission.csv', index=False)
