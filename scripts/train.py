# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import scipy
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV

train_path = "data/working/train_df.pkl"
train_df = pd.read_pickle(train_path)

X_train = train_df.drop("price", axis=1)
y_train = train_df.price

# define the intended features and type
num_cols = ["sqft_living", "sqft_lot", "sqft_above"]
cat_cols = ["bedrooms", "bathrooms", "waterfront", "zipcode", "condition", "view"]

# define our numerical and categorical pipelines
num_pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="mean")),
        ("standardize", StandardScaler()),
        ("scale", MinMaxScaler()),
    ]
)
cat_pipe = Pipeline(
    steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("one-hot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

# combine preprocessing pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", num_pipe, num_cols),
        ("categorical", cat_pipe, cat_cols),
    ]
)

# define estimator - TransformedTargetRegressor to normalize the target variable
estimator = TransformedTargetRegressor(
    regressor=Ridge(), func=np.log10, inverse_func=scipy.special.exp10
)

# construct full pipeline
full_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

# perform gridsearch to find best param settting
gscv = GridSearchCV(
    full_pipe,
    param_grid={"model__regressor__alpha": np.arange(0.1, 1, 0.1)},
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=1,
)

gscv.fit(X_train, y_train)
print(f"Best MAE: {gscv.best_score_}")

# save model
with open("model.pkl", "wb") as f:
    pickle.dump(gscv.best_estimator_, f)
