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

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import DateOffset

col_order = [
    "id",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "date_sold",
    "date_listed",
]


def random_day_offset(ts: pd._libs.tslibs.timestamps.Timestamp, max_days=60):
    """
    Given a pandas Timestamp, return a new timestep offset to an earlier date by
    a random number of days between 0 and max_days.
    """
    return ts - DateOffset(np.random.randint(0, max_days))


def get_active_feature_names(
    column_transformer,
):
    """Inspect the transformer steps in a given sklearn.ColumnTransformer to collect and
    return the names of all features that are not dropped as part of the pipeline."""

    active_steps = [
        k for k, v in column_transformer.named_transformers_.items() if v != "drop"
    ]

    return np.concatenate(
        [
            column_transformer.named_transformers_[step].feature_names_in_
            for step in active_steps
        ]
    ).tolist()


def outlier_removal(X, multiple, cols):
    """
    Replaces outliers in each column of a pd.DataFrame with np.Nan values.

    Outlier strictness is controlled by multiples of the IQR from each quantile.
    """

    X = pd.DataFrame(X).copy()

    for col in cols:

        x = pd.Series(X.loc[:, col]).copy()
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (multiple * iqr)
        upper = q3 + (multiple * iqr)

        X.loc[~X.loc[:, col].between(lower, upper, inclusive=True), col] = np.nan

    return X[~X.isna().any(axis=1)]


def scale_prices(df):
    """
    Scale prices from being denominated in dollars to hundreds of thousands of dollars.
    """
    copy = df.copy(deep=True)
    for col in ("ground_truth", "predicted_result"):
        copy[col] = copy[col] / 100_000

    return copy


def find_latest_report(report_dir):
    """
    Use date prefixed report titles located in a provided report_dir to identify the
    latest report and return the filename.

    Filename date prefixes should be in the format: "%Y-%m-%d"

    """
    reports = os.listdir(report_dir)
    date_map = {
        report.split("_")[0]: i
        for i, report in enumerate(reports)
        if report.split(".")[-1] == "html"
    }
    latest_report = max(date_map.keys(), key=lambda d: datetime.strptime(d, "%Y-%m-%d"))

    return reports[date_map[latest_report]]
