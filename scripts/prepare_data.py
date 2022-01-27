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

from src.utils import random_day_offset, outlier_removal

# Load raw data
df = pd.read_csv("data/raw/kc_house_data.csv")

# Drop duplicates
df = df.drop_duplicates(subset=["id"], keep="first")

# Create an artificial "listed" date to help mimic production scenario
np.random.seed(42)
df["date_sold"] = pd.to_datetime(df.date, infer_datetime_format=True)
df["date_listed"] = df.date_sold.apply(lambda x: random_day_offset(x))
df = df.drop(columns=["date"])
df.sort_values(by="date_listed", inplace=True)
df.reset_index(drop=True, inplace=True)

# remove price outliers for simplicity
df = outlier_removal(X=df, multiple=3, cols=["price"])

# Split out first 6 months of data for training, remaining for simulating a "production" scenario
min_sold_date = df.date_sold.min()
max_sold_date = (
    df.date_sold.max().to_period("M").to_timestamp()  # drop the partial last month
)

train_df = df[
    df.date_sold.between(min_sold_date, "2014-10-31", inclusive="both")
].sort_values("date_sold")

prod_df = df[
    df.date_sold.between("2014-10-31", max_sold_date, inclusive=False)
].sort_values("date_sold")

# Save off these dataframes
working_dir = "data/working"
os.makedirs(working_dir, exist_ok=True)
dfs = [("train", train_df), ("prod", prod_df)]
for name, dataframe in dfs:
    path = os.path.join(working_dir, f"{name}_df.pkl")
    dataframe.to_pickle(path)
