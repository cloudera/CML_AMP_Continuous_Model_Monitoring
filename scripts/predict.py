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

# Read the fitted model (scripts/train.py) from the file model.pkl
# and define a function that uses the model to make inference

# This version of the predict function is wrapped with the
# model_metrics decorator, enabling it to call .track_metrics()
# to store mathematical metrics associated with each prediction

import cdsw
import pickle
import pandas as pd

from src.utils import col_order

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# The model_metrics decorator equips the predict function to
# call .track_metrics(). It also changes the return type. If the
# raw predict function returns a value "result", the wrapped
# function will return eg
# {
#   "uuid": "612a0f17-33ad-4c41-8944-df15183ac5bd",
#   "prediction": "result"
# }


@cdsw.model_metrics
def predict(data_input):

    # Convert dict representation back to dataframe for inference
    df = pd.DataFrame.from_records([data_input["record"]])

    df = df[col_order].drop("price", axis=1)

    # Log raw input values of features used in inference pipeline
    # active_features = get_active_feature_names(model.named_steps["preprocess"]) # feature not compatible with Python 3.6
    active_features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "sqft_above",
        "waterfront",
        "zipcode",
        "condition",
        "view",
    ]
    cdsw.track_metric(
        "input_features", df[active_features].to_dict(orient="records")[0]
    )

    # Use pipeline to make inference on request
    result = model.predict(df).item()

    # Log the prediction
    cdsw.track_metric("predicted_result", result)

    return result
