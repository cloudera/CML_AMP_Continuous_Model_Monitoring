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
import logging
import numpy as np
import pandas as pd
from typing import Dict
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (
    DataDriftTab,
    NumTargetDriftTab,
    RegressionPerformanceTab,
)
import cml.metrics_v1 as metrics

from src.utils import scale_prices
from src.api import ApiUtility
from src.inference import ThreadedModelRequest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

log_file = "logs/simulation.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(file_handler)


class Simulation:
    """The main simulation routine to mimic a "production" monitoring use case.

    This simulation assumes a Model has already been deployed, and accepts that model
    name as input. The .run_simulation() method operates the main logic of this class.
    Namely, it:

        1. Scores all training data against the deployed model so we can query metrics
            from the Model Metrics database for evaluation
        2. Initializes a simulation clock, which is just a list of date ranges from the
            prod_df to iterate over. These batches mimic the cadence upon which new data
            "arrives" in a production setting.
        3. For simulation clock date_range, we:
            - Query the prod_df for newly *listed* recrods and score them using deployed model
            - Query the prod_df for newly *sold* records and add ground truths to metric store
            - Query the metric store for thoes newly *sold* records and generate new Evidently report
            - Redeploy the hosted Application to surface the new monitoring report

    Attributes:
        api (src.api.ApiUtility): utility class for help with CML APIv2 calls
        latest_deployment_details (dict): config info about deployed model
        tmr (src.inference.ThreadedModelRequest): utility for making concurrent model API calls
        master_id_uuid_mapping (dict): lookup between input data ID's and predictionUuids
        dev_mode (bool): flag for running simulation with 5% of total data
        sample_size (float): fraction of data to run simulation with

    """

    def __init__(self, model_name: str, dev_mode: bool = False):
        self.api = ApiUtility()
        self.latest_deployment_details = self.api.get_latest_deployment_details(
            model_name=model_name
        )
        self.tmr = ThreadedModelRequest(self.latest_deployment_details)
        self.master_id_uuid_mapping = {}
        self.dev_mode = dev_mode
        self.sample_size = 0.05 if self.dev_mode is True else 0.8

    def run_simulation(self, train_df, prod_df):
        """Operates the main logic to simulate a production scenario."""

        self.set_simulation_clock(prod_df, months_in_batch=1)

        # sample data
        train_df, prod_df = [
            self.sample_dataframe(df, self.sample_size) for df in (train_df, prod_df)
        ]

        # ------------------------ Training Data ------------------------
        # make inference on training data so records are query-able, add
        # ground truth prices to the metrics store, and query records for reporting
        logger.info("------- Starting Section: Train Data -------")

        train_inference_metadata = self.make_inference(train_df)
        formatted_metadata = self.format_metadata_for_delayed_metrics(
            train_df, is_train=True
        )
        self.add_delayed_metrics(*formatted_metadata)

        train_metrics_df = self.query_model_metrics(
            **{
                k: train_inference_metadata[k]
                for k in train_inference_metadata
                if k != "id_uuid_mapping"
            }
        )

        logger.info("------- Finished Section: Train Data -------")

        # ----------------------- Production Data -----------------------

        for i, date_range in tqdm(
            enumerate(self.date_ranges), total=len(self.date_ranges) + 1
        ):

            formatted_date_range = " <--> ".join(
                [ts.strftime("%m-%d-%Y") for ts in date_range]
            )

            logger.info(
                f"------- Starting Section {i+1}/{len(self.date_ranges)}: Prod Data ({formatted_date_range})-------"
            )

            # Query prod_df for newly *listed* records from this batch and make inference
            # TO-DO: refactor this first call into self.make_inference()
            new_listings_df = prod_df.loc[
                prod_df.date_listed.between(
                    date_range[0], date_range[1], inclusive="left"
                )
            ]
            inference_metadata = self.make_inference(new_listings_df)

            # Query prod_df for newly *sold* records from this batch and track ground truths
            formatted_metadata = self.format_metadata_for_delayed_metrics(
                prod_df, date_range, is_train=False
            )
            self.add_delayed_metrics(*formatted_metadata)

            # Query metric store and build Evidently report
            # Note: because we cant query by UUID, first query all records, then filter to new_sold by uuid
            metrics_df = self.query_model_metrics()
            new_sold_metrics_df = metrics_df[
                metrics_df.predictionUuid.isin(formatted_metadata[0])
            ]

            self.build_evidently_reports(
                reference_df=train_metrics_df,
                current_df=new_sold_metrics_df,
                current_date_range=date_range,
            )

            # Create/Refresh Monitoring Dashboard application
            app_name = "Price Regressor Monitoring Dashboard"

            if i == 0:
                self.api.deploy_monitoring_application(application_name=app_name)
            else:
                self.api.restart_running_application(application_name=app_name)

            logger.info(
                f"------- Finished Section {i+1}/{len(self.date_ranges)}: Prod Data ({formatted_date_range})-------"
            )

    def make_inference(self, df):
        """
        Uses the instance's ThreadedModelRequest object to make inference on each record in input dataframe
        by calling the deployed model endpoint.

        Additionally, this method updates the instance's master_id_uuid_mapping with new prediction metadata.

        Args:
            df (pd.DataFrame)

        Returns:
            dict: metadata about threaded API call including start/end timestamps and id-uuid mapping

                {'start_timestamp_ms': 1638308471198,
                 'end_timestamp_ms': 1638308472272,
                 'id_uuid_mapping': {1962200037: 'c213d9d2-ada7-404b-9d65-8066305fdf75',
                                      5561000190: 'cf7340ec-35b8-4201-b342-111cd184babd',
                                      7575600100: 'de41cf61-4977-450d-a718-c7585b7624ad',
                                      2025700730: '7a3e8c56-8a0b-4b67-91f7-097bc5b8ba7d',
                                      587550340: '74f5eb4f-e85b-4434-80f4-1c8ffb02821c'}
                }
        """

        records = self.cast_date_as_str_for_json(df).to_dict(orient="records")
        metadata = self.tmr.threaded_call(records)

        self.master_id_uuid_mapping.update(metadata["id_uuid_mapping"])
        logger.info(
            f'Made inference and updated the master_id_uuid_mapping with {len(metadata["id_uuid_mapping"])} records'
        )

        return metadata

    def set_simulation_clock(self, prod_df, months_in_batch=1):
        """
        Determine the number of "batches" of dates to simulate over the duration of the production dataset and
        set date ranges for each batch as class attribute.

        This is the equivalent of setting how often new data becomes available that you want to make inference
        on. In this case, we assume that every 2 months, we want to make inference on newly listed properties and
        collect delayed ground truths to update the monitoring dashboard with.

        Args:
            prod_df (pd.DataFrame)
            months_in_batch (int) - desired number of batches to simulate over

        """

        # find total number of months in prod set
        total_months = int(
            np.ceil(
                (prod_df.date_sold.max() - prod_df.date_sold.min())
                / np.timedelta64(1, "M")
            )
        )

        # construct date ranges to iterate through as simulation of time (include left, exclude right)
        date_ranges = [
            [
                (prod_df.date_sold.min() + DateOffset(months=n)),
                (prod_df.date_sold.min() + DateOffset(months=n + months_in_batch)),
            ]
            for n in range(0, total_months, months_in_batch)
        ]

        # increase first date range to account for records that listed during the train_df timeframe
        # but hadn't yet sold
        date_ranges[0][0] = date_ranges[0][0] - DateOffset(years=1)

        logger.info(f"Simulation clock set with {len(date_ranges)} batches")

        self.date_ranges = date_ranges

    def format_metadata_for_delayed_metrics(self, df, date_range=None, is_train=False):
        """
        In order to add delayed metrics to the metric store via metrics.track_delayed_metrics(), we must pass in
        list of metrics to track along with a list of corresponding uuids that the metrics should join to. This
        function curates the relevant metrics/uuids dependent on if metrics are being tracked for the train dataset
        (one time activity) or a batch from the production dataset.

        If train dataset:
            - use the full train_df to lookup the prediction_uuid's given the record id using the master_id_uuid_mapping

        If batch from production dataset:
            - first pull all the "sold" records within the current batch's date range
            - then lookup the prediction_uuid's given the record id using the master_id_uuid_mapping

        Args:
            df (pd.DataFrame): either the train_df or prod_df
            date_range (tuple): start and end dates; must be populated if is_train=False

        """

        if not is_train:

            assert date_range is not None

            # query records from prod_df that were newly "sold" in this batch
            new_sold_records = df.loc[
                df.date_sold.between(date_range[0], date_range[1], inclusive="left")
            ]
        else:
            # for train dataset, all records are "newly sold"
            new_sold_records = df.copy()

        # lookup uuids from newly sold records
        uuids = new_sold_records.id.apply(
            lambda x: self.master_id_uuid_mapping[x]
        ).tolist()

        # get list of ground truth prices for newly sold properties
        gts = df[df.id.isin(new_sold_records.id)].price.tolist()

        # get list of sold_dates for newly sold properties
        sold_dates = df[df.id.isin(new_sold_records.id)].date_sold.astype(str).tolist()

        return uuids, gts, sold_dates

    def add_delayed_metrics(self, uuids, ground_truths, sold_dates):
        """
        Add delayed metrics to CML Model Metrics database provided a list of prediction UUID's
        and corresponding list of metrics (ground truth values and sold dates).

        Args:
            uuids (list)
            ground_truths (list)
            sold_dates (list)

        """

        if len(uuids) != len(ground_truths) != len(sold_dates):
            raise ValueError(
                "UUIDs, ground_truths, and sold_dates must be of same length and correspond by index."
            )

        for uuid, gt, ds in zip(uuids, ground_truths, sold_dates):
            metrics.track_delayed_metrics(
                metrics={"ground_truth": gt, "date_sold": ds}, prediction_uuid=uuid
            )

        logger.info(f"Sucessfully added ground truth values to {len(uuids)} records")

    def query_model_metrics(self, **kwargs):
        """
        Use the metrics.read_metrics() functionality to query saved model metrics from the PostgresSQL database,
        and return details in formatted dataframe

        Query metrics for the model deployment saved in self.latest_deployment_details. Optionally, can pass
        additional arguments to indicate start/end timestamp.

        """

        ipt = {}
        ipt["model_deployment_crn"] = self.latest_deployment_details[
            "latest_deployment_crn"
        ]

        if kwargs:
            ipt.update(kwargs)

        return self.format_model_metrics_query(metrics.read_metrics(**ipt))

    @staticmethod
    def sample_dataframe(df, fraction):
        """
        Return a sample of the provided dataframe.

        Args:
            df (pd.DataFrame)
            fraction (float): sample size of dataframe desired

        Returns:
            pd.DataFrame

        """
        return df.sample(frac=fraction, random_state=42)

    @staticmethod
    def cast_date_as_str_for_json(df):
        """Given a dataframe, return the same dataframe with non-numeric columns cast as string"""

        for column, dt in zip(df.columns, df.dtypes):
            if dt.type not in [np.int64, np.float64]:
                df.loc[:, column] = df.loc[:, column].astype(str)
        return df

    @staticmethod
    def format_model_metrics_query(metrics: Dict):
        """
        Accepts the response dictionary from `metrics.read_metrics()`, filters out any non-"metrics" columns,
        and formats as Dataframe.

        Args:
            metrics (dict)

        Returns:
            pd.DataFrame
        """
        metrics = pd.json_normalize(metrics["metrics"])

        return metrics[
            [col for col in metrics.columns if col.split(".")[0] == "metrics"]
            + ["predictionUuid"]
        ].rename(columns={col: col.split(".")[-1] for col in metrics.columns})

    @staticmethod
    def build_evidently_reports(reference_df, current_df, current_date_range):
        """
        Constructs a set of Evidently.ai monitoring reports (Data Drift, Numerical
        Target Drift, and Regression Performance) provided a reference and current
        dataframe. Save the HTML reports to disk for use in an Application.

        Args:
            reference_df (pd.Dataframe)
            current_df (pd.Dataframe)
            current_date_range (tuple)

        """

        TARGET = "ground_truth"
        PREDICTION = "predicted_result"
        NUM_FEATURES = ["sqft_living", "sqft_lot", "sqft_above"]
        CAT_FEATURES = [
            "waterfront",
            "zipcode",
            "condition",
            "view",
            "bedrooms",
            "bathrooms",
        ]

        column_mapping = ColumnMapping()
        column_mapping.target = TARGET
        column_mapping.prediction = PREDICTION
        column_mapping.numerical_features = NUM_FEATURES
        column_mapping.categorical_features = CAT_FEATURES
        column_mapping.datetime = None

        report_dir = os.path.join(
            "apps/static/reports/",
            f'{current_date_range[0].strftime("%m-%d-%Y")}_{current_date_range[1].strftime("%m-%d-%Y")}',
        )
        os.makedirs(report_dir, exist_ok=True)

        reports = [
            ("data_drift", DataDriftTab()),
            ("num_target_drift", NumTargetDriftTab()),
            ("reg_performance", RegressionPerformanceTab()),
        ]

        for report_name, tab in reports:

            dashboard = Dashboard(tabs=[tab])

            dashboard.calculate(
                reference_data=scale_prices(reference_df)
                .sample(n=len(current_df), random_state=42)
                .set_index("date_sold", drop=True)
                .sort_index()
                .round(2),
                current_data=scale_prices(current_df)
                .set_index("date_sold", drop=True)
                .sort_index()
                .round(2),
                column_mapping=column_mapping,
            )

            report_path = os.path.join(report_dir, f"{report_name}_report.html")

            dashboard.save(report_path)
            logger.info(f"Generated new Evidently report: {report_path}")
