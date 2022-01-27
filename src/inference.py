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

import time
import cdsw
import json
import requests
import concurrent
import threading


class ThreadedModelRequest:
    """A utility for making concurrent model API calls

    Utilize multi-threading to achieve concurrency and speed up I/O bottleneck associated
    with making a large number of synchronous API calls to the model endpoint.

    Attributes:
        n_threads (int)
        deployment_details (dict): config info about deployed model
        model_service_url (str): deployed models API endpoint URL
        thread_local (_thread._local): A class that represents thread-local data

    """

    def __init__(self, deployment_details, n_threads=2):
        self.n_threads = n_threads
        self.deployment_details = deployment_details
        self.model_service_url = cdsw._get_model_call_endpoint()
        self.thread_local = threading.local()

    def get_session(self):
        if not hasattr(self.thread_local, "session"):
            self.thread_local.session = requests.Session()
        return self.thread_local.session

    def call_model(self, record):
        """'
        Use a self created payload object and the requests library to call the
        deployed model.

        Configuring the requests session manually allows for multithreading to
        work.

        """

        headers = {
            "Content-Type": "application/json",
        }
        data = {
            "accessKey": self.deployment_details["model_access_key"],
            "request": {"record": record},
        }

        session = self.get_session()
        response = session.post(
            url=self.model_service_url,
            headers=headers,
            data=json.dumps(data),
        ).json()

        return record["id"], response["response"]["uuid"]

    def call_model_cdsw(self, record):
        """
        Not Implemented - currently performs 42% slower than call_model.
        """

        response = cdsw.call_model(
            model_access_key=self.deployment_details["model_access_key"],
            ipt={"record": record},
        )

        return record["id"], response["response"]["uuid"]

    def threaded_call(self, records):
        """
        Utilize the call_model() method to make API calls to the deployed model
        for a batch of input records using multithreading for efficiency.

        """

        start_timestamp_ms = int(round(time.time() * 1000))

        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.n_threads
        ) as executor:
            completed = executor.map(self.call_model, records)

        results.extend(completed)

        end_timestamp_ms = int(round(time.time() * 1000))

        return {
            "start_timestamp_ms": start_timestamp_ms,
            "end_timestamp_ms": end_timestamp_ms,
            "id_uuid_mapping": dict(results),
        }
