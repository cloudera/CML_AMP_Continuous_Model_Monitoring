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
from datetime import datetime
from flask import Flask, render_template, jsonify


STATIC_PATH = "apps/static"
TEMPLATE_PATH = "apps/templates"

app = Flask(__name__, static_folder=STATIC_PATH, template_folder=TEMPLATE_PATH)
# app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/get_report_dates", methods=["GET"])
def get_report_dates():
    report_dates = sorted(
        os.listdir(os.path.join(STATIC_PATH, "reports")),
        key=lambda date: datetime.strptime(date.split("_")[0], "%m-%d-%Y"),
        reverse=True,
    )
    return jsonify(report_dates)


if __name__ == "__main__":
    # app.run(host="127.0.0.1", debug=True)
    app.run(host="127.0.0.1", port=os.environ.get("CDSW_READONLY_PORT"))
