/*
 * ****************************************************************************
 *
 *  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
 *  (C) Cloudera, Inc. 2021
 *  All rights reserved.
 *
 *  Applicable Open Source License: Apache 2.0
 *
 *  NOTE: Cloudera open source products are modular software products
 *  made up of hundreds of individual components, each of which was
 *  individually copyrighted.  Each Cloudera open source product is a
 *  collective work under U.S. Copyright Law. Your license to use the
 *  collective work is as provided in your written agreement with
 *  Cloudera.  Used apart from the collective work, this file is
 *  licensed for your use pursuant to the open source license
 *  identified above.
 *
 *  This code is provided to you pursuant a written agreement with
 *  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
 *  this code. If you do not have a written agreement with Cloudera nor
 *  with an authorized and properly licensed third party, you do not
 *  have any rights to access nor to use this code.
 *
 *  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
 *  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
 *  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
 *  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
 *  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
 *  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
 *  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
 *  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
 *  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
 *  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
 *  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
 *  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
 *  DATA.
 *
 * ***************************************************************************
 */

// get report dates from Flask server route
// and save to variable
const reportDates = [];

const fetchReportDates = async () => {
    try {
        const response = await axios.get('/get_report_dates')
        console.log("I FINISHED FETCHING REPORT DATES")
        reportDates.push.apply(reportDates, response.data)
    } catch (e) {
        console.log('Error getting dates from server.')
    }
}

// dynamically set form-select options with report dates
let dateSelect = document.querySelector("#dateRangeSelector");

const populateDateSelect = () => {
    for (const [i, date] of reportDates.entries()) {
        const newOption = document.createElement("option");
        newOption.text = date.replaceAll('-', '/').replaceAll('_', ' - ')
        newOption.value = date
        if (i===0){
            newOption.setAttribute("selected", "selected")
        }
        dateSelect.appendChild(newOption)
    }
    console.log('FINISHED POPULATING DATE SELECTOR')
}

// async-await to setup dashboard
// fetching report dates needs time
const setupDashboard = async () => {
    await fetchReportDates()
    populateDateSelect()

    const currentDateSelection = getActiveDate()
    const currentReportSelection = getActiveReport()
    updateReportUrl(currentDateSelection, currentReportSelection)
}

setupDashboard()

// update reportDisplay on new date selection event
dateSelect.addEventListener('change', function (event){
    const currentDateSelection = event.target.value
    const currentReportSelection = getActiveReport()
    updateReportUrl(currentDateSelection, currentReportSelection)
})

// update reportDisplay on new report selection event
document.querySelector('.nav-pills').addEventListener('click', function (event) {
    const currentReportSelection = event.target.value
    const currentDateSelection = getActiveDate()
    updateReportUrl(currentDateSelection, currentReportSelection)
})



// helper function to get current date selection
const getActiveDate  = () => {
    return document.querySelector("#dateRangeSelector").value
}
// helper function to get current report selection
const getActiveReport  = () => {
    return document.querySelectorAll('#reportTabSelector button.active')[0].value
}

// helper function that updates iframe source
// this gets called inside each eventListener
const updateReportUrl = function (date, report) {
    console.log('UPDATED THE REPORT URL SRC.')
    const reportUrl = `static/reports/${date}/${report}`
    document.querySelector('div#reportDisplay iframe').src = reportUrl;
}