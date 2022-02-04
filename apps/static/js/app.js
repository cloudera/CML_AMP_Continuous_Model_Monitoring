
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