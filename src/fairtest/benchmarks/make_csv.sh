#!/bin/bash

REPORT_PREFIX="$1"
OUTPUT_FILE="$2"

if [ -z ${REPORT_PREFIX} ]; then
    echo "Usage: $0 <results_dir> <filename>"
    exit -1
fi

function parse_report_metric() {
  local _report="$1"
  local _metric="$2"
  echo  "#set,init,train,test"
  cat ${_report}_* | grep -w "Instantiation" | grep -w "${_metric}" |\
        tr ':' ',' | cut -f2,4,6,8 -d','
}

function convert_to_percentage() {
  local _report="$1"

  echo  "#set,init,train,test,total"
  cat ${_report}_* | grep -w "Instantiation"  | tr ':' ',' | tr '-' ',' |\
    sed 's, ,,g' |cut -f1,2,3,4,6,8,10 -d',' | sed 's,\,, ,g' |\
    awk '{printf "%s-%s-%s-%s,%.2f,%.2f,%.2f,%d,%d\n",$2, $3, $4, $1, 100 * $5 / ($5+$6+$7),\
                                                      100 * $6 / ($5+$6+$7), 100 * $7 / ($5+$6+$7), ($5+$6+$7), NR-1}'
}

function _convert_to_percentage() {
  local _report="$1"

  cat ${_report}_* | grep -w "Instantiation"  | grep "Discovery" | tr ':' ',' | tr '-' ',' |\
    sed 's, ,,g' |cut -f1,2,3,4,6,8,10 -d','  | sed 's,\,, ,g' |\
    awk '{printf "%s,%.2f,%.2f,%.2f,%.2f,%d,D\n",$2, $5, $6, 100*$5/($5+$6), 100*$6/($5+$6), ($5+$6)}'

  cat ${_report}_* | grep -w "Instantiation"  | grep "Testing" | tr ':' ',' | tr '-' ',' |\
    sed 's, ,,g' |cut -f1,2,3,4,6,8,10 -d','  | sed 's,\,, ,g' |\
    awk '{printf "%s,%.2f,%.2f,%.2f,%.2f,%d,T\n",$2, $5, $6, 100*$5/($5+$6), 100*$6/($5+$6), ($5+$6)}'

  cat ${_report}_* | grep -w "Instantiation"  | grep "Error" | tr ':' ',' | tr '-' ',' |\
    sed 's, ,,g' |cut -f1,2,3,4,6,8,10 -d','  | sed 's,\,, ,g' |\
    awk '{printf "%s,%.2f,%.2f,%.2f,%.2f,%d,EP\n",$2, $5, $6, 100*$5/($5+$6), 100*$6/($5+$6), ($5+$6)}'
}

function __convert_to_percentage() {
  local _report="$1"

  echo "#set,train,test,ptrain,ptest,total,investigation,idx"
  _convert_to_percentage ${_report} | tr ',' ' ' | awk '{printf "%s,%.2f,%.2f,%.2f,%.2f,%ds,%s,%d\n", $1, $2, $3, $4, $5, $6, $7, NR-1}'
}

#
# Remove these from now
#
#parse_report_metric ${REPORT_PREFIX} "NMI" > init_train_test_timing_nmi.csv
#parse_report_metric ${REPORT_PREFIX} "Correlation" > init_train_test_timing_correlation.csv
#parse_report_metric ${REPORT_PREFIX} "Regression" > init_train_test_timing_regression.csv
#convert_to_percentage ${REPORT_PREFIX} > init_train_test_timing_percentages.csv

__convert_to_percentage ${REPORT_PREFIX} > ${OUTPUT_FILE}
