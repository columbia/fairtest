#San Francisco Crime Reports
----------------------------

###Methodology & Requirements

1. Map a subset of coordicates from "train.csv" into zipcodes to produce
   "coord2zipcodes,csv"
2. Use zipcodes (created at step one) to derive location demographics and
   produce "acs5demographics.csv"
3. Augment incidents of "train.csv" with demographics, provided that the
    coordinate of the incident is mapped with the respective zipcode. Note here
    that due to rate-limiting of the google geocoding API only 50% (or 430K
    incidents) of the original dateset are augmented with demographics.
4. Train a boosted regressor using only the original features (i.e, excluding
    the deomgraphics attained at step 3.) and register the logloss, each prediction
    acounts in the testing set, for as an additional feature.
5. Finally, merge the testing test incidents (which now contains logloss for each
   prediction) with the corresponding demographics and produce "crime_pred_logloss.csv"

Note: In case you want to actually query cencus.gov clone and install the
amazing sunlightlabs python wrapper for the census API, from: git@github.com:sunlightlabs/census.git

###Run
Run the regression and reproduce predictions: '''python logit.py'''


#Code Organisation
------------------

 File                       | Description
--------------------------- | ------------------------------------
coord2zipcode.py            | Maps coordinates to zipcodes, using Google APIs
coord2zipcode.csv           | Contains 25K coordinates mapped to the corresponding zipcode
acs4demographics.py         | Queries census for 5-years ACS5 2014 demographics report
acs4demographics.csv        | Zipcodes matched with the respective demographics
augment_train.py            | Maps coordinates to zipcodes and augements with demographics
augmented_train.py          | The augmented train set
logit.py                    | Multivariate logistic regression
crime_pred_logloss.csv      | The output predictions file, with incidents, demogprahics, and logloss
