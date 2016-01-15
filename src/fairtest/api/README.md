# FairTest API documentation

A REST-ful Application Programming Interface to enable developers use FairTest remotely over the Web. The current version of the API supports the management of application pools with records for which various FairTest investigations can be executed. An application pool, created from the system administrator upon user's request, is bound to one application and holds all the records of this application. Before running any investigation, a user must populate application records into the poll. The results of FairTest experiments (bug reports) are emitted in the corresponding application pool.

## Get Records of Application Pool
Retrieves all records of an application pool.

#### Specification
* _URL_: http://127.0.0.1:5000/_POOL_NAME
* _Method_: GET
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_:  None
* _Return_: Records of Pool (json format)

#### Return Codes
* 200 (OK) -- Resources successfully returned
* 404 (Not Found) -- Cannot find resource URL
    * Check spelling of URL
* 405 (Method not allowed) -- HTTP verb is not allowed
    * Verify that your client is issuing a GET request
* 500 (Internal Server Error) -- Something went wrong in the server
    * Open a github issue

## Post Record into Application Pool
Inserts a record into an existing application pool. Newly inserted records are integrated in
all experiments instantiated after their insertion, but are not integrated into any pending 
experiments.

#### Specification
* _URL_: http://127.0.0.1:5000/_POOL_NAME
* _Method_: POST
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: Record uid

#### Dictionary Fields

* record: The record to register as a json object.

#### Return Codes
* 200 (OK) -- Resources successfully returned
* 400 (Bad Request) -- Server cannot parse request
    * Verify that your client is sending a valid json object        
* 404 (Not Found) -- Cannot find resource URL
    * Check spelling of URL
* 405 (Method not allowed) -- HTTP verb is not allowed
    * Verify that your client is issuing a GET request
* 500 (Internal Server Error) -- Something went wrong in the server
    * Open a github issue.


## Delete Record from Application Pool
Removes a record from an application pool. Deleted records are not integrated to any future experiments
but are being used for any experiments pending prior to their deletion.

#### Specification
* _URL_: http://127.0.0.1:5000/_POOL_NAME/_RECORD_UID
* _Method_: DELETE
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: None
* _Return_: None

#### Return Codes
* 200 (OK) -- Resources successfully returned
* 400 TODO: 
* 404 (Not Found) -- Cannot find resource URL
    * Check spelling of URL
* 405 (Method not allowed) -- HTTP verb is not allowed
    * Verify that your client is issuing a GET request
* 500 (Internal Server Error) -- Something went wrong in the server
    * Open a github issue



## PUT Record into Application Pool
Updates a record into a application pool. Updated records are not integrated to any future experiments, but their old values are being used for any experiments pending prior to their update.

#### Specification
* _URL_: http://127.0.0.1:5000/_POOL_NAME/_RECORD_UID
* _Method_: PUT
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: Record uid

#### Dictionary Fields

* record: The new (updated) record.

#### Return Codes
* 200 (OK) -- Resources successfully returned
* 400 (Bad Request) -- Server cannot parse request
    * Verify that your client is sending a valid json object 
* 404 (Not Found) -- Cannot find resource URL
    * Check spelling of URL
* 405 (Method not allowed) -- HTTP verb is not allowed
    * Verify that your client is issuing a GET request
* 500 (Internal Server Error) -- Something went wrong in the server
    * Open a github issue


## Post FairTest Experiment
Instantiates a FairTest experiment into a Application Pool. The records currently in pool will be used as training and testing set of the experiment.

#### Specification
* _URL_: http://127.0.0.1:5000/experiments
* _Method_: POST
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: Experiment UID

#### Dictionary Fields

* pool_name: The name of an existing application pool, with valid records.
* sens: A list of names of sensitive attributes to check.
* targer: The name of the target (output) attribute.
* to_drop: A list of names of attributes to drop. This fields is optional.

#### Return Codes
* 200 (OK) -- Resources successfully returned
* 400 (Bad Request) -- Server cannot parse request
    * Verify that your client is sending a valid json object 
* 404 (Not Found) -- Cannot find resource URL
    * Check spelling of URL
* 405 (Method not allowed) -- HTTP verb is not allowed
    * Verify that your client is issuing a GET request
* 500 (Internal Server Error) -- Something went wrong in the server
    * Check that your dictionary is contains a valid application pool
    * Open a github issue if none of the above applies



## Get FairTest Report
Retrieves the bug-report report corresponding to an instantiated FairTest experiment.

#### Specification
* _URL_: http://127.0.0.1:5000/experiments/_EXPERIMENT_UID
* _Method_: GET
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: None
* _Return_: Dictionary

#### Dictionary Fields

* experiment_directory: The URL of the FairTest report for the experiment.

#### Return Codes
* 200 (OK) -- Resources successfully returned
* 400 TODO: 
* 404 (Not Found) -- Cannot find resource URL
    * Check spelling of URL
* 405 (Method not allowed) -- HTTP verb is not allowed
    * Verify that your client is issuing a GET request
* 500 (Internal Server Error) -- Something went wrong in the server
    * Check that the parameters of the experiment are right
    * Open a github issue if none of the above applies


## DEMO

```...
```
