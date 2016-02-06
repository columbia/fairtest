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
* 200 (OK) -- Resources successfully created
* 201 (Created) -- Resources successfully created
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
* 404 (Not Found) -- Cannot find resource URL
    * Check spelling of URL
* 405 (Method not allowed) -- HTTP verb is not allowed
    * Verify that your client is issuing a GET request
* 500 (Internal Server Error) -- Something went wrong in the server
    * Check that the parameters of the experiment are right
    * Open a github issue if none of the above applies

## Implementation Details

to be filled. Basically, it's a flask framework application implementing
the restful API with mongoDB support. Also, the backend workers are built
using redis...


## Demo

```python
import json
import requests


# GET registered records
url = 'http://127.0.0.1:5000/pools/demo_app'
r = requests.get(url)
print r.status_code

payloads = [
    {'record': 'distance,zipcode,city,state,gender,race,income,price'},
    {'record': 'near,33167,Miami,FL,F,Black or African American,income < 50K,low'},
    {'record': 'near,49202,Jackson,MI,M,Black or African American,income < 50K,low'},
    {'record': 'near,08863,Fords,NJ,F,Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,07011,Clifton,NJ,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,80247,Denver,CO,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,46077,Zionsville,IN,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,01832,Haverhill,MA,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'far,69138,Gothenburg,NE,M,White Not Hispanic or Latino,income >= 50K,high'},
    {'record': 'far,25918,Shady Spring,WV,M,White Not Hispanic or Latino,income >= 50K,high'},
    {'record': 'near,29566,Little River,SC,M,White Not Hispanic or Latino,income < 50K,low'}
]

# POST new records
url = 'http://127.0.0.1:5000/pools/demo_app'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
for payload in payloads:
    r = requests.post(url, data=json.dumps(payload), headers=headers)
    print r.status_code

# POST an experiment
url = 'http://127.0.0.1:5000/experiments'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
payload = {'sens': ['income'], 'target': 'price', 'to_drop':['zipcode', 'distance'], 'pool_name': 'demo_app'}
r = requests.post(url, data=json.dumps(payload), headers=headers)

# check response and GET experiment
if r.ok:
    data = json.loads(r.text)
    experiment_id = data['_id']
    url = 'http://127.0.0.1:5000/experiments/' + experiment_id
    r = requests.get(url)

    data = json.loads(r.text)
    experiment_dir = data['experiment_directory']
    print "Experiment_directory (reports and logs):\n\t\t\t", experiment_dir
else:
    print "Error code:", r.status_code

```
