# FairTest API documentation

A REST-ful Application Programming Interface to enable developers use FairTest remotely over the Web. The current version of the API supports the creation and management of a application pool with records for which various FairTest investigations can be ran.

## Create Application Pool
An application pool is bound to one application and holds all the records of the application. Before running any investigation, a user must instantiate a poll and
populate application records in it. The results of FairTest experiments (bug reports) are emitted in the corresponding application pool.

#### Specification
* _URL_: XXX
* _Method_: POST
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: None

#### Dictionary Fields

* pool_name: The name of the pool (must be unique)

#### Return Codes
* XXX


## Delete Application Pool
Removes an application pool by deleting all application entries from the DB and any pending experiments permanently.

#### Specification
* _URL_: XXX
* _Method_: DELETE
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: None

#### Dictionary Fields

* pool_name: The name of the pool

#### Return Codes
* XXX



## Get Records of Application Pool
Retrieves all records of an application pool.

#### Specification
* _URL_: XXX
* _Method_: GET
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_:  Dictionary (json format)
* _Return_: Records of Pool (json format)

#### Dictionary Fields

* pool_name: The name of the pool

#### Return Codes
* XXX


## Post Record into Application Pool
Inserts a record into an existing application pool. Newly inserted records are integrated in
all experiments instantiated after their insertion, but are not integrated into any pending 
experiments.

#### Specification
* _URL_: XXX
* _Method_: POST
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: Record id

#### Dictionary Fields

* pool_name: The name of the pool
* record: The record to register

#### Return Codes
* XXX




## Delete Record from Application Pool
Removes a record from an application pool. Deleted records are not integrated to any future experiments
but are being used for any experiments pending prior to their deletion.

#### Specification
* _URL_: XXX
* _Method_: DELETE
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: None

#### Dictionary Fields

* pool_name: The name of the pool
* record_id: The uid of the target record

#### Return Codes
* XXX



## PUT Record into Application Pool
Updates a record into a application pool. Updated record are not integrated to any future experiments, but their old values are being used for any experiments pending prior to their update.

#### Specification
* _URL_: XXX
* _Method_: PUT
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: None

#### Dictionary Fields

* pool_name: The name of the pool
* record_id: The uid of the target record
* record: The new (updated) record


#### Return Codes
* XXX
     
## Post FairTest Experiment
Instantiates a FairTest experiment into a Application Pool. The records currently in pool will be used as training and testing set of the experiment.

#### Specification
* _URL_: XXX
* _Method_: POST
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: The URL of the bug_report

#### Dictionary Fields

* XXX

#### Return Codes
* XXX


## Get FairTest Report
Retrieves the bug-report report corresponding to an instantiated FairTest experiment.

#### Specification
* _URL_: XXX
* _Method_: GET
* _Protocol_: HTTP, _Port_: 80
* _Input Parameters_: Dictionary (json format)
* _Return_: FairTest Report

#### Dictionary Fields

* XXX

#### Return Codes
* XXX

# Examples
`curl -k  -X GET https://aiermis.cern.ch/ldap/api/lbalias/list -u vatlidak`

`curl -k -H "Content-Type: application/json" -X POST -d '{"pool_name":"demo_pool"}' -L https://:@aiermis.cern.ch/krb/api/lbalias/add `

`curl -k -X DELETE   https://aiermis.cern.ch/ldap/api/pools/delete/demo_pool -u vatlidak`

`curl -k  -X GET https://aiermis.cern.ch/ldap/api/lbalias/list -u vatlidak`

curl -k  -H "Content-Type: application/json" -X POST -d '{"alias_name":"higgs-alias-3", "type":"external"}' -L https://aiermis.cern.ch/ldap/api/lbalias/add  -u vatlidak`

`curl -k -X DELETE   https://aiermis.cern.ch/ldap/api/lbalias/delete/higgs-alias-1 -u vatlidak`

`curl -k  -H "Content-Type: application/json" -X PUT -d '{"type":"external"}' -L https://aiermis.cern.ch/ldap/api/lbalias/update/higgs-alias-1 -u vatlidak`
  

`curl -k -H "Content-Type: application/json" -X POST -d '{"alias_name":"higgs-alias-2", "type":"external"}' -L https://:@aiermis.cern.ch/krb/api/lbalias/add `
