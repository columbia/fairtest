"""
Sample script to demostrate the use of the API

Usage: python ./demo.py
"""
import json
import requests


# GET registered records
url = 'http://127.0.0.1:5000/pools/demo_app'
r = requests.get(url)
print r.status_code


# header
payloads = [
    {'record': 'distance,zipcode,city,state,gender,race,income,price'},
]

# records
payloads += [
    {'record': 'near,33167,Miami,FL,F,Black or African American,income < 50K,low'},
    {'record': 'near,49202,Jackson,MI,M,Black or African American,income < 50K,low'},
    {'record': 'near,18863,Fords,NJ,F,Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,17011,Clifton,NJ,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,85614,Green Valley,AZ,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,63051,House Springs,MO,M,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,12368,Randolph,MA,M,Black or African American,income >= 50K,low'},
    {'record': 'near,11423,Hollis,NY,F,Black or African American,income < 50K,low'},
    {'record': 'near,94509,Antioch,CA,M,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,80247,Denver,CO,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,46077,Zionsville,IN,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,11832,Haverhill,MA,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'far,69138,Gothenburg,NE,M,White Not Hispanic or Latino,income >= 50K,high'},
    {'record': 'far,25918,Shady Spring,WV,M,White Not Hispanic or Latino,income >= 50K,high'},
    {'record': 'near,80247,Denver,CO,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,42001,Paducah,KY,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,33056,Opa Locka,FL,M,Black or African American,income < 50K,low'},
    {'record': 'far,11028,East Longmeadow,MA,M,Black or African American,income >= 50K,high'},
    {'record': 'near,18447,Olyphant,PA,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,46124,Edinburgh,IN,M,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,67212,Wichita,KS,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,83440,Rexburg,ID,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,77078,Houston,TX,F,Black or African American,income >= 50K,low'},
    {'record': 'near,38237,Martin,TN,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,75050,Grand Prairie,TX,M,Hispanic or Latino,income < 50K,low'},
    {'record': 'near,95405,Santa Rosa,CA,M,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,90731,San Pedro,CA,F,Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,71222,Bernice,LA,M,Black or African American,income < 50K,low'},
    {'record': 'near,21047,Fallston,MD,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,46131,Franklin,IN,M,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,68045,Oakland,NE,M,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,21012,Arnold,MD,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,89121,Las Vegas,NV,M,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,48138,Grosse Ile,MI,F,White Not Hispanic or Latino,income >= 50K,low'},
    {'record': 'near,60134,Geneva,IL,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,53718,Madison,WI,M,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'near,29006,Batesburg,SC,F,White Not Hispanic or Latino,income < 50K,low'},
    {'record': 'far,93555,Ridgecrest,CA,M,White Not Hispanic or Latino,income >= 50K,high'},
    {'record': 'near,29566,Little River,SC,M,White Not Hispanic or Latino,income < 50K,low'}
]*100

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
