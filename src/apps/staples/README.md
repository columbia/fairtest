Staples Pricing Scheme Simulation
==
## Requirements
```
$ pip3 install -r requirements.txt
```

## Creating database tables
```
$ python3 manage.py migrate
$ python3 manage.py populate_competitors -f data/office_depot_max_coordinates.csv
$ python3 manage.py populate_store -f data/staples_coordinates.csv
$ python3 manage.py populate_zipcodes -f data/zipcodes.csv
$ python3 manage.py populate_users -f data/users.csv (this may take some time)
```

## Running the simulation
```
$ python3 manage.py runserver (launches the server)
$ wget http://127.0.0.1:8000/bugreport/ (queries for the users - may take some time)
```

## Simulation results
```
$ The results are under simulation/staples.csv
$ Format is: #distance,zipcode,city,state,gender,race,income,price
```
