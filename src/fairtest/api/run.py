from eve import Eve
from bson.objectid import ObjectId
from pymongo import MongoClient
from flask import abort


settings = {
        'DOMAIN': {
            'pools/demo_app': {
                'allow_unknown': True
            },
            'experiments': {
                'allow_unknown': True,
                'cache_expires': 1
            }
            ## ADD HERE (don't forget comma seperators)
        },
        'IF_MATCH': False,
        'MONGO_DBNAME': 'fairtest_pools',
        'RESOURCE_METHODS': ['GET', 'POST'],
        'ITEM_METHODS': ['GET', 'PUT', 'DELETE']
    }


def _connect_to_client(hostname, port):
    try:
        client = MongoClient(hostname, port)
        return client
    except Exception, error:
        print error
        raise


def _get_db(client, db_name):
    try:
        db = client[db_name]
        return db
    except Exception, error:
        print error
        raise 



def validate_experiment(resource, request):
    '''
    Validate experiment on POST

    Check that experiment is submitted to an existing application pool
    (collection) which is also not empty
    '''
    if resource != 'experiments':
        return
    try:
        pool_name = 'pools/' + request.get_json()['pool_name']
    except Exception:
        abort(500, description='bad dict -- This is not 500')

    print "Validating experiment"
    client = _connect_to_client('localhost', 27017)
    db = _get_db(client, 'fairtest_pools')

    collection = db.get_collection(pool_name)
    if not collection.count():
        abort(500, description='Application pool empty. No entries registered')
    print resource, request, collection.count()
    print "Validating experiment -- OK"


def run_experiment(resource, items):
    '''
    When this event is fired up, the experiment is already (i) validated and
    (ii) introduced into the DB.

    Do the following:
    1. Launch running.
    2. Decide where to put reports.
    3. Return URI for the report.
    4. Remove experiment from DB when report is submitted.

    '''
    if resource != 'experiments':
        return

    print 'An experiment has just been sumbitted'



app = Eve(settings=settings)
app.on_pre_POST += validate_experiment
app.on_inserted += run_experiment

if __name__ == '__main__':
        app.run()
