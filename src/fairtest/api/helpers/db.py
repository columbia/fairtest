from flask import abort
from pymongo import MongoClient


def connect_to_client(hostname, port):
    """
    Connects to a MongoDB server
    """
    try:
        client = MongoClient(hostname, port)
        return client
    except Exception, error:
        print error
        # TODO: Add custom exception here
        raise


def get_db(client, db_name):
    """
    Gets a DB from a MongoDB server
    """
    try:
        db = client[db_name]
        return db
    except Exception, error:
        # TODO: Add custom exception here
        print error
        raise


def purge_data(storage, data):
    try:
        host, port, name, collection = storage
        client = connect_to_client(host, port)
        _db = get_db(client, name)
        collection = _db.get_collection(collection)
    except Exception, error:
        print error
        abort(500, description='Internal server error.')

    print "Collection size before purging used testing sets", collection.count()
    # TODO: Fix subtle bug which appears in case of zipcodes and other,
    # because Panda data-frames removes leading zeros from numerical values
    for d in data.itertuples(index=False):
        collection.delete_one({'record':','.join(map(str,list(d)))})
    print "Collection size after purging used testing sets", collection.count()
