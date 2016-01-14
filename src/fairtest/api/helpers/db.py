from pymongo import MongoClient


def connect_to_client(hostname, port):
    '''
    Connects to a MongoDB server
    '''
    try:
        client = MongoClient(hostname, port)
        return client
    except Exception, error:
        print error
        # TODO: Add custom exception here
        raise


def get_db(client, db_name):
    '''
    Gets a DB from a MongoDB server
    '''
    try:
        db = client[db_name]
        return db
    except Exception, error:
        # TODO: Add custom exception here
        print error
        raise
