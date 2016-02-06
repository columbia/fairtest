from eve import Eve
from rq import Queue
from redis import Redis
from helpers import worker
from helpers import experiments

settings = {
        'DOMAIN': {
            # Allow:
            #   (i) GET, POST of records into a application pool
            #   (ii) GET, PUT, DELETE of individual records
            'pools/demo_app': {
                'allow_unknown': True,
                'resource_methods': ['GET', 'POST'],
                'item_methods': ['GET', 'PUT', 'DELETE']
            },

            # Allow:
            #   (i) POST of an experiment into experiments pool
            #   (ii) GET of an individual experiement (whose id is known to
            #   the user)
            'experiments': {
                # TODO: Limit Rate
                'allow_unknown': True,
                'resource_methods': ['POST'],
                'item_methods': ['GET']
            }
            # ADD HERE (don't forget comma seperators)
        },
        # Global configuration
        'IF_MATCH': False,
        'MONGO_DBNAME': 'fairtest_pools',
        'CACHE_EXPIRES': 1
    }


app = Eve(settings=settings)
app.on_pre_POST += experiments.validate
app.on_inserted += experiments.run


if __name__ == '__main__':
        app.run(debug=True)

