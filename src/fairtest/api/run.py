from eve import Eve

my_settings = {
        'DOMAIN': {
            'demo_app': {
                'allow_unknown': True
            }
        },
        'IF_MATCH': False,
        'MONGO_DBNAME': 'fairtest_pools',
        'RESOURCE_METHODS': ['GET', 'POST', 'DELETE'],
        'ITEM_METHODS': ['GET', 'PATCH', 'PUT', 'DELETE']
    }


app = Eve(settings=my_settings)


if __name__ == '__main__':
        app.run()
