import os
from helpers import db
import multiprocessing
from flask import abort
from bson.objectid import ObjectId
from tempfile import mkdtemp, mkstemp

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource




def validate(resource, request):
    '''
    Validate experiment on POST

    Check that experiment is submitted to an existing application pool
    (collection) which is also not empty and mark experiment pending
    before it's being inserted in the DB
    '''
    if resource != 'experiments':
        return
    try:
        pool_name = 'pools/' + request.get_json()['pool_name']
    except Exception:
        # TODO: Fix this
        abort(500, description='bad dict -- This is not 500')

    print "Validating experiment"
    try:
        client = db.connect_to_client('localhost', 27017)
        _db = db.get_db(client, 'fairtest_pools')
        collection = _db.get_collection(pool_name)
    except Exception, error:
        print error
        # TODO: Fix this
        abort(500, description='Internal server error.')
    if not collection.count():
        abort(500, description='Application pool empty. No entries registered')

    # Mark experimenent pending and create temp dir for reports to be put
    try:
        request.get_json()['experiment_status'] = 'pending'
        report_dir = mkdtemp(prefix='fairtest_report_')
        request.get_json()['report_directory'] = report_dir
        os.chmod(report_dir, 0777)
    except Exception, error:
        print error
        abort(500)

    # print request.get_json()
    print "pre_POST hook: OK"


def run(resource, items):
    '''
    When this event is fired up, the experiment is already
    validated and  introduced into the DB.
    '''
    if resource != 'experiments':
        return

    report_dir = items[0]['report_directory']
    pool_name = 'pools/' + items[0]['pool_name']

    pool = multiprocessing.Pool(max(1,int(multiprocessing.cpu_count()/2)))
    # pool.apply_async(_run_async, [experiment_id, report_dir])
    pool.apply_async(_run, [report_dir, pool_name])


def _run(report_dir, pool_name):
    # DO:
    #   - prepare csv from records of DB application pool
    #   - run experiment and place report at proper place
    print "Horay -- Will run:", report_dir, pool_name

    # prepare csv
    filename =  _prepare_csv_from_pool(pool_name)
    print filename

    # run experiment and place report at proper place
    output_dir = report_dir
    data = prepare.data_from_csv(filename, to_drop=['zipcode', 'distance'])
    data_source = DataSource(data)

    expl = []
    sens = ['income']
    target = 'price'

    inv = Testing(data_source, sens, target, expl, random_state=0)
    train([inv])
    print "here"
    test([inv])
    report([inv], pool_name, output_dir)
    print "report in:", output_dir

    # remove csv
    # os.remove(filename)

    # change experiment's status


def _prepare_csv_from_pool(pool_name):
    '''
    Dump entries into csv fairtest-friendly format
    '''
    try:
        client = db.connect_to_client('localhost', 27017)
        _db = db.get_db(client, 'fairtest_pools')
        collection = _db.get_collection(pool_name)
    except Exception, error:
        print error
        # TODO: Fix this
        abort(500, description='Internal server error.')

    try:
        filename = mkstemp(prefix='experiment_csv_')[1]
        with open(filename, "w+") as f:
            for c in collection.find():
                # print c['record']
                print >> f, str(c['record'])
        return filename
    except Exception, error:
        print error
        raise
