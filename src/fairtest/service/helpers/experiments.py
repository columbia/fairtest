import os
import logging
from rq import Queue
from flask import abort
from redis import Redis
from tempfile import mkdtemp, mkstemp

from helpers import db, config
import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource


def validate(resource, request):
    """
    Validate experiment on POST -- This function is invoked by prepost hooks.

    Check that experiment is submitted to an existing application pool
    (collection) which is also not empty, and mark experiment pending
    before it's being inserted into the DB.
    """
    if resource != 'experiments':
        return
    try:
        _ = request.get_json()['sens']
        _ = request.get_json()['target']
        collection = 'pools/' + request.get_json()['pool_name']
    except Exception:
        abort(400, description='bad dictionary')

    conf = config.load_config("./config.yaml")
    host = conf['db_hostname']
    port = conf['db_port']
    name = conf['db_name']

    try:
        client = db.connect_to_client(host, port)
        _db = db.get_db(client, name)
        collection = _db.get_collection(collection)
    except Exception, error:
        print error
        abort(500, description='Internal server error.')

    if collection.count() <= 2:
        abort(500, description='Application pool empty. No entries registered')

    # Mark experiment pending and create temp dir to place repots
    try:
        request.get_json()['experiment_status'] = 'pending'
        experiment_dir = mkdtemp(prefix='fairtest_report_')
        request.get_json()['experiment_directory'] = experiment_dir
        os.chmod(experiment_dir, 0777)
    except Exception, error:
        print error
        abort(500)


def run(resource, items):
    """
    This function is invoke by on_inserted hooks.

    When this event is fired up, the experiment is already
    validated and  introduced into the DB.
    """
    if resource != 'experiments':
        return
    conf = config.load_config("./config.yaml")
    host = conf['redis_hostname']
    port = conf['redis_port']
    experiment_dict = items[0]
    collection = 'pools/' + items[0]['pool_name']
    redis_conn = Redis(host=host, port=port)
    q = Queue(connection=redis_conn)
    q.enqueue(_run, experiment_dict, collection)


def _run(experiment_dict, collection):
    """
    DO:
       - prepare csv from records of DB application pool
       - run experiment and place report at proper place
    """
    conf = config.load_config("./config.yaml")
    logfile = conf['logfile']
    host = conf['db_hostname']
    port = conf['db_port']
    name = conf['db_name']

    # prepare csv
    filename = _prepare_csv_from_collection(collection)

    # retrive dictionary parameteres
    experiment_dir = experiment_dict['experiment_directory']
    sens = experiment_dict['sens']
    target = experiment_dict['target']
    if 'to_drop' in experiment_dict:
        to_drop = experiment_dict['to_drop']
    else:
        to_drop = []
    if 'expl' in experiment_dict:
        expl = experiment_dict['expl']
    else:
        expl = []
    if 'random_state' in experiment_dict:
        random_state = experiment_dict['random_state']
    else:
        random_state = 0

    logging.basicConfig(
        filename=os.path.join(
            experiment_dir,
            logfile
        ),
        level=logging.DEBUG
    )
    print "Experiment parameters:", experiment_dict
    print "Input csv: %s\nOutput dir: %s" % (filename, experiment_dir)

    #TODO : support all three types investigations
    #TODO: index records in mongo DB
    #TODO: migrate to levelDB
    # run experiment and place report at proper place
    try:
        data = prepare.data_from_csv(filename)
        data_source = DataSource(data, storage=(host, port, name, collection))
        inv = Testing(
            data_source, sens, target, expl, random_state=random_state,
            to_drop=to_drop
        )
        train([inv])
        test([inv])
        report([inv], "", experiment_dir)
    except Exception, error:
        print error
        abort(500, description='Internal server error.')

    conf = config.load_config("./config.yaml")
    host = conf['db_hostname']
    port = conf['db_port']
    name = conf['db_name']

    # remove csv
    try:
        client = db.connect_to_client(host, port)
        _db = db.get_db(client, name)
        experiments = _db.get_collection('experiments')
    except Exception, error:
        print error
        abort(500, description='Internal server error.')

    experiment = experiments.find_one({'experiment_directory': experiment_dir})
    experiment['experiment_status'] = 'finished'
    experiments.save(experiment)
    # os.remove(filename)
    print "Experiment done. Input csv: %s removed\n" % (filename)


def _prepare_csv_from_collection(collection):
    """
    Dump entries into csv fairtest-friendly format
    """
    conf = config.load_config("./config.yaml")
    host = conf['db_hostname']
    port = conf['db_port']
    name = conf['db_name']
    try:
        client = db.connect_to_client(host, port)
        _db = db.get_db(client, name)
        collection = _db.get_collection(collection)
    except Exception, error:
        print error
        abort(500, description='Internal server error.')

    try:
        filename = mkstemp(prefix='experiment_csv_')[1]
        with open(filename, "w+") as f:
            for c in collection.find():
                print >> f, str(c['record'])
        return filename
    except Exception, error:
        print error
        raise


def demo_run(experiment_dict):
    """
    This function is invoke by on_inserted hooks.

    When this event is fired up, the experiment is already
    validated and  introduced into the DB.
    """
    # retrive dictionary parameteres
    output = experiment_dict['out']
    sens = [experiment_dict['sens']]
    dataset = experiment_dict['dataset']
    experiment_folder = experiment_dict['experiment_folder']

    if 'expl' in experiment_dict:
        expl = experiment_dict['expl']
    else:
        expl = []
    print "Experiment parameters:", experiment_dict

    # run experiment and place report at proper place
    try:
        print dataset
        try:
            data = prepare.data_from_csv(dataset)
        except Exception, error:
            print "Error:", error
        data_source = DataSource(data)
        inv = Testing(
            data_source, sens, output, expl
        )
        train([inv])
        test([inv])
        report([inv], os.path.basename(dataset).split('.')[0], experiment_folder)
    except Exception, error:
        print error
        abort(500, description='Internal server error.')


