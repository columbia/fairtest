"""
Demo server
"""
import os
import yaml


from flask import Flask
from flask import url_for
from flask import request
from flask import Response
from flask import redirect
from flask import send_file
from flask import render_template
from flask import send_from_directory

from werkzeug import secure_filename

import logging
from rq import Queue
from flask import abort
from redis import Redis
from tempfile import mkdtemp, mkstemp


UPLOAD_FOLDER = '/tmp/fairtest/datasets'
EXPERIMENTS_FOLDER = '/tmp/fairtest/experiments'
ALLOWED_EXTENSIONS = set(['csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def load_config(config):
    config = os.path.join("config", config)
    with open(config, 'r') as config_file:
        return yaml.load(config_file)


CONF = load_config("./config.yaml")
HOSTNAME = CONF['redis_hostname']
PORT = CONF['redis_port']
REDIS_CONN = Redis(host=HOSTNAME, port=PORT)
REDIS_QUEUE = Queue(connection=REDIS_CONN)


def allowed_file(filename):
    """
    Assert file tupes
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def make_tree(path):
    """
    List directory contents
    """
    tree = dict(name=os.path.basename(path), children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=name))
    return tree


app = Flask(__name__, static_folder='/tmp')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXPERIMENTS_FOLDER'] = EXPERIMENTS_FOLDER


from helpers import experiments

@app.route('/fairtest', methods=['GET', 'POST'])
def handler():
    """
    This is the main handler entry point
    """
    # POST request may require some work
    if request.method == 'POST':

        inv = None
        out = None
        sens = None
        upload_file = None
        expl = None
        report = None
        dataset = None

        # retrieve fields with set values
        try:
            upload_file = request.files['file']
        except Exception, error:
          pass
        try:
            dataset = request.form['dataset']
        except Exception, error:
            pass
        try:
            sens = request.form['sens']
        except Exception, error:
            pass
        try:
            expl = request.form['expl']
        except Exception, error:
            pass
        try:
            out = request.form['out']
        except Exception, error:
            pass
        try:
            inv = request.form['inv']
        except Exception, error:
            pass
        try:
            report = request.form['report']
        except Exception, error:
            pass

        # 1. upload  file(dataset registration)
        if upload_file:
            if upload_file and allowed_file(upload_file.filename):
                filename = secure_filename(upload_file.filename)
                upload_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # 2. post a new experiment
        if dataset:
            print dataset
            dataset = os.path.join(app.config['UPLOAD_FOLDER'], dataset)
            experiment_dict = {'dataset': dataset,
                'sens': sens,
                'expl': expl,
                'inv': inv,
                'out': out,
                'experiment_folder': EXPERIMENTS_FOLDER
            }
            print experiment_dict
            REDIS_QUEUE.enqueue(experiments.demo_run, experiment_dict)

#        # 3. request report for finished experiment
#        if report:
#            # construct filename path and check it exists
#            filename = os.path.join(app.config['EXPERIMENTS_FOLDER'], report)
#            if not os.path.isfile(filename):
#                raise Exception("Report file unavailable")
#            # respond with the attachment
#            print filename
#            # return  send_file(filename)
#            with open(filename, "r") as f:
#                content = f.read()
#            f.close()
#            os.remove(filename)
#            return Response(
#                content,
#                mimetype="text/plain",
#                headers={"Content-Disposition":
#                        "attachment;filename=" + filename
#                }
#            )

    return render_template("upload.html",
                           tree1=make_tree(app.config['UPLOAD_FOLDER']),
                           tree2=make_tree(app.config['EXPERIMENTS_FOLDER']),
                           experiments_folder=app.config['EXPERIMENTS_FOLDER'])


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
