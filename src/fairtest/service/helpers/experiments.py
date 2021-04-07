import os
from flask import abort
from tempfile import mkdtemp, mkstemp

import fairtest.utils.prepare_data as prepare
from fairtest import Testing, train, test, report, DataSource


def demo_run(experiment_dict):
    # retrive dictionary parameteres
    output = experiment_dict['out']
    sens = [experiment_dict['sens']]
    dataset = experiment_dict['dataset']
    experiment_folder = experiment_dict['experiments_folder']

    if 'expl' in experiment_dict:
        expl = experiment_dict['expl']
    else:
        expl = []

    # run experiment and place report at proper place
    try:
        try:
            data = prepare.data_from_csv(dataset)
        except Exception, error:
            print "Error:", error
        data_source = DataSource(data)
        inv = Testing(
            data_source, sens, output, expl
        )

        print "Experiment parameters:", experiment_dict
        train([inv])
        test([inv])
        report_name =  os.path.basename(dataset).split('.')[0]
        tmp_folder = mkdtemp(prefix="fairtest_")
        report([inv], report_name, tmp_folder)
        src_path = os.path.join(tmp_folder, "report_" + report_name + ".txt")
        dst_path = os.path.join(
            experiment_folder,
            report_name + "_" + sens[0] + "_" + output + ".txt"
        )
        print src_path, dst_path
        os.rename(src_path, dst_path)
        os.rmdir(tmp_folder)

    except Exception, error:
        print error
        abort(500, description='Internal server error.')

