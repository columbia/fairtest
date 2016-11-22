"""
Launch workers backend
"""
import sys
from rq import Queue
from redis import Redis
from helpers import worker, config


def main():
    conf = config.load_config("./config.yaml")
    workers = conf['workers']
    redis_conn = Redis()
    workers_handler = worker.WorkerHandler()
    queue = Queue(connection=redis_conn, default_timeout=18000)
    workers_handler.start_workers(redis_conn, queue, workers)


def usage(argv):
    print "Usage:%s <workers>" % argv[0]
    exit(-1)


if __name__ == '__main__':
    sys.exit(main())
