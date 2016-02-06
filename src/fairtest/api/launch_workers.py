"""
Launch workers backend

Usage: ./launch_workers <workers>
"""
import sys
from rq import Queue
from redis import Redis
from helpers import worker

def main(argv=sys.argv):
    if len(argv) != 2:
        usage(argv)

    workers = int(argv[1])
    redis_conn = Redis()
    workers_handler = worker.WorkerHandler()
    queue = Queue(connection=redis_conn)
    workers_handler.start_workers(redis_conn, queue, workers)

def usage(argv):
    print "Usage:%s <workers>" % argv[0]
    exit(-1)

if __name__ == '__main__':
    sys.exit(main())
