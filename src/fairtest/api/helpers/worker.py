import os
import signal
import multiprocessing as mp
from rq import Connection, Worker


def start(redis_connection, queues):
    with Connection(redis_connection):
        w = Worker(queues)
        w.work()


class WorkerHandler:

    def __init__(
        self
    ):
        self.pids_to_workers = {}

    def start_workers(self, redis_connection, queues, number_of_workers):
        pids_to_workers = self.pids_to_workers
        pids = []
        for i in range(number_of_workers):
            proc = mp.Process(
                target=start, kwargs={
                    "redis_connection": redis_connection,
                    "queues": queues
                }
            )
            proc.start()
            pid = proc.pid
            pids.append(pid)
            pids_to_workers[proc.pid] = proc
        return pids

    def terminate_workers(self, pids=None):
        pids_to_workers = self.pids_to_workers
        if pids is None:
            pids = list(pids_to_workers.keys())
        for pid in pids:
            try:
                worker = pids_to_workers[pid]
            except KeyError:
                print("No worker with pid {} found!".format(pid))
                continue
            worker.terminate()
            del pids_to_workers[pid]

    def kill_workers(self, pids=None):
        pids_to_workers = self.pids_to_workers
        if pids is None:
            pids = list(pids_to_workers.keys())
        for pid in pids:
            try:
                worker = pids_to_workers[pid]
            except KeyError:
                print("No worker with pid {} found!".format(pid))
                continue
            os.kill(pid, signal.SIGKILL)
            worker.join()
            del pids_to_workers[pid]
