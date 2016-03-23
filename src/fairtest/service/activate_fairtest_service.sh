#!/bin/bash
#
# Daemonize fairtest service
#

DIR="/tmp/fairtest"
DAEMON_LOGFILE="${DIR}/fairtest_daemon.log"
SERVER_LOGFILE="${DIR}/fairtest_server.log"
WORKERS_LOGFILE="${DIR}/fairtest_workers.log"

REDDISBIN="reddis-server"
WORKERSBIN="python launch_workers.py"
SERVERBIN="python launch_server.py"

function mkfs()
{
    [ -d ${DIR} ] || mkdir ${DIR}
    [ -d ${DIR}/datasets ] || mkdir ${DIR}/datasets
    [ -d ${DIR}/experiments ] || mkdir ${DIR}/experiments

    [ -f ${DAEMON_LOGFILE} ] || touch ${DAEMON_LOGFILE}
    [ -f ${SERVER_LOGFILE} ] || touch ${SERVER_LOGFILE}
    [ -f ${WORKERS_LOGFILE} ] || touch ${WORKERS_LOGFILE}
}

function launch_reddis()
{
    ${REDDISBIN}
    echo "launch redis"

}

function launch_workers()
{
    ${WORKERSBIN} 2>&1 >> ${WORKERS_LOGFILE}
    echo "launch workers"
}

function launch_server()
{
    ${SERVERBIN} 2>&1 >> ${SERVER_LOGFILE}
    echo "launch server"
}

function check_and_relaunch()
{
    date

    # always check directories
    mkfs

    # relaunch redis
    ps -FALL | grep -vw "grep" | grep -wq "redis-server"
    if [ "$?" -eq 0 ]; then
        echo "reddis server running"
    else
        launch_reddis &
    fi

    # relaunch workers
    ps -FALL | grep -vw "grep" | grep -wq "launch_workers\.py"
    if [ "$?" -eq 0 ]; then
        echo "workers running"
    else
        launch_workers &
    fi

    # relaunch server
    ps -FALL | grep -vw "grep" | grep -wq "launch_server\.py"
    if [ "$?" -eq 0 ]; then
        echo "fairtest server running"
    else
        launch_server &
    fi
}

check_and_relaunch 2>&1 >> ${DAEMON_LOGFILE}
