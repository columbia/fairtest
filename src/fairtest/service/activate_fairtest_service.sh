#!/bin/bash
#
# Daemonize fairtest service
#

TMPDIR="/tmp/fairtest"
CWD=`pwd`

DAEMON_LOGFILE="${TMPDIR}/fairtest_daemon.log"
SERVER_LOGFILE="${TMPDIR}/fairtest_server.log"
WORKERS_LOGFILE="${TMPDIR}/fairtest_workers.log"

REDDISBIN="reddis-server"
WORKERSBIN="python launch_workers.py"
SERVERBIN="python launch_server.py"

function mkfs()
{
    [ -d ${TMPDIR} ] || mkdir ${TMPDIR}
    [ -d ${TMPDIR}/datasets ] || mkdir ${TMPDIR}/datasets
    [ -d ${TMPDIR}/experiments ] || mkdir ${TMPDIR}/experiments

    [ -f ${DAEMON_LOGFILE} ] || touch ${DAEMON_LOGFILE}
    [ -f ${SERVER_LOGFILE} ] || touch ${SERVER_LOGFILE}
    [ -f ${WORKERS_LOGFILE} ] || touch ${WORKERS_LOGFILE}

    [ -d ${TMPDIR}/static ] || ln -s ${CWD}/static ${TMPDIR}/static
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


# always check directories
mkfs

# check and relaunch
check_and_relaunch 2>&1 >> ${DAEMON_LOGFILE}
