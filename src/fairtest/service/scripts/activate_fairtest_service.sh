#!/bin/bash
#
# Daemonize fairtest service
#


DIR="/tmp/fairtest"
REDDISBIN=
wORKERSBIN=
SERVERBIN=

function mkfs()
{
    [ -d ${DIR} || mkdir ${DIR} ]
}


function launch_reddis()
{
   echo "launch redis"

}

function launch_workers()
{
    chroot ${DIR} ${WORKERSBIN}
    echo "launch workers"
}

function launch_server()
{
    chroot ${DIR} ${SERVERBIN}
    echo "launch server"
}

function do_checks()
{
    # always check directories
    mkfs

    # relaunch redis
    ps -FALL | grep -vw "grep" | grep -wq "redis-server"
    if [ "$?" -eq 0 ]; then
        echo "reddis server running"
    else
        launch_reddis
    fi

    # relaunch workers
    ps -FALL | grep -vw "grep" | grep -wq "launch_workers\.py"
    if [ "$?" -eq 0 ]; then
        echo "workers running"
    else
        launch_workers
    fi

    # relaunch server
    ps -FALL | grep -vw "grep" | grep -wq "launch_server\.py"
    if [ "$?" -eq 0 ]; then
        echo "fairtest server running"
    else
        launch_server
    fi
}

do_checks #> /var/log/fairtest_daemon.log

