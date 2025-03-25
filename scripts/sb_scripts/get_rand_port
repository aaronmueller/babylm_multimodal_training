#/bin/bash

function random_unused_port {
    local port=$(shuf -i 2000-65000 -n 1)
    netstat -lat | grep $port > /dev/null
    if [[ $? == 1 ]] ; then
        export RANDOM_PORT=$port
    else
        random_unused_port
    fi
}

random_unused_port
