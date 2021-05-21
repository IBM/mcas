#!/bin/bash
if [ "$1" == "part2" ]
then
    echo "Deleting existing namespaces ..."
    sudo ndctl destroy-namespace all --force
    echo "Creating namespaces... this will take a while!"
    REGIONS=`ndctl list -R | egrep -oh 'region[0-9]+'`
    for R in $REGIONS
    do
	      echo "Creating namespace for region ($R)"
	      sudo ndctl create-namespace -t pmem -r $R -a 2M -f
    done
    sudo mkfs.xfs -f -d su=2m,sw=1 /dev/pmem0 -m reflink=0
    sudo mkfs.xfs -f -d su=2m,sw=1 /dev/pmem1 -m reflink=0
    sudo mkdir -p /mnt/pmem0
    sudo mkdir -p /mnt/pmem1
    sudo mount -o dax /dev/pmem0 /mnt/pmem0
    sudo mount -o dax /dev/pmem1 /mnt/pmem1
    sudo chmod a+rwx /mnt/pmem0
    sudo chmod a+rwx /mnt/pmem1
    sudo xfs_io -c "extsize 2m" /mnt/pmem0/
    sudo xfs_io -c "extsize 2m" /mnt/pmem1/
else
    read -p "Are you sure you want to erase AEP namespaces?" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "Creating goals..."
        sudo ndctl destroy-namespace all --force
        sudo ipmctl delete -goal
        sudo ipmctl create -goal PersistentMemoryType=AppDirect
        echo "Goal configuration complete. Reboot and then run this script with first argument \"part2\""
        exit
    fi
fi

