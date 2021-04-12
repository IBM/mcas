#!/usr/bin/python3

from dm import dm

# Default locations for devdax and fsdax stores are here: /dev/dax0.0, /mnt/pmem1

class fsdax(dm):
    """ fsdax specification (within a shard) """
    def __init__(self, region=1, accession=0):
        self.region = region
        self.accession = accession
        dm.__init__(self, {"path": "/mnt/pmem%d/a%d" % (region,accession)})
    def step(self,n):
        return fsdax(self.region, self.accession+n)

class devdax(dm):
    """ devdax specification (within a shard) """
    def __init__(self, region=0, accession=0):
        self.region = region
        self.accession = accession
        dm.__init__(self, {"path": "/dev/dax%d.%d" % (region, accession)})
    def step(self,n):
        return devdax(self.region, self.accession+n)

if __name__ == '__main__':
    print("fsdax:", fsdax().json())
    print("devdax:", devdax().json())
