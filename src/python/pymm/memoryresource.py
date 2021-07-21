# 
#    Copyright [2021] [IBM Corporation]
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

import pymmcore
import os

from .check import methodcheck

class MemoryReference():
    def __init__(self, internal_handle, memory_resource, memview, name):
        self.handle = internal_handle
        self.mr = memory_resource
        self.buffer = memview
        self.varname = name
        if os.getenv('PYMM_DEBUG') != None:
            self._debug_level = int(os.getenv('PYMM_DEBUG'))
        else:
            self._debug_level = 0

    def __del__(self):
        if self._debug_level > 0:
            print("releasing named memory {} @ {}".format(self.varname, hex(pymmcore.memoryview_addr(self.buffer))))
        self.mr.release_named_memory_by_handle(self.handle)

    def addr(self):
        '''
        For debugging. Get address of memory view
        '''
        return (hex(pymmcore.memoryview_addr(self.buffer)), len(self.buffer))

    def tx_begin(self):
        #print('tx_begin')

        #pymmcore.valgrind_trigger(1)
        # disable SW copy
        #self.__tx_begin_swcopy()
        pass

    def tx_commit(self):
        #print('tx_commit')
        #pymmcore.valgrind_trigger(2)
        # disable SW copy 
        #self.__tx_commit_swcopy()
        pass
    

    def __tx_begin_swcopy(self):
        '''
        Start consistent transaction (very basic copy-off undo-log)
        '''
        (self.tx_handle, mem) = self.mr._MemoryResource_create_named_memory(self.varname + '-tx', len(self.buffer))
        if self.tx_handle is None:
            raise RuntimeError('tx_begin failed')
        # copy data, then persist
        mem[:]= self.buffer;
        self.mr._MemoryResource_persist_memory_view(mem)
        if self._debug_level > 0:
            print('tx_begin: copy @ {}'.format(hex(pymmcore.memoryview_addr(mem)), len(mem)))

    def __tx_commit_swcopy(self):
        '''
        Commit consistent transaction
        '''
        self.mr.release_named_memory_by_handle(self.tx_handle)
        self.mr.erase_named_memory(self.varname + '-tx')
        if self._debug_level > 0:
            print('tx_commit OK!')
        
    def persist(self):
        '''
        Flush any cached memory (normally for persistence)
        '''
        self.mr._MemoryResource_persist_memory_view(self.buffer)


        
class MemoryResource(pymmcore.MemoryResource):
    '''
    MemoryResource represents a heap allocator and physical memory
    resources.  It is backed by an MCAS store component and corresponds
    to a pool.
    '''
    def __init__(self, name, size_mb, pmem_path, backend=None, mm_plugin=None, force_new=False):
        self._named_memory = {}
        super().__init__(pool_name=name, size_mb=size_mb, pmem_path=pmem_path, backend=backend, mm_plugin=mm_plugin, force_new=force_new)
        # todo check for outstanding transactions
        all_items = super()._MemoryResource_get_named_memory_list()
        recoveries = [val for val in all_items if val.endswith('-tx')]
        
        if len(recoveries) > 0:
            raise RuntimeError('detected outstanding undo log condition: recovery not yet implemented')

    @methodcheck(types=[])        
    def list_items(self):
        all_items = super()._MemoryResource_get_named_memory_list()
        # exclude values
        return [val for val in all_items if not val.endswith('-value')]
    
    @methodcheck(types=[str,int,int,bool])
    def create_named_memory(self, name, size, alignment=8, zero=False):
        '''
        Create a contiguous piece of memory and name it
        '''
        (handle, mem) = super()._MemoryResource_create_named_memory(name, size, alignment, zero)
        if handle == None:
            return None

        return MemoryReference(handle,self,mem,name)

    @methodcheck(types=[str])
    def open_named_memory(self, name):
        '''
        Open existing name memory
        '''
        (handle, mem) = super()._MemoryResource_open_named_memory(name)
        if handle == None:
            return None
        return MemoryReference(handle,self,mem,name)

    @methodcheck(types=[MemoryReference])
    def release_named_memory(self, ref : MemoryReference):
        '''
        Release a contiguous piece of memory (i.e. unlock)
        '''
        super()._MemoryResource_release_named_memory(ref.handle)

    @methodcheck(types=[int])
    def release_named_memory_by_handle(self, handle):
        '''
        Release a contiguous piece of memory (i.e. unlock)
        '''
        super()._MemoryResource_release_named_memory(handle)

    @methodcheck(types=[str])
    def erase_named_memory(self, name):
        '''
        Erase a named-memory object from the memory resource
        '''
        super()._MemoryResource_erase_named_memory(name)

    def put_named_memory(self, name, value):
        '''
        Copy-based crash-consistent put of named memory value
        '''
        if not isinstance(value, bytearray):
            raise RuntimeError('put_named_memory requires bytearray data')
            
        super()._MemoryResource_put_named_memory(name, value)

    @methodcheck(types=[str])
    def get_named_memory(self, name):
        '''
        Copy-based get of named memory value
        '''
        return super()._MemoryResource_get_named_memory(name)

    
    def get_percent_used(self):
        '''
        Get percentage of memory used in memory resource
        '''
        return super()._MemoryResource_get_percent_used()


    def atomic_swap_names(self, a: str, b: str):
        '''
        Swap the names of two named memories; must be released
        '''
        return super()._MemoryResource_atomic_swap_names(a,b)
