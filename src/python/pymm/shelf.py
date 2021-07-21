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
import pymm
import gc
import sys
import copy
import numpy
import torch
import weakref

import numpy as np
import torch

import PyMM.Meta.Header as Header
import PyMM.Meta.Constants as Constants
import PyMM.Meta.DataType as DataType

from .memoryresource import MemoryResource
from .check import methodcheck
from flatbuffers import util


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

def print_error(*args):
    print(colored(255,0,0,*args))

# common functions for shelved types
#

'''
Common superclass for shelved objects
'''
class ShelvedCommon:
    def __getattr__(self, name):
        if name == 'memory':
            return self._value_named_memory.addr()
        if name == 'namedmemory':
            return self._value_named_memory

#        else:
#            raise AttributeError()
            

class Shadow:
    '''
    Indicate type is a shadow type
    '''
    pass


class shelf():
    '''
    A shelf is a logical collection of variables held in CXL or persistent memory
    '''
    def __init__(self, name, pmem_path='/mnt/pmem0', size_mb=32, backend=None, mm_plugin=None, force_new=False):
        self.name = name
        self.mr = MemoryResource(name, size_mb, pmem_path=pmem_path, backend=backend, mm_plugin=mm_plugin, force_new=force_new)
        if self.mr == None:
            raise RuntimeError('shelf initialization failed')
        
        # todo iterate data and check value-metadata pairs
        items = self.mr.list_items()
        for varname in items:
            if not varname in self.__dict__:

                # read metadata header (always connect to name=key)
                buffer = self.mr.get_named_memory(varname)
                if buffer is None:
                    print("WARNING: no named memory for '{}'".format(varname))
                    continue
                    
                hdr_size = util.GetSizePrefix(buffer, 0)
                if hdr_size != 28:
                    print("WARNING: invalid header for '{}'; prior version?".format(varname))
                    continue

                root = Header.Header()
                hdr = root.GetRootAsHeader(buffer[4:], 0) # size prefix is 4 bytes
                
                if(hdr.Magic() != Constants.Constants().Magic):
                    print("WARNING: bad magic number")
                    continue
                
                stype = hdr.Type()

                # call appropriate existing_instance for detected type
                
                # type: pymm.string
                if (stype == DataType.DataType().AsciiString or
                    stype == DataType.DataType().Utf8String or
                    stype == DataType.DataType().Utf16String or
                    stype == DataType.DataType().Latin1String):
                    (existing, value) = pymm.string.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue
                    
                # type: pymm.ndarray
                elif (stype == DataType.DataType().NumPyArray):
                    (existing, value) = pymm.ndarray.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                # type: pymm.torch_tensor
                elif (stype == DataType.DataType().TorchTensor):
                    (existing, value) = pymm.torch_tensor.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                # type: pymm.float_number
                elif (stype == DataType.DataType().NumberFloat):
                    (existing, value) = pymm.float_number.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                # type: pymm.integer_number
                elif (stype == DataType.DataType().NumberInteger):
                    (existing, value) = pymm.integer_number.existing_instance(self.mr, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue
                    
                # type: pymm.linked_list (needs shelf)
                elif (stype == DataType.DataType().LinkedList):
                    (existing, value) = pymm.linked_list.existing_instance(self, varname)
                    if existing == True:
                        self.__dict__[varname] = value
                        print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                        continue

                    
                    
                # (existing, value) = pymm.pickled.existing_instance(self.mr, varname)
                # if existing == True:
                #     self.__dict__[varname] = value
                #     print("Value '{}' has been made available on shelf '{}'!".format(varname, name))
                #     continue

                print("Value '{}' is unknown type!".format(varname))


    def __del__(self):
        gc.collect()
        for i in self.__dict__:
            if i == 'mr':
                continue
            this_id = id(self.__dict__[i])
            for j in gc.get_objects():                
                if id(j) == this_id:
                    if sys.getrefcount(j) > 4:
                        #raise RuntimeError('trying to delete shelf with outstanding references to contents')
                        print_error('WARNING: deleting shelf with outstanding reference {}-->>{}'.format(i,sys.getrefcount(j)))
        

    def __setattr__(self, name, value):
        '''
        Handle attribute assignment
        '''
        # constant members
        if self.mr and name is 'mr':
            raise RuntimeError('invalid assignment')

        # selective pass through
        if name in ['items','name','mr']:
            self.__dict__[name] = value
            return
            
        if name in self.__dict__:
            if issubclass(type(value), pymm.ShelvedCommon):
                # this happens when an in-place __iadd__ or like operation occurs.  I'm not
                # quite sure why it happens?
                return
            elif name == 'name' or name == 'mr':
                raise RuntimeError('cannot change shelf attribute')

        # currently we allow reassignment of shelf variables
        # TODO: we might want to make a back up of it then later delete
        if name in self.__dict__:
            gc.collect()
            self.erase(name)
            
        # check for supported shadow types
        if self._is_supported_shadow_type(value):
            self.__dict__[name] = value.make_instance(self.mr, name)
            print("made instance '{}' on shelf".format(name))
        elif isinstance(value, pymm.linked_list):
            # pass shelf itself as param
            self.__dict__[name] = value.make_instance(self, name)
            print("made instance '{}' on shelf".format(name))
        elif isinstance(value, numpy.ndarray): # perform a copy instantiation (ndarray)
            self.__dict__[name] = pymm.ndarray.build_from_copy(self.mr, name, value)
            print("made ndarray instance from copy '{}' on shelf".format(name))
        elif isinstance(value, torch.Tensor): # perform a copy instantiation (ndarray)
            self.__dict__[name] = pymm.torch_tensor.build_from_copy(self.mr, name, value)
            print("made torch_tensor instance from copy '{}' on shelf".format(name))
        elif isinstance(value, str):
            self.__dict__[name] = pymm.string.build_from_copy(self.mr, name, value)
        elif isinstance(value, float):
            self.__dict__[name] = pymm.float_number.build_from_copy(self.mr, name, value)
        elif isinstance(value, int):
            self.__dict__[name] = pymm.integer_number.build_from_copy(self.mr, name, value)
        elif issubclass(type(value), pymm.ShelvedCommon):
            raise RuntimeError('persistent reference not yet supported - use a volatile one!')            
        elif type(value) == type(None):
            pass
        else:
            raise RuntimeError('non-pymm type ({}, {}) cannot be put on the shelf'.format(name,type(value)))

    def __getattr__(self, name):
        '''
        Handle attribute access
        '''
        if name == 'items':
            return self.get_item_names()
        elif name == 'used':
            return self.mr.get_percent_used()
        if name in ['mr']:
            if not name in self.__dict__:
                return None
            return self.__dict__[name]
        if not name in self.__dict__:
            raise RuntimeError('invalid member {}'.format(name))
        else:
            return self.__dict__[name]
#            return weakref.ref(self.__dict__[name])
            
    @methodcheck(types=[])
    def get_item_names(self):
        '''
        Get names of items on the shelf
        '''
        items = []
        for s in self.__dict__:
            if issubclass(type(self.__dict__[s]), pymm.ShelvedCommon):
                items.append(s) # or to add object itself...self.__dict__[s]
            elif issubclass(type(self.__dict__[s]), torch.Tensor):
                items.append(s)
                
        return items
            
        
    @methodcheck(types=[str])
    def erase(self, name):
        '''
        Erase and remove variable from the shelf
        '''
        # check the thing we are trying to erase is on the shelf
        if not name in self.__dict__:
            raise RuntimeError('attempting to erase something that is not on the shelf')

        # sanity check
        if gc.is_tracked(self.__dict__[name]):
            gc.collect() # force gc        
            count = sys.getrefcount(self.__dict__[name])
            if count != 2: 
                raise RuntimeError('erase failed due to outstanding references ({})'.format(count))
            
            self.__dict__.pop(name)
            gc.collect() # force gc
        
        # then remove the named memory from the store
        self.mr.erase_named_memory(name)


    @methodcheck(types=[])
    def get_percent_used(self):
        return self.mr.get_percent_used()

    def _is_supported_shadow_type(self, value):
        '''
        Helper function to return True if value is of a shadow type
        '''
        return (isinstance(value, pymm.ndarray) or
                isinstance(value, pymm.string) or
                isinstance(value, pymm.torch_tensor) or
                isinstance(value, pymm.float_number) or
                isinstance(value, pymm.integer_number)
        )

    def supported_types(self):
        return ["pymm.ndarray", "pymm.torch_tensor", "pymm.string", "pymm.float_number"]



# NUMPY
#
# Note: all invocations define a transaction boundary
#-----------------------------------------------------
# import numpy as np
# import pyarrow as pa
#
# myShelf = pymm.shelf("/mnt/pmem0/data.pm")
#
# >> create an array in persistent memory
#
# myShelf.x = pymm.ndarray((9,9),dtype=np.uint8)
#
# >> create reference to array in persistent memoty
#
# X = myShelf.x
#
# >> in place modification (zero-copy)
#
# X.fill(8)
# X[1,1] = 99
#
# >> copy to DRAM
#
# c = X.copy()
#
# >> replication in persistent memory
#
# myShelf.z = X   or myshelf.z = myshelf.x
#
# >> remove from persistent memory (requires no outstanding references)
#
# myShelf.erase('z')




# APACHE ARROW
#------------------------------------------------
# import numpy as np
# import pyarrow as pa
#
# myShelf = pymm.shelf("/mnt/pmem0/data.pm")
#
# >> create an array of strings builder
#
# myShelf.x = pymm.arrow.StringBuilder()
#
# X = myShelf.x
#
# X.append("hello")
# X.append("world");
#
# X.finish();
#
# >> access immutable array (StringScalar type)
#
# print(X[0]) --> <pyarrow.StringScalar: 'hello'>
#
# >> create a schema ...
#

