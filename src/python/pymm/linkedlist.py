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
import flatbuffers
import struct
import gc
import numpy as np

import PyMM.Meta.Header as Header
import PyMM.Meta.Constants as Constants
import PyMM.Meta.DataType as DataType

from flatbuffers import util
from .memoryresource import MemoryResource
from .shelf import Shadow
from .shelf import ShelvedCommon

from .float_number import float_number


def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


class linked_list(Shadow):
    '''
    Floating point number that is stored in the memory resource. Uses value cache
    '''
    def __init__(self):
        print(colored(255,0,0, 'WARNING: linked_list is experimental and unstable!'))

    def make_instance(self, shelf, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_linked_list(shelf, name)

    def existing_instance(shelf, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''
        memory_resource = shelf.mr
        buffer = memory_resource.get_named_memory(name)
        if buffer is None:
            return (False, None)

        hdr_size = util.GetSizePrefix(buffer, 0)
        if(hdr_size != 28):
            return (False, None)

        root = Header.Header()
        hdr = root.GetRootAsHeader(buffer[4:], 0) # size prefix is 4 bytes

        if(hdr.Magic() != Constants.Constants().Magic):
            return (False, None)

        if (hdr.Type() == DataType.DataType().LinkedList):
            return (True, shelved_linked_list(shelf, name))

        # not a string
        return (False, None)

    def build_from_copy(memory_resource: MemoryResource, name: str, value):
        raise RuntimeError("not implemented!")
#        return shelved_linked_list(memory_resource, name, value=value)

MEMORY_INCREMENT_SIZE=128*1024

class shelved_linked_list(ShelvedCommon):
    '''
    Shelved floating point number
    '''
    def __init__(self, shelf, name):

        self._shelf = shelf # retain reference to shelf
        self._tag = 0
        
        memory_resource = shelf.mr
        memref = memory_resource.open_named_memory(name)

        if memref == None:

            # create metadata
            builder = flatbuffers.Builder(32)
            Header.HeaderStart(builder)
            Header.HeaderAddMagic(builder, Constants.Constants().Magic)
            Header.HeaderAddVersion(builder, Constants.Constants().Version)
            Header.HeaderAddType(builder, DataType.DataType().LinkedList)
            hdr = Header.HeaderEnd(builder)
            builder.FinishSizePrefixed(hdr)
            hdr_ba = builder.Output()

            # allocate named memory for metadata
            hdr_len = len(hdr_ba)
            memref = memory_resource.create_named_memory(name, hdr_len, 1, False)
            memref.tx_begin()
            memref.buffer[0:hdr_len] = hdr_ba
            memref.tx_commit()

            self._metadata_named_memory = memref

            # initialize internal structure with allocated memory
            self._value_named_memory = memory_resource.create_named_memory(name + '-value', MEMORY_INCREMENT_SIZE, 1, False)
            self._internal = pymmcore.List(buffer=self._value_named_memory.buffer, rehydrate=False)

        else:

            hdr_size = util.GetSizePrefix(memref.buffer, 0)
            if hdr_size != 28:
                raise RuntimeError("invalid header for '{}'; prior version?".format(varname))
            
            root = Header.Header()
            hdr = root.GetRootAsHeader(memref.buffer[4:], 0) # size prefix is 4 bytes
            
            if(hdr.Magic() != Constants.Constants().Magic):
                raise RuntimeError("bad magic number - corrupt data?")

            self._metadata_named_memory = memref
            # rehydrate internal structure
            self._value_named_memory = memory_resource.open_named_memory(name + '-value')
            self._internal = pymmcore.List(buffer=self._value_named_memory.buffer, rehydrate=True)

        # save name
        self._name = name

        
    def append(self, element):
        '''
        Add element to end of list
        '''
        if issubclass(type(element), ShelvedCommon): # implies it already on the shelf
            return self._internal.append(element=None, name=element._name)
        elif (isinstance(element, float) or isinstance(element, int)): # inline value
            return self._internal.append(element)
        elif (isinstance(element, np.ndarray) or
              isinstance(element, str)):
            # use shelf to store value
            self._tag += 1
            tag = self._tag
            name = '_' + self._name + '_' + str(tag)
            self._shelf.__setattr__(name, element)
            return self._internal.append(element=None, tag=tag)

        raise RuntimeError('unhandled type')

    def __len__(self):
        return self._internal.size()

    def __getitem__(self, item):
        if isinstance(item, int):
            value, item_in_index = self._internal.getitem(item)
            if item_in_index:
                name = '_' + self._name + '_' + str(value)
                try:
                    return self._shelf.__getattr__(name)
                except RuntimeError:
                    raise RuntimeError('list member is missing from main index; did it get deleted?')
            return value
        else:
            raise RuntimeError('slice criteria not supported')
              
        

