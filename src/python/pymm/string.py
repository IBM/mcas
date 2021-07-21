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
import gc

import PyMM.Meta.Header as Header
import PyMM.Meta.Constants as Constants
import PyMM.Meta.DataType as DataType

from flatbuffers import util
from .memoryresource import MemoryResource
from .shelf import Shadow
from .shelf import ShelvedCommon

class string(Shadow):
    '''
    String object that is stored in the memory resource.
    '''
    def __init__(self, string_value, encoding='utf-8'):
        self.string_value = string_value
        self.encoding = encoding

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_string(memory_resource, name, string_value=self.string_value, encoding=self.encoding)

    def existing_instance(memory_resource: MemoryResource, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''                        
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

        stype = hdr.Type()
        
        if stype == DataType.DataType().AsciiString:
            return (True, shelved_string(memory_resource, name, buffer[hdr_size + 4:], 'ascii'))
        elif stype == DataType.DataType().Utf8String:
            return (True, shelved_string(memory_resource, name, buffer[hdr_size + 4:], 'utf-8'))
        elif stype == DataType.DataType().Utf16String:
            return (True, shelved_string(memory_resource, name, buffer[hdr_size + 4:], 'utf-16'))
        elif stype == DataType.DataType().Latin1String:
            return (True, shelved_string(memory_resource, name, buffer[hdr_size + 4:], 'latin-1'))

        # not a string
        return (False, None)

    def build_from_copy(memory_resource: MemoryResource, name: str, value):
        return shelved_string(memory_resource, name, string_value=value, encoding='utf-8')


class shelved_string(ShelvedCommon):
    '''Shelved string with multiple encoding support'''
    def __init__(self, memory_resource, name, string_value, encoding):

        if not isinstance(name, str):
            raise RuntimeException("invalid name type")

        memref = memory_resource.open_named_memory(name)

        if memref == None:
            # create new value
            builder = flatbuffers.Builder(32)
            # create header
            Header.HeaderStart(builder)
            Header.HeaderAddMagic(builder, Constants.Constants().Magic)
            Header.HeaderAddVersion(builder, Constants.Constants().Version)

            if encoding == 'ascii':
                Header.HeaderAddType(builder, DataType.DataType().AsciiString)
            elif encoding == 'utf-8':
                Header.HeaderAddType(builder, DataType.DataType().Utf8String)
            elif encoding == 'utf-16':
                Header.HeaderAddType(builder, DataType.DataType().Utf16String)
            elif encoding == 'latin-1':
                Header.HeaderAddType(builder, DataType.DataType().Latin1String)
            else:
                raise RuntimeException('shelved string does not recognize encoding {}'.format(encoding))
                    
            hdr = Header.HeaderEnd(builder)
            builder.FinishSizePrefixed(hdr)
            hdr_ba = builder.Output()

            # allocate memory
            hdr_len = len(hdr_ba)
            value_len = len(string_value) + hdr_len

            memref = memory_resource.create_named_memory(name, value_len, 1, False)
            # copy into memory resource
            memref.tx_begin()
            memref.buffer[0:hdr_len] = hdr_ba
            memref.buffer[hdr_len:] = bytes(string_value, encoding)
            memref.tx_commit()

            self.view = memoryview(memref.buffer[hdr_len:])
        else:
            self.view = memoryview(memref.buffer[32:])

        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._value_named_memory = memref
        self.encoding = encoding
        self._name = name

    def __repr__(self):
        # TODO - some how this is keeping a reference? gc.collect() clears it.
        #
        # creates a new string
        return str(self.view,self.encoding)

    def __len__(self):
        return len(self.view)

    def __getitem__(self, key):
        '''
        Magic method for slice handling
        '''
        s = str(self.view,'utf-8')
        if isinstance(key, int):
            return s[key]
        if isinstance(key, slice):
            return s.__getitem__(key)
        else:
            raise TypeError

    def persist(self):
        '''
        Flush cache and persistent all value memory
        '''
        self._value_named_memory.persist()        
    
    def __getattr__(self, name):
        if name not in ("encoding"):
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self),name))
        else:
            return self.__dict__[name]

    def __add__(self, value): # in-place append e.g, through +=

        # build a new value with different name, then swap & delete
        new_str = str(self.view,self.encoding).__add__(value)
        memory = self._memory_resource
        # create new value
        builder = flatbuffers.Builder(32)
        # create header
        Header.HeaderStart(builder)
        Header.HeaderAddMagic(builder, Constants.Constants().Magic)
        Header.HeaderAddVersion(builder, Constants.Constants().Version)

        if self.encoding == 'ascii':
            Header.HeaderAddType(builder, DataType.DataType().AsciiString)
        elif self.encoding == 'utf-8':
            Header.HeaderAddType(builder, DataType.DataType().Utf8String)
        elif self.encoding == 'utf-16':
            Header.HeaderAddType(builder, DataType.DataType().Utf16String)
        elif self.encoding == 'latin-1':
            Header.HeaderAddType(builder, DataType.DataType().Latin1String)
        else:
            raise RuntimeException('shelved string does not recognize encoding {}'.format(encoding))
            
        hdr = Header.HeaderEnd(builder)
        builder.FinishSizePrefixed(hdr)
        hdr_ba = builder.Output()

        # allocate memory
        hdr_len = len(hdr_ba)
        value_len = len(new_str) + hdr_len

        memref = memory.create_named_memory(self._name + "-tmp", value_len, 1, False)
        
        # copy into memory resource
        #memref.tx_begin() # don't need transaction here?
        memref.buffer[0:hdr_len] = hdr_ba
        memref.buffer[hdr_len:] = bytes(new_str, self.encoding)
        #memref.tx_commit()

        del memref # this will force release
        del self._value_named_memory # this will force release
        gc.collect()

        # swap names
        memory.atomic_swap_names(self._name, self._name + "-tmp")

        # erase old data
        memory.erase_named_memory(self._name + "-tmp")

        # open new data
        memref = memory.open_named_memory(self._name)
        self._value_named_memory = memref
        self.view = memoryview(memref.buffer[hdr_len:])
        return self

    def __eq__(self, value): # == operator
        return (str(self.view,self.encoding).__eq__(value))

    def __ge__(self, value): # == operator
        return (str(self.view,self.encoding).__ge__(value))

    def __gt__(self, value): # == operator
        return (str(self.view,self.encoding).__gt__(value))

    def __le__(self, value): # == operator
        return (str(self.view,self.encoding).__le__(value))

    def __lt__(self, value): # == operator
        return (str(self.view,self.encoding).__lt__(value))

    def __ne__(self, value): # == operator
        return (str(self.view,self.encoding).__ne__(value))

    def __contains__(self, value): # in / not in operator
        return (value in str(self.view,self.encoding))

    def capitalize(self):
        return str(self.view,self.encoding).capitalize()

    def center(self, width, fillchar=' '):
        return str(self.view,self.encoding).center(width, fillchar)
    
    def casefold(self):
        return str(self.view,self.encoding).casefold()

    def count(self, start=0, end=0):
        if end > 0:
            return str(self.view,self.encoding).count(start, end)
        else:
            return str(self.view,self.encoding).count(start)
        
    def encode(self, encoding='utf-8', errors='strict'):
        return str(self.view,self.encoding).encode(encoding, errors)

    # TODO MOSHIK TO FINISH ..

# string object methods
#
#  ['__add__', '__class__', '__contains__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 'translate', 'upper', 'zfill']
