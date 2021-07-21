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

import PyMM.Meta.Header as Header
import PyMM.Meta.Constants as Constants
import PyMM.Meta.DataType as DataType

from flatbuffers import util
from .memoryresource import MemoryResource
from .shelf import Shadow
from .shelf import ShelvedCommon

class float_number(Shadow):
    '''
    Floating point number that is stored in the memory resource. Uses value cache
    '''
    def __init__(self, number_value):
        self.number_value = number_value

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_float_number(memory_resource, name, number_value=self.number_value)

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

        if (hdr.Type() == DataType.DataType().NumberFloat):
            return (True, shelved_float_number(memory_resource, name, buffer[hdr_size + 4:]))

        # not a string
        return (False, None)

    def build_from_copy(memory_resource: MemoryResource, name: str, value):
        return shelved_float_number(memory_resource, name, number_value=value)



class shelved_float_number(ShelvedCommon):
    '''
    Shelved floating point number
    '''
    def __init__(self, memory_resource, name, number_value):

        memref = memory_resource.open_named_memory(name)

        if memref == None:
            # create new value
            builder = flatbuffers.Builder(32)
            # create header
            Header.HeaderStart(builder)
            Header.HeaderAddMagic(builder, Constants.Constants().Magic)
            Header.HeaderAddVersion(builder, Constants.Constants().Version)
            Header.HeaderAddType(builder, DataType.DataType().NumberFloat)

            hdr = Header.HeaderEnd(builder)
            builder.FinishSizePrefixed(hdr)
            hdr_ba = builder.Output()

            # allocate memory
            hdr_len = len(hdr_ba)
            value_bytes = str.encode(number_value.hex())
            value_len = hdr_len + len(value_bytes) 

            memref = memory_resource.create_named_memory(name, value_len, 1, False)
            # copy into memory resource
            memref.tx_begin()
            memref.buffer[0:hdr_len] = hdr_ba
            memref.buffer[hdr_len:] = value_bytes
            memref.tx_commit()
        else:

            hdr_size = util.GetSizePrefix(memref.buffer, 0)
            if hdr_size != 28:
                raise RuntimeError("invalid header for '{}'; prior version?".format(varname))
            
            root = Header.Header()
            hdr = root.GetRootAsHeader(memref.buffer[4:], 0) # size prefix is 4 bytes
            
            if(hdr.Magic() != Constants.Constants().Magic):
                raise RuntimeError("bad magic number - corrupt data?")
                
            self._type = hdr.Type()

        # set up the view of the data
        # materialization alternative - self._view = memoryview(memref.buffer[32:])
        self._cached_value = number_value
        self._name = name
        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._value_named_memory = memref

    def _atomic_update_value(self, value):
        if not isinstance(value, float):
            raise TypeError('bad type for atomic_update_value')

        # create new float 
        builder = flatbuffers.Builder(32)
        # create header
        Header.HeaderStart(builder)
        Header.HeaderAddMagic(builder, Constants.Constants().Magic)
        Header.HeaderAddVersion(builder, Constants.Constants().Version)
        Header.HeaderAddType(builder, DataType.DataType().NumberFloat)

        hdr = Header.HeaderEnd(builder)
        builder.FinishSizePrefixed(hdr)
        hdr_ba = builder.Output()

        # allocate memory
        hdr_len = len(hdr_ba)
        value_bytes = str.encode(value.hex())
        value_len = hdr_len + len(value_bytes) 

        memory = self._memory_resource
        memref = memory.create_named_memory(self._name + '-tmp', value_len, 1, False)
        # copy into memory resource
        memref.tx_begin()
        memref.buffer[0:hdr_len] = hdr_ba
        memref.buffer[hdr_len:] = value_bytes
        memref.tx_commit()

        del memref # this will force release
        del self._value_named_memory # this will force release
        gc.collect()

        # swap names
        memory.atomic_swap_names(self._name, self._name + "-tmp")

        # erase old data
        memory.erase_named_memory(self._name + "-tmp")
        
        memref = memory.open_named_memory(self._name)
        self._value_named_memory = memref
        self._cached_value = value
        # materialization alternative - self._view = memoryview(memref.buffer[32:])
        return self

        
    def _get_value(self):
        '''
        Materialize the value either from persistent memory or cached value
        '''
        return self._cached_value
        # materialization alternative - return float.fromhex((bytearray(self._view)).decode())


    def __repr__(self):
        return str(self._get_value())

    def __float__(self):
        return float(self._get_value())

    def __int__(self):
        return int(self._get_value())

    def __bool__(self):
        return bool(self._get_value())

    # in-place arithmetic
    def __iadd__(self, value): # +=
        return self._atomic_update_value(float(self._get_value()).__add__(value))

    def __imul__(self, value): # *=
        return self._atomic_update_value(float(self._get_value()).__mul__(value))

    def __isub__(self, value): # -=
        return self._atomic_update_value(float(self._get_value()).__sub__(value))

    def __itruediv__(self, value): # /=
        return self._atomic_update_value(float(self._get_value()).__truediv__(value))

    def __imod__(self, value): # %=
        return self._atomic_update_value(float(self._get_value()).__mod__(value))

    def __ipow__(self, value): # **=
        return self._atomic_update_value(float(self._get_value()).__pow__(value))

    def __ilshift__(self, value): # <<=
        return self._atomic_update_value(float(self._get_value()).__lshift__(value))

    def __irshift__(self, value): # >>=
        return self._atomic_update_value(float(self._get_value()).__rshift__(value))

    def __iand__(self, value): # &=
        return self._atomic_update_value(float(self._get_value()).__and__(value))

    def __ixor__(self, value): # ^=
        return self._atomic_update_value(float(self._get_value()).__xor__(value))

    def __ior__(self, value): # |=
        return self._atomic_update_value(float(self._get_value()).__or__(value))
        
        

    # arithmetic operations
    # not using magic method seems to help with implicit casting
    def __add__(self, value):
        return self._get_value() + value

    def __radd__(self, value):
        return value + self._get_value()

    def __sub__(self, value):
        return self._get_value() - value

    def __rsub__(self, value):
        return value - self._get_value()

    def __mul__(self, value):
        return self._get_value() * value

    def __rmul__(self, value):
        return value * self._get_value()

    def __truediv__(self, value):
        return self._get_value() / value

    def __rtruediv__(self, value):
        return self._get_value() / value

    def __floordiv__(self, value):
        return self._get_value() // value

    def __rfloordiv__(self, value):
        return value // self._get_value()
    
    def __divmod__(self, x):
        return divmod(self._get_value(),x)

    def __eq__(self, x):
        return self._get_value() == x

    def __le__(self, x):
        return self._get_value() <= x

    def __lt__(self, x):
        return self._get_value() < x

    def __ge__(self, x):
        return self._get_value() >= x

    def __gt__(self, x):
        return self._get_value() > x

    def __ne__(self, x):
        return self._get_value() != x



    # TODO: MOSHIK TO FINISH
    
    def hex(self):
        return float(self._get_value()).hex()
        
# ['__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__class__', '__delattr__', '__dir__', '__divmod__', '__doc__', '__eq__', '__float__', '__floor__', '__floordiv__', '__format__', '__ge__', '__getattribute__', '__getnewargs__', '__gt__', '__hash__', '__index__', '__init__', '__init_subclass__', '__int__', '__invert__', '__le__', '__lshift__', '__lt__', '__mod__', '__mul__', '__ne__', '__neg__', '__new__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__reduce_ex__', '__repr__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__', '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setattr__', '__sizeof__', '__str__', '__sub__', '__subclasshook__', '__truediv__', '__trunc__', '__xor__', 'bit_length', 'conjugate', 'denominator', 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']
