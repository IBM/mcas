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

#    Notes: not all methods hooked.

import pymmcore
import numpy as np
import copy

from numpy import uint8, ndarray, dtype, float
from .memoryresource import MemoryResource
from .shelf import Shadow
from .shelf import ShelvedCommon

dtypedescr = np.dtype
    
# shadow type for ndarray
#
class ndarray(Shadow):
    '''
    ndarray that is stored in a memory resource
    '''
    def __init__(self, shape, dtype=float, strides=None, order='C', zero=False):

        # todo check params
        # todo check and invalidate param 'buffer'
        # save constructor parameters and type
        self.__p_shape = shape
        self.__p_dtype = dtype
        self.__p_strides = strides
        self.__p_order = order
        self.__p_zero = zero

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_ndarray(memory_resource,
                               name,
                               shape = self.__p_shape,
                               dtype = self.__p_dtype,
                               strides = self.__p_strides,
                               order = self.__p_order,
                               zero = self.__p_zero)

    def existing_instance(memory_resource: MemoryResource, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''
        metadata = memory_resource.get_named_memory(name)
        if metadata is None:
            return (False, None)
        
        if pymmcore.ndarray_read_header(memoryview(metadata)) == None:
            return (False, None)
        else:
            return (True, shelved_ndarray(memory_resource, name, shape = None))

    def __str__(self):
        print('shadow ndarray')


    def build_from_copy(memory_resource: MemoryResource, name: str, array):
        new_array = shelved_ndarray(memory_resource,
                                    name,
                                    shape = array.shape,
                                    dtype = array.dtype,
                                    strides = array.strides)

        # now copy the data
        #new_array[:] = array
        np.copyto(new_array, array)
        return new_array



    
# concrete subclass for ndarray
#
class shelved_ndarray(np.ndarray, ShelvedCommon):
    '''ndarray that is stored in a memory resource'''
    __array_priority__ = -100.0 # what does this do?

    def __new__(subtype, memory_resource, name, shape, dtype=float, strides=None, order='C', type=0, zero=False):

        # determine size of memory needed
        #
        descr = dtypedescr(dtype)
        _dbytes = descr.itemsize

        if shape != None:
            size = np.intp(1)  # avoid default choice of np.int_, which might overflow

            if isinstance(shape, tuple) or isinstance(shape, list):
                if isinstance(shape, tuple) and isinstance(shape[0], list):
                    shape = shape[0]

                for k in shape:
                    size *= k
            else:
                raise RuntimeError('unhandled condition in shelved_ndarray shape handling (shape={})'.format(shape))

        # the meta data is always accessible by the key name
        value_key = name + '-value'
        metadata_key = name

        value_named_memory = memory_resource.open_named_memory(value_key)

        if value_named_memory == None: # does not exist yet
            #
            # create a newly allocated named memory from MemoryResource
            #
            msize = int(size*_dbytes)
            if msize < 8:
                alignment = 1
            else:
                alignment = 8
            value_named_memory = memory_resource.create_named_memory(value_key,
                                                                     msize,
                                                                     alignment,
                                                                     zero) # zero memory
            # construct array using supplied memory
            #        shape, dtype=float, buffer=None, offset=0, strides=None, order=None
            self = np.ndarray.__new__(subtype, dtype=dtype, shape=shape, buffer=value_named_memory.buffer,
                                      strides=strides, order=order)

            # create and store metadata header
            metadata = pymmcore.ndarray_header(self,np.dtype(dtype).str, type=type)
            memory_resource.put_named_memory(metadata_key, metadata)

        else:
            # entity already exists, load metadata            
            metadata = memory_resource.get_named_memory(metadata_key)
            hdr = pymmcore.ndarray_read_header(memoryview(metadata),type=type)
            self = np.ndarray.__new__(subtype, dtype=hdr['dtype'], shape=hdr['shape'], buffer=value_named_memory.buffer,
                                      strides=hdr['strides'], order=order)

        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._value_named_memory = value_named_memory
        self._metadata_key = metadata_key
        self._value_key = value_key
        self.name = name
        return self

    def __delete__(self, instance):
        raise RuntimeError('cannot delete item: use shelf erase')

    def __array_wrap__(self, out_arr, context=None):
        # Handle scalars so as not to break ndimage.
        # See http://stackoverflow.com/a/794812/1221924
        if out_arr.ndim == 0:
            return out_arr[()]
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getattr__(self, name):
        if name == 'addr':
            return self._value_named_memory.addr()
        elif name not in super().__dict__:
            raise AttributeError("'{}' object has no attribute '{}'".format(type(self),name))
        else:
            return super().__dict__[name]

    def asndarray(self):
        return self.view(np.ndarray)

    def dim(self):
        return len(super().shape)
    
    def update_metadata(self, array):
        metadata = pymmcore.ndarray_header(array,np.dtype(dtype).str)
        self._memory_resource.put_named_memory(self._metadata_key, metadata)

    # each type will handle its own transaction methodology.  this
    # is because metadata may be dealt with differently
    #
    def _value_only_transaction(self, F, *args):
        if self is None:
            return
        self._value_named_memory.tx_begin()
        result = F(*args)
        self._value_named_memory.tx_commit()
        return result

    # all methods that perform writes are implicitly used to define transaction
    # boundaries (at least most fine-grained)
    #
    # reference: https://numpy.org/doc/stable/reference/routines.array-manipulation.html
    #

    # in-place methods need to be transactional
    def fill(self, value):
        return self._value_only_transaction(super().fill, value)

    def byteswap(self, inplace):
        if inplace == True:
            return self._value_only_transaction(super().byteswap, True)
        else:
            return super().byteswap(False)

    # in-place arithmetic
    def __iadd__(self, value): # +=
        return self._value_only_transaction(super().__iadd__, value)

    def __imul__(self, value): # *=
        return self._value_only_transaction(super().__imul__, value)

    def __isub__(self, value): # -=
        return self._value_only_transaction(super().__isub__, value)

    def __idiv__(self, value): # /=
        return self._value_only_transaction(super().__idiv__, value)

    def __imod__(self, value): # %=
        return self._value_only_transaction(super().__imod__, value)

    def __ipow__(self, value): # **=
        return self._value_only_transaction(super().__ipow__, value)

    def __ilshift__(self, value): # <<=
        return self._value_only_transaction(super().__ilshift__, value)

    def __irshift__(self, value): # >>=
        return self._value_only_transaction(super().__irshift__, value)

    def __iand__(self, value): # &=
        return self._value_only_transaction(super().__iand__, value)

    def __ixor__(self, value): # ^=
        return self._value_only_transaction(super().__ixor__, value)

    def __ior__(self, value): # |=
        return self._value_only_transaction(super().__ior__, value)

    
    # out-of-place we need to convert back to ndarray
    def __add__(self, value):
        return super().__add__(value).asndarray()

    def __mul__(self, value):
        return super().__mul__(value).asndarray()

    def __sub__(self, value):
        return super().__sub__(value).asndarray()

    def __div__(self, value):
        return super().__div__(value).asndarray()

    def __mod__(self, value):
        return super().__mod__(value).asndarray()

    def __pow__(self, value):
        return super().__pow__(value).asndarray()

    def __lshift__(self, value):
        return super().__lshift__(value).asndarray()

    def __rshift__(self, value):
        return super().__rshift__(value).asndarray()

    def __and__(self, value):
        return super().__and__(value).asndarray()

    def __or__(self, value):
        return super().__or__(value).asndarray()
    
    def __xor__(self, value):
        return super().__xor__(value).asndarray()


    # set item, e.g. x[2] = 2
    # if we want transactionality on this we override it - but beware, it will cost!
#    def __setitem__(self, position, x):
#        return self._value_only_transaction(super().__setitem__, position, x)

    def flip(self, m, axis=None):
        return self._value_only_transaction(super().flip, m, axis)


    # operations that return new views on same data.  we want to change
    # the behavior to give a normal volatile version
#    def reshape(self, shape, order='C'):
#        x = np.array(self) # copy constructor
#        return x.reshape(shape)
        

    def __array_finalize__(self, obj):
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        if obj is None: return

        # From view casting and new-from-template
        self.info = getattr(obj, 'info', None)
        self._memory_resource = getattr(obj, '_memory_resource', None)
        self._value_named_memory = getattr(obj, '_value_named_memory', None)
        self._metadata_key = getattr(obj, '_metadata_key', None)
        self.name = getattr(obj, 'name', None)

    def persist(self):
        '''
        Flush cache and persistent all value memory
        '''
        self._value_named_memory.persist()
        
