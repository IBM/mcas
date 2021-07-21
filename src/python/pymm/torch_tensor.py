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

#    Notes: highly experimental and likely defective

import torch
import pymmcore
import numpy as np
import copy

from numpy import uint8, ndarray, dtype, float
from .memoryresource import MemoryResource
from .ndarray import ndarray, shelved_ndarray
from .shelf import Shadow
from .shelf import ShelvedCommon

def colored(r, g, b, text):
    return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)

dtypedescr = np.dtype
    
# shadow type for torch_tensor
#
class torch_tensor(Shadow):
    '''
    PyTorch tensor that is stored in a memory resource
    '''
    def __init__(self, shape, dtype=float, strides=None, order='C'):

        print(colored(255,0,0, 'WARNING: torch_tensor is experimental and unstable!'))
        # todo check params
        # todo check and invalidate param 'buffer'
        # save constructor parameters and type
        self.__p_shape = shape
        self.__p_dtype = dtype
        self.__p_strides = strides
        self.__p_order = order

    def make_instance(self, memory_resource: MemoryResource, name: str):
        '''
        Create a concrete instance from the shadow
        '''
        return shelved_torch_tensor(memory_resource,
                                    name,
                                    shape = self.__p_shape,
                                    dtype = self.__p_dtype,
                                    strides = self.__p_strides,
                                    order = self.__p_order)

    def existing_instance(memory_resource: MemoryResource, name: str):
        '''
        Determine if an persistent named memory object corresponds to this type
        '''
        metadata = memory_resource.get_named_memory(name)
        if metadata is None:
            return (False, None)
        
        if pymmcore.ndarray_read_header(memoryview(metadata), type=1) == None:
            return (False, None)
        else:
            return (True, shelved_torch_tensor(memory_resource, name, shape = None))

    def __str__(self):
        print('shadow torch_tensor')


    def build_from_copy(memory_resource: MemoryResource, name: str, tensor):
        new_tensor = shelved_torch_tensor(memory_resource,
                                          name,
                                          shape = tensor.shape,
                                          dtype = tensor.dtype)
        
        # now copy the data
        new_tensor[:] = tensor
        return new_tensor

    
# concrete subclass for torch tensor
#
class shelved_torch_tensor(torch.Tensor, ShelvedCommon):
    '''PyTorch tensor that is stored in a memory resource'''
    __array_priority__ = -100.0 # what does this do?

    def __new__(subtype, memory_resource, name, shape, dtype=float, strides=None, order='C'):

        torch_to_numpy_dtype_dict = {
            torch.bool  : np.bool,
            torch.uint8 : np.uint8,
            torch.int8  : np.int8,
            torch.int16 : np.int16,
            torch.int32 : np.int32,
            torch.int64 : np.int64,
            torch.float16 : np.float16,
            torch.float32 : np.float32,
            torch.float64 : np.float64,
            torch.complex64 : np.complex64,
            torch.complex128 : np.complex128,
        }
        np_dtype = torch_to_numpy_dtype_dict.get(dtype,None)

        # determine size of memory needed
        #
        descr = dtypedescr(np_dtype)
        _dbytes = descr.itemsize

        if shape != None:
            size = np.intp(1)  # avoid default choice of np.int_, which might overflow

            # if it is not a shape converted from tuple, then we may have to convert
            if not isinstance(shape, torch.Size):
                shape = get_shape(shape) # convert shape

            for k in shape:
                size *= k
                
        value_key = name + '-value'
        metadata_key = name
        
        value_named_memory = memory_resource.open_named_memory(value_key)

        if value_named_memory == None: # does not exist yet

            # use shelved_ndarray under the hood and make a subclass from it
            base_ndarray = shelved_ndarray(memory_resource, name=name, shape=shape, dtype=np_dtype, type=1)

            # create and store metadata header
            metadata = pymmcore.ndarray_header(base_ndarray, np.dtype(np_dtype).str, type=1) # type=1 indicate torch_tensor
            memory_resource.put_named_memory(metadata_key, metadata)

        else:
            # entity already exists, load metadata
            del value_named_memory # the shelved_ndarray ctor will need to reopen it
            metadata = memory_resource.get_named_memory(metadata_key)
            hdr = pymmcore.ndarray_read_header(memoryview(metadata), type=1) # type=1 indicate torch_tensor

            base_ndarray = shelved_ndarray(memory_resource, name=name, dtype=hdr['dtype'], shape=hdr['shape'],
                                           strides=hdr['strides'], order=order, type=1)
            
        self = torch.Tensor._make_subclass(subtype, torch.as_tensor(base_ndarray))
        self._base_ndarray = base_ndarray
        self._value_named_memory = base_ndarray._value_named_memory

        # hold a reference to the memory resource
        self._memory_resource = memory_resource
        self._metadata_key = metadata_key
        self.name_on_shelf = name
        return self

    def __delete__(self, instance):
        raise RuntimeError('cannot delete item: use shelf erase')

    def __array_wrap__(self, out_arr, context=None):
        # Handle scalars so as not to break ndimage.
        # See http://stackoverflow.com/a/794812/1221924
        if out_arr.ndim == 0:
            return out_arr[()]
        return torch.tensor.__array_wrap__(self, out_arr, context)

    def __getattr__(self, name):
#        print("__getattr__ :", name)
        if name == 'addr':
            return self._value_named_memory.addr()
        else:
            return super().__dict__[name]

    # def __array_ufunc__(ufunc, method, *inputs, **kwargs):
    #     print('__array_ufunc__')
    #     return super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        

    def as_tensor(self):
        '''Cast class to torch.Tensor subclass'''
        return self.as_subclass(torch.Tensor)
    
    def update_metadata(self, array):
        metadata = pymmcore.ndarray_header(array,np.dtype(dtype).str, type=1)
        self._memory_resource.put_named_memory(self._metadata_key, metadata)

    # each type will handle its own transaction methodology.  this
    # is because metadata may be dealt with differently
    #
    def _value_only_transaction(self, F, *args):
        if self is None:
            return
        if '_value_named_memory' in self.__dict__:
            self._value_named_memory.tx_begin()
            result = F(*args)
            self._value_named_memory.tx_commit()            
        else:
            result = F(*args)
        return result

    def _tx_begin(self):
        self._value_named_memory.tx_begin()

    def _tx_commit(self):
        self._value_named_memory.tx_commit()

    # all methods that perform writes are implicitly used to define transaction
    # boundaries (at least most fine-grained)
    #
    # reference: https://numpy.org/doc/stable/reference/routines.array-manipulation.html
    #

    # in-place methods need to be transactional
    def fill(self, value):
        return self._value_only_transaction(super().fill_, value)
    
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

    

    # TODO... more

    # set item, e.g. x[2] = 2    
#    def __setitem__(self, position, x):
#        return self._value_only_transaction(super().__setitem__, position, x)


#    def __getitem__(self, key):
#        '''
#        Magic method for slice handling. Pytorch tensor does something strange
#        with the slicing, the self (object id) changes, so we have to hand
#        off the base ndarray
#        '''
#        try:
#            # there may be things that tensor allows that ndarray does not?
#            return torch.as_tensor(self._base_ndarray.__getitem__(key))
#        except AttributeError:
#            return super().__getitem__(key)


    # operations that return new views on same data.  we want to change
    # the behavior to give a normal volatile version
    #
    #def reshape(self, shape, order='C'):
    #    x = self.clone() # copy
    #    return x.reshape(shape)
        

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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # NOTE: this is some PyTorch magic to intercept everything
#        print('TORCH FUNCTION {} {}'.format(cls, func))
        
        if kwargs is None:
            kwargs = {}
            
        r = super().__torch_function__(func, types, args, kwargs)

        # nicely, in-place functions have a underscore suffix
        if func.__name__[-1:] == '_':
            return r

        if isinstance(r, shelved_torch_tensor):
            return r.as_subclass(torch.Tensor) # cast to plain tensor
        else:
            return r # result may not be a tensor

    def persist(self):
        '''
        Flush cache and persistent all value memory
        '''
        self._value_named_memory.persist()




from collections.abc import Sequence

def get_shape(lst):
    def ishape(lst):
        shapes = [ishape(x) if isinstance(x, list) else [] for x in lst]
        shape = shapes[0]
        if shapes.count(shape) != len(shapes):
            raise ValueError('Ragged list')
        shape.append(len(lst))
        return shape
    return tuple(reversed(ishape(lst)))
