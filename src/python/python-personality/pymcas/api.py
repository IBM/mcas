# Copyright [2017-2021] [IBM Corporation]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
# See the License for the specific language governing permissions and
# limitations under the License.

# Client side API for Python Personality

#from mcas import *
import mcas
import pymcascore
import pymcas
import flatbuffers
import sys
import numpy as np
import inspect
import pickle
import os
import binascii

from flatbuffers import util

# protocol API
from pymcas.Proto.Element import *
from pymcas.Proto.DataType import *
from pymcas.Proto.DataDescriptor import *
from pymcas.Proto.CodeType import *
from pymcas.Proto.InvokeReply import *
from pymcas.Proto.InvokeRequest import *
from pymcas.Proto.Operation import *
from pymcas.Proto.Message import *

# decorator function to type and range check parameters
def paramcheck(types, ranges=None):
    def __f(f):
        def _f(*args, **kwargs):
            for a, t in zip(args, types):
                if not isinstance(a, t):
                    raise TypeError("expected type %s got %r" % (t, a))
            for a, r in zip(args, ranges or []):
                if r and not r[0] <= a <= r[1]:
                    raise ValueError("expected value in range %r: %r" % (r, a))
            return f(*args, **kwargs)
        return _f
    return __f


def methodcheck(types, ranges=None):
    def __f(f):
        def _f(*args, **kwargs):
            for a, t in zip(args[1:], types):
                if not isinstance(a, t):
                    raise TypeError("method expected type %s got %r" % (t, a))
            for a, r in zip(args[1:], ranges or []):
                if r and not r[0] <= a <= r[1]:
                    raise ValueError("method expected value in range %r: %r" % (r, a))
            return f(*args, **kwargs)
        return _f
    return __f


@paramcheck(types=[str,int,str,(str,type(None)),int], ranges=[None, None, None, None, None])
def create_session(ip, port, device="mlx5_0", extra=None, debug=0):
    return Session(ip, port, device, extra, debug)


class Session():
    """
    Session class represents connection to MCAS server
    """       
    __session = None

    @methodcheck(types=[str, int, str, (str, type(None)), int], ranges=[None, None, None, None, None])
    def __init__(self, ip, port, device="mlx5_0", extra=None, debug=0):
        self.__session = mcas.Session(ip, port, device, extra, debug)

    @methodcheck(types=[str, int, int, bool])
    def create_pool(self, name, size=32000000, count=1000, create_only=False):
        return Pool(self.__session.create_pool(name, size, count, create_only))

    
class ManagedArray():
    """
    Managed array couples memory with ndarray
    """
    def __init__(self, arr, raw_memory_view, pool, global_name):
        if not isinstance(arr, np.ndarray):
            raise TypeError("bad array parameter type")
        
        self.array = arr
        self.__raw = raw_memory_view
        self.__pool = pool
        self.global_name = global_name
        
    def __array__(self):
        return self.array

    def __del__(self):
        self.__array = None
        self.__pool.free_direct_memory(self.__raw)

    # type extension through embedding (to complete)
    def __add__(self, x):
        return self.array.__add__(x)
    
    def __sub__(self, x):
        return self.array.__sub__(x)

    def __rsub__(self, x):
        return self.array.__rsub__(x)
    
    def __mul__(self, x):
        return self.array.__mul__(x)
    
    def __div__(self, x):
        return self.array.__div__(x)

    def __rdiv__(self, x):
        return self.array.__rdiv__(x)

    def __getattr__(self, name):
        return self.array.__getattribute__(name)
  

class Pool():
    """
    Pool class
    """
    @methodcheck(types=[mcas.Pool])
    def __init__(self, pool):
        self.__pool = pool

    def __getattr__(self, name):
        return self.__pool.__getattribute__(name)

    @methodcheck(types=[str])
    def erase(self, key):
        """
        Erase object from pool
        """
        return self.__pool.erase(key)
        
    @methodcheck(types=[str])
    def save(self, key, value):
        """
        Save Python object to MCAS pool
        """
        if isinstance(value, np.ndarray):

            #
            # [ fb metadata | numpy hdr | numpy data ]
            #
            
            # create opaque header (byte array)
            hdr = pymcascore.ndarray_header(value)
            
            builder = flatbuffers.Builder(256) # initial size
            key_fb = builder.CreateString(key)

            DataDescriptorStart(builder)
            DataDescriptorAddGlobalName(builder, key_fb)
            DataDescriptorAddType(builder, DataType().NumPyArray)
            DataDescriptorAddHeaderLength(builder, len(hdr))
            DataDescriptorAddDataLength(builder, len(value))
            dd = DataDescriptorEnd(builder)

            MessageStart(builder)
            MessageAddElementType(builder, Element.DataDescriptor)
            MessageAddElement(builder, dd)
            msg = MessageEnd(builder)
            
            builder.FinishSizePrefixed(msg)

            # until we have scatter put, we need a contiguous byte array
            # or memoryview

            # construct message
            data = builder.Output()      # flatbuffer info
            data.extend(hdr)             # ndarray header
            data.extend(value.tobytes()) # ndarray actual data
        else:
            # default to pickle
            #
            # [ fb metadata | pickled data ]
            #
            pickled_value = pickle.dumps(value)
            print("pickled: len={0}".format(len(pickled_value)))
            
            builder = flatbuffers.Builder(256) # initial size
            key_fb = builder.CreateString(key)

            DataDescriptorStart(builder)
            DataDescriptorAddGlobalName(builder, key_fb)
            DataDescriptorAddType(builder, DataType().Pickled)
            DataDescriptorAddHeaderLength(builder, 0)
            DataDescriptorAddDataLength(builder, len(pickled_value))
            dd = DataDescriptorEnd(builder)

            MessageStart(builder)
            MessageAddElementType(builder, Element.DataDescriptor)
            MessageAddElement(builder, dd)
            msg = MessageEnd(builder)
            
            builder.FinishSizePrefixed(msg)

            # until we have scatter put, we need a contiguous byte array
            # or memoryview

            # construct message
            data = builder.Output()    # flatbuffer header
            data.extend(pickled_value) # pickled data

        # put direct into store
        print('save: total len = {0}'.format(len(data)))
        self.__pool.put_direct(key, data) # not sure this works without special memory?



    @methodcheck(types=[str])            
    def load(self, key):
        """
        Load Python object from MCAS pool
        """
        # using get_direct will get size first, allocate memory
        # and then issue get.  Return type is <memoryview>
        raw_obj = self.__pool.get_direct(key)

        print("load: raw length of {0} is {1}".format(key, len(raw_obj)))

        # extract 4 byte size prefix
        msg_size = util.GetSizePrefix(raw_obj, 0)
        print("msg_size:", msg_size + 4, " (including 4B prefix)")

        # parse Message
        root = pymcas.Proto.Message.Message()
        msg = root.GetRootAsMessage(raw_obj[4:], 0) # size prefix is 4 bytes
        print("version : {0}".format(int(msg.Version())))

        # check magic
        if (msg.Magic() != 0xc0ffee0):
            raise ValueError("bad magic")

        if (msg.ElementType() != Element.DataDescriptor):
            print(msg)
            raise ValueError("bad packet format (expected DataDescriptor)")
            
        dd = pymcas.Proto.DataDescriptor.DataDescriptor()
        dd.Init(msg.Element().Bytes, msg.Element().Pos)

        
        global_name = dd.GlobalName()
        print("element - name={0}, hdrlen={1} datalen={2}".format(global_name, dd.HeaderLength(), dd.DataLength()))
        
        if dd.Type() == DataType.NumPyArray: # handle NumPyArray data type
            ndarray_data = raw_obj[msg_size:]
            print("ndarray_data: len={0} {1}".format(len(ndarray_data),type(ndarray_data)))
            arr = pymcascore.ndarray_from_bytes(raw_obj[msg_size+4:], # zero-copy slice, add 4 for prefix
                                                dd.HeaderLength()) #, dd.DataLength())
            # managed array ensures that memory is freed appropriately
            return ManagedArray(arr, raw_obj, self.__pool, global_name)
        elif dd.Type() == DataType.Pickled: # handle pickled data
            print("loading pickled data object..")
            print(raw_obj[msg_size+4:].tobytes())
            print("pickled data: len={0}".format(dd.DataLength()))
            return pickle.loads(raw_obj[msg_size+4:])
        else:
            raise TypeError("unhandled data descriptor type")

    @methodcheck(types=[str])            
    def erase(self, key):
        '''
        Erase object from pool
        '''
        self.__pool.erase(key)

    @methodcheck(types=[str])            
    def invoke(self, key, function, parameters=None):
        '''
        Invoke ADO
        '''
        if inspect.isfunction(function) == False:
            raise TypeError("invoke requires (key, function); detected not function type")
        
        code = inspect.getsource(function)

        builder = flatbuffers.Builder(4096) # initial size
        key_fb = builder.CreateString(key)
        function_name_fb = builder.CreateString(function.__name__)
        code_str =  builder.CreateString(code)

        # additional parameters; convert to UU-encoded string
        params = binascii.b2a_base64(pickle.dumps(parameters)).rstrip()
        params_str = builder.CreateString(params)

        print('-ADO params + function------------>')
        print('params:', pickle.loads(binascii.a2b_base64(params)))
        print(code)
        print('---------------------------------->')
        
        OperationStart(builder)
        OperationAddCodeType(builder, CodeType().CPython)
        OperationAddFunction(builder, function_name_fb)
        OperationAddCode(builder, code_str)
        OperationAddAdditionalParams(builder, params_str)
        op = OperationEnd(builder)
        
        InvokeRequestStart(builder)
        InvokeRequestAddOp(builder, op)
        ir = InvokeRequestEnd(builder)
                
        MessageStart(builder)
        MessageAddElementType(builder, Element.InvokeRequest)
        MessageAddElement(builder, ir)
        msg = MessageEnd(builder)
            
        builder.FinishSizePrefixed(msg)

        # until we have scatter put, we need a contiguous byte array
        # or memoryview

        # construct message
        data = builder.Output()      # flatbuffer info
    
        print('invoke: total len = {0}'.format(len(data)))
        return pickle.loads(self.__pool.invoke_ado(key, data))
        
        

# -----------------------------------------------------------------------------------------
# TESTING AREA
# -----------------------------------------------------------------------------------------

# Notes:
#
# The API firstly provides load and save operations for python
# objects. The implementation will try to user zero-copy but will
# revert to pickle if needed.  To perform near-data operations
# (i.e. execution of python code in the ADO), the user calls the
# invoke function, passing target, function and params. If no params
# are given, None is passed.  Parameters are pickled and base64 encoded.
#
def basic_test_0():
    session = pymcas.create_session(os.getenv('SERVER_IP'), 11911, debug=3)
    if sys.getrefcount(session) != 2:
        raise ValueError("session ref count should be 2")
    pool = session.create_pool("myPool")
    if sys.getrefcount(pool) != 2:
        raise ValueError("pool ref count should be 2")

    a = np.identity(10)
    print(a)
    key = 'object-A'
    pool.save(key, a)

    b = pool.load(key)
    print("test0 OK!?")
    print("result of load:")
    print(b)
    return b


def test_dict_work(target_dict, params):
    target_dict['plant'] = ["triffid"]
    for k,v in params.items():
        target_dict[k] = v
    # this is sent back to client as invoke result
    return {'newhousehold': target_dict} 


def test_dict():
    """
    Test for non-NdArray types, e.g., dictionary. Tests parameter passing also.
    """
    session = pymcas.create_session(os.getenv('SERVER_IP'), 11911, debug=3)
    if sys.getrefcount(session) != 2:
        raise ValueError("session ref count should be 2")
    pool = session.create_pool("myPool")
    if sys.getrefcount(pool) != 2:
        raise ValueError("pool ref count should be 2")

    mydict = {}
    mydict['cat'] = ["jenny", "jazmine"]
    mydict['dog'] = ["violet"]
    mydict['chickens'] = ["bob", "ferdy", "rosemary"]

    pool.save('household', mydict);

    result = pool.invoke('household', test_dict_work, {"children" : ["ben", "molly"]})
    print("New household->", result)
    

def sobel_filter(image, params):
    from skimage import data, io, filters
    edges = filters.sobel(image)
    ado.save('image0-edges', edges) # save image
    # this is sent back to client as invoke result
    return edges.shape

def test_skimage_0():
    """
    Test using NdArray data and Scikit-Image to perform Sobel filter in ADO
    """
    session = pymcas.create_session(os.getenv('SERVER_IP'), 11911, debug=3)
    if sys.getrefcount(session) != 2:
        raise ValueError("session ref count should be 2")
    pool = session.create_pool("myPool")
    if sys.getrefcount(pool) != 2:
        raise ValueError("pool ref count should be 2")

    # save image
    from skimage import data, io, filters
    image = data.coins()
    pool.save('image0', image)

    # perform ADO invocation
    shape = pool.invoke('image0', sobel_filter)
    print("shape:{0}".format(shape))

    # recall and display image
    edges = pool.load('image0-edges')
    print(type(edges))
    io.imshow(edges)
    io.show()


def blend_ado_function(target_image, delta):
    brick = ado.load('brick')
    blended = target_image+(delta*brick)
    ado.save('blended', blended)
    return blended.shape

def test_skimage_1():
    """
    Push two images, perform ADO function to combine and return image
    """
    session = pymcas.create_session(os.getenv('SERVER_IP'), 11911, debug=3)
    if sys.getrefcount(session) != 2:
        raise ValueError("session ref count should be 2")
    pool = session.create_pool("myPool2",int(1e9),100)
    if sys.getrefcount(pool) != 2:
        raise ValueError("pool ref count should be 2")

    # save image
    from skimage import data, io, filters

    pool.save('camera', data.camera())
    pool.save('brick', data.brick())

    # # perform ADO invocation
    shape = pool.invoke('camera', blend_ado_function, 0.5) # param is delta
    print("shape:{0}".format(shape))

    blend = pool.load('blended')
    
    io.imshow(blend)
    io.show()

    
def test_ref_cnt():
    """
    Test reference counting
    """
    session = mcas.Session(os.getenv('SERVER_IP'), 11911)
    pool = session.create_pool('fooPool', 100)
    session = 0
    pool = 0

def test_erase():
    session = pymcas.create_session(os.getenv('SERVER_IP'), 11911, debug=3)
    if sys.getrefcount(session) != 2:
        raise ValueError("session ref count should be 2")
    pool = session.create_pool("myPool")
    if sys.getrefcount(pool) != 2:
        raise ValueError("pool ref count should be 2")

    from skimage import data, io, filters
    image = data.coins()
    pool.save('image0', image)

    pool.erase('image0')

    try:
        pool.load('image0')
    except:
        print("TEST: correctly erased - object cannot be loaded!")
    

    
