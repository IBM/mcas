/*
   Copyright [2021] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <vector>
#include <string>
#include <sstream>

#include <common/utils.h>
#include <common/dump_utils.h>
#include <Python.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

namespace global
{
unsigned debug_level = 3;
}

/* forward decls */
PyObject * unmarshall_nparray(byte * ptr);
void create_ndarray_header(PyArrayObject * src_ndarray, std::string& out_hdr, const char * dtype_str = nullptr);  

PyObject * pymcas_ndarray_header_size(PyObject * self,
                                      PyObject * args,
                                      PyObject * kwargs)
{
  import_array();
  static const char *kwlist[] = {"array",
                                 NULL};

  PyObject * src_array = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "O",
                                    const_cast<char**>(kwlist),
                                    &src_array)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }
  
  if (! src_array) {
    PyErr_SetString(PyExc_RuntimeError,"array parameter missing");
    return NULL;
  }

  if (! PyArray_Check(src_array)) {
    PyErr_SetString(PyExc_RuntimeError,"not ndarray type");
    return NULL;
  }

  PyArrayObject * src_ndarray = reinterpret_cast<PyArrayObject*>(src_array);

  /* sanity checks */
  if (! PyArray_ISBEHAVED(src_ndarray)) {
    PyErr_SetString(PyExc_RuntimeError,"mis-behaving ndarray type not supported");
    return NULL;
  }

  if (! PyArray_ISONESEGMENT(src_ndarray)) {
    PyErr_SetString(PyExc_RuntimeError,"only single-segment ndarray supported");
    return NULL;
  }

  size_t size = 0;

  /* number of dimensions */
  int ndims = PyArray_NDIM(src_ndarray);
  size += sizeof(int);

  /* item size */
  size += sizeof(npy_intp);

  /* dimensions and strides */
  size += (sizeof(npy_intp) * ndims) * 2;

  /* selected flags and type */
  size += sizeof(int) * 2;
  return PyLong_FromUnsignedLong(size);
}


PyObject * pymcas_ndarray_header(PyObject * self,
                                 PyObject * args,
                                 PyObject * kwargs)
{
  import_array();
  static const char *kwlist[] = {"source",
                                 "dtype_str",
                                 NULL};

  PyObject * src_obj = nullptr;
  const char * dtype_str = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "O|s",
                                    const_cast<char**>(kwlist),
                                    &src_obj,
                                    &dtype_str)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (! PyArray_Check(src_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"not ndarray type");
    return NULL;
  }

  PyArrayObject * src_ndarray = reinterpret_cast<PyArrayObject*>(src_obj);

  /* sanity checks */
  if (! PyArray_ISBEHAVED(src_ndarray)) {
    PyErr_SetString(PyExc_RuntimeError,"un-behaving ndarray type not supported");
    return NULL;
  }

  if (! PyArray_ISONESEGMENT(src_ndarray)) {
    PyErr_SetString(PyExc_RuntimeError,"only single-segment ndarray supported");
    return NULL;
  }

  std::string hdr;
  create_ndarray_header(src_ndarray, hdr, dtype_str);
  
  return PyByteArray_FromStringAndSize(hdr.c_str(), hdr.size());
}


PyObject * pymcas_ndarray_from_bytes(PyObject * self,
                                     PyObject * args,
                                     PyObject * kwargs)
{
  static const char *kwlist[] = {"data",
                                 "header_length",
                                 NULL};

  PyObject * bytes_memory_view  = nullptr;
  Py_ssize_t header_length = 0;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "On",
                                    const_cast<char**>(kwlist),
                                    &bytes_memory_view,
                                    &header_length)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments!");
    return NULL;
  }

  if (! PyMemoryView_Check(bytes_memory_view)) {
    PyErr_SetString(PyExc_RuntimeError,"data should be type <memoryview>");
    return NULL;
  }

  Py_INCREF(bytes_memory_view); /* increment reference count to hold data */

  Py_buffer * buffer = PyMemoryView_GET_BUFFER(bytes_memory_view);
  byte * ptr = (byte *) buffer->buf;

  if(global::debug_level > 1) {
    PLOG("header_length = %lu", header_length);
    hexdump(ptr, header_length);
  }

  /* unmarsh the ndarray from metadata - zero copy */
  auto nparray = unmarshall_nparray(ptr);
  
  return nparray;
}


/** 
 * Create header for array, return as string
 * 
 */
void create_ndarray_header(PyArrayObject * src_ndarray, std::string& out_hdr, const char * dtype_str)
{
  std::stringstream hdr;

  /* number of dimensions */
  int ndims = PyArray_NDIM(src_ndarray);
  hdr.write(reinterpret_cast<const char*>(&ndims), sizeof(ndims));

  /* item size */
  npy_intp item_size = PyArray_ITEMSIZE(src_ndarray);
  hdr.write(reinterpret_cast<const char*>(&item_size), sizeof(item_size));

  /* dimensions */
  npy_intp * dims = PyArray_DIMS(src_ndarray);
  hdr.write(reinterpret_cast<const char*>(dims), sizeof(npy_intp) * ndims);

  /* strides */
  npy_intp * strides = PyArray_STRIDES(src_ndarray);
  hdr.write(reinterpret_cast<const char*>(strides), sizeof(npy_intp) * ndims);

  if(global::debug_level > 1) {
    PLOG("saving ndims=%d", ndims);
    for(int d=0; d < ndims; d++)
      PLOG("dim=%ld", dims[d]);
  }

  /* selected flags */
  int select_flags =
    PyArray_IS_C_CONTIGUOUS(src_ndarray) |
    PyArray_IS_F_CONTIGUOUS(src_ndarray) |
    PyArray_ISALIGNED(src_ndarray);

  hdr.write(reinterpret_cast<const char*>(&select_flags), sizeof(select_flags));

  /* dtype */
  if(dtype_str) {
    std::string dtype(dtype_str);
    unsigned short dtype_marker = 0xFFFF;
    size_t dtype_len = dtype.size();
    hdr.write(reinterpret_cast<const char*>(&dtype_marker), sizeof(dtype_marker));
    hdr.write(reinterpret_cast<const char*>(&dtype_len), sizeof(dtype_len));
    hdr.write(dtype.c_str(), dtype_len);
  }
  else { /* for backwards compatibility with pymcas */
    int type = PyArray_TYPE(src_ndarray);
    hdr.write(reinterpret_cast<const char*>(&type), sizeof(type));
  }

  out_hdr = hdr.str();

  if(global::debug_level > 1) {
    PLOG("ndarray with metadata header:");
    hexdump(out_hdr.c_str(), out_hdr.length());
  }
}


PyObject * unmarshall_nparray(byte * ptr)
{
  import_array();
  
  int ndims = *(reinterpret_cast<int*>(ptr));
  ptr += sizeof(ndims);

  npy_intp item_size = *(reinterpret_cast<npy_intp*>(ptr));
  ptr += sizeof(item_size);

  std::vector<npy_intp> dims;
  for(int i=0; i < ndims; i++) {
    npy_intp dim = *(reinterpret_cast<npy_intp*>(ptr));
    ptr += sizeof(dim);
    dims.push_back(dim);
  }

  std::vector<npy_intp> strides;
  for(int i=0; i < ndims; i++) {
    npy_intp stride = *(reinterpret_cast<npy_intp*>(ptr));
    ptr += sizeof(stride);
    strides.push_back(stride);
  }

  int flags = *(reinterpret_cast<int*>(ptr));
  ptr += sizeof(flags);

  assert(flags == 1);
  
  int type =  *(reinterpret_cast<int*>(ptr));
  ptr += sizeof(type);

  if(global::debug_level > 2) {
    PLOG("ndims=%d, flags=%d, type=%d", ndims, flags, type);
    for(auto d: dims) PLOG("dim=%ld", d);
    for(auto s: strides) PLOG("stride=%ld", s);
  }

  PyObject* nparray = PyArray_New(&PyArray_Type,
                                  ndims,
                                  dims.data(),
                                  type,
                                  strides.data(),
                                  ptr, // TO DO check with header length?
                                  boost::numeric_cast<int>(item_size),
                                  flags,
                                  NULL);
  return nparray;
}


PyObject * pymcas_ndarray_read_header(PyObject * self,
                                      PyObject * args,
                                      PyObject * kwargs)
{
  import_array();
  
  static const char *kwlist[] = {"data",
                                 NULL};

  PyObject * bytes_memory_view  = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "O",
                                    const_cast<char**>(kwlist),
                                    &bytes_memory_view)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments!");
    return NULL;
  }

  if (! PyMemoryView_Check(bytes_memory_view)) {
    PyErr_SetString(PyExc_RuntimeError,"data should be type <memoryview>");
    return NULL;
  }

  Py_buffer * buffer = PyMemoryView_GET_BUFFER(bytes_memory_view);
  byte * ptr = (byte *) buffer->buf;

  int ndims = *(reinterpret_cast<int*>(ptr));
  ptr += sizeof(ndims);

  npy_intp item_size = *(reinterpret_cast<npy_intp*>(ptr));
  ptr += sizeof(item_size);

  std::vector<npy_intp> dims;
  for(int i=0; i < ndims; i++) {
    npy_intp dim = *(reinterpret_cast<npy_intp*>(ptr));
    ptr += sizeof(dim);
    dims.push_back(dim);
  }

  std::vector<npy_intp> strides;
  for(int i=0; i < ndims; i++) {
    npy_intp stride = *(reinterpret_cast<npy_intp*>(ptr));
    ptr += sizeof(stride);
    strides.push_back(stride);
  }

  int flags = *(reinterpret_cast<int*>(ptr));
  ptr += sizeof(flags);

  assert(flags == 1);

  unsigned short dtype_marker = *(reinterpret_cast<unsigned short*>(ptr));
  ptr += sizeof(dtype_marker);

  /* pymcas does not use this yet */
  if(dtype_marker != 0xFFFF) throw General_exception("bad dtype marker");
  size_t dtype_str_len = *(reinterpret_cast<int*>(ptr));
  ptr += sizeof(dtype_str_len);
  std::string dtype(reinterpret_cast<const char*>(ptr), dtype_str_len);
  ptr += dtype_str_len;

  int type =  *(reinterpret_cast<int*>(ptr));
  ptr += sizeof(type);

  if(global::debug_level > 2) {
    PLOG("ndims=%d, flags=%d, type=%d", ndims, flags, type);
    for(auto d: dims) PLOG("dim=%ld", d);
    for(auto s: strides) PLOG("stride=%ld", s);
  }

  /* put attributes into a dictionary */
  auto dict = PyDict_New();
  PyDict_SetItemString(dict, "ndims", PyLong_FromLong(ndims));
  PyDict_SetItemString(dict, "item_size", PyLong_FromLong(item_size));

  auto dims_tuple = PyTuple_New(dims.size());
  Py_ssize_t i=0;
  for( auto d : dims ) {
    PyTuple_SetItem(dims_tuple, i, PyLong_FromLong(d));
    i++;
  }
  PyDict_SetItemString(dict, "shape", dims_tuple);

  auto strides_tuple = PyTuple_New(strides.size());
  i=0;
  for( auto d : strides ) {
    PyTuple_SetItem(strides_tuple, i, PyLong_FromLong(d));
    i++;
  }
  PyDict_SetItemString(dict, "strides", strides_tuple);


  PyDict_SetItemString(dict, "flags", PyLong_FromLong(flags));
  PyDict_SetItemString(dict, "dtype", PyUnicode_FromString(dtype.c_str()));

  return dict;
}

