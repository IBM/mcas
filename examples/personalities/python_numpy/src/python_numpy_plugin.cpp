/*
  Copyright [2017-2019] [IBM Corporation]
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion-null"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wcast-qual"

#include <numpy/arrayobject.h>
#include <Python.h>

#include <libpmem.h>
#include <common/logging.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/cycles.h>
#include <api/interfaces.h>
#include <string.h>
#include "python_numpy_plugin.h"

ADO_python_numpy_plugin::ADO_python_numpy_plugin()
{
  Py_Initialize();

  PLOG("Python intialized");
}

ADO_python_numpy_plugin::~ADO_python_numpy_plugin()
{
  Py_Finalize();
}

status_t ADO_python_numpy_plugin::register_mapped_memory(void * shard_vaddr,
                                                         void * local_vaddr,
                                                         size_t len)
{
  PLOG("ADO_python_numpy_plugin: register_mapped_memory (%p, %p, %lu)",
       shard_vaddr, local_vaddr, len);
  /* we would need a mapping if we are not using the same virtual
     addresses as the Shard process */

  return S_OK;
}


status_t ADO_python_numpy_plugin::do_work(const uint64_t work_key,
                                          const char * key,
                                          size_t key_len,
                                          IADO_plugin::value_space_t& values,
                                          const void *in_work_request,
                                          const size_t in_work_request_len,
                                          bool new_root,
                                          response_buffer_vector_t& response_buffers)
{
  auto value = values[0].ptr;
  auto value_len = values[0].len;

  std::string wr(reinterpret_cast<const char*>(in_work_request), in_work_request_len);

  if(_debug_level > 2) {
    PLOG("key:%s value:%p value_len:%lu newroot=%s",
         key, value, value_len, new_root ? "y":"n");
    PLOG("work_request: (%s)", wr.c_str());
  }

  /* load pickle mod_pickle */
  PyObject *mod_pickle = PyImport_ImportModule("pickle");
  if(mod_pickle == nullptr)
    throw General_exception("unable to import pickle");

  PyObject * bytes_object = PyBytes_FromStringAndSize(reinterpret_cast<const char *>(value), value_len);
  if(!bytes_object)
    throw General_exception("unable to convert metadata to bytes object");

  PyObject * metadata = PyObject_CallMethodObjArgs(mod_pickle,
                                                   PyUnicode_FromString("loads"),
                                                   bytes_object,
                                                   nullptr);
  if(!metadata)
    throw General_exception("unable to unpickle metadata");

  if(!PyTuple_Check(metadata))
    throw General_exception("metadata is not a tuple?");

  auto shape = PyTuple_GetItem(metadata,0);
  auto ndims = PyTuple_Size(shape);
  npy_intp* dims = new npy_intp[ndims];
  PyObject * obj;
  for(Py_ssize_t pos = 0; pos < ndims; pos++) {
    obj = PyTuple_GetItem(shape, pos);
    assert(PyLong_Check(obj));
    dims[pos] = PyLong_AsLong(obj);
    Py_DECREF(obj);
  }
  Py_DECREF(shape);

  int type_num = (int) PyLong_AsLong(PyTuple_GetItem(metadata, 1));

  import_array ();

  /* retrieve the actual matrix data from MCAS */
  void * matrix_data = nullptr;
  size_t matrix_data_len = 0;
  std::string key_prefix(key, key_len);
  if(_cb.open_key(work_key, key_prefix + "-data", 0, matrix_data, matrix_data_len, nullptr, nullptr) != S_OK)
    throw General_exception("could not read matrix ");

  auto new_array = PyArray_SimpleNewFromData(ndims, dims, type_num, matrix_data);
  if(!new_array)
    throw General_exception("PyArray_SimpleNew failed");

  /* prepare array as global variable */
  auto local_dict = PyDict_New();
  auto global_dict = PyDict_New();
  PyDict_SetItemString(global_dict, "matrix", new_array);
  PyDict_SetItemString(global_dict, "__builtins__", PyEval_GetBuiltins());

  /* now run the supplied program/operation */
  wr = "import numpy as np\n" + wr; // + "\n";

  auto cc = Py_CompileString(wr.c_str(), "jitcode", Py_file_input);
  if(!cc)
    throw General_exception("unable to compile code");

  /* PyEval_EvalCode takes global and local variable dictionaries.
     We can get output back out of the execution through the local_dict.
  */
  PyObject* result = PyEval_EvalCode(cc, global_dict, local_dict);
  if(!result)
    throw General_exception("PyEval_EvalCode failed unexpectedly");

  /* get back the matrix object */
  auto post_op_matrix = PyDict_GetItemString(local_dict, "matrix");
  if(post_op_matrix) {

    if(!PyArray_Check(post_op_matrix))
      throw General_exception("don't support transform of matrix to non-ndarray type");

    /* Update key-value with new matrix */
    PWRN("TODO: Update key-value pair");

    /* this means that matrix has been reassigned */
    Py_DECREF(post_op_matrix);
  }
  /* otherwise, matrix was changed in-place */

  Py_DECREF(result);
  Py_DECREF(local_dict);
  Py_DECREF(global_dict);

  /* get meta data */
  return S_OK;
}

status_t ADO_python_numpy_plugin::shutdown()
{
  /* here you would put graceful shutdown code if any */
  return S_OK;
}

/**
 * Factory-less entry point
 *
 */
extern "C" void * factory_createInstance(component::uuid_t& interface_iid)
{
  if(interface_iid == interface::ado_plugin)
    return static_cast<void*>(new ADO_python_numpy_plugin());
  else return NULL;
}

#pragma GCC diagnostic pop
