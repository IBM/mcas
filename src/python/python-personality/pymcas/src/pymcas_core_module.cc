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

#define PYMCAS_API_VERSION "v0.1alpha"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL pymcascore_ARRAY_API

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wconversion"

#include <numpy/arrayobject.h>
#include <Python.h>
#include <structmember.h>
#include <object.h>
#include <objimpl.h>
#include <sstream>
#include <vector>

#include <common/types.h>
#include <common/dump_utils.h>
#include <common/logging.h>

#if defined(__linux__)
#include <execinfo.h>
#endif

#include "ndarray_helpers.h"

/* documentation strings */
PyDoc_STRVAR(pymcas_version_doc,
             "version() -> Get module version");
PyDoc_STRVAR(pymcas_ndarray_header_doc,
             "ndarray_header(array_value) -> Get metadata header for Numpy array");
PyDoc_STRVAR(pymcas_ndarray_from_bytes_doc,
             "ndarray_from_bytes(header+payload) -> Zero-copy create NumpyArray from existing bytearray");


/** 
 * Get API version number
 * 
 * @return String version
 */
static PyObject * pymcas_version(PyObject * self,
                                 PyObject * args,
                                 PyObject * kwargs);


static PyMethodDef pymcas_methods[] =
  {
   {"version", (PyCFunction) pymcas_version, METH_NOARGS, pymcas_version_doc },
   {"ndarray_header", (PyCFunction) pymcas_ndarray_header,
    METH_VARARGS | METH_KEYWORDS, pymcas_ndarray_header_doc },
   {"ndarray_from_bytes", (PyCFunction) pymcas_ndarray_from_bytes,
    METH_VARARGS | METH_KEYWORDS, pymcas_ndarray_from_bytes_doc },
   {NULL, NULL, 0, NULL}        /* sentinel */
  };


static PyModuleDef pymcas_module = {
                                    PyModuleDef_HEAD_INIT,
                                    "pymcascore",
                                    "PyMCAS core functions",
                                    -1,
                                    pymcas_methods,
                                    NULL, NULL, NULL, NULL
};

                                     
PyMODINIT_FUNC
PyInit_pymcascore(void)
{  
  PyObject *m;

  PLOG("Init PyMCAS Python extension");

  /* imports */
  import_array();

  /* register module */
#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&pymcas_module);
#else
#error "Extension for Python 3 only."
#endif

  return m;
}


static PyObject * pymcas_version(PyObject * self,
                                 PyObject * args,
                                 PyObject * kwds)
{
  return PyUnicode_FromString(PYMCAS_API_VERSION);
}


#pragma GCC diagnostic pop
