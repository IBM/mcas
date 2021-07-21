#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <vector>
#include <string>
#include <sstream>

#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/rand.h> /* user-level PRNG */
#include <Python.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

#include "meta_generated.h"

PyObject * pymmcore_create_metadata(PyObject * self,
                                    PyObject * args,
                                    PyObject * kwargs)
{
  static const char *kwlist[] = {"buffer",
                                 "type",
                                 NULL};

  PyObject * memoryview_object = nullptr;
  PyObject * type = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "OO",
                                    const_cast<char**>(kwlist),
                                    &memoryview_object,
                                    &type)) {
     PyErr_SetString(PyExc_RuntimeError, "pymmcore_create_metadata unable to parse args");
     return -1;
  }

  if (! PyMemoryView_Check(memoryview_object)) {
    PyErr_SetString(PyExc_RuntimeError, "pymmcore_create_metadata ctor parameter is not a memory view");
     return -1;
  }

  Py_RETURN_NONE;
}

