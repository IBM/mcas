#define MCAS_API_VERSION "v0.3"
#define PAGE_SIZE 4096
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL mcas_ARRAY_API

#include <Python.h>
#include <structmember.h>
#include <objimpl.h>
#include <pythread.h>

#include <api/components.h>
#include <api/kvstore_itf.h>

#if defined(__linux__)
#include <execinfo.h>
#endif

#include <list>
#include <common/logging.h>

namespace global
{
unsigned debug_level = 0;
}

// forward declaration of custom types
//
extern PyTypeObject SessionType;
extern PyTypeObject PoolType;

PyDoc_STRVAR(mcas_version_doc,
             "version() -> Get module version");
PyDoc_STRVAR(mcas_allocate_direct_memory_doc,
             "allocate_direct_memory(s) -> Returns 4K page-aligned memory view (experimental)");
PyDoc_STRVAR(mcas_free_direct_memory_doc,
             "free_direct_memory(s) -> Free memory previously allocated with allocate_direct_memory (experimental)");


static PyObject * mcas_version(PyObject * self,
                               PyObject * args,
                               PyObject * kwargs);

static PyObject * mcas_allocate_direct_memory(PyObject * self,
                                              PyObject * args,
                                              PyObject * kwargs);

static PyObject * mcas_free_direct_memory(PyObject * self,
                                          PyObject * args,
                                          PyObject * kwargs);


static PyMethodDef mcas_methods[] =
  {
   {"version",
    (PyCFunction) mcas_version, METH_NOARGS, mcas_version_doc },
   {"allocate_direct_memory",
    (PyCFunction) mcas_allocate_direct_memory, METH_VARARGS | METH_KEYWORDS, mcas_allocate_direct_memory_doc },
   {"free_direct_memory",
    (PyCFunction) mcas_free_direct_memory, METH_VARARGS | METH_KEYWORDS, mcas_free_direct_memory_doc },
   {NULL, NULL, 0, NULL}        /* Sentinel */
  };


static PyModuleDef mcas_module = {
    PyModuleDef_HEAD_INIT,
    "mcas",
    "MCAS client API module",
    -1,
    mcas_methods,
    NULL, NULL, NULL, NULL
};

                                     
PyMODINIT_FUNC
PyInit_mcas(void)
{  
  PyObject *m;

  PLOG("Init mcas Python extension");

  SessionType.tp_base = 0; // no inheritance
  if(PyType_Ready(&SessionType) < 0) {
    assert(0);
    return NULL;
  }

  PoolType.tp_base = 0; // no inheritance
  if(PyType_Ready(&PoolType) < 0) {
    assert(0);
    return NULL;
  }


  /* register module */
#if PY_MAJOR_VERSION >= 3
  m = PyModule_Create(&mcas_module);
#else
#error "Extension for Python 3 only."
#endif

  if (m == NULL)
    return NULL;

  /* add types */
  int rc;

  Py_INCREF(&SessionType);
  rc = PyModule_AddObject(m, "Session", (PyObject *) &SessionType);
  if(rc) return NULL;
  
  Py_INCREF(&PoolType);
  rc = PyModule_AddObject(m, "Pool", (PyObject *) &PoolType);
  if(rc) return NULL;


  return m;
}

/** 
 * Allocated memory view of aligned memory.  This memory
 * will not be garbage collected and should be explicitly 
 * freed
 * 
 * @param self 
 * @param args: size(size in bytes to allocate), zero(zero memory)
 * @param kwds 
 * 
 * @return memoryview object
 */
static PyObject * mcas_allocate_direct_memory(PyObject * self,
                                              PyObject * args,
                                              PyObject * kwds)
{
  static const char *kwlist[] = {"size",
                                 "zero",
                                 NULL};

  unsigned long nsize = 0;
  int zero_flag = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "k|p",
                                    const_cast<char**>(kwlist),
                                    &nsize,
                                    &zero_flag)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }
  
  char * ptr = static_cast<char*>(aligned_alloc(PAGE_SIZE, nsize));

  if(zero_flag)
    memset(ptr, 0x0, nsize);
  
  if(ptr == NULL) {
    PyErr_SetString(PyExc_RuntimeError,"aligned_alloc failed");
    return NULL;
  }
  
  //  PNOTICE("%s allocated %lu at %p", __func__, nsize, ptr);
  return PyMemoryView_FromMemory(ptr, nsize, PyBUF_WRITE);
}



/** 
 * Free direct memory allocated with allocate_direct_memory
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return 
 */
static PyObject * mcas_free_direct_memory(PyObject * self,
                                          PyObject * args,
                                          PyObject * kwds)
{
  static const char *kwlist[] = {"memory",
                                 NULL};

  PyObject * memview = NULL;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "O",
                                    const_cast<char**>(kwlist),
                                    &memview)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (PyMemoryView_Check(memview) == 0) {
    PyErr_SetString(PyExc_RuntimeError,"argument should be memoryview type");
    return NULL;
  }
  
  Py_buffer * buffer = PyMemoryView_GET_BUFFER(memview);
  buffer->len = 0;
  PyBuffer_Release(buffer);
  Py_RETURN_NONE;
}

static PyObject * mcas_version(PyObject * self,
                               PyObject * args,
                               PyObject * kwds)
{
  return PyUnicode_FromString(MCAS_API_VERSION);
}
