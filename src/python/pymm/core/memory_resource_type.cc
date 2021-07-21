#include <assert.h>
#include <stdlib.h>
#include <common/logging.h>
#include <common/utils.h>
#include <api/mcas_itf.h>
#include <api/kvstore_itf.h>
#include <Python.h>
#include <structmember.h>
#include <libpmem.h>

#include "pymm_config.h"

/* defaults */
constexpr const char * DEFAULT_PMEM_PATH = "/mnt/pmem0";
constexpr const char * DEFAULT_POOL_NAME = "default";
constexpr const char * DEFAULT_BACKEND = "hstore-cc";
constexpr uint64_t DEFAULT_LOAD_ADDR = 0x900000000;

using namespace component;

/* Python type */
typedef struct {
  PyObject_HEAD
  IKVStore *       _store;
  IKVStore::pool_t _pool;
} MemoryResource;

/** 
 * Class to manage backend instance.  You can't open the same resource
 * multiple times, so you have to share.
 * 
 */
class Backend_instance_manager
{
public:
  IKVStore * get(const std::string backend,
                 const std::string path,                 
                 const addr_t load_addr,
                 const unsigned debug_level,
                 const std::string mm_plugin_path)
  {
    PLOG("Backend_instance_mgr: (%s) (%s) (%s)", backend.c_str(), path.c_str(), mm_plugin_path.c_str());
    auto iter = _map.find(backend);
    std::string key = path + mm_plugin_path;
    
    if((iter == _map.end()) || (_map[backend].find(key) == _map[backend].end()))   {
      /* create new entry */
      IKVStore * itf = load_backend(backend, path, mm_plugin_path, load_addr, debug_level);
      assert(itf);
      _map[backend][key] = itf;
      itf->add_ref(); /* extra ref to hold it open */
      return itf;
    }
    else {
      PLOG("Matching backend exists!");
    }

    auto itf = _map[backend][key];
    assert(itf);
    itf->add_ref();
    return itf;
  }
  
private:
  IKVStore * load_backend(const std::string& backend,
                          const std::string& path,
                          const std::string& mm_plugin_path,
                          const addr_t load_addr,
                          const unsigned debug_level);

  using backend_t = std::string;
  using path_t = std::string;
  
  std::map<backend_t, std::map<path_t, IKVStore*>> _map;
  
} g_store_map;




static PyObject *
MemoryResource_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  auto self = (MemoryResource *)type->tp_alloc(type, 0);
  assert(self);
  return (PyObject*)self;
}

/** 
 * tp_dealloc: Called when reference count is 0
 * 
 * @param self 
 */
static void
MemoryResource_dealloc(MemoryResource *self)
{
  assert(self);

  if(self->_store) {
    if(self->_pool)
      self->_store->close_pool(self->_pool);

    self->_store->release_ref();
  }

  if(globals::debug_level > 0)
    PLOG("MemoryResource: closed (%p)", self);
  
  assert(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

IKVStore * Backend_instance_manager::load_backend(const std::string& backend,
                                                  const std::string& path,
                                                  const std::string& mm_plugin_path,
                                                  const uint64_t load_addr,
                                                  const unsigned debug_level)
{
  PLOG("load_backend: (%s) (%s) (%s)", backend.c_str(), path.c_str(), mm_plugin_path.c_str());
  IBase* comp = nullptr;
  if(backend == "hstore") {
    comp = load_component("libcomponent-hstore.so", hstore_factory);
  }
  else if (backend == "hstore-cc") {
    comp = load_component("libcomponent-hstore-cc.so", hstore_factory);
  }
  else if (backend == "hstore-mc") {
    comp = load_component("libcomponent-hstore-mc.so", hstore_factory);
  }
  else if (backend == "hstore-mr") {
    comp = load_component("libcomponent-hstore-mr.so", hstore_factory);
  }    
  else if (backend == "mapstore") {
    comp = load_component("libcomponent-mapstore.so", mapstore_factory);
  }
  else {
    PNOTICE("invalid backend (%s)", backend.c_str());
    return nullptr;
  }

  /* try adding default path if needed */
  std::string checked_mm_plugin_path = mm_plugin_path;
  if(checked_mm_plugin_path != "") /* empty means use default */
  {   
    std::string path = mm_plugin_path;
    if(access(path.c_str(), F_OK) != 0) { /* is not accessible */
        path = LIB_INSTALL_PATH + path;
        if(access(path.c_str(), F_OK) != 0) {
          PERR("inaccessible plugin path (%s) and (%s)", mm_plugin_path.c_str(), path.c_str());
          throw General_exception("unable to open mm_plugin");
        }
        checked_mm_plugin_path = path;
    }
  }


  IKVStore* store = nullptr;
  auto fact = make_itf_ref(static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid())));
  assert(fact);  

  if(backend == "hstore-mc" || backend == "hstore-mr") {
    
    std::stringstream ss;
    ss << "[{\"path\":\"" << path << "\",\"addr\":" << load_addr << "}]";
    PLOG("dax config: %s", ss.str().c_str());
    
    store = fact->create(debug_level,
                         {
                          {+component::IKVStore_factory::k_debug, std::to_string(debug_level)},
                          {+component::IKVStore_factory::k_dax_config, ss.str()},
                          {+component::IKVStore_factory::k_mm_plugin_path, checked_mm_plugin_path}
                         });
  }
  else if(backend == "hstore" || backend == "hstore-cc") {

    std::stringstream ss;
    ss << "[{\"path\":\"" << path << "\",\"addr\":" << load_addr << "}]";
    PLOG("dax config: %s", ss.str().c_str());
    
    store = fact->create(debug_level,
                         {
                          {+component::IKVStore_factory::k_debug, std::to_string(debug_level)},
                          {+component::IKVStore_factory::k_dax_config, ss.str()}
                         });
  }
  else {
    /* mapstore */
    if(mm_plugin_path == "") {
      /* use default plugin */
      store = fact->create(debug_level,
                         {
                          {+component::IKVStore_factory::k_debug, std::to_string(debug_level)},
                         });
    }
    else {
      store = fact->create(debug_level,
                         {
                          {+component::IKVStore_factory::k_debug, std::to_string(debug_level)},
                          {+component::IKVStore_factory::k_mm_plugin_path, checked_mm_plugin_path}
                         });      
    }
  }
  assert(store);   
  return store;
}

static int MemoryResource_init(MemoryResource *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"pool_name",
                                 "size_mb",
                                 "pmem_path",
                                 "load_addr",
                                 "backend",
                                 "mm_plugin",
                                 "force_new",
                                 NULL,
  };

  char * p_pool_name = nullptr;
  uint64_t size_mb = 32;
  char * p_path = nullptr;
  char * p_addr = nullptr;
  PyObject * p_backend = nullptr; /* None to use default */
  PyObject * p_mm_plugin = nullptr;
  int force_new = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s|nssOOp",
                                    const_cast<char**>(kwlist),
                                    &p_pool_name,
                                    &size_mb,
                                    &p_path,
                                    &p_addr,
                                    &p_backend,
                                    &p_mm_plugin,
                                    &force_new)) {
    PyErr_SetString(PyExc_RuntimeError, "bad arguments");
    PWRN("bad arguments or argument types to MemoryResource constructor");
    return -1;
  }

  uint64_t load_addr = DEFAULT_LOAD_ADDR;
  if(p_addr)
    load_addr = ::strtoul(p_addr,NULL,16);
  
  const std::string pool_name = p_pool_name ? p_pool_name : DEFAULT_POOL_NAME;
  const std::string path = p_path ? p_path : DEFAULT_PMEM_PATH;
  const std::string backend = (!p_backend || p_backend == Py_None || !PyUnicode_Check(p_backend)) ? DEFAULT_BACKEND : PyUnicode_AsUTF8(p_backend);
  const std::string mm_plugin = (!p_mm_plugin || p_mm_plugin == Py_None || !PyUnicode_Check(p_mm_plugin)) ? "" : PyUnicode_AsUTF8(p_mm_plugin);

  self->_store = g_store_map.get(backend, path, load_addr, globals::debug_level, mm_plugin);
  
  assert(self->_store);

  if(force_new) {
    if(globals::debug_level > 0)
      PLOG("forcing new.");

    try {
      self->_pool = self->_store->delete_pool(pool_name);
    }
    catch(...){}
  }

  if((self->_pool = self->_store->create_pool(pool_name, MiB(size_mb))) == 0) {
    PyErr_SetString(PyExc_RuntimeError, "unable to create/open pool");
    return -1;
  }
  
  return 0;
}

/** 
 * MemoryResource_create_named_memory - create a KV pair
 * from the backend store (fail if already existing)
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return Tuple (key_handle, memoryview)
 */
static PyObject * MemoryResource_create_named_memory(PyObject * self,
                                                     PyObject * args,
                                                     PyObject * kwds)
{
  static const char *kwlist[] = {"name",
                                 "size",
                                 "alignment",
                                 "zero",
                                 NULL};

  char * name = nullptr;
  size_t alignment = 0;
  size_t size = 0;
  int zero = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "sk|kp",
                                    const_cast<char**>(kwlist),
                                    &name,
                                    &size,
                                    &alignment,
                                    &zero)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (alignment > size) {
    PyErr_SetString(PyExc_RuntimeError,"alignment greater than size");
    return NULL;
  }

  if (strlen(name) < 1) {
    PyErr_SetString(PyExc_RuntimeError,"bad name argument");
    return NULL;
  }

  auto mr = reinterpret_cast<MemoryResource *>(self);

  /* get memory */
  void * ptr = nullptr;
  IKVStore::key_t key_handle;

  status_t s;

  s = mr->_store->lock(mr->_pool,
                       name,
                       IKVStore::STORE_LOCK_WRITE,
                       ptr,
                       size,
                       alignment,
                       key_handle);

  if (s == S_OK) {
    mr->_store->unlock(mr->_pool, key_handle);
    PyErr_SetString(PyExc_RuntimeError,"named memory already exists");
    return NULL;
  }
  
  if (s == E_LOCKED) {
    PyErr_SetString(PyExc_RuntimeError,"variable already assigned");
    return NULL;
  } 

  /* optionally zero memory */
  if(zero)
    ::memset(ptr, 0x0, size);
  
  /* build a tuple (memory view, memory handle) */
  auto mview = PyMemoryView_FromMemory(static_cast<char*>(ptr), size, PyBUF_WRITE);
  auto tuple = PyTuple_New(2);
  PyTuple_SetItem(tuple, 0, PyLong_FromVoidPtr(key_handle));
  PyTuple_SetItem(tuple, 1, mview);
  return tuple;
}



/** 
 * Open an existing named memory resource
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return None if there is no corresponding named memory
 */
static PyObject * MemoryResource_open_named_memory(PyObject * self,
                                                   PyObject * args,
                                                   PyObject * kwds)
{
  static const char *kwlist[] = {"name",
                                 NULL};

  char * name = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &name)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  auto mr = reinterpret_cast<MemoryResource *>(self);

  /* get memory */
  void * ptr = nullptr;
  IKVStore::key_t key_handle;
  size_t value_len = 0;
  
  status_t s;

  {
    std::vector<uint64_t> v;
    std::string key = name;
    s = mr->_store->get_attribute(mr->_pool, IKVStore::Attribute::VALUE_LEN, v, &key);

    if(s != S_OK) {
      auto tuple = PyTuple_New(2);
      PyTuple_SetItem(tuple, 0, Py_None);
      PyTuple_SetItem(tuple, 1, Py_None);

      return tuple;
    }
  }

  s = mr->_store->lock(mr->_pool,
                       name,
                       IKVStore::STORE_LOCK_WRITE,
                       ptr,
                       value_len,
                       0, // alignment
                       key_handle);

  if (s == E_LOCKED) {
    PyErr_SetString(PyExc_RuntimeError,"named memory already open");
    return NULL;
  }

  if (s != S_OK) {
    PyErr_SetString(PyExc_RuntimeError,"failed to lock KV pair in open");
    return NULL;
  }
  /* build a tuple (memory view, memory handle) */
  auto mview = PyMemoryView_FromMemory(static_cast<char*>(ptr), value_len, PyBUF_WRITE);
  auto tuple = PyTuple_New(2);
  PyTuple_SetItem(tuple, 0, PyLong_FromVoidPtr(key_handle));
  PyTuple_SetItem(tuple, 1, mview);
  return tuple;
}



/** 
 * Release a previously locked KV pair on the backend store
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return None
 */
static PyObject * MemoryResource_release_named_memory(PyObject * self,
                                                      PyObject * args,
                                                      PyObject * kwds)
{
  static const char *kwlist[] = {"lock_handle",
                                 NULL};

  unsigned long handle = 0;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "k",
                                    const_cast<char**>(kwlist),
                                    &handle)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  auto mr = reinterpret_cast<MemoryResource *>(self);
  auto key_handle = reinterpret_cast<IKVStore::key_t>(handle);

  status_t s;

  s = mr->_store->unlock(mr->_pool, key_handle);

  if (s != S_OK) {
    PyErr_SetString(PyExc_RuntimeError,"unlock failed unexpectedly");
    return NULL;
  }

  Py_RETURN_NONE;
}


/** 
 * Erase named memory object from store
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return 
 */
static PyObject * MemoryResource_erase_named_memory(PyObject * self,
                                                    PyObject * args,
                                                    PyObject * kwds)
{
  static const char *kwlist[] = {"name",
                                 NULL};

  char * name = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &name)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  auto mr = reinterpret_cast<MemoryResource *>(self);
  
  auto status = mr->_store->erase(mr->_pool, name);
  if(status != S_OK) {
    PyErr_SetString(PyExc_RuntimeError,"erase failed unexpectedly");
    return NULL;
  }

  /* optional erase of split-value */
  std::string vname = name;
  vname += "-value";
  mr->_store->erase(mr->_pool, vname.c_str());

  return PyLong_FromLong(0);
}

/** 
 * Copy-based crash-consistent put of a value
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return 
 */
static PyObject * MemoryResource_put_named_memory(MemoryResource *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"name",
                                 "data",
                                 NULL};

  char * name = nullptr;
  PyObject * data = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "sO",
                                    const_cast<char**>(kwlist),
                                    &name,
                                    &data)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (! PyByteArray_Check(data)) {
    PyErr_SetString(PyExc_RuntimeError,"data parameter should be bytearray type");
    return NULL;
  }

  auto data_len = PyByteArray_Size(data);
  if(data_len == 0) {
    PyErr_SetString(PyExc_RuntimeError,"data bytearray should not be empty");
    return NULL;
  }
    
  auto data_ptr = PyByteArray_AsString(data);
  assert(data_ptr);
  
  auto mr = reinterpret_cast<MemoryResource *>(self);

  /* overwrites existing values */
  /* we might have to create another and swap keys - then we have a backup
     to tie to the metadata+value composite transaction */
  status_t s = mr->_store->put(mr->_pool,
                               name,
                               data_ptr,
                               data_len);
  
  return PyLong_FromLong(s);
}


static PyObject * MemoryResource_get_named_memory(MemoryResource *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"name",
                                 NULL};

  char * name = nullptr;
  
  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "s",
                                    const_cast<char**>(kwlist),
                                    &name)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  void * data_ptr = nullptr;
  size_t data_len = 0;
  auto mr = reinterpret_cast<MemoryResource *>(self);

  status_t s = mr->_store->get(mr->_pool,
                               name,
                               data_ptr,
                               data_len);

  if(s != S_OK) {
    Py_RETURN_NONE;
  }


  auto result = PyByteArray_FromStringAndSize(static_cast<const char *>(data_ptr), data_len);
  //  ::free(data_ptr);
  
  return result;
}

/** 
 * Persist a memory view with libpmem
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return 
 */
static PyObject * MemoryResource_persist_memory_view(MemoryResource *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"memoryview",
                                 NULL,
  };

  PyObject * mview = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "O",
                                    const_cast<char**>(kwlist),
                                    &mview)) {
    PyErr_SetString(PyExc_RuntimeError, "bad arguments");
    return NULL;
  }

  if (! PyMemoryView_Check(mview)) {
    PyErr_SetString(PyExc_RuntimeError, "bad arguments");
    return NULL;
  }

  Py_buffer * pybuffer = PyMemoryView_GET_BUFFER(mview);

  PNOTICE("persisting %p %lu", pybuffer->buf, pybuffer->len);
  pmem_persist(pybuffer->buf, pybuffer->len); /* part of libpmem */
  
  return PyLong_FromLong(0);
}

/** 
 * Get list of named memory items (keys) belonging to resource (pool)
 * 
 * @param self 
 * @param args 
 * 
 * @return List of names
 */
static PyObject * MemoryResource_get_named_memory_list(MemoryResource *self, PyObject *args)
{
  auto mr = reinterpret_cast<MemoryResource *>(self);

  auto pool = mr->_pool;
  auto pi = mr->_store->open_pool_iterator(pool);

  status_t s = S_OK;
  PyObject* result = PyList_New(0);
  
  do {
    bool time_match;
    KVStore::pool_reference_t pool_ref;

    s = mr->_store->deref_pool_iterator(pool, pi, 0, 0, pool_ref, time_match, true);

    if(s != S_OK) break;

    PyObject * strobj = PyUnicode_FromString(pool_ref.get_key().c_str());
    if(PyList_Append(result, strobj))
      throw General_exception("PyList_Append failed unexpectedly");    

  }
  while(s == S_OK);

  mr->_store->close_pool_iterator(pool, pi);
  
  return result;
}


/** 
 * Get percent used of memory resource
 * 
 * @param self 
 * @param args 
 * 
 * @return Percent used
 */
static PyObject * MemoryResource_get_percent_used(MemoryResource *self, PyObject *args)
{
  auto mr = reinterpret_cast<MemoryResource *>(self);

  auto pool = mr->_pool;

  std::vector<uint64_t> value;
  auto status = mr->_store->get_attribute(pool, IKVStore::Attribute::PERCENT_USED, value);
  if(status != S_OK) {
    PyErr_SetString(PyExc_RuntimeError,"get attribute percent used failed");
    return NULL;
  }
  assert(value.size() > 0);
  return PyLong_FromUnsignedLong(value[0]);
}

/** 
 * Atomically swap names of two memories
 * 
 * @param self 
 * @param args 
 * @param kwds 
 * 
 * @return 
 */
static PyObject * MemoryResource_atomic_swap_names(MemoryResource *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"left",
                                 "right",
                                 NULL,
  };

  const char * left_name = nullptr;
  const char * right_name = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "ss",
                                    const_cast<char**>(kwlist),
                                    &left_name, &right_name)) {
    PyErr_SetString(PyExc_RuntimeError, "bad arguments");
    return NULL;
  }

  auto mr = reinterpret_cast<MemoryResource *>(self);
  auto pool = mr->_pool;
  status_t s = mr->_store->swap_keys(pool, left_name, right_name);
  if(s != S_OK)
    PWRN("swap_keys failed (%d)", s);
  return PyLong_FromLong(s);
}

static PyMemberDef MemoryResource_members[] =
  {
   //  {"port", T_ULONG, offsetof(MemoryResource, _port), READONLY, "Port"},
   {NULL}
  };


//MemoryResource_create_named_memory
static PyMethodDef MemoryResource_methods[] =
  {
   /* single prefix makes protected */
   {"_MemoryResource_create_named_memory",  (PyCFunction) MemoryResource_create_named_memory, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_create_named_memory(name,size,alignment,zero)"},
   {"_MemoryResource_open_named_memory",  (PyCFunction) MemoryResource_open_named_memory, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_open_named_memory(name,size,alignment,zero)"},   
   {"_MemoryResource_release_named_memory", (PyCFunction) MemoryResource_release_named_memory, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_release_named_memory(handle)"},
   {"_MemoryResource_erase_named_memory", (PyCFunction) MemoryResource_erase_named_memory, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_erase_named_memory(handle)"},
   {"_MemoryResource_persist_memory_view", (PyCFunction) MemoryResource_persist_memory_view, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_persist_memory_view(memview)"},
   {"_MemoryResource_put_named_memory", (PyCFunction) MemoryResource_put_named_memory, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_put_named_memory(data)"},
   {"_MemoryResource_get_named_memory", (PyCFunction) MemoryResource_get_named_memory, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_get_named_memory(data)"},
   {"_MemoryResource_atomic_swap_names", (PyCFunction) MemoryResource_atomic_swap_names, METH_VARARGS | METH_KEYWORDS,
    "MemoryResource_atomic_swap_name(a,b)"},
   {"_MemoryResource_get_named_memory_list", (PyCFunction) MemoryResource_get_named_memory_list, METH_NOARGS,
    "MemoryResource_get_named_memory_list()"},
   {"_MemoryResource_get_percent_used", (PyCFunction) MemoryResource_get_percent_used, METH_NOARGS,
    "MemoryResource_get_percent_used()"},   
   {NULL}
  };


PyTypeObject MemoryResourceType = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "pymm.MemoryResource",   /* tp_name */
  sizeof(MemoryResource),  /* tp_basicsize */
  0,                       /* tp_itemsize */
  (destructor) MemoryResource_dealloc,      /* tp_dealloc */
  0,                       /* tp_print */
  0,                       /* tp_getattr */
  0,                       /* tp_setattr */
  0,                       /* tp_reserved */
  0,                       /* tp_repr */
  0,                       /* tp_as_number */
  0,                       /* tp_as_sequence */
  0,                       /* tp_as_mapping */
  0,                       /* tp_hash */
  0,                       /* tp_call */
  0,                       /* tp_str */
  0,                       /* tp_getattro */
  0,                       /* tp_setattro */
  0,                       /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
  "MemoryResource",        /* tp_doc */
  0,                       /* tp_traverse */
  0,                       /* tp_clear */
  0,                       /* tp_richcompare */
  0,                       /* tp_weaklistoffset */
  0,                       /* tp_iter */
  0,                       /* tp_iternext */
  MemoryResource_methods,  /* tp_methods */
  MemoryResource_members,  /* tp_members */
  0,                       /* tp_getset */
  0,                       /* tp_base */
  0,                       /* tp_dict */
  0,                       /* tp_descr_get */
  0,                       /* tp_descr_set */
  0,                       /* tp_dictoffset */
  (initproc)MemoryResource_init,  /* tp_init */
  0,            /* tp_alloc */
  MemoryResource_new,             /* tp_new */
  0, /* tp_free */
};

