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

#ifndef __NDARRAY_HELPERS_H__
#define __NDARRAY_HELPERS_H__

/** 
 * Extract NumPy ndarray metadata into byte array
 * 
 * @param self 
 * @param args 
 * @param kwargs 
 * 
 * @return Bytearray metadata header
 */
PyObject * pymcas_ndarray_header(PyObject * self,
                                 PyObject * args,
                                 PyObject * kwargs);

/** 
 * Get size of header for ndarray
 * 
 * @param self 
 * @param args 
 * @param kwargs 
 * 
 * @return Size of header in bytes
 */
PyObject * pymcas_ndarray_header_size(PyObject * self,
                                      PyObject * args,
                                      PyObject * kwargs);

/** 
 * Create an NumPy ndarray from existing memory
 * 
 * @param self 
 * @param args 
 * @param kwargs 
 * 
 * @return 
 */
PyObject * pymcas_ndarray_from_bytes(PyObject * self,
                                     PyObject * args,
                                     PyObject * kwargs);


/** 
 * Read ndarray meta data into Python dictionary
 * 
 * @param self 
 * @param args 
 * @param kwargs 
 * 
 * @return 
 */
PyObject * pymcas_ndarray_read_header(PyObject * self,
                                      PyObject * args,
                                      PyObject * kwargs);



#endif // __NDARRAY_HELPERS_H__
