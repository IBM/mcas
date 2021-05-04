A memory manager "plugin" is a lightweight shared library that can be
load at runtime. Unlike components, plugins do not have interface
management, reference counting etc.  They can be written in pure C.

Plugins can be used with the libmm interposition library.



