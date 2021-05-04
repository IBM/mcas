
# Basic Use

Use LD_PRELOAD to overload symbols:

LD_PRELOAD=dist/lib/libmm.so src/lib/libmm/mmwrapper-test-prog

Add LD_DEBUG=all to check out loading sequence.

# Debugging with GDB

```
gdb -ex 'set env LD_PRELOAD dist/lib/libmm.so' src/lib/libmm/mmwrapper-test-prog
```

# Debugging with Core file

Run program.

```
coredumpctl list
coredumpctl dump <pid> --output core
gdb src/lib/libmm/mmwrapper-test-prog core
bt
```
