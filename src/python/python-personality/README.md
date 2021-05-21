# Experimental Python Personality

**This feature is still in testing.**

Tested with Python 3.6 and beyond.

The Python Personality (PP) provides: i.) a client API to load and
save Python objects from the MCAS store, ii.) support for Python execution
in the ADO.  In the case of Numpy
arrays, zero-copy is possible (through the direct APIs).  For other
data structures, the standard pickling function is used.

PP does not yet support in-place modification of objects in the MCAS store.

## Running basic test

This test will produce an X-window and thus cannot be fully run from a text
only console.  Code for test is in api.py:

```
SERVER_IP=<your server IP address> python3
```

Then type interactively:

```python3
> import pymcas
> pymcas.test_skimage_0()
```


## Example

This example (from api.py) shows how to load and save python objects
both from the client and from the ADO plugin.

```python3
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
    pool = session.create_pool("myPool",int(1e9),100)
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
```
