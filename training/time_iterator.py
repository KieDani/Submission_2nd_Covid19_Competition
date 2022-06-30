import time

def itertime(iterator):
    t0 = time.time()
    for item in iterator:
        t1 = time.time()
        yield (t1 - t0), item
        t0 = time.time()
