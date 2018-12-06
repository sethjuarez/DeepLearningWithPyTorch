import os
import time
import tensorflow as tf
from functools import wraps
from inspect import getargspec
# pylint: disable-msg=E0611
from tensorflow.python.tools import freeze_graph as freeze
# pylint: enable-msg=E0611

def info(msg, char = "#", width = 75):
    print("")
    print(char * width)
    print(char + "   %0*s" % ((-1*width)+5, msg) + char)
    print(char * width)

def print_info(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        info('-> {}'.format(f.__name__))
        print('Parameters:')
        ps = list(zip(getargspec(f).args, args))
        width = max(len(x[0]) for x in ps) + 1
        for t in ps:
            items = str(t[1]).split('\n')
            print('   {0:<{w}} ->  {1}'.format(t[0], items[0], w=width))
            for i in range(len(items) - 1):
                print('   {0:<{w}}       {1}'.format(' ', items[i+1], w=width))
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print('\n -- Elapsed {0:.4f}s\n'.format(te-ts))
        return result
    return wrapper

def print_args(args):
    info('Arguments')
    ps = args.__dict__.items()
    width = max(len(k) for k, _ in ps) + 1
    for k, v in ps:
        print('   {0:<{w}} ->  {1}'.format(k, str(v), w=width))