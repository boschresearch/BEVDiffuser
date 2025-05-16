# This source code is from BEVFormer
#   (https://github.com/fundamentalvision/BEVFormer)
# Copyright (c) 2022 BEVFormer authors, licensed under the Apache-2.0 license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import functools
import time
from collections import defaultdict
import torch
time_maps = defaultdict(lambda :0.)
count_maps = defaultdict(lambda :0.)
def run_time(name):
    def middle(fn):
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.time()
            res = fn(*args, **kwargs)
            torch.cuda.synchronize()
            time_maps['%s : %s'%(name, fn.__name__) ] += time.time()-start
            count_maps['%s : %s'%(name, fn.__name__) ] +=1
            print("%s : %s takes up %f "% (name, fn.__name__,time_maps['%s : %s'%(name, fn.__name__) ] /count_maps['%s : %s'%(name, fn.__name__) ] ))
            return res
        return wrapper
    return middle
    