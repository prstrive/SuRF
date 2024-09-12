import lmdb
from plyfile import PlyData, PlyElement
import os
import numpy as np
import torch

DATA_DIR = "/amax/data/cas_training/PseudoMVSDepth"
SPLIT_PATH = "/home/pengr/Documents/SparseRecon/datasets/dtu_split/all.txt"
OUTPUT_ROOT = "/amax/data/cas_training/PseudoMVSDepth/lmdb"

with open(SPLIT_PATH) as f:
    scans = f.readlines()
    scenes = [line.rstrip() for line in scans]

def dump_and_compress(obj):
    import pickle
    import lz4.frame as lz4f

    stream = pickle.dumps(obj, -1)
    stream = lz4f.compress(stream)
    return stream


db_env = lmdb.open(OUTPUT_ROOT,
                       lock=False,
                       readahead=False,
                       map_size=1024 ** 4 * 64)

db_txn = db_env.begin(write=True)
keys_cache = []
count = 0

for scan in scenes:
    pcd = PlyData.read(os.path.join(DATA_DIR, "mvsnet{:0>3}_l3.ply".format(int(scan[4:]))))
    px = pcd['vertex']['x']
    py = pcd['vertex']['y']
    pz = pcd['vertex']['z']
    pxyz_ori = np.stack([px, py, pz], axis=1)
    idxs = np.arange(pxyz_ori.shape[0])
    np.random.shuffle(idxs)
    idxs_split = np.array_split(idxs, 3, axis=0)
    
    for i in range(3):
        pxyz = pxyz_ori[idxs_split[i]]
        stream = dump_and_compress(pxyz)
        
        key_byte = (scan+"_{}".format(i)).encode()
        db_txn.put(key_byte, stream)
        keys_cache.append(key_byte)
        
    count += 1
    if count % 10 == 0:
        db_txn.commit()
        db_txn = db_env.begin(write=True)
        count = 0
    print("save {}, length: {}".format(scan, len(keys_cache)))

db_txn.put("__point_keys_list__".encode(), dump_and_compress(keys_cache))
db_txn.commit()
db_env.close()
