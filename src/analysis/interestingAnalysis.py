import numpy as np
import torch
from os.path import basename, exists
import os
def _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb):
    num_bb = max(min_bb, (img_dump['conf'] > conf_th).sum())
    num_bb = min(max_bb, num_bb)
    return int(num_bb)

def load_npz(conf_th, max_bb, min_bb, num_bb, fname, keep_all=False):
    try:
        img_dump = np.load(fname, allow_pickle=True)
        if keep_all:
            nbb = None
        else:
            nbb = _compute_nbb(img_dump, conf_th, max_bb, min_bb, num_bb)
        dump = {}
        for key, arr in img_dump.items():
            if arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            if arr.ndim == 2:
                dump[key] = arr[:nbb, :]
            elif arr.ndim == 1:
                dump[key] = arr[:nbb]
            else:
                raise ValueError('wrong ndim')
    except Exception as e:
        # corrupted file
        print(f'corrupted file {fname}', e)
        dump = {}
        nbb = 0

    name = basename(fname)
    return name, dump, nbb

def _get_img_feat(filename):
        name, dump, nbb = load_npz(0.7,
                                   100,
                                   0,
                                   36,
                                   filename)
        # img_feat = dump['features']
        # img_bb = dump['norm_bb']
        # soft_labels = dump['soft_labels']

        # img_feat = torch.tensor(img_feat[:nbb, :]).float()
        # img_bb = torch.tensor(img_bb[:nbb, :]).float()

        # img_bb = torch.cat([img_bb, img_bb[:, 4:5] * img_bb[:, 5:]], dim=-1)

        return dump

def count_objects():
    data_dir = "/data-xxx/data/xxx/dataset/MMS_ROI/images_test/"
    useless_id_file = "mms_useless_ids.txt"
    with open(useless_id_file, 'r') as f:
        useless_ids = f.read()
        useless_ids = useless_ids.strip().split(' ')
        useless_ids = [int(id) for id in useless_ids]
        f.close()
    # useless_ids =[42,119,124,115,269,24,121,467,57,349,101,70,88,99]
    # useful_ids =[1,2,16,4,5,6,7,8,9,10,11,12,13,14,43]

    num_bbs0 = []
    num_bbs1 = []
    for i in range(2000):
        cur_dump = _get_img_feat(data_dir+'{}.npz'.format(i+1))
        if i+1 in useless_ids:
            num_bbs0.append(len(cur_dump['conf']))
        else:
            num_bbs1.append(len(cur_dump['conf']))
    print("useless: ", sum(num_bbs0)/len(num_bbs0))
    print("useful: ", sum(num_bbs1)/len(num_bbs1))
    print("num_bbs0: ", num_bbs0)
    print("num_bbs1: ", num_bbs1)

def count_copywords():
    def count_copy_num(slist, rlist):
        cur_num = 0
        for wr in rlist:
            if wr in slist:
                cur_num = cur_num + 1
        return cur_num

    
    # useless_id_file = "mms_useless_ids.txt"
    # with open(useless_id_file, 'r') as f:
    #     useless_ids = f.read()
    #     useless_ids = useless_ids.strip().split(' ')
    #     useless_ids = [int(id) for id in useless_ids]
    #     f.close()
    useless_ids =[42,119,124,115,269,24,121,467,57,349,101,70,88,92,99]
    useful_ids =[1,2,16,4,5,6,7,8,9,10,11,12,13,14,43]

    src_file = "/data-xxx/data/xxx/dataset/MMS/corpus/test_sent.txt"
    ref_file = "/data-xxx/data/xxx/dataset/MMS/corpus/test_title.txt"
    with open(src_file, 'r') as f:
        src_lines = f.readlines()
        src_lines = [line.strip().split(' ') for line in src_lines]
        f.close()
    with open(ref_file, 'r') as f:
        ref_lines = f.readlines()
        ref_lines = [line.strip().split(' ') for line in ref_lines]
        f.close()
    num_bbs0 = []
    num_bbs1 = []
    for i in range(2000):
        numc = count_copy_num(src_lines[i], ref_lines[i])
        if i+1 in useless_ids:
            num_bbs0.append(numc)
        elif i+1 in useful_ids:
            num_bbs1.append(numc)
    print("------------copy words------------")
    print("useless: ", sum(num_bbs0)/len(num_bbs0))
    print("useful: ", sum(num_bbs1)/len(num_bbs1))

def count_reflens():
    useless_id_file = "mms_useless_ids.txt"
    with open(useless_id_file, 'r') as f:
        useless_ids = f.read()
        useless_ids = useless_ids.strip().split(' ')
        useless_ids = [int(id) for id in useless_ids]
        f.close()
    # useless_ids =[42,119,124,115,269,24,121,467,57,349,101,70,88,92,99]
    # useful_ids =[1,2,16,4,5,6,7,8,9,10,11,12,13,14,43]

    src_file = "/data-xxx/data/xxx/dataset/MMS/corpus/test_sent.txt"
    ref_file = "/data-xxx/data/xxx/dataset/MMS/corpus/test_title.txt"
    with open(src_file, 'r') as f:
        src_lines = f.readlines()
        src_lines = [line.strip().split(' ') for line in src_lines]
        f.close()
    with open(ref_file, 'r') as f:
        ref_lines = f.readlines()
        ref_lines = [line.strip().split(' ') for line in ref_lines]
        f.close()
    num_bbs0 = []
    num_bbs1 = []
    for i in range(2000):
        numc = len(ref_lines[i])
        if i+1 in useless_ids:
            num_bbs0.append(numc)
        else:
            num_bbs1.append(numc)
    print("------------ref lens------------")
    print("useless: ", sum(num_bbs0)/len(num_bbs0))
    print("useful: ", sum(num_bbs1)/len(num_bbs1))

if __name__ == '__main__':
    count_objects()
    count_copywords()
    count_reflens()
    
