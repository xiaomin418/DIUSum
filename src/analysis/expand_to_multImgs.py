import pickle

data_dir = "/data-xxx/xxx/IGMS_checkpoints/model_msmo_07-26-10:44:33/"
hyp_files = ("hyp_img1_12.txt","hyp_img2_12.txt","hyp_img3_12.txt")
guide_files = ("img1_12_guide.pickle", "img2_12_guide.pickle", "img3_12_guide.pickle")
save_max_guide_refs = "hyp_12_max_guide.txt"

def fetch_one_data(index):
    data =[]
    with open(data_dir+hyp_files[index],'r') as f:
        lines = f.readlines()
        f.close()
    with open(data_dir+guide_files[index],'rb') as f:
        guide_score = pickle.load(f)
        f.close()
    for g,lin in zip(guide_score, lines):
        data.append((g,lin))
    return data

def merge_data(datas_list, res_num):
    res_data = []
    ds_num = len(datas_list)
    for i in range(res_num):
        cur_ex = []
        for j in range(ds_num):
            cur_ex.append(datas_list[j][i])
        # import pdb
        # pdb.set_trace()
        cur_ex = sorted(cur_ex, key=lambda x:x[0])
        res_data.append(cur_ex[0])
    return res_data

datas_list = []
for index in range(len(hyp_files)):
    dt = fetch_one_data(index)
    datas_list.append(dt)
res_data = merge_data(datas_list, len(datas_list[0]))
del datas_list
with open(data_dir + save_max_guide_refs, 'w') as f_max_guide:
    for d in res_data:
        f_max_guide.write(d[1])
    f_max_guide.close()

