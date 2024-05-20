import pickle


multi_file = "/data-xxx/xxx/IGMS_checkpoints/base_multimodal/hyp_baseMulti_19.pickle"
txt_file = "/data-xxx/xxx/IGMS_checkpoints/base_textonly/hyp_textonly.pickle"
norm_file = "/data-xxx/xxx/IGMS_checkpoints/base_multimodal/norm_img_.pkl"

with open(norm_file, 'rb') as f:
    norm_imgs = pickle.load(f)
    f.close()
with open(multi_file, 'rb') as f:
    multi_score = pickle.load(f)
    f.close()
with open(txt_file, 'rb') as f:
    txt_score = pickle.load(f)
    f.close()

interval = 10
eps = 1e-3
min_n = min(norm_imgs)
max_n = max(norm_imgs)
inter = (max_n - min_n)/interval
min_n = min_n + eps

fenduan_norm = [[] for _ in range(interval)]
fenduan = [[] for _ in range(interval)]
fenduan_txt = [[] for _ in range(interval)]
for i in range(2000):
    c_index = int((norm_imgs[i] - min_n)/inter)
    key_i = "./images_test/{}.jpg".format(i+1)
    fenduan_norm[c_index].append(norm_imgs[i])
    fenduan[c_index].append(multi_score[key_i][0])
    fenduan_txt[c_index].append(txt_score[key_i][0])
for i in range(interval):
    if len(fenduan_norm[i])==0:
        print("{} is empty".format(i))
        continue
    avg_multi = sum(fenduan[i])/len(fenduan[i])
    avg_txt = sum(fenduan_txt[i])/len(fenduan_txt[i])
    print("i:{}-{}, norm:{}, multi:{}, txt:{}, diff: {}".format(i, 
                                                   len(fenduan_norm[i]),
                                                   sum(fenduan_norm[i])/len(fenduan_norm[i]),
                                                   avg_multi,
                                                   avg_txt,
                                                   avg_multi- avg_txt
                                                   ))



