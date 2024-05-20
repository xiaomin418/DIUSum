import pickle

ex_file = "/data-xxx/data/xxx/dataset/MSMO_Finished/exID_bert_test.pickle"
data_dir ="/data-xxx/data/xxx/dataset/MSMO_Finished/"

with open(ex_file, 'rb') as f:
    ext_img_ex = pickle.load(f)
    f.close()
fhyp = open('/data-xxx/xxx/IGMS_checkpoints/Lead/hyp_msmo_lead.txt','w')
for index in range(10256):
    cur_path = ext_img_ex[0][index]
    # print(cur_path)
    cur_path = cur_path.split('-')
    cur_path = data_dir + cur_path[0] + '/article_bert/' + cur_path[1]  + '.pickle'
    with open(cur_path, 'rb') as fc:
        cur_d = pickle.load(fc)
        fc.close()
    source_text, target_text = cur_d[0], cur_d[1]
    lead_hyp = source_text.split('@body')[1:][0].split('.')[:3]
    lead_hyp = "".join(lead_hyp)
    lead_hyp = lead_hyp.replace('\n','')
    fhyp.write(lead_hyp+'\n')
    # print(cur_path)
fhyp.close()



    