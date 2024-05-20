import pickle

example_path = "/data-xxx/data/xxx/dataset/MSMO_Finished/exID_bert_test.pickle"
data_dir = "/data-xxx/data/xxx/dataset/MSMO_Finished/"
save_test_cpl = "/data-xxx/data/xxx/dataset/MSMO_Finished/test2000-cpl.txt"
with open(example_path, 'rb') as f:
    test_ex = pickle.load(f)
    f.close()
len_ex = len(test_ex[0])
summaries = []
for index in range(len_ex):
    cur_path = test_ex[0][index]
    cur_path = cur_path.split('-')
    cur_path = data_dir + cur_path[0] + '/article_bert/' + cur_path[1]  + '.pickle'
    with open(cur_path, 'rb') as f:
        cur_d = pickle.load(f)
        f.close()
    source_text, target_text = cur_d[0], cur_d[1]
    summaries.append(target_text)
with open(save_test_cpl, 'w') as f:
    for line in summaries:
        f.write(line+'\n')
    f.close()