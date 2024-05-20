# To get one sentence rouge score.
# [test]-hyp-text-gate-model_7_15000.txt(0.43914)
# [test]-hyp-text-global-gate-model_9_5000.txt(0.43741)
# [dev]-hyp-text-gate-model_7_15000.txt(0.45541)
# [dev]-hyp-text-global-gate-model_9_5000.txt(0.46395)
import os
import pickle
import numpy as np
# from matplotlib import pyplot as plt
import math
import sys
sys.path.append('../')

tmp_ref_dir = '../result/tmp-ref/'
tmp_hyp_dir = '../result/tmp-hyp/'

def generate_sent_file(hyp_file, ref_file):
    hyp_lines = open(hyp_file,'r').readlines()
    ref_lines = open(ref_file,'r').readlines()
    for i, hl, rl in zip(range(len(hyp_lines)), hyp_lines, ref_lines):
        with open(tmp_hyp_dir+str(i+1)+'.txt', 'w') as f:
            f.write(hl)
            f.close()
        with open(tmp_ref_dir+str(i+1)+'.txt', 'w') as f:
            f.write(rl)
            f.close()

def get_sents_scores(prefix_path):
    scores = {}
    for i in range(2000):
        cur_hyp_path = tmp_hyp_dir + str(i+1) + '.txt'
        cur_ref_path = tmp_ref_dir + str(i+1) + '.txt'
        cmd_str = 'files2rouge ' + cur_ref_path + ' ' + cur_hyp_path
        df = os.popen(cmd_str).read()
        if len(df.split('\n'))<5:
            scores[prefix_path + str(i + 1) + '.jpg'] = [0.0, 0.0, 0.0]
            continue
        rouge1, rouge2, rougeL = df.split('\n')[5].split(' ')[3], df.split('\n')[9].split(' ')[3], \
                                 df.split('\n')[13].split(' ')[3]
        rouge1, rouge2, rougeL = float(rouge1), float(rouge2), float(rougeL)
        if i%5==0:
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
            print("The {} rouge score is: {}".format(cur_hyp_path, rouge1))
        scores[prefix_path+str(i+1)+'.jpg'] = [rouge1, rouge2, rougeL]
    return scores

def get_detail_sentence_rouge(hyp_file, ref_file, prefix_path, mode):
    generate_sent_file(hyp_file, ref_file)
    scores = get_sents_scores(prefix_path)
    scores['mode'] = mode
    import pickle
    hyp_dir = "/".join(hyp_file.split('/')[:-2]) # /data/xxx/PreEMMS_checkpoints/0328-textonly/hyps
    with open('{}/{}.pickle'.format(hyp_dir,hyp_file.split('/')[-1].split('.')[0]), 'wb') as f:
        pickle.dump(scores, f)
        f.close()
    return scores

def get_multimodal_better_img_ids_from_compare_file():
    import pickle
    data = pickle.load(open('compare.pickle','rb'))
    multimodal_scores = {}
    textonly_scores = {}
    for d in data:
        if d['mode']=='multimodal':
            multimodal_scores = dict(multimodal_scores, **d)
        else:
            textonly_scores = dict(textonly_scores, **d)
    multimodal_better_imgs = []
    for k,v in multimodal_scores.items():
        if k=='mode' or 'dev' in k:
            continue
        else:
            if v > textonly_scores[k]:
                multimodal_better_imgs.append(k)
    print(multimodal_better_imgs)
    print("len better: ",len(multimodal_better_imgs))


def get_mlm_better_avg_rouge2(aiming):
    import pickle
    multimodal_scores = pickle.load(open('../result/multimodal-rouge.pickle','rb'))
    textonly_scores = pickle.load(open('../result/textonly-rouge.pickle','rb'))
    multi_origin = [43.61, 22.77, 41.03]
    text_orgin = [44.00, 22.74, 41.30]
    print("******Sampled: ******")
    multimodal_scores_list1 = []
    textonly_scores_list1 = []
    multimodal_scores_list2 = []
    textonly_scores_list2 = []
    multimodal_scores_listL = []
    textonly_scores_listL = []
    for k in multimodal_scores.keys():
        if k in aiming:
            multimodal_scores_list1.append(multimodal_scores[k][0])
            textonly_scores_list1.append(textonly_scores[k][0])
            multimodal_scores_list2.append(multimodal_scores[k][1])
            textonly_scores_list2.append(textonly_scores[k][1])
            multimodal_scores_listL.append(multimodal_scores[k][2])
            textonly_scores_listL.append(textonly_scores[k][2])

    m1 = round(sum(multimodal_scores_list1) / len(multimodal_scores_list1)*100,2)
    m2 = round(sum(multimodal_scores_list2) / len(multimodal_scores_list2)*100,2)
    mL = round(sum(multimodal_scores_listL) / len(multimodal_scores_listL)*100,2)
    t1 = round(sum(textonly_scores_list1) / len(textonly_scores_list1)*100,2)
    t2 = round(sum(textonly_scores_list2) / len(textonly_scores_list2)*100,2)
    tL = round(sum(textonly_scores_listL) / len(textonly_scores_listL)*100,2)
    cur_str = str(m1) + '(' + str(round(m1-multi_origin[0],2)) + ')' + '\t' + \
              str(m2) + '(' + str(round(m2 - multi_origin[1],2)) + ')' + '\t' + \
              str(mL) + '(' + str(round(mL - multi_origin[2],2)) + ')' + '\n' + \
              str(t1) + '(' + str(round(t1 - text_orgin[0],2)) + ')' + '\t' + \
              str(t2) + '(' + str(round(t2 - text_orgin[1],2)) + ')' + '\t' + \
              str(tL) + '(' + str(round(tL - text_orgin[2],2)) + ')' + '\n'+ \
              str(t1) + '(' + str(round((m1 - multi_origin[0])-(t1 - text_orgin[0]), 2)) + ')' + '\t' + \
              str(t2) + '(' + str(round((m2 - multi_origin[1])-(t2 - text_orgin[1]), 2)) + ')' + '\t' + \
              str(tL) + '(' + str(round((mL - multi_origin[2])-(tL - text_orgin[2]), 2)) + ')' + '\n'
    print("len: ", len(multimodal_scores_list1))
    print(cur_str)
    print("*******Not Sampled:********")
    multimodal_scores_list1 = []
    textonly_scores_list1 = []
    multimodal_scores_list2 = []
    textonly_scores_list2 = []
    multimodal_scores_listL = []
    textonly_scores_listL = []
    for k in multimodal_scores.keys():
        if k == 'mode':
            continue
        if k not in aiming:
            multimodal_scores_list1.append(multimodal_scores[k][0])
            textonly_scores_list1.append(textonly_scores[k][0])
            multimodal_scores_list2.append(multimodal_scores[k][1])
            textonly_scores_list2.append(textonly_scores[k][1])
            multimodal_scores_listL.append(multimodal_scores[k][2])
            textonly_scores_listL.append(textonly_scores[k][2])
    m1 = round(sum(multimodal_scores_list1) / len(multimodal_scores_list1) * 100, 2)
    m2 = round(sum(multimodal_scores_list2) / len(multimodal_scores_list2) * 100, 2)
    mL = round(sum(multimodal_scores_listL) / len(multimodal_scores_listL) * 100, 2)
    t1 = round(sum(textonly_scores_list1) / len(textonly_scores_list1) * 100, 2)
    t2 = round(sum(textonly_scores_list2) / len(textonly_scores_list2) * 100, 2)
    tL = round(sum(textonly_scores_listL) / len(textonly_scores_listL) * 100, 2)
    cur_str = str(m1) + '(' + str(round(m1 - multi_origin[0], 2)) + ')' + '\t' + \
              str(m2) + '(' + str(round(m2 - multi_origin[1], 2)) + ')' + '\t' + \
              str(mL) + '(' + str(round(mL - multi_origin[2], 2)) + ')' + '\n' + \
              str(t1) + '(' + str(round(t1 - text_orgin[0], 2)) + ')' + '\t' + \
              str(t2) + '(' + str(round(t2 - text_orgin[1], 2)) + ')' + '\t' + \
              str(tL) + '(' + str(round(tL - text_orgin[2], 2)) + ')' + '\n'+ \
              str(t1) + '(' + str(round((m1 - multi_origin[0])-(t1 - text_orgin[0]), 2)) + ')' + '\t' + \
              str(t2) + '(' + str(round((m2 - multi_origin[1])-(t2 - text_orgin[1]), 2)) + ')' + '\t' + \
              str(tL) + '(' + str(round((mL - multi_origin[2])-(tL - text_orgin[2]), 2)) + ')' + '\n'
    print("len: ", len(multimodal_scores_list1))
    print(cur_str)

def get_rouge_diff(modelA_score_file, modelB_score_file):
    Ascore = pickle.load(open(modelA_score_file, 'rb'))
    Bscore = pickle.load(open(modelB_score_file, 'rb'))
    AbetterB_score  = dict()
    for k,a_s in Ascore.items():
        if k=='mode':
            continue
        if k in Bscore:
            b_s = Bscore[k]
            aMb = a_s[2] - b_s[2]
            aMb = 1/(1+math.exp((-1)*aMb))
            AbetterB_score[k.split('/')[-1]] = aMb
    return AbetterB_score

if __name__=='__main__':
    hyp_file, ref_file = ('/data/xxx/ReAttnMMS_checkpoints/1206-c2-ot-l6l9l3/hyps/hyp_model_12_62000_1670233244.txt',
                          '/home/xxx/document/MMSS4.0/corpus/test_title.txt')
    get_detail_sentence_rouge(hyp_file, ref_file, './images_test/', 'multimodal')

