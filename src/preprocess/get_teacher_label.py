import os
import pickle

def get_rouge(train_hyp_file, train_ref_file, score_save, tmp_dir):
    f_hyp = open(train_hyp_file, 'r')
    f_ref = open(train_ref_file, 'r')
    hyp_line = f_hyp.readline()
    ref_line = f_ref.readline()
    score = []
    count = 0
    while hyp_line and ref_line:
        tmp_hyp_file = open('{}hyp.txt'.format(tmp_dir),'w')
        tmp_ref_file = open('{}ref.txt'.format(tmp_dir),'w')
        tmp_hyp_file.write(hyp_line)
        tmp_ref_file.write(ref_line)
        tmp_hyp_file.close()
        tmp_ref_file.close()
        df = os.popen('files2rouge --ignore_empty_reference {} {}'.format('{}ref.txt'.format(tmp_dir), '{}hyp.txt'.format(tmp_dir))).read()
        if len(df.split('\n')) < 5:
            score.append(-1)
            count = count + 1
            hyp_line = f_hyp.readline()
            ref_line = f_ref.readline()
        else:
            rouge1, rouge2, rougeL = df.split('\n')[5].split(' ')[3], df.split('\n')[9].split(' ')[3], \
                                    df.split('\n')[13].split(' ')[3]
            rouge1, rouge2, rougeL = float(rouge1), float(rouge2), float(rougeL)
            score.append(rouge1)
            count = count + 1
            hyp_line = f_hyp.readline()
            ref_line = f_ref.readline()
            if count>0 and count%100==0:
                print("{}/{}".format(count, 2000))
    with open(score_save, 'wb') as f:
        pickle.dump(score, f)
        f.close()
    
def get_teacher_label(textonly_score_file, multi_score_file, teacher_save_file):
    with open(textonly_score_file, 'rb') as f:
        text_score = pickle.load(f)
        f.close()
    with open(multi_score_file, 'rb') as f:
        multi_score = pickle.load(f)
        f.close()
    tech_labels = []
    # tech_scores = []
    for ts, ms in zip(text_score, multi_score):
        if ms >= ts:
            tech_labels.append(1)
            # tech_scores.append(ms)
        else:
            tech_labels.append(0)
            # tech_scores.append(ts)
    with open(teacher_save_file, 'wb') as f:
        pickle.dump(tech_labels, f)
        f.close()

if __name__ == '__main__':
    #1. train
    # text_hyp_file = "/home/xxx/document/MultiSum/IGMS/result/hyp_textonly_trainset.txt"
    # train_ref_file = "/data/xxx/dataset/MMS/corpus/train_title.txt"
    # text_score_save = "/data/xxx/dataset/MMS/corpus/merge_guidance/train_textScore.pickle"
    # text_tmp_dir = "../tmp/"

    # multi_hyp_file = "/home/xxx/document/MultiSum/IGMS/result/hyp_multi_trainset.txt"
    # train_ref_file = "/data/xxx/dataset/MMS/corpus/train_title.txt"
    # multi_score_save = "/data/xxx/dataset/MMS/corpus/merge_guidance/train_multiScore.pickle"
    # multi_tmp_dir = "../tmp2/"

    # teacher_save_file = "/data/xxx/dataset/MMS/corpus/merge_guidance/train_teacher.pickle"

    #2. test
    text_hyp_file = "/data-xxx/xxx/IGMS_checkpoints/msmo_textonly/hyp_msmo_textonly_19.txt"
    test_ref_file = "/data/xxx/dataset/MSMO_Finished/test2000.txt"
    text_score_save = "/data-xxx/xxx/IGMS_checkpoints/msmo_textonly/testonly_19.pickle"
    text_tmp_dir = "../tmp/"

    multi_hyp_file = "/data-xxx/xxx/IGMS_checkpoints/msmo_base_img1/hyp_msmo_img1_19.txt"
    test_ref_file = "/data/xxx/dataset/MSMO_Finished/test2000.txt"
    multi_score_save = "/data-xxx/xxx/IGMS_checkpoints/msmo_base_img1/hyp_msmo_img1_19.pickle"
    multi_tmp_dir = "../tmp2/"

    #3. dev
    # text_hyp_file = "/home/xxx/document/MultiSum/IGMS/result/hyp_textonly_devset.txt"
    # test_ref_file = "/data/xxx/dataset/MMS/corpus/dev_title.txt"
    # text_score_save = "/data/xxx/dataset/MMS/corpus/merge_guidance/dev_textScore.pickle"
    # text_tmp_dir = "../tmp/"

    # multi_hyp_file = "/home/xxx/document/MultiSum/IGMS/result/hyp_multi_devset.txt"
    # test_ref_file = "/data/xxx/dataset/MMS/corpus/dev_title.txt"
    # multi_score_save = "/data/xxx/dataset/MMS/corpus/merge_guidance/dev_multiScore.pickle"
    # multi_tmp_dir = "../tmp2/"

    teacher_save_file = "/data/xxx/dataset/MMS/corpus/merge_guidance/dev_teacher.pickle"
    get_rouge(text_hyp_file, test_ref_file, text_score_save, text_tmp_dir)
    # get_rouge(multi_hyp_file, test_ref_file, multi_score_save, multi_tmp_dir)
    # get_teacher_label(text_score_save, multi_score_save, teacher_save_file)

