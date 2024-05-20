import csv

def Levenshtein_Distance_Recursive(str1, str2):

    if len(str1) == 0:
        return len(str2)
    elif len(str2) == 0:
        return len(str1)
    elif str1 == str2:
        return 0

    if str1[len(str1)-1] == str2[len(str2)-1]:
        d = 0
    else:
        d = 1
    
    return min(Levenshtein_Distance_Recursive(str1, str2[:-1]) + 1,
                Levenshtein_Distance_Recursive(str1[:-1], str2) + 1,
                Levenshtein_Distance_Recursive(str1[:-1], str2[:-1]) + d)

def get_LD(ref_file, hyp_file,k):
    with open(ref_file,'r') as f:
        ref_lines = f.readlines()
        f.close()
    
    with open(hyp_file,'r') as f:
        hyp_lines = f.readlines()
        f.close()
    lds = []
    for rl, hl in zip(ref_lines, hyp_lines):
        cur_ld = Levenshtein_Distance_Recursive(rl.split(' ')[:k], hl.split(' ')[:k])
        lds.append(cur_ld)
    avg = sum(lds)/len(lds)
    return lds, avg


def write_to_csv(csv_file, dt_list):
    f = open(csv_file, 'w')
    writer = csv.writer(f, dialect='excel')
    writer.writerow(['ed'])
    for d in dt_list:
        writer.writerow([d])
    f.close()

ref_file = "/data-xxx/data/xxx/dataset/MMS/corpus/test_title.txt"

our_file = "/data-xxx/xxx/IGMS_checkpoints/mmss_f4_w05/hyp__16.txt"
base_hyp_file = "/data-xxx/xxx/IGMS_checkpoints/base_multimodal/hyp_baseMulti_19.txt"
hyp_textonly_file = "/data-xxx/xxx/IGMS_checkpoints/base_textonly/hyp_textonly.txt"
cfsum_file = "/data-xxx/data/xxx/ReAttnMMS_checkpoints/1206-c2-ot-l6l9l3/hyps/hyp12.txt"
matt_file = "/data-xxx/data/xxx/MMSS_checkpoints/1019-multimodal.txt"
mmss_lead = "/data-xxx/xxx/IGMS_checkpoints/Lead/hyp_lead.txt"

msmo_ref2000_file = "/data-xxx/data/xxx/dataset/MSMO_Finished/test2000.txt" #2000
msmo_ref_file = "/data-xxx/data/xxx/dataset/MSMO_Finished/test-10256.txt"

msmo_cfsum_file = "/home/xxx/xxx/document/MultiSum/IGMS/evaluation/tmp/msmo_cfsum.txt" #2000
msmo_ism_txt_file = "/data-xxx/xxx/IGMS_checkpoints/msmo_textonly_cpl/hyp_msmo_txt_14.txt"
msmo_ism_basemulti_file = "/data-xxx/xxx/IGMS_checkpoints/msmo_base_img1_cpl/hyp_msmo_img1_14.txt"
msmo_ism_file = "/data-xxx/xxx/IGMS_checkpoints/msmo_assist_w0.5/hyp_msmo_correct_6.txt"
msmo_pg_file = "/home/xxx/xxx/document/MultiSum/IGMS/evaluation/tmp/msmo_pg.txt"
msmo_lead_file = "/data-xxx/xxx/IGMS_checkpoints/Lead/hyp_msmo_lead.txt"

kk = 5

# lds, avg = get_LD(ref_file, base_hyp_file,kk)
# print("base multi avg: ", avg)
# write_to_csv('tmp2.csv',lds)
# avg = get_LD(ref_file, hyp_textonly_file,kk)
# print("textonly avg: ", avg)
# avg = get_LD(ref_file, cfsum_file,kk)
# print("cfsum avg: ", avg)
# avg = get_LD(ref_file, matt_file,kk)
# print("matt avg: ", avg)
lds, avg = get_LD(ref_file, our_file,kk)
print("ours avg: ", avg)
write_to_csv('tmp2.csv',lds)
# avg = get_LD(ref_file, mmss_lead,kk)
# print("mmss_lead avg: ", avg)


# kk = 5
# avg = get_LD(msmo_ref2000_file, msmo_cfsum_file,kk)
# print("msmo cfsum avg: ", avg)

# avg = get_LD(msmo_ref_file, msmo_ism_basemulti_file,kk)
# print("msmo base avg: ", avg)
# avg = get_LD(msmo_ref_file, msmo_ism_file,kk)
# print("our ISM avg: ", avg)
# avg = get_LD(msmo_ref_file, msmo_pg_file,kk)
# print("our ISM avg: ", avg)
# avg = get_LD(msmo_ref_file, msmo_ism_txt_file,kk)
# print("msmo base avg: ", avg)
# avg = get_LD(msmo_ref_file, msmo_lead_file,kk)
# print("msmo base avg: ", avg)

