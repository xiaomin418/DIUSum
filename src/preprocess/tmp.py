import pickle

textfile = "/data-xxx/xxx/IGMS_checkpoints/msmo_textonly/textonly_19.pickle"
multifile = "/data-xxx/xxx/IGMS_checkpoints/msmo_base_img1/hyp_msmo_img1_19.pickle"
with open(textfile, 'rb') as f:
    text_scores = pickle.load(f)
    f.close()
with open(multifile, 'rb') as f:
    multi_scores = pickle.load(f)
    f.close()

# text_scores = [text_scores['./images_test/{}.jpg'.format(i+1)][0] for i in range(2000)]
# multi_scores = [multi_scores['./images_test/{}.jpg'.format(i+1)][0] for i in range(2000)]
# new_socres = []
# for i in range(2000):
#     if text_scores[i] > multi_scores[i]:
#         new_socres.append(text_scores[i])
#     else:
#         new_socres.append(multi_scores[i])
# print(sum(text_scores)/2000)
# print(sum(multi_scores)/2000)
# print(sum(new_socres)/2000)
text_scores = [text_scores[i] for i in range(2000)]
multi_scores = [multi_scores[i] for i in range(2000)]
new_socres = []
for i in range(2000):
    if text_scores[i] > multi_scores[i]:
        new_socres.append(text_scores[i])
    else:
        new_socres.append(multi_scores[i])
print(sum(text_scores)/2000)
print(sum(multi_scores)/2000)
print(sum(new_socres)/2000)