import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
import os
from tqdm import tqdm, trange
import json
import pickle

from preprocess import LCSTSProcessor, convert_examples_to_features, create_dataset
from models.model_msmo_selfupdate_expandMultiImagey import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import rouge
from datasets.dataset_msmo import YourDataSetClass, get_dataset
BATCH_SIZE = 8
torch.manual_seed(123)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
ground_file_path = "/data-yifan/data/meihuan2/dataset/MSMO_Finished/test-10256.txt"
import glob

def decode_one(config, device, decode_name, decode_iter):
    model = BertAbsSum(args, args.bert_model, config['decoder_config'], device)
    model_dir = args.model_path.split('/')[:-1]
    cur_model_path = "/".join(model_dir) + '/BertAbsSum_{}.bin'.format(decode_iter)
    res_file_path = "/".join(model_dir) + '/rouge_{}_{}.txt'.format(decode_name, decode_iter)
    if not os.path.exists(cur_model_path):
        return 0
    
    model.load_state_dict(torch.load(cur_model_path))
    model.to(device)
    
    processor = LCSTSProcessor()
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, 'vocab.txt'))
    # tokenizer,
        # model_params["MAX_SOURCE_TEXT_LENGTH"],
        # model_params["MAX_TARGET_TEXT_LENGTH"]
    test_set = get_dataset(args, tokenizer, 'test')
    test_sampler = SequentialSampler(test_set)
    test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=BATCH_SIZE, drop_last=True, shuffle=False)

    logger.info('Loading complete. Writing results to %s' % (args.result_path))

    model.eval()
    # f_hyp = open(os.path.join(args.result_path, 'hyp.txt'), 'w', encoding='utf-8')
    hyp_file = os.path.join("/".join(model_dir), '{}_{}.txt'.format(decode_name, decode_iter))
    guide_file = os.path.join("/".join(model_dir), '{}_{}.pickle'.format(decode_iter, 'guide'))
    f_hyp = open(hyp_file, 'w', encoding='utf-8')
    # f_ref = open(os.path.join(args.result_path, 'ref.txt'), 'w', encoding='utf-8')
    hyp_list = []
    ref_list = []
    img_guides = []
    for batch in tqdm(test_dataloader, desc="Iteration"):
        batch = tuple(t.to(device) if len(t)!=0 else [] for t in batch)
        img, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask = batch
        pred, pred_img_guide = model.beam_decode(img, src_ids, src_mask, 3, 3)
        img_guides = img_guides + pred_img_guide[:,0,1].cpu().tolist()
        for i in range(BATCH_SIZE):
            # sample_src = " ".join(tokenizer.convert_ids_to_tokens(src[i].cpu().numpy())).split('[CLS]')[1].split('[SEP]')[0] + '\n'
            if 'sum' in args.mode:
                sample_tgt = " ".join(tokenizer.convert_ids_to_tokens(tgt_ids[i].cpu().numpy())).split('[CLS]')[1].split('[SEP]')[0] + '\n'
            elif 'key' in args.mode:
                sample_tgt = " ".join(tokenizer.convert_ids_to_tokens(key_ids[i].cpu().numpy())).split('[CLS]')[1].split('[SEP]')[0] + '\n'
            sample_pred = " ".join(tokenizer.convert_ids_to_tokens(pred[i][0])).split('[SEP]')[0] + '\n'
            sample_tgt = sample_tgt.replace(" ##", '')
            sample_pred = sample_pred.replace(" ##", '')
            
            # f_ref.write(target_texts[i] + '\n')
            f_hyp.write(sample_pred)
            hyp_list.append(sample_pred)
            ref_list.append(sample_tgt)
        print("Ground truth: ", sample_tgt)
        print("Generated: ", sample_pred)
    with open(guide_file, 'wb') as f_guide:
        pickle.dump(img_guides, f_guide)
        f_guide.close()
    f_hyp.close()
    df = os.popen('files2rouge --ignore_empty_summary --ignore_empty_reference {} {}'.format(ground_file_path, hyp_file)).read()
    with open(res_file_path,'w') as fres:
        fres.write(df)
        fres.close()
    return 1

def sort_rouge(model_dir):
    rouge_paths = glob.glob(model_dir + '/rouge_hyp*')
    all_result_dict = {}
    for rouge_p in rouge_paths:
        with open(rouge_p, 'r') as frouge:
            df = frouge.read()
            frouge.close()
        print("Compute", rouge_p)
        rouge1, rouge2, rougeL = df.split('\n')[5].split(' ')[3], df.split('\n')[9].split(' ')[3], \
                                 df.split('\n')[13].split(' ')[3]
        rouge1, rouge2, rougeL = float(rouge1), float(rouge2), float(rougeL)
        all_result_dict[rouge_p] = [rouge1, rouge2, rougeL]
    sorted_result = sorted(all_result_dict.items(), key=lambda x: x[1][2])

    fresult = open(model_dir + '/rouge.txt', 'w')
    for model_pth ,rouges in sorted_result:
        fresult.write(model_pth+' '+str(rouges[0])+' '+str(rouges[1])+' '+str(rouges[2])+'\n')
        print(model_pth+' '+str(rouges[0])+' '+str(rouges[1])+' '+str(rouges[2])+'\n')
    fresult.close()

if __name__ == "__main__":
    import sys
    config_name = sys.argv[1]
    sys.argv = sys.argv[:-1]
    with open(config_name, 'r') as f:
        data = json.load(f)
        f.close()
    args = parser.parse_args()
    args.__dict__.update(data)
    if args.GPU_index != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    decode_step = [6,8,10,12,14,16] #
    decode_name = "hyp12-8"
    for ds in decode_step:
        decode_one(config, device, decode_name, ds)
    model_dir = args.model_path.split('/')[:-1]
    model_dir = "/".join(model_dir)
    sort_rouge(model_dir)
    # with open(os.path.join(args.bert_model, 'bert_config.json'), 'r') as f:
    #     bert_config = json.load(f)
    # config = {'bert_config': bert_config, 'decoder_config': decoder_config}
    

        

