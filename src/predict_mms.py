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
from models.model_selfupdate import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import rouge
from datasets.dataset_selfupdate import YourDataSetClass, get_dataloader
BATCH_SIZE = 8
torch.manual_seed(123)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
import os
from tqdm import tqdm, trange
import json
import pickle
import glob

from preprocess import LCSTSProcessor, convert_examples_to_features, create_dataset
from models.model_selfupdate import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils import rouge
from datasets.dataset_selfupdate import YourDataSetClass, get_dataloader
BATCH_SIZE = 8
torch.manual_seed(123)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
ground_file_path = "/data-yifan/data/meihuan2/dataset/MMS/corpus/test_title.txt"

def decode_one(config, device, decode_name, decode_iter):
    model = BertAbsSum(args, args.bert_model, config['decoder_config'], device, 'imgtxt-sum')
    model_dir = args.model_path.split('/')[:-1]
    cur_model_path = "/".join(model_dir) + '/BertAbsSum_{}.bin'.format(decode_iter)
    res_file_path = "/".join(model_dir) + '/rouge_{}_{}.txt'.format(decode_name, decode_iter)
    if not os.path.exists(cur_model_path):
        return 0
    # model.load_state_dict(torch.load(cur_model_path), strict=False)
    model.load_state_dict(torch.load(cur_model_path))
    model.to(device)
    processor = LCSTSProcessor()
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, 'vocab.txt'))
    test_set, test_dataloader = get_dataloader(args, 'test', tokenizer)

    logger.info('Loading complete. Writing results to %s' % (args.result_path))

    model.eval()
    hyp_file = os.path.join("/".join(model_dir), '{}_{}.txt'.format(decode_name, decode_iter))
    f_hyp = open(hyp_file, 'w', encoding='utf-8')
    # f_ref = open(os.path.join(args.result_path, 'ref.txt'), 'w', encoding='utf-8')
    hyp_list = []
    ref_list = []
    pred_guides = []
    for batch in tqdm(test_dataloader, desc="Iteration"):
        batch = tuple(t.to(device) if len(t)!=0 else [] for t in batch)
        index, img, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask, img_guide = batch
        pred, pred_max_guide = model.beam_decode(img, src_ids, src_mask, 3, 3)
        pred_guides = pred_guides + pred_max_guide[:,0,1].tolist()
        # import pdb
        # pdb.set_trace()
        for i in range(BATCH_SIZE):
            # sample_src = " ".join(tokenizer.convert_ids_to_tokens(src[i].cpu().numpy())).split('[CLS]')[1].split('[SEP]')[0] + '\n'
            sample_tgt = " ".join(tokenizer.convert_ids_to_tokens(tgt_ids[i].cpu().numpy())).split('[CLS]')[1].split('[SEP]')[0] + '\n'
            sample_pred = " ".join(tokenizer.convert_ids_to_tokens(pred[i][0])).split('[SEP]')[0] + '\n'
            sample_tgt = sample_tgt.replace(" ##", '')
            sample_pred = sample_pred.replace(" ##", '')
            
            # f_ref.write(target_texts[i] + '\n')
            f_hyp.write(sample_pred)
            hyp_list.append(sample_pred)
            ref_list.append(sample_tgt)
        print("Ground truth: ", sample_tgt)
        print("Generated: ", sample_pred)
    print(sum(pred_guides))
    file_pred_guides = os.path.join("/".join(model_dir), '{}_{}.pickle'.format("pred_guides", decode_iter))
    with open(file_pred_guides, 'wb') as f:
        pickle.dump(pred_guides, f)
        f.close()

    f_hyp.close()
    df = os.popen('files2rouge --ignore_empty_reference --ignore_empty_summary {} {}'.format(ground_file_path, hyp_file)).read()
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
    # decode_step = [14,15]
    # decode_name = "hyp_selfupdate" 
    decode_step = [10,12,14,16,18] #
    decode_name = "hyp_"
    for ds in decode_step:
        decode_one(config, device, decode_name, ds)
    model_dir = args.model_path.split('/')[:-1]
    model_dir = "/".join(model_dir)
    sort_rouge(model_dir)
    
    

        

