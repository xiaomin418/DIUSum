import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
import os
import json
import time
import pickle
import torch.nn.functional as F
from preprocess import LCSTSProcessor, convert_examples_to_features, create_dataset
from models.model_selfupdate import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm, trange
from transformer import Constants
from datasets.dataset_selfupdate import YourDataSetClass, get_dataloader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_FILE = 'train_big.csv'

parser = argparse.ArgumentParser()
torch.manual_seed(123)
'''
TODO: beam/greedy search, eval, copy, rouge
'''
def Levenshtein_Distance_Recursive_org(str1, str2):

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

def Levenshtein_Distance_Recursive(str1, str2):
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1,len(str1)+1):
        for j in range(1,len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)
    return matrix[len(str1)][len(str2)]

def get_LD(logits, ground, k):
    hyps = logits.max(dim=-1).indices[:,:k].tolist()
    refs = ground[:,1:1+k].tolist()
    lds = []
    for x, y in zip(hyps, refs):
        # print(x,y)
        cur_ld = Levenshtein_Distance_Recursive(x,y)
        lds.append(cur_ld)
        # print(cur_ld)
    lds = torch.tensor(lds)
    lds = lds.to(logits.device)
    return lds
    

def cal_performance(logits, ground, smoothing=True):
    k = 5
    # top3_presion = (logits.max(dim=-1).indices[:,:k]==ground[:,1:1+k]).sum(dim=-1)
    ld_distance = get_LD(logits, ground, k)
    ld_distance =  - ld_distance
    batch_size = logits.shape[0]
    ground = ground[:, 1:]
    logits = logits.view(-1, logits.size(-1))
    ground = ground.contiguous().view(-1)

    loss, bs_loss = cal_loss(logits, ground, batch_size)

    pad_mask = ground.ne(Constants.PAD)
    pred = logits.max(-1)[1]
    correct = pred.eq(ground)
    correct = correct.masked_select(pad_mask).sum().item()
    # return loss, top3_presion
    return loss, ld_distance

def cal_loss(logits, ground, batch_size):
    def label_smoothing(logits, labels):
        eps = 0.1
        num_classes = logits.size(-1)

        # >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
        # >>> z
        # tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
        #        [ 0.0000,  0.0000,  0.0000,  1.2300]])
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
        log_prb = F.log_softmax(logits, dim=1)
        non_pad_mask = ground.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)

        bs_loss = loss.view(batch_size, -1)
        tgt_no_pad = non_pad_mask.long().view(batch_size, -1)
        bs_loss = bs_loss * tgt_no_pad
        bs_loss = bs_loss.sum(dim=-1)/(tgt_no_pad.sum(dim=-1))

        loss = loss.masked_select(non_pad_mask).mean()
        
        
        return loss, bs_loss
    loss, bs_loss= label_smoothing(logits, ground)
    
    return loss, bs_loss

def cal_use_performance(logits, ground, pad_mask):
    logits = logits.view(-1, logits.size(-1))
    ground = ground.contiguous().view(-1)

    loss = cal_use_loss(logits, ground, pad_mask)
    # pad_mask = ground.ne(-1)
    # pred = logits.max(-1)[1]
    # correct = pred.eq(ground)
    # correct = correct.masked_select(pad_mask).sum().item()
    return loss

def cal_use_loss(logits, ground, non_pad_mask):
    def label_smoothing(logits, labels):
        eps = 0.00
        num_classes = logits.size(-1)

        # >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
        # >>> z
        # tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
        #        [ 0.0000,  0.0000,  0.0000,  1.2300]])
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
        log_prb = logits.log()
        # non_pad_mask = ground.ne(-1)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss*non_pad_mask.float()
        loss = loss.mean()
        return loss
    
    def cross_entropy(logits, ground, non_pad_mask):
        eps = 1e-12
        gold_probs = torch.gather(logits, 1, ground.unsqueeze(1)).squeeze()
        pure_step_losses = -torch.log(gold_probs + eps)
        step_loss = pure_step_losses * non_pad_mask.float()
        loss = step_loss.mean()
        return loss
    loss = label_smoothing(logits, ground)
    # loss = F.cross_entropy(logits, ground)
    # loss = cross_entropy(logits, ground, non_pad_mask)
    return loss

def get_optimizer(model, learning_rate, num_train_optimization_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)
    return optimizer

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
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    assert args.train_batch_size % n_gpu == 0
    logger.info(f'Using device:{device}, n_gpu:{n_gpu}')

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_path = os.path.join(args.output_dir, time.strftime('model_%m-%d-%H:%M:%S', time.localtime()))
        os.mkdir(model_path)
        logger.info(f'Saving model to {model_path}.')
        
    with open(os.path.join(args.bert_model, 'config.json'), 'r') as f:
            bert_config = json.load(f)
            decoder_config = {}
            decoder_config['len_max_seq'] = args.max_tgt_len
            decoder_config['d_word_vec'] = bert_config['hidden_size']
            decoder_config['n_layers'] = 8
            decoder_config['n_head'] = 12
            decoder_config['d_k'] = 64
            decoder_config['d_v'] = 64
            decoder_config['d_model'] = bert_config['hidden_size']
            decoder_config['d_inner'] = bert_config['hidden_size']
            decoder_config['vocab_size'] = bert_config['vocab_size']
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
                
    # train data preprocess
    processor = LCSTSProcessor()
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, 'vocab.txt'))
    logger.info('Loading train examples...')
    
    logger.info('Building dataloader...')
    # train_path = ["/home/xxx/dataset/MMS/corpus/train_sent.txt", "/home/xxx/dataset/MMS/corpus/train_title.txt"]
    # dev_path = ["/home/xxx/dataset/MMS/corpus/dev_sent.txt", "/home/xxx/dataset/MMS/corpus/dev_title.txt"]
    # tokenizer,
        # model_params["MAX_SOURCE_TEXT_LENGTH"],
        # model_params["MAX_TARGET_TEXT_LENGTH"]
    
    val_set, eval_dataloader = get_dataloader(args, 'dev', tokenizer)
    training_set, train_dataloader = get_dataloader(args, 'train', tokenizer)
    num_train_optimization_steps = int(len(training_set) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs


    # model
    model = BertAbsSum(args, args.bert_model, decoder_config, device, 'imgtxt-sum')
    model.init_enc_crfer(args.multi_encoder_init)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # optimizer
    optimizer = get_optimizer(model, args.learning_rate, num_train_optimization_steps)
    
    # training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(training_set))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    global_step = 0

    config = {'bert_config': bert_config, 'decoder_config': decoder_config}
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config, f)
        
    
    devloader_correct = True
    for i in range(int(args.num_train_epochs)):
        # do training
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        iter_bar = tqdm(train_dataloader, desc="Iteration")
        if args.usefuler_assist:
            use_loss_all = []
            multi_loss_all = []
            txt_loss_all = []
        else:
            loss_all = []

        train_guide = True
        second_update = False
        if i>=args.start_train_guide_epoch:
            train_guide = True
        else:
            train_guide = False
        if i%args.guide_update_interval==0:
            update_guide = True
            if i>0:
                second_update = True
        else:
            update_guide = False
        second_update = second_update and args.guide_correct

        if update_guide:
            new_image_guidance = {}
        fenge = 6
        if i<=fenge:
            for step, batch in enumerate(iter_bar):
                batch = tuple(t.to(device) if len(t)!=0 else [] for t in batch)
                index, img, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask, img_guide = batch
                cur_mode = 'txt-sum'
                cur_batch = batch[1:] + tuple([cur_mode, True])
                txt_logits, use_logits = model(*cur_batch)
                txt_loss, txt_bs_loss = cal_performance(txt_logits, tgt_ids)

                cur_mode = 'imgtxt-sum'
                cur_batch = batch[1:] + tuple([cur_mode, True])
                multi_logits, multi_use_logits = model(*cur_batch)
                multi_loss, bs_loss = cal_performance(multi_logits, tgt_ids)
                
                # non_pad_mask = (txt_loss.detach()-bs_loss+args.loss_bias)>0
                non_pad_mask = (bs_loss.detach()- txt_bs_loss.detach())>=0
                non_pad_mask_grd = non_pad_mask.long()
                new_pad_mask = torch.ones_like(non_pad_mask_grd)
                multi_use_pred = multi_use_logits.max(dim=-1).indices.squeeze(1)
                if update_guide:
                    index_k = index.tolist()
                    mg = multi_use_pred.bool()
                    guide_v = (mg & non_pad_mask).long() + (~mg & ~non_pad_mask).long()
                    guide_v = guide_v.tolist()
                    for k,v in zip(index_k, guide_v):
                        new_image_guidance[k] = v
                if train_guide:
                    use_loss = cal_use_performance(multi_use_logits, non_pad_mask_grd, new_pad_mask)
                    use_loss = use_loss * args.assist_loss_w
                    loss = multi_loss + use_loss
                else:
                    loss = multi_loss
                    use_loss = torch.tensor(0)
                optimizer.zero_grad()
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    
                txt_loss_all.append(txt_loss.cpu().item())
                multi_loss_all.append(multi_loss.cpu().item())
                use_loss_all.append(use_loss.cpu().item())
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item()) 
                
                if (step + 1) % args.print_every == 0:
                    logger.info(f'Epoch {i}, step {step}, loss {loss.item()}.')
                    logger.info(f'Ground: {" ".join(tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy()))}')
                    logger.info(f'Generated: {" ".join(tokenizer.convert_ids_to_tokens(multi_logits[0].max(-1)[1].cpu().numpy()))}')
        else:
            # if i==fenge+1:
            #     new_lr = 1e-4
            #     optimizer = get_optimizer(model.decoder, new_lr, num_train_optimization_steps)
            model.eval()
            model.decoder.train()
            for step, batch in enumerate(iter_bar):
                batch = tuple(t.to(device) if len(t)!=0 else [] for t in batch)
                index, img, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask, img_guide = batch
                cur_mode = 'txt-sum'
                cur_batch = batch[1:] + tuple([cur_mode, False])
                txt_logits, use_logits = model(*cur_batch)
                txt_loss, txt_bs_loss = cal_performance(txt_logits, tgt_ids)

                cur_mode = 'imgtxt-sum'
                cur_batch = batch[1:] + tuple([cur_mode, False])
                multi_logits, multi_use_logits = model(*cur_batch)
                multi_loss, bs_loss = cal_performance(multi_logits, tgt_ids)
                
                # non_pad_mask = (txt_loss.detach()-bs_loss+args.loss_bias)>0
                non_pad_mask = (bs_loss.detach()- txt_loss.detach())>=0
                multi_use_pred = multi_use_logits.max(dim=-1).indices.squeeze(1)
                if update_guide:
                    index_k = index.tolist()
                    mg = multi_use_pred.bool()
                    guide_v = (mg & non_pad_mask).long() + (~mg & ~non_pad_mask).long()
                    guide_v = guide_v.tolist()
                    for k,v in zip(index_k, guide_v):
                        new_image_guidance[k] = v
                if train_guide:
                    use_loss = cal_use_performance(multi_use_logits, 1- multi_use_pred, ~non_pad_mask)
                    use_loss = use_loss * args.assist_loss_w
                    loss = multi_loss
                else:
                    loss = multi_loss
                    use_loss = torch.tensor(0)
                optimizer.zero_grad()
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    

                txt_loss_all.append(txt_loss.cpu().item())
                multi_loss_all.append(multi_loss.cpu().item())
                use_loss_all.append(use_loss.cpu().item())
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item()) 
                
                if (step + 1) % args.print_every == 0:
                    logger.info(f'Epoch {i}, step {step}, loss {loss.item()}.')
                    logger.info(f'Ground: {" ".join(tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy()))}')
                    logger.info(f'Generated: {" ".join(tokenizer.convert_ids_to_tokens(multi_logits[0].max(-1)[1].cpu().numpy()))}')

        # do evaluation
        if update_guide:
            new_g_list = []
            upd_num = 0
            for eid in range(62000):
                if eid in new_image_guidance:
                    new_g_list.append(new_image_guidance[eid])
                    upd_num = upd_num + 1
                else:
                    new_g_list.append(1)
            train_dataloader.dataset.image_guidance = new_g_list
            with open(model_path+'/{}_guide.pickle'.format(i),'wb') as fg:
                pickle.dump(new_g_list, fg)
                fg.close()
            logger.info("Update image guidance num: {}/62000".format(upd_num))

        if args.usefuler_assist:
            logger.info("Epoch: {} ImgLoss: {} MultiLoss: {} TxtLoss: {}".format(i, sum(use_loss_all)/len(use_loss_all), 
                                                                                 sum(multi_loss_all)/len(multi_loss_all), 
                                                                                 sum(txt_loss_all)/len(txt_loss_all)))
        if args.output_dir is not None:
            state_dict = model.module.state_dict() if n_gpu > 1 else model.state_dict()
            torch.save(state_dict, os.path.join(model_path, 'BertAbsSum_{}.bin'.format(i)))
            logger.info('Model saved')
        if eval_dataloader is not None:
            model.eval()
            batch = next(iter(eval_dataloader))
            batch = tuple(t.to(device) if len(t)!=0 else [] for t in batch)
            index, img, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask, img_guide = batch
            # beam_decode
            if n_gpu > 1:
                pred, _ = model.module.beam_decode(img, src_ids, src_mask, 3, 3)
            else:
                pred, _ = model.beam_decode(img, src_ids, src_mask, 3, 3)
            # logger.info(f'Source: {" ".join(tokenizer.convert_ids_to_tokens(batch[0][0].cpu().numpy()))}')
            logger.info(f'Ground: {" ".join(tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy()))}')
            logger.info(f'Beam Generated: {" ".join(tokenizer.convert_ids_to_tokens(pred[0][0]))}')
            # if n_gpu > 1:
            #     pred = model.module.greedy_decode(batch[0], batch[1])
            # else:
            #     pred = model.greedy_decode(batch[0], batch[1])
            # logger.info(f'Beam Generated: {tokenizer.convert_ids_to_tokens(pred[0].cpu().numpy())}')
        logger.info(f'Epoch {i} finished.')
    with open(os.path.join(args.bert_model, 'bert_config.json'), 'r') as f:
        bert_config = json.load(f)
    config = {'bert_config': bert_config, 'decoder_config': decoder_config}
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config, f)
    logger.info('Training finished')


