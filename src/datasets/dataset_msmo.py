#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/11/29
# project = Dataset

import torch
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
import base_config
import numpy as np


transform = transforms.Compose(
        [transforms.Resize((base_config.IMG_SIZE, base_config.IMG_SIZE)),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)]
    )

class YourDataSetClass(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """

    def __init__(
            self, args, example, txt2img, key_targets, data_dir, img_dir, tokenizer, max_source_len, max_summ_len,img_len
    ):
        """
        Initializes a Dataset class
        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.args = args
        self.example = example
        self.mode = args.mode
        self.txt2img = txt2img
        self.key_targets = key_targets
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_summ_len = max_summ_len
        self.img_len = img_len
        

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.example)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        cur_path = self.example[index] #data1-adac9fb88f20f54f4bec37c267254eb94bbbeafe
        imgs, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask = [], [], [], [], [], [], []
        if "img" in self.mode:
            # imgs = self.get_img_item(cur_path)
            imgs = self.get_image_from_pkl(cur_path)
            imgs = imgs[:self.img_len,:]
            imgs = imgs.to(dtype=torch.float)
        
        cur_path = cur_path.split('-')
        cur_path = self.data_dir + cur_path[0] + '/article_bert/' + cur_path[1]  + '.pickle'
        with open(cur_path, 'rb') as f:
            cur_d = pickle.load(f)
            f.close()
        source_text, target_text = cur_d[0], cur_d[1]
        if 'txt' in self.mode:
            src_ids, src_mask = self.convert_source_to_feature(source_text, self.tokenizer, self.max_source_len)
            src_ids, src_mask = src_ids.to(dtype=torch.long), src_mask.to(dtype=torch.long)
        if 'key' in self.mode:
            target_id = self.key_targets[index]
            key_ids, key_mask = self._convert_target_to_feature(target_id, self.tokenizer, self.max_summ_len)
            key_ids, key_mask = key_ids.to(dtype=torch.long), key_mask.to(dtype=torch.long)
        if 'sum' in self.mode:
            tgt_ids, tgt_mask = self.convert_target_to_feature(target_text, self.tokenizer, self.max_summ_len)
            tgt_ids, tgt_mask = tgt_ids.to(dtype=torch.long), tgt_mask.to(dtype=torch.long)
        
        
        return (imgs, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask)
    
    def get_img_item(self, txt_name):
        def preprocess_image(img_path):
            img = Image.open(os.path.join(img_path)).convert("RGB")
            img = transform(img)
            return img
        
        imgs = []
        img_path = self.txt2img[txt_name]
        if len(img_path) == 0:
            return torch.randn(self.img_len, 3, base_config.IMG_SIZE, base_config.IMG_SIZE)
        img_path = img_path[0]
        img_path = self.img_dir + img_path
        imgs.append(preprocess_image(img_path))
        if self.img_len > 1:
            cur_img_ex = self.txt2img[txt_name][1:]
            
            for i, dt in enumerate(cur_img_ex):
                if i == self.img_len -1:
                    break
                img_path = self.img_dir + img_path
                imgs.append(preprocess_image(img_path))
        
        imgs = [torch.tensor(im).squeeze(0) for im in imgs]
        imgs = torch.stack(imgs)
        return imgs

    def get_image_from_pkl(self, txt_name):
        cur_path = txt_name.split('-')
        cur_path = self.data_dir + cur_path[0] + '/img_global/' + cur_path[1]  + '.npz'
        if not os.path.exists(cur_path):
            return torch.randn(self.img_len, self.args.img_global_dim)
        try:
            with open(cur_path, 'rb') as f:
                img_feats = np.load(f)
                f.close()
        except:
            img_feats = []
        img_feats = torch.tensor(img_feats)
        if img_feats.shape[0] < self.img_len:
            comple_randn = torch.randn(self.img_len-img_feats.shape[0], self.args.img_global_dim)
            img_feats = torch.cat((img_feats, comple_randn), dim=0)
        return img_feats

    def convert_exmpale_to_feature(self, source_text, target_text, tokenizer, src_max_seq_length, tgt_max_seq_length):
        src_tokens = tokenizer.tokenize(source_text)
        tgt_tokens = tokenizer.tokenize(target_text)
        if len(src_tokens) > src_max_seq_length - 2:
            src_tokens = src_tokens[:(src_max_seq_length - 2)]
        if len(tgt_tokens) > tgt_max_seq_length - 2:
            tgt_tokens = tgt_tokens[:(tgt_max_seq_length - 2)]
        src_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]
        tgt_tokens = ["[CLS]"] + tgt_tokens + ["[SEP]"]
        # no need to generate segment ids here because if we do not provide
        # bert model will generate dafault all-zero ids for us
        # and we regard single text as one sentence

        src_ids = tokenizer.convert_tokens_to_ids(src_tokens)
        tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        src_mask = [1] * len(src_ids)
        tgt_mask = [1] * len(tgt_ids)
        # Zero-pad up to the sequence length.
        src_padding = [0] * (src_max_seq_length - len(src_ids))
        tgt_padding = [0] * (tgt_max_seq_length - len(tgt_ids))
        src_ids += src_padding
        src_mask += src_padding
        tgt_ids += tgt_padding
        tgt_mask += tgt_padding

        assert len(src_ids) == src_max_seq_length
        assert len(tgt_ids) == tgt_max_seq_length
        
        return (torch.tensor(src_ids), torch.tensor(src_mask), torch.tensor(tgt_ids), torch.tensor(tgt_mask))
    
    def convert_target_to_feature(self, target_text, tokenizer, tgt_max_seq_length):
        tgt_tokens = tokenizer.tokenize(target_text)
        if len(tgt_tokens) > tgt_max_seq_length - 2:
            tgt_tokens = tgt_tokens[:(tgt_max_seq_length - 2)]
        tgt_tokens = ["[CLS]"] + tgt_tokens + ["[SEP]"]

        tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        tgt_mask = [1] * len(tgt_ids)
        # Zero-pad up to the sequence length.
        tgt_padding = [0] * (tgt_max_seq_length - len(tgt_ids))
        tgt_ids += tgt_padding
        tgt_mask += tgt_padding

        assert len(tgt_ids) == tgt_max_seq_length
        
        return (torch.tensor(tgt_ids), torch.tensor(tgt_mask))
    
    def _convert_target_to_feature(self, tgt_ids, tokenizer, tgt_max_seq_length):
        if len(tgt_ids) > tgt_max_seq_length - 2:
            tgt_ids = tgt_ids[:(tgt_max_seq_length - 2)]
        tgt_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + tgt_ids + tokenizer.convert_tokens_to_ids(["[SEP]"])
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        tgt_mask = [1] * len(tgt_ids)
        # Zero-pad up to the sequence length.
        tgt_padding = [0] * (tgt_max_seq_length - len(tgt_ids))
        tgt_ids += tgt_padding
        tgt_mask += tgt_padding

        assert len(tgt_ids) == tgt_max_seq_length
        
        return (torch.tensor(tgt_ids), torch.tensor(tgt_mask))
    
    def convert_source_to_feature(self, source_text, tokenizer, src_max_seq_length):
        src_tokens = tokenizer.tokenize(source_text)
        if len(src_tokens) > src_max_seq_length - 2:
            src_tokens = src_tokens[:(src_max_seq_length - 2)]
        src_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]

        src_ids = tokenizer.convert_tokens_to_ids(src_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        src_mask = [1] * len(src_ids)
        # Zero-pad up to the sequence length.
        src_padding = [0] * (src_max_seq_length - len(src_ids))
        src_ids += src_padding
        src_mask += src_padding

        assert len(src_ids) == src_max_seq_length
        
        return (torch.tensor(src_ids), torch.tensor(src_mask))

def get_dataset(args, tokenizer, type):
    comma_str = ""
    if args.with_comma == True:
        comma_str = "_comma"
    if type == 'dev':
        with open(args.val_ex_path, 'rb') as f:
            val_ex = pickle.load(f)
            val_ex = val_ex[0][:100] #data1-adac9fb88f20f54f4bec37c267254eb94bbbeafe
            f.close()
        with open(args.val_txt2img_path, 'rb') as f:
            txt2img = pickle.load(f)
            # txt2img = txt2img[1]
            f.close()
        key_ids_path = args.keyids_dir + 'valid_sum{}{}.pickle'.format(args.len_sum, comma_str)
        if 'key' in args.mode:
            with open(key_ids_path, 'rb') as f:
                key_ids = pickle.load(f)
                # txt2img = txt2img[1]
                f.close()
        else:
            key_ids = []
        val_set = YourDataSetClass(
        args,
        val_ex,
        txt2img,
        key_ids,
        args.data_dir,
        args.img_dir,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        args.img_len
        )
        return val_set
    
    elif type == 'train':
        with open(args.train_ex_path, 'rb') as f:
            train_ex = pickle.load(f)
            train_ex = train_ex[0]
            f.close()
        with open(args.train_txt2img_path, 'rb') as f:
            txt2img = pickle.load(f)
            # txt2img = txt2img[1]
            f.close()
        key_ids_path = args.keyids_dir + 'train_sum{}{}.pickle'.format(args.len_sum,comma_str)
        if 'key' in args.mode:
            with open(key_ids_path, 'rb') as f:
                key_ids = pickle.load(f)
                # txt2img = txt2img[1]
                f.close()
        else:
            key_ids = []
        train_set = YourDataSetClass(
        args,
        train_ex, 
        txt2img,
        key_ids,
        args.data_dir, 
        args.img_dir,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        args.img_len
        )
        return train_set
    else:
        with open(args.test_ex_path, 'rb') as f:
            test_ex = pickle.load(f)
            # test_ex = test_ex[0][:2000]
            test_ex = test_ex[0]
            f.close()
        with open(args.test_txt2img_path, 'rb') as f:
            txt2img = pickle.load(f)
            # txt2img = txt2img[1]
            f.close()
        key_ids_path = args.keyids_dir + 'test_sum{}{}.pickle'.format(args.len_sum, comma_str)
        if 'key' in args.mode:
            with open(key_ids_path, 'rb') as f:
                key_ids = pickle.load(f)
                # txt2img = txt2img[1]
                f.close()
        else:
            key_ids = []
        test_set = YourDataSetClass(
        args,
        test_ex, 
        txt2img,
        key_ids,
        args.data_dir, 
        args.img_dir,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        args.img_len
        )
        return test_set