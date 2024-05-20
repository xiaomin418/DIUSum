#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/11/29
# project = Dataset

import torch
import pickle
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import os
import spacy
eng_model = spacy.load('en_core_web_sm')
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
import base_config

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
            self, args, source_file, target_file, target_feat_file, tokenizer,
            img_dir, ext_ex_file, merge_guidance_file, 
            max_source_len, max_summ_len,img_len
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
        self.mode = self.args.mode
        self.source_file = source_file
        self.target_file = target_file
        self.target_feat_file = target_feat_file
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_summ_len = max_summ_len
        self.img_len = img_len

        self.img_dir = img_dir
        with open(self.source_file, 'r') as f:
            self.sources = f.readlines()
            f.close()
        with open(self.target_file, 'r') as f:
            self.targets = f.readlines()
            f.close()

        if os.path.exists(self.target_feat_file):
            with open(self.target_feat_file, 'rb') as f:
                self.target_keys = pickle.load(f)
                f.close()
        else:
            self.target_keys = self._get_ids_and_lens(self.targets)
            with open(self.target_feat_file, 'wb') as f:
                pickle.dump(self.target_keys, f)
                f.close()

        with open(ext_ex_file, 'rb') as f:
            self.ext_img_ex = pickle.load(f)
            f.close()
        
        # with open(merge_guidance_file, 'rb') as f:
        #     self.image_guidance = pickle.load(f)
        #     f.close()
        

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.sources)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        imgs, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask = [], [], [], [], [], [], []
        if "img" in self.mode:
            imgs = self.get_img_item(index)
            imgs = imgs.to(dtype=torch.float)
        if 'txt' in self.mode:
            src_ids, src_mask = self.convert_source_to_feature(self.sources[index], self.tokenizer, self.max_source_len)
            src_ids, src_mask = src_ids.to(dtype=torch.long), src_mask.to(dtype=torch.long)
        if 'key' in self.mode:
            target_id = self.target_keys[index]
            key_ids, key_mask = self._convert_target_to_feature(target_id, self.tokenizer, self.max_summ_len)
            key_ids, key_mask = key_ids.to(dtype=torch.long), key_mask.to(dtype=torch.long)
        if 'sum' in self.mode:
            tgt_ids, tgt_mask = self.convert_target_to_feature(self.targets[index], self.tokenizer, self.max_summ_len)
            tgt_ids, tgt_mask = tgt_ids.to(dtype=torch.long), tgt_mask.to(dtype=torch.long)
        
        img_guide = -1
        return (index, imgs, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask, img_guide)
        

    def get_img_item(self, index):
        def preprocess_image(img_path):
            img = Image.open(os.path.join(img_path)).convert("RGB")
            img = transform(img)
            return img
        
        imgs = []
        img_path = self.img_dir + '/{}.jpg'.format(index+1)
        imgs.append(preprocess_image(img_path))
        if self.img_len > 1:
            cur_img_ex = self.ext_img_ex[index]
            if len(cur_img_ex) >= 4:
                cur_img_ex = cur_img_ex[2:]
            
            for i, dt in enumerate(cur_img_ex):
                if i == self.img_len -1:
                    break
                dir1, dir2, name = dt[0].split('-')
                img_path = "{}{}/{}/{}".format(self.ext_img_dir, dir1, dir2, name)
                imgs.append(preprocess_image(img_path))
        imgs = [torch.tensor(im).squeeze(0) for im in imgs]
        imgs = torch.stack(imgs)
        return imgs
    
    def _get_ids_and_lens(self, summaris):
        dec_ids = []
        target_str = []

        effective_pos = ['NOUN','VERB', 'ADJ'] #ADP去掉

        for ind in range(len(summaris)):
            summ = summaris[ind].strip()
            org_pos = eng_model(summ)
            # summ = summ.split(" ")
            summ = [w.text for w in org_pos]
            org_pos = [w.pos_ for w in org_pos]
            choose_dec = [] # ('word',index, prior)
            for ii in range(len(org_pos)):
                if org_pos[ii] in effective_pos:
                    # print("ii:{}, len(summ):{}".format(ii, len(summ)))
                    choose_dec.append((summ[ii], ii,1))
                else:
                    choose_dec.append((summ[ii], ii,0))
            choose_dec = sorted(choose_dec, key= lambda x:x[2],reverse=True)
            choose_dec = choose_dec[: self.args.len_sum]
            choose_dec = sorted(choose_dec, key = lambda x:x[1])
            choose_dec = [x[0] for x in choose_dec]
            choose_str = " ".join(choose_dec)
            target_str.append(choose_str)
            dec = self.tokenizer.tokenize(choose_str)
            dec = self.tokenizer.convert_tokens_to_ids(dec)
            dec_ids.append(dec)
            # dec_poses.append(dec_pos)
        
        return dec_ids
    
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

def get_dataloader(args, mode, tokenizer):
    if mode=='dev':
        val_set = YourDataSetClass(
        args, 
        args.dev_source, 
        args.dev_target, 
        args.target_feat_dir+"/dev_sum{}.pickle".format(args.len_sum),
        tokenizer,
        args.dev_img_dir,
        args.dev_ext_ex_file,
        args.merge_guidance_dir+ '/dev_teacher.pickle', 
        args.max_src_len,
        args.max_tgt_len,
        args.img_len
        )
        eval_sampler = RandomSampler(val_set)
        eval_dataloader = DataLoader(val_set, sampler=eval_sampler, batch_size=args.train_batch_size, drop_last=True)
        return val_set, eval_dataloader
    elif mode=='train':
        training_set = YourDataSetClass(
        args, 
        args.train_source, 
        args.train_target, 
        args.target_feat_dir+"/train_sum{}.pickle".format(args.len_sum), 
        tokenizer,
        args.train_img_dir,
        args.train_ext_ex_file,
        args.merge_guidance_dir+ '/train_teacher.pickle', 
        args.max_src_len,
        args.max_tgt_len,
        args.img_len
        )
        train_sampler = RandomSampler(training_set)
        train_dataloader = DataLoader(training_set, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)
        return training_set, train_dataloader
    else:
        test_set = YourDataSetClass(
        args, 
        args.test_source, 
        args.test_target, 
        args.target_feat_dir+"/test_sum{}.pickle".format(args.len_sum),
        tokenizer,
        args.test_img_dir,
        args.test_ext_ex_file,
        args.merge_guidance_dir+ '/test_teacher.pickle', 
        args.max_src_len,
        args.max_tgt_len,
        args.img_len
        )
        # train_data = create_dataset(train_features)
        test_sampler = SequentialSampler(test_set)
        test_dataloader = DataLoader(test_set, sampler=test_sampler, batch_size=8, drop_last=True, shuffle=False)
        return test_set, test_dataloader