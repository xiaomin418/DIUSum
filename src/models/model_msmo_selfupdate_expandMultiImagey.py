import torch.nn as nn
import torch
import os
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer import Constants
from transformer.Models import get_non_pad_mask, get_sinusoid_encoding_table, get_attn_key_pad_mask, get_subsequent_mask
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.modeling import BertEmbeddings
from torch.utils import checkpoint
import pdb
from random import random
from transformer.Beam import Beam
import torchvision.models as models

CONFIG_NAME = 'bert_config.json'
import base_config

class BertDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, decoder_config, embedding, device, dropout=0.1):

        super().__init__()
        self.len_max_seq = decoder_config['len_max_seq']
        d_word_vec = decoder_config['d_word_vec']
        n_layers = decoder_config['n_layers']
        n_head = decoder_config['n_head']
        d_k = decoder_config['d_k']
        d_v = decoder_config['d_v']
        d_model = decoder_config['d_model']
        d_inner = decoder_config['d_inner']  # should be equal to d_model
        vocab_size = decoder_config['vocab_size']

        self.device = device
        self.n_position = self.len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.n_position, d_word_vec, padding_idx=0),
            freeze=True)
        
        self.embedding = embedding

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.last_linear = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_seq, src_seq, enc_output):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)
            
        tgt_pos = torch.arange(1, tgt_seq.size(-1) + 1).unsqueeze(0).repeat(tgt_seq.size(0), 1).to(self.device)
        # -- Forward
        dec_output = self.embedding(tgt_seq) + self.position_enc(tgt_pos)
        
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)
        
        return self.last_linear(dec_output)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=base_config.trunc_norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=base_config.trunc_norm_init_std)

class VggEncoder(nn.Module):

    def __init__(self,global_dim, hidden_dim, train_CNN=False):
        super(VggEncoder, self).__init__()
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        # self.train_CNN = train_CNN
        # self.vgg19 = models.vgg19(pretrained=True)
        # self.vgg19=self.vgg19.eval()
        self.W_h = nn.Linear(self.global_dim , self.hidden_dim, bias=False)
        init_linear_wt(self.W_h)

        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, global_features):
        # Fine tuning, we don't want to train
        """
        local_features = self.vgg19.features(images)
        local_features = local_features.detach()
        x = self.vgg19.avgpool(local_features)
        x = torch.flatten(x,1)
        global_features = self.vgg19.classifier[:6](x)
        global_features = global_features.detach()


        
        local_outputs = local_features.view(-1, self.tk_dim, self.hidden_dim) #B x t_k x hidden_dim
        local_features = local_outputs.view(-1, self.hidden_dim)  # B * t_k x 2*hidden_dim
        local_features = self.W_h(local_features)
        # local_features = self.dropout(local_features)
        return local_outputs, local_features,global_features
        """
        global_features = self.W_h(global_features)


        return global_features

class UseFuler(nn.Module):

    def __init__(self, feature_dim, num_classes):
        super(UseFuler, self).__init__()
        self.multi_prj = nn.Linear(feature_dim*2, feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(feature_dim, num_classes),
        )
        self.softmax = nn.Softmax(dim = -1)
        # self._initialize_weights()

    def forward(self, txt_enc, img_enc):
        import pdb
        pdb.set_trace()
        multi_enc = torch.cat((txt_enc, img_enc), dim = -1)
        multi_enc = self.multi_prj(multi_enc)
        fea_all = torch.cat((multi_enc, txt_enc), dim = -1)
        out = self.classifier(fea_all)
        out = self.softmax(out)
        return out
    
    def forward_mulitImage(self, txt_enc, img_enc):
        # import pdb
        # pdb.set_trace()
        bs, img_len, hid = img_enc.shape
        txt_expand = txt_enc.expand(bs, img_len, hid)
        multi_enc = torch.cat((txt_expand, img_enc), dim = -1)
        multi_enc = self.multi_prj(multi_enc)
        fea_all = torch.cat((multi_enc, txt_expand), dim = -1)
        out = self.classifier(fea_all)
        out = self.softmax(out)
        max_img_ids = out[:,:,1].max(dim=-1).indices
        max_img_ids_t = max_img_ids.unsqueeze(1).expand(8,2).unsqueeze(1)
        max_img_ids_h = max_img_ids.unsqueeze(1).expand(8,768).unsqueeze(1)
        gout = torch.gather(out, 1, max_img_ids_t)
        return gout, max_img_ids_h

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class BertAbsSum(nn.Module):
    def __init__(self, args, bert_model_path, decoder_config, device):
        super().__init__()

        # self.bert_encoder = BertModel.from_pretrained(bert_model_path)
        self.args = args
        self.mode = args.mode
        self.fusion_method = args.fusion_method
        if 'txt' in self.mode:
            self.bert_encoder = BertModel.from_pretrained(bert_model_path)
        if 'img' in self.mode:
            self.image_encoder = VggEncoder(args.img_global_dim, args.image_hidden_dim)
        bert_config_file = os.path.join(bert_model_path, CONFIG_NAME)
        bert_config = BertConfig.from_json_file(bert_config_file) 
        self.device = device
        self.bert_emb = BertEmbeddings(bert_config)
        self.decoder = BertDecoder(decoder_config, self.bert_emb, device) 
        self.teacher_forcing = 0.5
        if args.usefuler_assist == True:
            self.assist = UseFuler(args.image_hidden_dim, 2)
    
    def init_enc_crfer(self, bert_init_file):
        if bert_init_file != "None":
            self.load_state_dict(torch.load(bert_init_file), strict=False)
        
    def forward(self, img, src_ids, src_mask, key_ids, key_mask, tgt_ids, tgt_mask, cur_mode, img_all_use):
        # src/tgt: [batch_size, seq_len]
        # shift right
        if len(key_ids)!=0:
            key_ids = key_ids[:, :-1]
            key_mask = key_mask[:, :-1]
        if len(tgt_ids)!=0:
            tgt_ids = tgt_ids[:, :-1]
            tgt_mask = tgt_mask[:, :-1]
        # bert input: BertModel.forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True)
        # token_type_ids is not important since we only have one sentence so we can use default all zeros
        # import pdb
        # pdb.set_trace()
        if 'img' in cur_mode:
            # img = img.reshape(-1,3,self.args.IMG_SIZE, self.args.IMG_SIZE)
            img_encoded = self.image_encoder(img) # [b, img_num, hidden_size]
            b = img_encoded.size(0)
            src_imgs_nonE = torch.ones(b, self.args.img_len).to(device=img.device)
        if 'txt' in cur_mode:
            bert_encoded = self.bert_encoder(src_ids, attention_mask=src_mask, output_all_encoded_layers=False)[0]
        else:
            src_ids = torch.tensor([]).to(device=img.device)
            bert_encoded = torch.tensor([]).to(device=img.device)
        
        if 'img' in cur_mode:
            if self.args.usefuler_assist:
                pred_img_guide = self.assist(bert_encoded[:,:1,:], img_encoded) #pred_img_guide = pred_img_guide[:,:,1:]
                if img_all_use==False:
                    # img_encoded = img_encoded.view(b, 1, -1)
                    img_encoded = img_encoded * pred_img_guide[:,:,1:]
                    # expand_txt = bert_encoded[:,:1,:].expand(b, self.args.img_len, -1).contiguous()
                    # expand_txt = expand_txt.view(b,1, -1)
                    img_pad = torch.cat((bert_encoded[:,:1,:],img_encoded),dim=1)
                    with torch.no_grad():
                        pred_max_guide = pred_img_guide.max(dim=-1).indices
                        pred_max_guide = torch.zeros_like(pred_img_guide.squeeze(1)).scatter(1,pred_max_guide,1).unsqueeze(1)
                    img_encoded = torch.matmul(pred_max_guide, img_pad)
                    # img_encoded = img_encoded.view(b, self.args.img_len, -1)

            else:
                pred_img_guide = None
        else:
            img_encoded = torch.zeros_like(bert_encoded[:,:self.args.img_len,:])
            b = bert_encoded.size(0)
            src_imgs_nonE = torch.ones(b, self.args.img_len).to(device=img.device)
            pred_img_guide = None
        
            # print("bert_encoded shape: {}, img_encoded shape: {}".format(bert_encoded.shape, img_encoded.shape))
        txt_logits = self.decoder(tgt_ids, src_ids, bert_encoded)

        return (txt_logits, pred_img_guide)
    
    def beam_decode(self, img, src_seq, src_mask, beam_size, n_best):
        ''' Translation work in one batch '''

        def get_inst_idx_to_tensor_position_map(inst_idx_list):
            ''' Indicate the position of an instance in a tensor. '''
            return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

        def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, beam_size):
            ''' Collect tensor parts associated to active instances. '''

            _, *d_hs = beamed_tensor.size()
            n_curr_active_inst = len(curr_active_inst_idx)
            new_shape = (n_curr_active_inst * beam_size, *d_hs)

            beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
            beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
            beamed_tensor = beamed_tensor.view(*new_shape)

            return beamed_tensor

        def collate_active_info(
                src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list):
            # Sentences which are still active are collected,
            # so the decoder will not run on completed sentences.
            n_prev_active_inst = len(inst_idx_to_position_map)
            active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
            active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)

            active_src_seq = collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, beam_size)
            active_src_enc = collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, beam_size)
            active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        def beam_decode_step(
                inst_dec_beams, len_dec_seq, src_seq, enc_output, inst_idx_to_position_map, beam_size):
            ''' Decode and update beam status, and then return active beam idx '''

            def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
                dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
                dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
                dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
                return dec_partial_seq

            def predict_word(dec_seq, src_seq, enc_output, n_active_inst, beam_size):
                dec_output = self.decoder(dec_seq, src_seq, enc_output)
                dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                word_prob = nn.functional.log_softmax(dec_output, dim=1)
                word_prob = word_prob.view(n_active_inst, beam_size, -1)

                return word_prob

            def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map):
                active_inst_idx_list = []
                for inst_idx, inst_position in inst_idx_to_position_map.items():
                    is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                    if not is_inst_complete:
                        active_inst_idx_list += [inst_idx]

                return active_inst_idx_list

            n_active_inst = len(inst_idx_to_position_map)

            dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
            word_prob = predict_word(dec_seq, src_seq, enc_output, n_active_inst, beam_size)

            # Update the beam with predicted word prob information and collect incomplete instances
            active_inst_idx_list = collect_active_inst_idx_list(
                inst_dec_beams, word_prob, inst_idx_to_position_map)

            return active_inst_idx_list

        def collect_hypothesis_and_scores(inst_dec_beams, n_best):
            all_hyp, all_scores = [], []
            for inst_idx in range(len(inst_dec_beams)):
                scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
                all_scores += [scores[:n_best]]

                hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
                all_hyp += [hyps]
            return all_hyp, all_scores

        def decode_for_img(img, beam_size, n_best):
            with torch.no_grad():
                #-- Encode
                # img = img.reshape(-1,3,self.args.IMG_SIZE, self.args.IMG_SIZE)
                img_encoded = self.image_encoder(img) # [b, img_num, hidden_size]
                src_enc = img_encoded.reshape(-1, self.args.img_len, self.args.image_hidden_dim)
                b = src_enc.size(0)
                # transformer input: BertDecoder.forward(self, tgt_seq_embedded, tgt_pos, src_seq, enc_output, return_attns=False)
                src_imgs_nonE = torch.ones(b, self.args.img_len).to(device=img.device)
                # src_enc = self.bert_encoder(src_seq, attention_mask=src_mask, output_all_encoded_layers=False)[0]

                #-- Repeat data for beam search
                n_inst, len_s, d_h = src_enc.size()
                src_seq = src_imgs_nonE.repeat(1, beam_size).view(n_inst * beam_size, len_s)
                src_enc = src_enc.repeat(1, beam_size, 1).view(n_inst * beam_size, len_s, d_h)

                #-- Prepare beams
                inst_dec_beams = [Beam(beam_size, device=self.device) for _ in range(n_inst)]

                #-- Bookkeeping for active or not
                active_inst_idx_list = list(range(n_inst))
                inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

                #-- Decode
                for len_dec_seq in range(1, self.decoder.len_max_seq + 1):

                    active_inst_idx_list = beam_decode_step(
                        inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, beam_size)

                    if not active_inst_idx_list:
                        break  # all instances have finished their path to <EOS>

                    src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                        src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)
            return batch_hyp, batch_scores

        def decode_for_txt(src_seq, src_mask, beam_size, n_best):
            with torch.no_grad():
                #-- Encode
                src_enc = self.bert_encoder(src_seq, attention_mask=src_mask, output_all_encoded_layers=False)[0]

                #-- Repeat data for beam search
                n_inst, len_s, d_h = src_enc.size()
                src_seq = src_seq.repeat(1, beam_size).view(n_inst * beam_size, len_s)
                src_enc = src_enc.repeat(1, beam_size, 1).view(n_inst * beam_size, len_s, d_h)

                #-- Prepare beams
                inst_dec_beams = [Beam(beam_size, device=self.device) for _ in range(n_inst)]

                #-- Bookkeeping for active or not
                active_inst_idx_list = list(range(n_inst))
                inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

                #-- Decode
                for len_dec_seq in range(1, self.decoder.len_max_seq + 1):

                    active_inst_idx_list = beam_decode_step(
                        inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, beam_size)

                    if not active_inst_idx_list:
                        break  # all instances have finished their path to <EOS>

                    src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                        src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)
            return batch_hyp, batch_scores

        def decode_for_imgtxt(img, src_seq, src_mask, beam_size, n_best):
            with torch.no_grad():
                #-- Encode img
                # img = img.reshape(-1,3,self.args.IMG_SIZE, self.args.IMG_SIZE)
                cur_img_len = self.args.img_len
                cur_img_len = 1
                img_encoded = self.image_encoder(img) # [b, img_num, hidden_size]
                img_encoded = img_encoded.reshape(-1, self.args.img_len, self.args.image_hidden_dim)
                b = img_encoded.size(0)
                src_imgs_nonE = torch.ones(b, cur_img_len).to(device=img.device)
                
                #-- Encode txt 
                src_enc = self.bert_encoder(src_seq, attention_mask=src_mask, output_all_encoded_layers=False)[0]
                if self.args.usefuler_assist:
                    # import pdb
                    # pdb.set_trace()
                    pred_img_guide, max_indexs = self.assist.forward_mulitImage(src_enc[:,:1,:], img_encoded) #pred_img_guide = pred_img_guide[:,:,1:]
                    img_encoded = torch.gather(img_encoded, 1, max_indexs)

                    # pred_img_guide = self.assist(src_enc[:,:1,:], img_encoded) #pred_img_guide = pred_img_guide[:,:,1:]
                    
                    # img_encoded = img_encoded.view(b, 1, -1)
                    img_encoded = img_encoded * pred_img_guide[:,:,1:]
                    # expand_txt = src_enc[:,:1,:].expand(b, cur_img_len, -1).contiguous()
                    # expand_txt = expand_txt.view(b,1, -1)
                    img_pad = torch.cat((src_enc[:,:1,:],img_encoded),dim=1)
                    with torch.no_grad():
                        pred_max_guide = pred_img_guide.max(dim=-1).indices
                        pred_max_guide = torch.zeros_like(pred_img_guide.squeeze(1)).scatter(1,pred_max_guide,1).unsqueeze(1)
                    img_encoded = torch.matmul(pred_max_guide, img_pad)
                    # img_encoded = img_encoded.view(b, cur_img_len, -1)
                #-- concat
                src_seq = torch.cat((src_imgs_nonE,src_seq),dim=1).to(dtype=torch.long)
                src_enc = torch.cat((img_encoded,src_enc),dim=1)

                #-- Repeat data for beam search
                n_inst, len_s, d_h = src_enc.size()
                src_seq = src_seq.repeat(1, beam_size).view(n_inst * beam_size, len_s)
                src_enc = src_enc.repeat(1, beam_size, 1).view(n_inst * beam_size, len_s, d_h)

                #-- Prepare beams
                inst_dec_beams = [Beam(beam_size, device=self.device) for _ in range(n_inst)]

                #-- Bookkeeping for active or not
                active_inst_idx_list = list(range(n_inst))
                inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

                #-- Decode
                for len_dec_seq in range(1, self.decoder.len_max_seq + 1):

                    active_inst_idx_list = beam_decode_step(
                        inst_dec_beams, len_dec_seq, src_seq, src_enc, inst_idx_to_position_map, beam_size)

                    if not active_inst_idx_list:
                        break  # all instances have finished their path to <EOS>

                    src_seq, src_enc, inst_idx_to_position_map = collate_active_info(
                        src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, n_best)
            return batch_hyp, pred_img_guide

        if self.fusion_method == 'textonly':
            batch_hyp, batch_scores = decode_for_txt(src_seq, src_mask, beam_size, n_best)
        elif self.fusion_method == 'imgonly':
            batch_hyp, batch_scores = decode_for_img(img, beam_size, n_best)
        else:
            batch_hyp, batch_scores = decode_for_imgtxt(img, src_seq, src_mask, beam_size, n_best)

        return batch_hyp, batch_scores

    def greedy_decode(self, src_seq, src_mask):
        enc_output = self.bert_encoder(src_seq, attention_mask=src_mask, output_all_encoded_layers=False)[0]
        dec_seq = torch.full((src_seq.size(0), ), Constants.BOS).unsqueeze(-1).type_as(src_seq)

        for i in range(self.decoder.len_max_seq):
            dec_output = self.decoder(dec_seq, src_seq, enc_output, 1)
            dec_output = dec_output.max(-1)[1]
            dec_seq = torch.cat((dec_seq, dec_output[:, -1].unsqueeze(-1)), 1)
        return dec_seq











        
