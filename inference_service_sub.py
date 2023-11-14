# inference_service_sub.py
import os
import logging
import io
import numpy as np
from inference_service import *

from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer
from t3qai_client import DownloadFile
import t3qai_client as tc
from t3qai_client import T3QAI_TRAIN_OUTPUT_PATH, T3QAI_TRAIN_MODEL_PATH, T3QAI_TRAIN_DATA_PATH, T3QAI_TEST_DATA_PATH, T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH

def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[search log] ★ Files and directories in {} :'.format(path))
    logging.info('[search log] ★ dir_list : {}'.format(dir_list)) 
    
list_files_directories(os.getcwd())

logging.info('[Search log] ★ Files and directories in [T3QAI_TRAIN_MODEL_PATH]')
list_files_directories(T3QAI_TRAIN_MODEL_PATH)

logging.info('[Search log] ★ Files and directories in [T3QAI_MODULE_PATH]')
list_files_directories(T3QAI_MODULE_PATH)

logging.info('[Search log] ★ Files and directories in [T3QAI_INIT_MODEL_PATH]')
list_files_directories(T3QAI_INIT_MODEL_PATH)

logging.info('[Search log] ★ Files and directories in [T3QAI_TRAIN_OUTPUT_PATH]')
list_files_directories(T3QAI_TRAIN_OUTPUT_PATH)

logging.info('[Search log] ★ Files and directories in [T3QAI_TEST_DATA_PATH]')
list_files_directories(T3QAI_TEST_DATA_PATH)
#########################################################

 
token_path = os.path.join(os.getcwd(), '') + 'mods/algo_30/1/src/hugging_tokenizer' 
logging.info('[token_path]:{}'.format(token_path))
#pretrain_model = AutoTokenizer.from_pretrained(token_path, local_files_only=True)
#logging.info('[why!!!! ]: {}'.format(pretrain_model))

py_file_location = "/data/aip/logs/t3qai/000820703792611374/mods/algo_30/1/ext/models/1103_1718/model_step_10000.pt"
sys.path.append(os.path.abspath(py_file_location))

#현재 디렉토리 경로가져오기
list_files_directories(os.getcwd())
os.chdir('kobertsum')
os.chdir('src')
dir_path=os.getcwd()
list_files_directories(os.getcwd())

# logging.info('[Search log] ★ /work/kobertsum/src 현재 경로 : {} :'.format(dir_path)
    
# Imports  
import copy
import torch
import torch.nn as nn
import argparse
import math
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from torch.nn.init import xavier_uniform_
from transformers import AutoModel, AutoTokenizer

from models.model_builder import *
from models.encoder import *

from prepro.tokenization_kobert import *
from prepro.tokenization_kobert import KoBertTokenizer
from kss import split_sentences

# 저장 파일 확인
def list_files_directories(path):
    # Get the list of all files and directories in current working directory
    dir_list = os.listdir(path)
    logging.info('[Search log] ★ Files and directories in {} :'.format(path))
    logging.info('[Search log] ★ dir_list : {}'.format(dir_list))  

# 인자 설정
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
parser.add_argument("-model_path", default='../models/')
parser.add_argument("-result_path", default='../results/cnndm')
parser.add_argument("-temp_dir", default='../temp')

parser.add_argument("-batch_size", default=140, type=int)
parser.add_argument("-test_batch_size", default=200, type=int)

parser.add_argument("-max_pos", default=512, type=int)
parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-load_from_extractive", default='', type=str)

parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
parser.add_argument("-lr_bert", default=2e-3, type=float)
parser.add_argument("-lr_dec", default=2e-3, type=float)
parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
parser.add_argument("-dec_dropout", default=0.2, type=float)
parser.add_argument("-dec_layers", default=6, type=int)
parser.add_argument("-dec_hidden_size", default=768, type=int)
parser.add_argument("-dec_heads", default=8, type=int)
parser.add_argument("-dec_ff_size", default=2048, type=int)
parser.add_argument("-enc_hidden_size", default=512, type=int)
parser.add_argument("-enc_ff_size", default=512, type=int)
parser.add_argument("-enc_dropout", default=0.2, type=float)
parser.add_argument("-enc_layers", default=6, type=int)

parser.add_argument("-pretrained_model", default='bert', type=str)

parser.add_argument("-mode", default='', type=str)
parser.add_argument("-select_mode", default='greedy', type=str)
parser.add_argument("-map_path", default='../../data/')
parser.add_argument("-raw_path", default='../../line_data')
parser.add_argument("-save_path", default='../../data/')

parser.add_argument("-shard_size", default=2000, type=int)
parser.add_argument('-min_src_nsents', default=1, type=int)    # 3
parser.add_argument('-max_src_nsents', default=120, type=int)    # 100
parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)    # 5
parser.add_argument('-max_src_ntokens_per_sent', default=300, type=int)    # 200
parser.add_argument('-min_tgt_ntokens', default=1, type=int)    # 5
parser.add_argument('-max_tgt_ntokens', default=500, type=int)    # 500

parser.add_argument("-lower", type=str2bool, nargs='?',const=True,default=True)
parser.add_argument("-use_bert_basic_tokenizer", type=str2bool, nargs='?',const=True,default=False)

parser.add_argument('-log_file', default='../../logs/cnndm.log')

parser.add_argument('-dataset', default='')

parser.add_argument('-n_cpus', default=2, type=int)

# params for EXT
parser.add_argument("-ext_dropout", default=0.2, type=float)
parser.add_argument("-ext_layers", default=2, type=int)
parser.add_argument("-ext_hidden_size", default=768, type=int)
parser.add_argument("-ext_heads", default=8, type=int)
parser.add_argument("-ext_ff_size", default=2048, type=int)

parser.add_argument("-label_smoothing", default=0.1, type=float)
parser.add_argument("-generator_shard_size", default=32, type=int)
parser.add_argument("-alpha",  default=0.6, type=float)
parser.add_argument("-beam_size", default=5, type=int)
parser.add_argument("-min_length", default=15, type=int)
parser.add_argument("-max_length", default=150, type=int)
parser.add_argument("-max_tgt_len", default=140, type=int)

args = parser.parse_args('')

def exec_init_model():
    model_path = os.path.join('/data/nas/search/t3q-dl/aip/logs/t3qai/000820703792611374/mods/algo_30/1/kobertsum/ext/models/1103_1718/', 'model_step_10000.pt')
    model = model_path
    model_info_dict = {
        "model": model

    }
    return model_info_dict

def exec_inference_dataframe(df, model_info_dict):
    
    logging.info('[Search log] the start line of the function [exec_inference_dataframe]')
    
    ## 학습 모델 준비
    model = model_info_dict['model'] 
    logging.info('[model_ready_] : {}'.format(model))

    result = []
    
    import kss
    
    for new_doc in df:
        logging.info(f'new_doc : {new_doc}')
        list_text = new_doc
        str_text = ''.join(list_text)
        logging.info('[Search log] ★  df check : {} :'.format(str_text))
        logging.info('[Search log] ★  df check : {} :'.format(str_text.__class__))
        
        kss_text = kss.split_sentences(str_text) # 분류 후 kss (len 13)
        logging.info('[kss_text] : {}'.format(len(kss_text)))
        summarize_result = summarize(kss_text)
        final_summarize = [kss_text[i] for i in summarize_result[0][:len(kss_text)//3]]
        result.append(final_summarize)
        
    logging.info('summarize : {}'.format(result))
    
    return {
        "summarize" : result
    }

 
    #########################################################################################
    #########################################################################################
# Class
# BertData 변환
class BertData():
    def __init__(self):
        # self.tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        # self.tokenizer = pretrain_model
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        # self.sep_vid = self.tokenizer.token2idx[self.sep_token]
        # self.cls_vid = self.tokenizer.token2idx[self.cls_token]
        # self.pad_vid = self.tokenizer.token2idx[self.pad_token]

        self.sep_vid = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        self.cls_vid = self.tokenizer.convert_tokens_to_ids(self.cls_token)
        self.pad_vid = self.tokenizer.convert_tokens_to_ids(self.pad_token)

    def preprocess(self, src):

        if (len(src) == 0):
            return None

        original_src_txt = [' '.join(s) for s in src]
        idxs = [i for i, s in enumerate(src) if (len(s) > 1)]

        src = [src[i][:2000] for i in idxs]
        src = src[:1000]

        if (len(src) < 3):
            return None

        src_txt = [' '.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        #src_subtokens = src_subtokens[:4094]  ## 512가 최대인데 [SEP], [CLS] 2개 때문에 510
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = None
        src_txt = [original_src_txt[i] for i in idxs]
        tgt_txt = None
        
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt_txt
    

# network
class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, query, inputs, mask):
        if (iter != 0):
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)

class ExtTransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, num_inter_layers=0):
        super(ExtTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_inter_layers = num_inter_layers
        self.pos_emb = PositionalEncoding(dropout, d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, top_vecs, mask):
        """ See :obj:`EncoderBase.forward()`"""

        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.num_inter_layers):
            x = self.transformer_inter[i](i, x, x, ~ mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
    
class BigBird(nn.Module):
    def __init__(self, temp_dir, finetune=False):
        super(BigBird, self).__init__()
        # self.model = AutoModel.from_pretrained("monologg/kobigbird-bert-base") # 모델 변경
        self.model = AutoModel.from_pretrained(token_path)
        self.finetune = finetune

    def forward(self, x, mask):
        if self.finetune:
            outputs = self.model(x, attention_mask=mask)
            top_vec = outputs.last_hidden_state
        else:
            self.eval()
            with torch.no_grad():
                outputs = self.model(x, attention_mask=mask)
                top_vec = outputs.last_hidden_state
        return top_vec 
    
class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        logging.info('[args:{}'.format(self.args))
        logging.info('[device:{}'.format(self.device))  
        
        self.bigbird = AutoModel.from_pretrained(token_path)
        logging.info('[pretrain_model:{}'.format(self.bigbird))
      
        self.ext_layer = ExtTransformerEncoder(self.bigbird.config.hidden_size, args.ext_ff_size, args.ext_heads,
                                               args.ext_dropout, args.ext_layers)

        if(args.max_pos>512):
            my_pos_embeddings = nn.Embedding(args.max_pos, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(args.max_pos-512,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings


        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=False)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bigbird(src)[0]
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
    
##########################################################################################
# summarize 함수 선언
def summarize(text):
    
    def txt2input(text):
        bertdata = BertData()
        txt_data = bertdata.preprocess(text)
        data_dict = {"src":txt_data[0],
                    "labels":[0,1,2],
                    "segs":txt_data[2],
                    "clss":txt_data[3],
                    "src_txt":txt_data[4],
                    "tgt_txt":None}
     
        input_data = []
        input_data.append(data_dict)
        return input_data
    
    input_data = txt2input(text)
    device = torch.device("cuda")
    
    def _pad(data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data
    
    pre_src = [x['src'] for x in input_data]
    pre_segs = [x['segs'] for x in input_data]
    pre_clss = [x['clss'] for x in input_data]

    src = torch.tensor(_pad(pre_src, 0)).cuda()
    segs = torch.tensor(_pad(pre_segs, 0)).cuda()
    mask_src = ~(src == 0)

    clss = torch.tensor(_pad(pre_clss, -1)).cuda()
    mask_cls = ~(clss == -1)
    clss[clss == -1] = 0

    clss.to(device).long()
    mask_cls.to(device).long()
    segs.to(device).long()
    mask_src.to(device).long()
    
    # py_file_location
    # checkpoint = torch.load("./kobertsum/ext/models/model_step_27000.pt")  # BigBird
    checkpoint = torch.load(py_file_location)
    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    with torch.no_grad():
        sent_scores, mask = model(src, segs, clss, mask_src, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        print(sent_scores)
        selected_ids = np.argsort(-sent_scores, 1)
        print(selected_ids)
    
    return selected_ids
#########################################################################################

def exec_inference_file(files, model_info_dict):
    
    """파일기반 추론함수는 files와 로드한 model을 전달받습니다."""
    
    """구현 패스"""
    logging.info('[hunmin log] the start line of the function [exec_inference_file]')
    
    model = model_info_dict['model']
 
    inference_result = []
    
    for one_file in files:
        logging.info(f'[hunmin log] inference: {one_file.filename}')
        inference_file = one_file.file
        new_doc = inference_file

        logging.info(f'[hunmin log] predict: {one_file.filename}')

        inference_result.append(topics)
        
    result = {'inference' : inference_result}
    return result



    