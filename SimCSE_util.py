# SimCSE++ のプログラムを書く

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import collections
import math
import pickle
import warnings
from enum import Enum
import numpy as np

from transformers import BatchEncoding, DataCollatorForLanguageModeling, SpecialTokensMixin, Trainer
from transformers.trainer_pt_utils import DistributedLengthGroupedSampler, DistributedSamplerWithLoop, LengthGroupedSampler
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertPreTrainingHeads, BertOnlyMLMHead, BertLMPredictionHead, BertForPreTrainingOutput
from transformers.utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, get_torch_version, logging, replace_return_docstrings

from datasets import Dataset
from dataclasses import dataclass
from transformers.utils.generic import ModelOutput
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union, Iterator # 追加
from collections.abc import Mapping # 追加
import sys # 追加


@dataclass
class BertForSimCSEpp(BertPreTrainedModel) :
    def __init__(self, config) :
        super().__init__(config)

        self.bert = BertModel(config)
        # self.cls = BertPreTrainingHeads(config)

        # クラストークンをMLPに入れて汎化性能向上を期待
        # self.cls = nn.Linear(config.hidden_size, config.hidden_size)

        # 次元ごとのSimCSEのlossの反映係数
        self.alpha = 0.1 # default:0.1(in paper)

        # cosine similarity
        self.cos = nn.CosineSimilarity(dim=-1)

        # cross entropy loss
        self.criterion = nn.CrossEntropyLoss()

        # ネットワークの初期化
        self.post_init()


    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # drop out 有り
        outputs1 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output1, pooled_output1 = outputs1[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量

        # drop out 無し
        for name, layer in self.bert.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.eval()

        outputs2 = self.bert( 
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        for name, layer in self.bert.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.train()
        
        sequence_output1, pooled_output2 = outputs2[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量

    
        # SimCSE++ loss
        # cl_loss = self.data_infoNCE_loss(pooled_output1, pooled_output2, temp=0.05)
        # dcl_loss = self.dim_NeXent_loss(pooled_output1.T, pooled_output2.T, temp=5)
        
        cl_loss = self.infoNCE_loss(pooled_output1, pooled_output2, temp=0.05, type="data")
        dcl_loss = self.infoNCE_loss(pooled_output1.T, pooled_output2.T, temp=5, type="dim")

        total_loss = cl_loss + self.alpha * dcl_loss
    
        
        return BertForSimCSEppOutput(
            loss=total_loss,
            contrastive_learning_loss=cl_loss,
            dimensino_contrastive_learning_loss=dcl_loss,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )
    
    # Contrastive Learning loss を実装
    def infoNCE_loss(self, dropout_emb, not_dropout_emb, temp=0.05, type="data") :
        
        feature1 = F.normalize(dropout_emb, dim=1)
        feature2 = F.normalize(not_dropout_emb, dim=1)

        if type == "dim" :
            temp = 5
            similarity_matrix = torch.matmul(feature1, feature2.T)
        elif type == "data" :
            similarity_matrix = self.cos(feature1.unsqueeze(0), feature2.unsqueeze(1))
        else :
            print("You fogget info_NCE type setting")
            sys.exit(1)

        mask = torch.eye(feature1.shape[0], dtype=torch.bool).to(dropout_emb.device)

        positives = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
        negatives = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dropout_emb.device)

        logits = logits / temp

        cl_loss = self.criterion(logits, labels).to(dropout_emb.device)

        return cl_loss
       
    def data_infoNCE_loss(self, dropout_emb, not_dropout_emb, temp=0.05, trading_off_ratio=0.9) :
        
        pos_ratio = dropout_emb.size(0) / 78 # 12C2 + 12
        neg_ratio = 1 - pos_ratio

        feature1 = F.normalize(dropout_emb, dim=1)
        feature2 = F.normalize(not_dropout_emb, dim=1)

        pos_sim = pos_ratio * (self.cos(feature1, feature2) / temp)
        pos_loss = -1 * pos_sim

        cos_sim = 1 * self.cos(feature1.unsqueeze(0), feature2.unsqueeze(1)) / temp
        neg_mask = torch.eye(feature1.size(0), device=feature1.device)
        neg_sim1 = torch.sum(cos_sim * neg_mask, dim=1)
        neg_sim2 = trading_off_ratio * torch.sum(cos_sim * (torch.ones_like(cos_sim, device=feature1.device) - neg_mask), dim=1)
        neg_loss = torch.log(neg_sim1 + neg_sim2)

        cl_loss = torch.mean(pos_loss + neg_loss)

        return cl_loss
    
    def dim_NeXent_loss(self, dropout_emb, not_dropout_emb, temp=5) :
        
        dropout_emb_mean = torch.mean(dropout_emb, dim=1, keepdim=True)
        not_dropout_emb_mean = torch.mean(not_dropout_emb, dim=1, keepdim=True)
        dropout_emb_std = torch.std(dropout_emb, dim=1, keepdim=True)
        not_dropout_emb_std = torch.std(not_dropout_emb, dim=1, keepdim=True)

        similarity_matrix = torch.matmul(
            ((dropout_emb - dropout_emb_mean) / dropout_emb_std),
            ((not_dropout_emb - not_dropout_emb_mean) / not_dropout_emb_std).T
            ) / temp
        
        pos_loss = -1 * (similarity_matrix * torch.eye(similarity_matrix.size(0), device=dropout_emb.device))
        neg_loss = torch.sum(similarity_matrix, dim=1)
        cl_loss = torch.mean(pos_loss + neg_loss)

        return cl_loss

#@dataclass
class BertForSimCSE(BertPreTrainedModel) :
    def __init__(self, config) :
        super().__init__(config)

        self.bert = BertModel(config)
        # self.cls = BertPreTrainingHeads(config)

        # クラストークンをMLPに入れて汎化性能向上を期待
        # self.cls = nn.Linear(config.hidden_size, config.hidden_size)

        # cosine similarity
        self.cos = nn.CosineSimilarity(dim=-1)

        # cross entropy loss
        self.criterion = nn.CrossEntropyLoss()

        # ネットワークの初期化
        self.post_init()

        # not dropout
        self.not_dropout = False
    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # drop out 有り
        outputs1 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output1, pooled_output1 = outputs1[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量

        
        if self.not_dropout == True :
            # drop out 無しの場合
            for name, layer in self.bert.named_modules():
                if isinstance(layer, nn.Dropout):
                    layer.eval()
                else :
                    pass
        else :
            pass

        outputs2 = self.bert( 
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # drop out 無しの場合，以下を実行
        if self.not_dropout == True :
            for name, layer in self.bert.named_modules():
                if isinstance(layer, nn.Dropout):
                    layer.train()
                else :
                    pass
        else :
            pass
        
        sequence_output1, pooled_output2 = outputs2[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量

        # SimCSE loss
        # cl_loss = self.data_infoNCE_loss(pooled_output1, pooled_output2, temp=0.05)
        cl_loss = self.infoNCE_loss(pooled_output1, pooled_output2, temp=0.05)

        
        return BertForSimCSEOutput(
            loss=cl_loss,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )
    
    # Contrastive Learning loss を実装
    def infoNCE_loss(self, dropout_emb, not_dropout_emb, temp=0.10) :
        
        if self.not_dropout == True :
            feature1 = F.normalize(dropout_emb, dim=1)
            feature2 = F.normalize(not_dropout_emb, dim=1)
            
            similarity_matrix = self.cos(feature1.unsqueeze(0), feature2.unsqueeze(1))
            
            mask = torch.eye(feature1.shape[0], dtype=torch.bool).to(dropout_emb.device)

            positives = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
            negatives = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)
        
        else :
            labels = torch.cat([torch.arange(dropout_emb.size(0)) for i in range(2)], dim=0).to(dropout_emb.device)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.to(dropout_emb.device)

            features = torch.cat([dropout_emb, not_dropout_emb], dim=0)
            features = F.normalize(features, dim=1)

            similarity_matrix = self.cos(features.unsqueeze(0), features.unsqueeze(1))

            mask = torch.eye(labels.shape[0], dtype=torch.bool, device=dropout_emb.device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
            
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(dropout_emb.device)

        logits = logits / temp

        cl_loss = self.criterion(logits, labels).to(dropout_emb.device)

        return cl_loss
       
    def data_infoNCE_loss(self, dropout_emb, not_dropout_emb, temp=0.05, trading_off_ratio=1.0) :
        
        pos_ratio = dropout_emb.size(0) / 78 # 12C2 + 12
        neg_ratio = 1 - pos_ratio

        feature1 = F.normalize(dropout_emb, dim=1)
        feature2 = F.normalize(not_dropout_emb, dim=1)

        pos_sim = pos_ratio * (self.cos(feature1, feature2) / temp)
        pos_loss = -1 * pos_sim

        cos_sim = 1 * self.cos(feature1.unsqueeze(0), feature2.unsqueeze(1)) / temp
        neg_mask = torch.eye(feature1.size(0), device=feature1.device)
        neg_sim1 = torch.sum(cos_sim * neg_mask, dim=1)
        neg_sim2 = trading_off_ratio * torch.sum(cos_sim * (torch.ones_like(cos_sim, device=feature1.device) - neg_mask), dim=1)
        neg_loss = torch.log(neg_sim1 + neg_sim2)

        cl_loss = torch.mean(pos_loss + neg_loss)

        return cl_loss

class BertForMaskedMLandSimCSEpp(BertPreTrainedModel) :
    def __init__(self, config) :
        super().__init__(config)

        # Bertモデルの定義
        self.bert = BertModel(config)
        
        # MLM と SimCSEpp ヘッドの定義
        self.cls = BertMLMandSimCSEppHeads(config)

        # クラストークンをMLPに入れて汎化性能向上を期待
        # self.cls = nn.Linear(config.hidden_size, config.hidden_size)

        # 次元ごとのSimCSEのlossの反映係数
        self.alpha = 0.1 # default:0.1(in paper)
        
        # MLM と SimCSEpp の loss の割合 (beta=0.1: mlm*0.1+simcsepp)
        self.beta = 1.0

        # MLM と SimCSEpp の loss の割合（gamma=2.0： mlm+2.0*simcsepp）
        self.gamma = 2.0

        # cosine similarity
        self.cos = nn.CosineSimilarity(dim=-1)

        # cross entropy loss
        self.criterion = nn.CrossEntropyLoss()

        # ネットワークの初期化
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias    

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # drop out 有り
        outputs1 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output1, pooled_output1 = outputs1[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量

        # drop out 無し
        for _, layer in self.bert.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.eval()

        outputs2 = self.bert( 
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        for _, layer in self.bert.named_modules():
            if isinstance(layer, nn.Dropout):
                layer.train()
        
        sequence_output2, pooled_output2 = outputs2[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量

        prediction_scores, data_logits, data_labels, dim_logits, dim_labels = self.cls(sequence_output1, pooled_output1, pooled_output2)

    
        # MLM loss
        masked_lm_loss = self.criterion(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        # SimCSE++ loss
        cl_loss = self.criterion(data_logits, data_labels)
        dcl_loss = self.criterion(dim_logits, dim_labels)
        

        total_loss = self.beta * masked_lm_loss + self.gamma * (cl_loss + self.alpha * dcl_loss)
    
        
        return BertForMLMandSimCSEppOutput(
            loss=total_loss,
            masked_lm_loss=masked_lm_loss,
            contrastive_learning_loss=cl_loss,
            dimensino_contrastive_learning_loss=dcl_loss,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )

class BertForMaskedMLandSimCSE(BertPreTrainedModel) :
    def __init__(self, config) :
        super().__init__(config)

        # not dropout
        self.not_dropout = False

        # Bert model の定義
        self.bert = BertModel(config)

        # MLM と SimCSE ヘッドの定義
        self.cls = BertMLMandSimCSEHeads(config, self.not_dropout)

        # クラストークンをMLPに入れて汎化性能向上を期待
        # self.cls = nn.Linear(config.hidden_size, config.hidden_size)

        # MLM と SimCSE の loss の割合 (beta=0.1: mlm*0.1+simcse)
        self.beta = 1.0

        # MLM と SimCSE の loss の割合（gamma=1.0： mlm+1.0*simcse）
        self.gamma = 1.0

        # cosine similarity
        self.cos = nn.CosineSimilarity(dim=-1)

        # cross entropy loss
        self.criterion = nn.CrossEntropyLoss()

        # ネットワークの初期化
        self.post_init()
    
    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings
        self.cls.predictions.bias = new_embeddings.bias

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # drop out 有り
        outputs1 = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output1, pooled_output1 = outputs1[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量


        if self.not_dropout == True :
            # drop out 無しの場合
            for name, layer in self.bert.named_modules():
                if isinstance(layer, nn.Dropout):
                    layer.eval()
                else :
                    pass
        else :
            pass

        outputs2 = self.bert( 
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # drop out 無しの場合，以下を実行
        if self.not_dropout == True :
            for name, layer in self.bert.named_modules():
                if isinstance(layer, nn.Dropout):
                    layer.train()
                else :
                    pass
        else :
            pass
        
        sequence_output2, pooled_output2 = outputs2[:2] # pooled_output が cls トークンで，sequence_output が文章の埋め込み特徴量

        prediction_scores, cl_logits, cl_labels = self.cls(sequence_output1, pooled_output1, pooled_output2)


        # MLM loss
        masked_lm_loss = self.criterion(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        # SimCSE loss      
        cl_loss = self.criterion(cl_logits, cl_labels)

        
        total_loss = self.beta * masked_lm_loss + self.gamma * cl_loss
    
        
        return BertForMLMandSimCSEOutput(   
            loss=total_loss,
            masked_lm_loss=masked_lm_loss,
            contrastive_learning_loss=cl_loss,
            hidden_states=outputs1.hidden_states,
            attentions=outputs1.attentions,
        )


class BertMLMandSimCSEppHeads(BertOnlyMLMHead) :
    def __init__(self, config):
        super().__init__(config)
        self.predictions = BertLMPredictionHead(config)
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def forward(self, sequence_output, pooled_output1, pooled_output2):
        prediction_scores = self.predictions(sequence_output)
        data_logits, data_labels = self.infoNCE(pooled_output1, pooled_output2, temp=0.05, type="data")
        dim_logits, dim_labels = self.infoNCE(pooled_output1, pooled_output2, temp=5, type="dim")
        return prediction_scores, data_logits, data_labels, dim_logits, dim_labels
    
    def infoNCE(self, feature1, feature2, temp=0.05, type="data") :

        feature1 = F.normalize(feature1, dim=1)
        feature2 = F.normalize(feature2, dim=1)

        if type == "dim" :
            temp = 5
            similarity_matrix = torch.matmul(feature1.T, feature2)
        elif type == "data" :
            similarity_matrix = self.cos(feature1.unsqueeze(0), feature2.unsqueeze(1))
        else :
            print("You fogget info_NCE type setting")
            sys.exit(1)

        mask = torch.eye(similarity_matrix.shape[0], dtype=torch.bool).to(feature1.device)

        positives = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
        negatives = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(feature1.device)

        logits = logits / temp
        
        return logits, labels


class BertMLMandSimCSEHeads(BertOnlyMLMHead) :
    def __init__(self, config, not_dropout=False) :
        super().__init__(config)
        self.not_dropout = not_dropout
        self.predictions = BertLMPredictionHead(config)
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def forward(self, sequence_output, pooled_output1, pooled_output2):
        prediction_scores = self.predictions(sequence_output)
        data_logits, data_labels = self.infoNCE(pooled_output1, pooled_output2, temp=0.05)
        return prediction_scores, data_logits, data_labels
    
    def infoNCE(self, feature1, feature2, temp=0.05) :

        if self.not_dropout == True :
            feature1 = F.normalize(feature1, dim=1)
            feature2 = F.normalize(feature2, dim=1)
            
            similarity_matrix = self.cos(feature1.unsqueeze(0), feature2.unsqueeze(1))
            
            mask = torch.eye(feature1.shape[0], dtype=torch.bool).to(feature1.device)

            positives = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
            negatives = similarity_matrix[~mask.bool()].view(similarity_matrix.shape[0], -1)
        
        else :
            labels = torch.cat([torch.arange(feature1.size(0)) for i in range(2)], dim=0).to(feature1.device)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            labels = labels.to(feature1.device)

            features = torch.cat([feature1, feature2], dim=0)
            features = F.normalize(features, dim=1)

            similarity_matrix = self.cos(features.unsqueeze(0), features.unsqueeze(1))

            mask = torch.eye(labels.shape[0], dtype=torch.bool, device=feature1.device)
            labels = labels[~mask].view(labels.shape[0], -1)
            similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

            positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
            negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
            
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(feature1.device)

        logits = logits / temp
        
        return logits, labels


@dataclass
class BertForSimCSEppOutput(ModelOutput) :
    
    loss: Optional[torch.FloatTensor] = None
    contrastive_learning_loss: torch.FloatTensor = None
    dimensino_contrastive_learning_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BertForSimCSEOutput(ModelOutput) :
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BertForMLMandSimCSEppOutput(ModelOutput) :
    
    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: torch.FloatTensor = None
    contrastive_learning_loss: torch.FloatTensor = None
    dimensino_contrastive_learning_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class BertForMLMandSimCSEOutput(ModelOutput) :
    
    loss: Optional[torch.FloatTensor] = None
    masked_lm_loss: torch.FloatTensor = None
    contrastive_learning_loss: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None