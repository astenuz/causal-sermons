"""
An extensible implementation of the Causal Bert model from 
"Adapting Text Embeddings for Causal Inference" 
    (https://arxiv.org/abs/1905.12741)
"""
from collections import defaultdict
import os
import pickle

import scipy
from sklearn.model_selection import KFold

from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer
from transformers import BertModel, BertPreTrainedModel, BertConfig
from transformers import get_linear_schedule_with_warmup

from transformers import DistilBertTokenizer
from transformers import DistilBertModel, DistilBertPreTrainedModel

from torch.nn import CrossEntropyLoss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from scipy.special import softmax
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm
import math

from src.causal_sermons.ate import all_ate_estimators

CUDA = (torch.cuda.device_count() > 0)
MASK_IDX = 103


def platt_scale(outcome, probs):
    logits = logit(probs)
    logits = logits.reshape(-1, 1)
    log_reg = LogisticRegression(penalty='none', warm_start=True, solver='lbfgs')
    log_reg.fit(logits, outcome)
    return log_reg.predict_proba(logits)


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def make_bow_vector(ids, vocab_size, use_counts=False):
    """ Make a sparse BOW vector from a tensor of dense ids.
    Args:
        ids: torch.LongTensor [batch, features]. Dense tensor of ids.
        vocab_size: vocab size for this tensor.
        use_counts: if true, the outgoing BOW vector will contain
            feature counts. If false, will contain binary indicators.
    Returns:
        The sparse bag-of-words representation of ids.
    """
    vec = torch.zeros(ids.shape[0], vocab_size)
    ones = torch.ones_like(ids, dtype=torch.float)
    if CUDA:
        vec = vec.cuda()
        ones = ones.cuda()
        ids = ids.cuda()

    vec.scatter_add_(1, ids, ones)
    vec[:, 1] = 0.0  # zero out pad
    if not use_counts:
        vec = (vec != 0).float()
    return vec


class DragonHeads(nn.Module):
    def __init__(self, 
                 num_outcomes, hidden_size, num_confounders):
        """
        Dragon heads for Q(0), Q(1) and g

        - Can predict multiple potential outcomes at the same time via num_outcomes
        - Only one binary treatment, still 
        - Also accomodates for numeric confounders
        """
        super().__init__()

        self.num_outcomes = num_outcomes
        self.hidden_size = hidden_size
        self.num_confounders = num_confounders
        
        self.Q_cls = nn.ModuleDict()
    
        for T in range(2):
            # ModuleDict keys have to be strings.
            self.Q_cls['%d' % T] = nn.Sequential(
                nn.Linear(hidden_size + num_confounders, 200),
                nn.ReLU(),
                nn.Linear(200, num_outcomes))

        self.g_cls = nn.Linear(hidden_size + num_confounders, 1)

    def forward(self, inputs, T, Y):
        # g logits
        g = self.g_cls(inputs)

        # conditional expected outcome logits: 
        # run each example through its corresponding T matrix
        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)

        # TODO add sigmoid if binary outcomes
        Q0 = Q_logits_T0
        Q1 = Q_logits_T1

        if self.training:
            # g loss
            g_loss = F.binary_cross_entropy_with_logits(g.view(-1), T.view(-1))

            # Q loss
            T0_ws = (T == 0.0).float().unsqueeze(1).repeat(1, self.num_outcomes)
            T1_ws = (T == 1.0).float().unsqueeze(1).repeat(1, self.num_outcomes)

            Q_loss_T0 = T0_ws * F.mse_loss(Q0, Y, reduction='none')
            Q_loss_T1 = T1_ws * F.mse_loss(Q1, Y, reduction='none')
            
            Q_loss = Q_loss_T0.mean() + Q_loss_T1.mean()
        else:
            g_loss = 0.0
            Q_loss = 0.0

        return g, Q0, Q1, g_loss, Q_loss


class CausalDistilBert(DistilBertPreTrainedModel):
    """The model itself."""
    def __init__(self, config, num_outcomes=1, num_confounders=2):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.num_outcomes = num_outcomes
        self.num_confounders = num_confounders

        self.bert = DistilBertModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.dragonheads = DragonHeads(self.num_outcomes, self.config.hidden_size, self.num_confounders)

        self.init_weights()

    def forward(self, W_ids, W_len, W_mask, C, T, Y=None, use_mlm=True):
        if use_mlm:
            W_len = W_len.unsqueeze(1) - 2 # -2 because of the +1 below
            mask_class = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
            mask = (mask_class(W_len.shape).uniform_() * W_len.float()).long() + 1 # + 1 to avoid CLS
            target_words = torch.gather(W_ids, 1, mask)
            mlm_labels = torch.ones(W_ids.shape).long() * -100
            if CUDA:
                mlm_labels = mlm_labels.cuda()
            mlm_labels.scatter_(1, mask, target_words)
            W_ids.scatter_(1, mask, MASK_IDX)

        outputs = self.bert(W_ids, attention_mask=W_mask)
        seq_output = outputs[0]
        pooled_output = seq_output[:, 0]
        # seq_output, pooled_output = outputs[:2]
        # pooled_output = self.dropout(pooled_output)

        if use_mlm and self.training:
            prediction_logits = self.vocab_transform(seq_output)  # (bs, seq_length, dim)
            prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
            prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

            mlm_loss = CrossEntropyLoss()(
                prediction_logits.view(-1, self.vocab_size), mlm_labels.view(-1))
        else:
            mlm_loss = 0.0

        # constructing input of language model + confounders
        # previous version that used a categorical C
        C = C.to(pooled_output.dtype)
        T = T.to(pooled_output.dtype)

        if Y is not None:
            Y = Y.to(pooled_output.dtype)

        inputs = torch.cat((pooled_output, C), 1)

        g, Q0, Q1, g_loss, Q_loss = self.dragonheads(inputs, T, Y)

        return F.sigmoid(g), Q0, Q1, g_loss, Q_loss, mlm_loss


class CausalDataset(Dataset):
    def __init__(self, 
                 texts, confounds, treatments, outcomes, tokenizer,
                 max_length=256):
        self.texts = np.array(texts)
        self.confounds = np.array(confounds)
        self.treatments = np.array(treatments) if treatments is not None else np.full((len(self.confounds), 2), -1)
        self.outcomes = np.array(outcomes) if outcomes is not None else np.full(len(self.treatments), -1)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        return self.preprocess_data(
            self.texts[idx], self.confounds[idx], self.treatments[idx], self.outcomes[idx])

    def __len__(self):
        return len(self.texts)

    def preprocess_data(self, W, C, T, Y):
        encoded_sent = self.tokenizer.encode_plus(
            W, 
            add_special_tokens=True, 
            padding='max_length', 
            max_length=self.max_length,
            truncation=True)

        W_id = torch.tensor(encoded_sent['input_ids'])
        W_mask = torch.tensor(encoded_sent['attention_mask'])
        W_len = torch.tensor(sum(encoded_sent['attention_mask']))
        T = torch.tensor(T)
        Y = torch.tensor(Y)
        C = torch.tensor(C)
        return W_id, W_len, W_mask, C, T, Y


class CausalModelWrapper:
    """Model wrapper in charge of training and inference."""

    def __init__(self, 
                 model,
                 g_weight=1.0, Q_weight=0.1, mlm_weight=1.0,
                 batch_size=32, max_length=128, num_workers=1):
        
        self.model = model
        
        if CUDA:
            self.model = self.model.cuda()

        self.loss_weights = {
            'g': g_weight,
            'Q': Q_weight,
            'mlm': mlm_weight
        }
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

    def train_step(self, step, batch):
        if CUDA: 
            batch = (x.cuda() for x in batch)

        W_ids, W_len, W_mask, C, T, Y = batch
        self.model.zero_grad()

        g, Q0, Q1, g_loss, Q_loss, mlm_loss = self.model(W_ids, W_len, W_mask, C, T, Y)
        
        loss = self.loss_weights['g'] * g_loss + \
                self.loss_weights['Q'] * Q_loss + \
                self.loss_weights['mlm'] * mlm_loss
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss

    def train(self, texts, confounds, treatments, outcomes,
            learning_rate=2e-5, epochs=3):
        
        dataloader = self.build_dataloader(
            texts, confounds, treatments, outcomes)

        self.model.train()

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        
        total_steps = len(dataloader) * epochs
        warmup_steps = total_steps * 0.1
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        for epoch in range(epochs):
            losses = []
            self.model.train()
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                    loss = self.train_step(step, batch)
                    losses.append(loss.detach().cpu().item())
                # print(np.mean(losses))
                    # if step > 5: continue

        return self
    
    def inference_step(self, step, batch):
        if CUDA: 
            batch = (x.cuda() for x in batch)
        W_ids, W_len, W_mask, C, T, Y = batch

        # not passing Y
        g, Q0, Q1, _, _, _ = self.model(W_ids, W_len, W_mask, C, T, use_mlm=False)

        return g, Q0, Q1, T, Y

    def inference(self, texts, confounds, treatments=None, outcomes=None):
        self.model.eval()

        dataloader = self.build_dataloader(
            texts, confounds, treatments, outcomes,
            sampler='sequential')
        
        gs = []
        Q0s = []
        Q1s = []
        Ts = []
        Ys = []
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            g, Q0, Q1, T, Y = self.inference_step(step, batch)

            # so that T matches pattern of others
            T = T.unsqueeze(1)

            gs.append(g.detach().cpu())
            Q0s.append(Q0.detach().cpu())
            Q1s.append(Q1.detach().cpu())
            Ts.append(T.detach().cpu())
            Ys.append(Y.detach().cpu())

            # if step > 5: break

        # probs = np.array(list(zip(Q0s, Q1s)))
        # preds = np.argmax(probs, axis=1)  # this did not make much sense

        return (
            torch.cat(gs, dim=0).numpy(), 
            torch.cat(Q0s, dim=0).numpy(), 
            torch.cat(Q1s, dim=0).numpy(), 
            torch.cat(Ts, dim=0).numpy(), 
            torch.cat(Ys, dim=0).numpy()
        )

    def ATE(self, texts, confounds, treatments, outcomes=None):
        g, Q0, Q1, T, Y = self.inference(
            texts=texts, confounds=confounds, treatments=treatments, outcomes=outcomes)

        return all_ate_estimators(Q0, Q1, g, T, Y)
    
    def ATE_old(self, C, W, Y=None, platt_scaling=False):
        Q_probs, _, Ys = self.inference(W, C, outcome=Y)
        if platt_scaling and Y is not None:
            Q0 = platt_scale(Ys, Q_probs[:, 0])[:, 0]
            Q1 = platt_scale(Ys, Q_probs[:, 1])[:, 1]
        else:
            Q0 = Q_probs[:, 0]
            Q1 = Q_probs[:, 1]

        return np.mean(Q0 - Q1)

    def build_dataloader_old(self, 
                         texts, confounds, treatments=None, outcomes=None,
                         tokenizer=None, sampler='random'):
        def collate_CandT(data):
            # sort by (C, T), so you can get boundaries later
            # (do this here on cpu for speed)
            data.sort(key=lambda x: (x[1], x[2]))
            return data
        
        confounds = confounds.values
        if outcomes is not None:
            outcomes = outcomes.values

        # fill with dummy values
        if treatments is None:
            treatments = np.full((len(confounds), 2), -1)  # like this to fit expected format
        if outcomes is None:
            outcomes = np.full(len(treatments), -1)

        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased', do_lower_case=True)
            
        out = defaultdict(list)
        for i, (W, C, T, Y) in enumerate(zip(texts, confounds, treatments, outcomes)):
            # out['W_raw'].append(W)
            encoded_sent = tokenizer.encode_plus(
                W, 
                add_special_tokens=True,
                padding='max_length',
                max_length=self.max_length,
                truncation=True)

            out['W_ids'].append(encoded_sent['input_ids'])
            out['W_mask'].append(encoded_sent['attention_mask'])
            out['W_len'].append(sum(encoded_sent['attention_mask']))
            out['Y'].append(Y)
            out['T'].append(T)
            out['C'].append(C)

        data = (torch.tensor(out[x]) for x in ['W_ids', 'W_len', 'W_mask', 'C', 'T', 'Y'])

        data = TensorDataset(*data)
        sampler = RandomSampler(data) if sampler == 'random' else SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)
            # collate_fn=collate_CandT)

        return dataloader
    
    def build_dataloader(self, 
                         texts, confounds, treatments=None, outcomes=None,
                         tokenizer=None, sampler='random'):

        if tokenizer is None:
            tokenizer = DistilBertTokenizer.from_pretrained(
                'distilbert-base-uncased', do_lower_case=True)

        dataset = CausalDataset(texts, confounds, treatments, outcomes, tokenizer, max_length=self.max_length)
        sampler = RandomSampler(dataset) if sampler == 'random' else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, num_workers=self.num_workers)

        return dataloader


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('testdata.csv')

    # original form
    model = CausalDistilBert.from_pretrained(
            "distilbert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False)

    cb = CausalModelWrapper(model, batch_size=2,
        g_weight=0.1, Q_weight=0.1, mlm_weight=1)
    print(df.T)
    cb.train(df['text'], df['C'], df['T'], df['Y'], epochs=1)
    print(cb.ATE(df['C'], df.text, platt_scaling=True))
