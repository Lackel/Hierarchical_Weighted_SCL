import os
import random
from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from transformers import BertModel, get_linear_schedule_with_warmup
# from cuml import KMeans
from sklearn.cluster import KMeans
from util import clustering_score

class ModelManager:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.set_seed()
        self.model = BertForModel(args, data.n_fine, data.n_coarse)
        self.model_m = BertForModel(args, data.n_fine, data.n_coarse)
        if args.freeze_bert_parameters:
            self.freeze_parameters(self.model)
        self.freeze_parameters_m(self.model_m)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_m.to(self.device)
        self.optimizer = self.get_optimizer(args)
        self.num_training_steps = int(
            len(data.train_examples) / args.train_batch_size) * args.num_train_epochs
        self.num_warmup_steps= int(args.warmup_proportion * self.num_training_steps) 
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=self.num_warmup_steps, num_training_steps=self.num_training_steps) 
        self.m = args.momentum_factor

    def set_seed(self):
        seed = self.args.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def freeze_parameters(self, model):
        """
        Freezing the weights of different Transformer layers.
        We only freeze the embedding layer by default.
        """
        for name, param in model.bert.named_parameters():
            param.requires_grad = True
            if "embeddings" in name:
                param.requires_grad = False
    
    def freeze_parameters_m(self, model):
        """
        Freeze all the weights of Momentum BERT.
        """
        for _, param in model.named_parameters():
            param.requires_grad = False
    
    def get_features_labels(self, dataloader, model, args):
        """
        Getting features and labels for clustering.
        """
        model.eval()
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_fine_labels = torch.empty(0,dtype=torch.long).to(self.device)
        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, _, label_id_fine = batch
            with torch.no_grad():
                feature = model(input_ids, segment_ids, input_mask, mode = 'extract')

            total_features = torch.cat((total_features, feature))
            total_fine_labels = torch.cat((total_fine_labels, label_id_fine))
        return total_features, total_fine_labels

    def get_optimizer(self, args):
        """
        Setting the optimizer with weight decay for BERT.
        """
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
        return optimizer

    def train(self):
        """
        Training the model.
        """
        for epoch in range(1, int(self.args.num_train_epochs)+1):
            self.model.train()
            tr_loss = 0
            tr_steps = 0
            for _, batch in enumerate(self.data.train_dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_id_coarse, _ = batch
                k = self.model_m(input_ids, segment_ids, input_mask, label_id_coarse, mode='extract')
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_id_coarse, mode="train", k=k)
                    loss.backward()
                    tr_loss += loss.item()

                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)   
                    self.optimizer.step()
                    self.scheduler.step()
                    self.momentum_update_encoder_m()
                    self.optimizer.zero_grad()

                    tr_steps += 1

            loss = tr_loss / tr_steps
            print('Epoch {} train_loss: {}'.format(epoch, loss))
            
    def test(self):
        """
        Testing trained model on the test sets by clustering.
        """
        self.model.eval()

        feats, labels = self.get_features_labels(self.data.test_dataloader, self.model, self.args)

        feats = feats.cpu().numpy()
        km = KMeans(n_clusters = self.data.n_fine, n_init=20).fit(feats)

        y_pred = km.labels_
        y_true = labels.cpu().numpy()
        results_all = clustering_score(y_true, y_pred)
        return results_all
    
    def momentum_update_encoder_m(self):
        """
        Updating the Momentum BERT.
        We only update the last four layers by default.
        """
        for (name, param_q), (_, param_m) in zip(self.model.bert.named_parameters(), self.model_m.bert.named_parameters()):
            if "encoder.layer.11" in name or "encoder.layer.10" in name or "encoder.layer.9" in name or "encoder.layer.8" in name or "pooler" in name:
                param_m.data = param_m.data * self.m + param_q.data * (1. - self.m)


class BertForModel(nn.Module):
    def __init__(self, args, num_fine, num_coarse):
        super(BertForModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_fine
        self.args = args
        self.bert = BertModel.from_pretrained(args.model_name)
        self.config = self.bert.config
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.temperature = args.temperature
        self.classifier = nn.Linear(self.config.hidden_size, num_fine)
        self.classifier_test = nn.Linear(self.config.hidden_size, num_fine)
        self.classifier_coarse = nn.Linear(self.config.hidden_size, num_coarse)
        self.queue = torch.zeros((num_coarse, self.args.train_batch_size, self.config.hidden_size), requires_grad=False).to(self.device)

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , labels=None, mode=None, k=None):
        """
        Deep features: pool_d from the output layer.
        Shallow features: pool_s from the layer_num-th Transformer layer.
        """
        encoded_layer_d, _, encoded_layer= self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True, return_dict=False)
        pooled_layer_d = self.dense(encoded_layer_d.mean(dim = 1))
        pooled_output_d = self.activation(pooled_layer_d)
        pool_d = self.dropout(pooled_output_d)
        logits = self.classifier_coarse(pool_d)

        encoded_layer_s = encoded_layer[self.args.layer_num]
        pooled_layer_s = self.dense(encoded_layer_s.mean(dim = 1))
        pooled_output_s = self.activation(pooled_layer_s)
        pool_s = self.dropout(pooled_output_s)
        logits_coarse = self.classifier_coarse(pool_s)
        

        if mode == 'train':
            contrastiveLoss = self.contrastiveLoss(pool_s, pool_d, labels, k)
            loss = nn.CrossEntropyLoss()(logits_coarse, labels)
            loss_2 = nn.CrossEntropyLoss()(logits, labels)
            return  loss_2 + self.args.gamma1 * loss + self.args.gamma2 * contrastiveLoss

        elif mode == 'extract':
            return pool_d

    def contrastiveLoss(self, pool_s, pool_d, labels, k):
        """
        Weighted Self-contrastive Loss.
        """
        batch_size = pool_s.shape[0]
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(self.device)
        mask_w = (self.args.alpha_same - self.args.alpha_diff) * mask  + self.args.alpha_diff * torch.ones(batch_size).float().to(self.device) - torch.eye(batch_size).float().to(self.device)  # weights for negative samples in the shallow and deep layer
        mask_e = torch.eye(batch_size).float().to(self.device) # for positive samples
        logits_shallow = F.cosine_similarity(pool_d.unsqueeze(1), pool_s.unsqueeze(0), dim=2) /self.temperature   # batch_size * batch_size
        logits_deep = F.cosine_similarity(pool_d.unsqueeze(1), pool_d.unsqueeze(0), dim=2) /self.temperature
        negative = self.query_queue(k, labels)
        logits_momentum = F.cosine_similarity(pool_d.unsqueeze(1), negative, dim=2) /self.temperature
        self.update_queue(k, labels)
        exp_logits_s = torch.exp(logits_shallow) * mask_w # batch_size * batch_size
        exp_logits_d = torch.exp(logits_deep) * mask_w
        exp_logits_m = torch.exp(logits_momentum) * self.args.alpha_m
        log_prob = logits_shallow - torch.log(exp_logits_s.sum(1, keepdim=True)) - torch.log(exp_logits_d.sum(1, keepdim=True)) - torch.log(exp_logits_m.sum(1, keepdim=True))
        mean_log_prob_pos = (log_prob * mask_e).sum(1)  # batch_size * 1
        loss_1 = - mean_log_prob_pos
        loss = loss_1.mean()       
        return loss

    def query_queue(self, k, labels):
        """
        Getting the momentum negative samples from the dynamic queue.
        """
        size = k.shape[0]
        temp_queue = torch.empty((size, self.args.train_batch_size, self.config.hidden_size)).float().to(self.device)
        labels = labels.cpu().numpy()
        for index, label in enumerate(labels):
            temp_queue[index,:,:] = self.queue[int(label),:,:]
        return temp_queue.clone().detach()

    def update_queue(self, pool_d, labels):
        """
        Updating the dynamic queue with the latest samples.
        """
        pool_d = pool_d.clone().detach()
        labels = labels.cpu().numpy()
        for index, label in enumerate(labels):
            self.queue[int(label),:,:] = torch.cat([self.queue[int(label),1:,:], pool_d[index, :].unsqueeze(0)])

