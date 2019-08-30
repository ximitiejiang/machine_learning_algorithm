#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 18:42:30 2019

@author: ubuntu
"""
import numpy as np
from utils.dataloader import batch_iterator

class Trainer:
    
    def __init__(self, model, feats, labels, batch_size, val_feats=None, val_labels=None):
        self.model = model
        self.feats = feats
        self.labels = labels
        self.val_feats = val_feats
        self.val_labels = val_labels
        self.batch_size = batch_size
    
    
    def batch_operation_train(self, x, y):
        """基于每一个batch的数据分别进行前向计算和反向计算"""
        # 前向计算
        y_pred = self.forward_pass(x=x, training=True)
        losses = self.loss_function.loss(y, y_pred)
        loss = np.mean(losses)
        acc = self.loss_function.acc(y, y_pred)
        # 反向传播
        loss_grad = self.loss_function.gradient(y, y_pred)
        self.backward_pass(grad = loss_grad)
        return loss, acc
    
    def batch_operation_val(self, val_x, val_y):
        """在每个batch做完都做一次验证，比较费时，但可实时看到验证集的acc变化，可用于评估是否过拟合。
        注意：val操作时跟test一样，不需要反向传播。
        """
        # BN和dropout在训练和验证过程中需要有不同的操作差异，(其他层不受影响)，所以在val和test时，需要闯入training flag给每一个层。      
        y_pred = self.forward_pass(val_x, training=False) # 在验证时是假定已经训练完成，但由于该步是在train中执行，必须手动设置trained=False
        losses = self.loss_function.loss(val_y, y_pred)
        loss = np.mean(losses)
        acc = self.loss_function.acc(val_y, y_pred)
        return loss, acc
        
    def train(self):
        total_iter = 1
        all_losses = {'train':[], 'val':[]}
        all_accs = {'train':[], 'val':[]}
        for i in range(self.n_epochs):
            it = 1
            for x_batch, y_batch in batch_iterator(self.feats, self.labels, 
                                                   batch_size = self.batch_size):
                # 训练数据的计算
                loss, acc = self.batch_operation_train(x_batch, y_batch)
                all_losses['train'].append([total_iter,loss])
                all_accs['train'].append([total_iter, acc])
                log_text = "iter %d/epoch %d: batch_loss=%f, batch_acc=%f"%(it, i+1, loss, acc)
                # 验证数据的计算
                if self.val_feats is not None:
                    val_loss, val_acc = self.batch_operation_val(self.val_feats, self.val_labels)
                    all_losses['val'].append([total_iter, val_loss])
                    all_accs['val'].append([total_iter, val_acc])
                    log_text += ", val_loss=%f, val_acc=%f"%(val_loss, val_acc)
                # 显示loss
                if it % 5 == 0:
                    print(log_text)

                it += 1
                total_iter += 1
                
        self.model.vis_loss(all_losses['train'], all_accs['train'], title='train')
        if self.val_feats is not None:
            self.model.vis_loss(all_losses['val'], all_accs['val'], title='val')
        self.trained = True  # 完成training则可以预测
        return all_losses, all_accs
    
    def evaluation(self, x, y, title=None):
        """对一组数据进行预测精度"""
        if title is None:
            title = "evaluate"
        if self.trained:
            y_pred = self.forward_pass(x, training=False)  # (1280, 10)            
            y_pred = np.argmax(y_pred, axis=1)  # (1280,)
            y_label = np.argmax(y, axis=1)      # (1280,)
            acc = np.sum(y_pred == y_label, axis=0) / len(y_label)
            print(title + " acc: %f"%acc)
            return acc, y_pred
        else:
            raise ValueError("model not trained, can not predict or evaluate.")
            

if __name__ == "__main__":
    
    
    
    
    trainer = Trainer()
    trainer.train()