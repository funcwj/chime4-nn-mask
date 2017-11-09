#!/usr/bin/env python
# coding=utf-8
# wujian@17.11.8

import tqdm
import os
import logging
import torch as th
import torch.nn as nn

from torch.autograd import Variable

logfmt  = "%(filename)s[%(lineno)d] %(asctime)s %(levelname)s: %(message)s"
datefmt = "%Y-%M-%d %T"
logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt)

class BatchNormRNN(nn.Module):
    """
        A batchnorm wrapper for single RNN layer.
    """
    def __init__(self, input_size, output_size, rnn_cell=nn.LSTM, 
                bidirectional=False, dropout=0.0):
        super(BatchNormRNN, self).__init__()
        self.inner_rnn = rnn_cell(
            input_size=input_size, 
            hidden_size=output_size,
            dropout=dropout
        ) 
        self.batch_norm = nn.BatchNorm1d(output_size)

    def forward(self, x):
        """
            Parameter x: M x T x N
                M: number of channels
                T: number of frames
                N: number of frequency bins
        """
        # go through RNN
        x, _ = self.inner_rnn(x)
        # time first
        t, n = x.size(0), x.size(1)
        # reshape to apply batchnorm
        x = x.contiguous().view(n * t, -1)
        x = self.batch_norm(x)
        # following is fully connection layer, so do not restore the shape
        return x

class BatchNormAffine(nn.Module):
    """
        A single fully connected layer with dropout and batchnorm, dropout
        is not necessary
    """
    def __init__(self, input_size, output_size, dropout=0.0, activate='relu'):
        super(BatchNormAffine, self).__init__()
        assert dropout >= 0, "dropout rate must >= 0"
        assert activate in ['relu', 'sigmoid'], "activate only support ReLU/Sigmoid"
        self.linear_transform = nn.Linear(input_size, output_size)
        self.apply_batchnorm  = nn.BatchNorm1d(output_size)
        self.apply_dropout = None
        self.activate = nn.ReLU() if activate == 'relu' else nn.Sigmoid()
        if dropout != 0.0:
            self.apply_dropout    = nn.Dropout(p=dropout)

    def forward(self, x):
        """
            linear transform => dropout => activate => batchnorm
        """
        x = self.linear_transform(x)
        if self.apply_dropout:
            x = self.apply_dropout(x)
        x = self.activate(x)
        x = self.apply_batchnorm(x)
        return x

class MaskEstimator(nn.Module):
    """
        Reference:
            Heymann J, Drude L, Haebumbach R. Neural network based spectral mask estimation 
            for acoustic beamforming.[J]. IEEE Transactions on Industrial Electronics, 
            2016, 46(3):544-553.
    """
    def __init__(self, num_bins):
        super(MaskEstimator, self).__init__()
        self.batchnorm_blstm  = BatchNormRNN(num_bins, 256, bidirectional=True, dropout=0.5)
        self.fully_connection = nn.Sequential(
            BatchNormAffine(256, num_bins, dropout=0.5),
            BatchNormAffine(num_bins, num_bins, dropout=0.5)
        )
        self.noise_mask  = nn.Linear(num_bins, num_bins)
        self.clean_mask  = nn.Linear(num_bins, num_bins)
    
    def forward(self, x):
        x = self.batchnorm_blstm(x)
        x = self.fully_connection(x)
        mask_n = self.noise_mask(x)
        mask_x = self.clean_mask(x)
        return mask_n, mask_x

def offload_to_gpu(cpu_var):
    return Variable(cpu_var.cuda())

class LRScheduler(object):
    """
        Wrapper to implement learning rate decay.
        It's a simple version of torch.optim.ReduceLROnPlateau
    """
    def __init__(self, optimizer, init_cvloss=None, factor=0.5):
        self.optimizer = optimizer
        self.factor = factor
        self.prev_loss = init_cvloss
    
    def _apply_lr_decay(self):
        for group in self.optimizer.param_groups:
            prev_lr = float(group['lr'])
            next_lr = self.factor * prev_lr
            group['lr'] = next_lr
            logging.info("schedule lr {:.4e} => {:.4e}".format(prev_lr, next_lr))
    
    def step(self, loss):
        if self.prev_loss and loss > self.prev_loss:
            self._apply_lr_decay()
            return True
        else:
            self.prev_loss = loss
            return False

class EstimatorTrainer(object):
    def __init__(self, num_bins, checkout_dir, learning_rate=0.001, momentum=0.9):
        self.estimator = MaskEstimator(num_bins)
        logging.info('estimator structure: {}'.format(self.estimator))
        self.estimator.cuda()
        self.optimizer = th.optim.RMSprop(self.estimator.parameters(), \
                lr=learning_rate, momentum=momentum) 
        self.checkout_dir = checkout_dir
        if not os.path.exists(checkout_dir):
            os.makedirs(checkout_dir)
        

    def run_one_epoch(self, data_loader, training=False):
        """
            Train/Evaluate model through the feeding DataLoader
            return avarage loss, for logging or learning rate schedule
        """
        average_loss = 0.0 
        for specs_feats, clean_label, noise_label in tqdm.tqdm(data_loader, \
                desc=('training' if training else 'evaluate')):
            if training:
                self.optimizer.zero_grad()
            specs_feats = offload_to_gpu(specs_feats)
            noise_label = offload_to_gpu(noise_label)
            clean_label = offload_to_gpu(clean_label)
            loss = self._calculate_loss(specs_feats, noise_label, clean_label)
            average_loss += float(loss.cpu().data.numpy())
            if training:
                loss.backward() 
                self.optimizer.step()
        return average_loss / len(data_loader)
    
    def train(self, training_loader, evaluate_loader, epoch=10):
        evaluate_loss = self.run_one_epoch(evaluate_loader, training=False)
        scheduler = LRScheduler(self.optimizer, init_cvloss=evaluate_loss)
        logging.info('evaluate loss with initial weights: {:.4f}'.format(evaluate_loss))
        for e in range(1, epoch + 1):
            training_loss = self.run_one_epoch(training_loader, training=True)
            evaluate_loss = self.run_one_epoch(evaluate_loader, training=False)
            logging.info('epoch: {:3d} training loss: {:.4f}, evaluate loss: {:.4f}'.format(e, \
                    training_loss, evaluate_loss))
            if scheduler.step(evaluate_loss):
                logging.info('reject model and apply lr decay')
            else:
                th.save(self.estimator.state_dict(), os.path.join(self.checkout_dir, \
                        'estimator_{:.4f}.pkl'.format(evaluate_loss)))
        
    def _calculate_loss(self, input_specs, noise_label, clean_label):
        """
            Calculate loss from two parts(noise estimator and clean estimator), and
            using average of them as final loss.
            binary_cross_entropy_with_logits combine sigmoid and BCE criterion
        """
        mask_n, mask_x = self.estimator(input_specs)
        # sigmoid with BCE
        loss_n = nn.functional.binary_cross_entropy_with_logits(mask_n, noise_label)
        loss_x = nn.functional.binary_cross_entropy_with_logits(mask_x, clean_label)
        return (loss_n + loss_x) / 2
    

class MaskComputer(object):
    def __init__(self, model_structure, state_file):
        self.estimator = model_structure
        self.estimator.load_state_dict(th.load(state_file))
        # setting evaluate mode
        self.estimator.eval()
        self.estimator.cuda()
    
    def compute_masks(self, input_specs):
        input_specs = Variable(th.from_numpy(input_specs))
        t, n = input_specs.size(0), input_specs.size(1)
        # output of estimator do not apply sigmoid
        mask_n, mask_x = self.estimator(input_specs) 
        mask_n = nn.functional.sigmoid(mask_n).view(t, n, -1).data.numpy()
        mask_x = nn.functional.sigmoid(mask_x).view(t, n, -1).data.numpy()
        return mask_n, mask_x

