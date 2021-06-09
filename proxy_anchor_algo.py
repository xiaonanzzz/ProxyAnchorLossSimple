import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data.sampler import BatchSampler

from tqdm import *

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # set random seed for all gpus

def binarize(T, nb_classes):
    device = T.device
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(device)
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class ProxyAnchorLoss(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha

    def forward(self, X, T):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss


def prepare_optimizer(args, param_groups):
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(param_groups, lr=float(args.lr), weight_decay=args.weight_decay, momentum=0.9,
                              nesterov=True)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr=float(args.lr), alpha=0.9, weight_decay=args.weight_decay,
                                  momentum=0.9)
    elif args.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=args.weight_decay)
    return opt

class ProxyAnchorLossEmbeddingModelTrainer():
    def __init__(self):
        self.mrg = 0.1
        self.alpha = 32
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.optimizer = 'adam'
        self.lr_decay_step = 10
        self.lr_decay_gamma = 0.5
        self.nb_epochs = 60
        self.nb_workers = 8    # number of cpus process for loading data
        self.batch_size = 16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self, model, train_dataset, sz_embedding, nb_classes):
        args = self

        criterion = ProxyAnchorLoss(nb_classes=nb_classes, sz_embed=sz_embedding, mrg=self.mrg,
                                        alpha=self.alpha)

        param_groups = [{'params': model.parameters(), 'lr': float(args.lr) * 1},
                        {'params': criterion.proxies, 'lr': float(args.lr) * 100}]

        opt = prepare_optimizer(args, param_groups)

        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)


        print("Training parameters: {}".format(vars(args)))

        dl_tr = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=args.nb_workers,
            pin_memory=True
        )

        for epoch in range(0, args.nb_epochs):
            losses_per_epoch = []

            model.train()
            model.to(self.device)
            criterion.to(self.device)
            pbar = tqdm(enumerate(dl_tr))
            for batch_idx, (x, y) in pbar:
                m = model(x.to(self.device))
                loss = criterion(m, y.to(self.device))

                opt.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

                losses_per_epoch.append(loss.data.cpu().numpy())
                opt.step()

                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        epoch, batch_idx + 1, len(dl_tr),
                        100. * batch_idx / len(dl_tr),
                        np.mean(losses_per_epoch)))
            scheduler.step()

            self.epoch_end_hook()

    def epoch_end_hook(self):
        pass


class LinearEmbedder(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearEmbedder, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        o = self.fc(x)
        o = self.l2_norm(o)
        return o

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

def debug_proxy_anchor_loss_algo():

    model = LinearEmbedder(300, 200)
    x = torch.randn(100, 300)
    y = torch.randint(0, 10, (100, ))
    print('data shape', x.shape, y.shape,)
    dataset = torch.utils.data.TensorDataset(x, y)

    trainer = ProxyAnchorLossEmbeddingModelTrainer()
    trainer.nb_epochs = 10
    trainer.train(model, dataset, 200, 10)

if __name__ == '__main__':
    debug_proxy_anchor_loss_algo()
