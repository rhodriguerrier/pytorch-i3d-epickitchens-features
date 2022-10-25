import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from kitchens_dataset import SequentialClassKitchens
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from kitchens_dataset import EpicKitchensDataset
from pytorch_i3d import InceptionI3d
import argparse
import logging


class SequentialKitchenActionsExtraction:
    def __init__(self, labels_path, domain_id, train_domain_id, class_num, is_flow=False):
        self.domain_id = domain_id
        self.is_flow = is_flow
        self.class_num = class_num
        if is_flow:
            self.model = InceptionI3d(8, in_channels=2)
            self.model.cuda()
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(f'/user/work/rg16964/flow_{train_domain_id}_train.pt'))
        else:
            self.model = InceptionI3d(8, in_channels=3)
            self.model.cuda()
            self.model = nn.DataParallel(self.model)
            self.model.load_state_dict(torch.load(f'/user/work/rg16964/rgb_{train_domain_id}_train.pt'))
        self.dataset = SequentialClassKitchens(labels_path=labels_path, class_num=self.class_num, temporal_window=16, is_flow=is_flow)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)


    def extract(self):
        self.model.eval()
        total_features = torch.tensor([])
        total_labels = []
        counter = 0
        for (labels, img_inputs) in self.dataloader:
            counter += 1
            seq_action_concat = torch.tensor([])
            for img_input in img_inputs:
                seq_action_concat = torch.cat((seq_action_concat, img_input))
            inputs = torch.tensor(seq_action_concat).float()
            print(inputs.size())
            inputs = Variable(inputs.cuda())
            features = self.model.module.extract_features(inputs)
            ft_reshape = torch.reshape(features, (features.size()[0], features.size()[1])).detach().cpu()
            total_features = torch.cat((total_features, ft_reshape))
            total_labels.extend([float(labels[0]) for i in range(inputs.size(0))])
            if counter == 3:
                break
        return total_features, total_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--train_domain_id", action="store", dest="train_domain_id", default="D1")
    parser.add_argument("--domain_id", action="store", dest="domain_id", default="D1")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="10")
    parser.add_argument("--flow", action="store_true", dest="is_flow")
    parser.add_argument("--class_num", action="store_true", dest="class_num", default="0")
    parser.set_defaults(is_flow=False)
    args = parser.parse_args()
    model = SequentialKitchenActionsExtraction(
        labels_path=f"/user/work/rg16964/label_lookup/{args.domain_id}_test.pkl",
        domain_id=args.domain_id,
        train_domain_id=args.train_domain_id,
        class_num=int(args.class_num),
        is_flow=args.is_flow
    )
    features, labels = model.extract()
    features = features.detach().cpu().numpy()
    low_dim_data = TSNE(
        n_components=2,
        init='random'
    ).fit_transform(features)
    plt.scatter(low_dim_data[:,0], low_dim_data[:,1], 15, c=labels)
    plt.savefig('test_plot_seq_features.png')
