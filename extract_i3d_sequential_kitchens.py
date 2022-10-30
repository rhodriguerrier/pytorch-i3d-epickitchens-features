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
    def __init__(self, labels_path, domain_id, train_domain_id, class_num, class_sample_num, is_flow=False):
        self.domain_id = domain_id
        self.is_flow = is_flow
        self.class_num = class_num
        self.class_sample_num = class_sample_num
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
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=4)


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
            inputs = Variable(inputs.cuda())
            features = self.model.module.extract_features(inputs)
            ft_reshape = torch.reshape(features, (features.size()[0], features.size()[1])).detach().cpu()
            total_features = torch.cat((total_features, ft_reshape))
            total_labels.extend([f"{self.domain_id}-{labels[0]}" for i in range(inputs.size(0))])
            if counter == self.class_sample_num:
                break
        return total_features, total_labels


def format_ft_data(labels, features):
    unique_labels = set(labels)
    ft_labels_dict = {label: {"class_label": label.split("-")[-1], "domain_label": label.split("-")[0], "features": np.empty((0, 2))} for label in unique_labels}
    for i, val in enumerate(features):
        collected_ft = ft_labels_dict[labels[i]]["features"]
        ft_labels_dict[labels[i]]["features"] = np.append(collected_ft, np.array([val]), axis=0)
    return ft_labels_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--train_domain_id", action="store", dest="train_domain_id", default="D1")
    parser.add_argument("--primary_domain_id", action="store", dest="primary_domain_id", default="D1")
    parser.add_argument("--secondary_domain_id", action="store", dest="secondary_domain_id", default="D2")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="10")
    parser.add_argument("--flow", action="store_true", dest="is_flow")
    parser.add_argument("--class_sample_num", action="store_true", dest="class_sample_num", default="4")
    parser.set_defaults(is_flow=False)
    args = parser.parse_args()

    primary_domain_ft = torch.tensor([])
    primary_domain_labels = []
    for i in range(8):
        class_model = SequentialKitchenActionsExtraction(
            labels_path=f"/user/work/rg16964/label_lookup/{args.primary_domain_id}_test.pkl",
            domain_id=args.primary_domain_id,
            train_domain_id=args.train_domain_id,
            class_num=i,
            class_sample_num=int(args.class_sample_num),
            is_flow=args.is_flow
        )
        features, labels = class_model.extract()
        primary_domain_ft = torch.cat((primary_domain_ft, features))
        primary_domain_labels.extend(labels)


    secondary_domain_ft = torch.tensor([])
    secondary_domain_labels = []
    for i in range(8):
        class_model = SequentialKitchenActionsExtraction(
            labels_path=f"/user/work/rg16964/label_lookup/{args.primary_domain_id}_test.pkl",
            domain_id=args.secondary_domain_id,
            train_domain_id=args.train_domain_id,
            class_num=i,
            class_sample_num=int(args.class_sample_num),
            is_flow=args.is_flow
        )
        features, labels = class_model.extract()
        secondary_domain_ft = torch.cat((secondary_domain_ft, features))
        secondary_domain_labels.extend(labels)

    # Plot graphs
    fig, axs = plt.subplots(3, 2)
    fig.suptitle(f"tSNE Plot of Sequential Features")
    test_class_plot = "0"
    concat_ft = torch.cat((primary_domain_ft, secondary_domain_ft)).numpy()
    concat_labels = primary_domain_labels
    concat_labels.extend(secondary_domain_labels)
    low_dim_data = TSNE(
        n_components=2,
        init='random'
    ).fit_transform(concat_ft)
    primary_domain_classes = np.empty((0, 3))
    secondary_domain_classes = np.empty((0, 3))
    ft_by_domain = np.empty((0, 3))
    ft_labels_dict = format_ft_data(concat_labels, low_dim_data)
    class_by_colour = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:cyan"
    ]
    for label in ft_labels_dict:
        if ft_labels_dict[label]["class_label"].split("_")[0] == test_class_plot and ft_labels_dict[label]["domain_label"] == args.primary_domain_id:
            axs[0, 0].plot(
                ft_labels_dict[label]["features"][:, 0],
                ft_labels_dict[label]["features"][:, 1],
                marker="o",
                markersize=2,
                label=ft_labels_dict[label]["class_label"],
                linestyle=":"
            )
        if ft_labels_dict[label]["class_label"].split("_")[0] == test_class_plot and ft_labels_dict[label]["domain_label"] == args.secondary_domain_id:
            axs[0, 1].plot(
                ft_labels_dict[label]["features"][:, 0],
                ft_labels_dict[label]["features"][:, 1],
                marker="o",
                markersize=2,
                label=ft_labels_dict[label]["class_label"],
                linestyle=":"
            )
        if ft_labels_dict[label]["domain_label"] == args.primary_domain_id:
            axs[1, 0].plot(
                ft_labels_dict[label]["features"][:, 0],
                ft_labels_dict[label]["features"][:, 1],
                marker="o",
                markersize=2,
                label=ft_labels_dict[label]["domain_label"],
                linestyle=":",
                c=class_by_colour[int(ft_labels_dict[label]["class_label"].split("_")[0])]
            )
            axs[2, 0].plot(
                ft_labels_dict[label]["features"][:, 0],
                ft_labels_dict[label]["features"][:, 1],
                marker="o",
                markersize=2,
                label=ft_labels_dict[label]["domain_label"],
                linestyle=":",
                color="blue"
            )
        if ft_labels_dict[label]["domain_label"] == args.secondary_domain_id:
            axs[1, 1].plot(
                ft_labels_dict[label]["features"][:, 0],
                ft_labels_dict[label]["features"][:, 1],
                marker="o",
                markersize=2,
                label=ft_labels_dict[label]["domain_label"],
                linestyle=":",
                c=class_by_colour[int(ft_labels_dict[label]["class_label"].split("_")[0])]
            )
            axs[2, 0].plot(
                ft_labels_dict[label]["features"][:, 0],
                ft_labels_dict[label]["features"][:, 1],
                marker="o",
                markersize=2,
                label=ft_labels_dict[label]["domain_label"],
                linestyle=":",
                color="red"
            )
    axs[0, 0].set_title(f"{args.primary_domain_id} Features from Class {test_class_plot}")
    axs[0, 1].set_title(f"{args.secondary_domain_id} Features from Class {test_class_plot}")
    axs[1, 0].set_title(f"All {args.primary_domain_id} Feature Classes")
    axs[1, 1].set_title(f"All {args.secondary_domain_id} Feature Classes")
    axs[2, 0].set_title(f"{args.primary_domain_id} vs {args.secondary_domain_id}")
    fig.tight_layout()
    fig.savefig(f"seq_ft_comp_{args.primary_domain_id}_and_{args.secondary_domain_id}.png")
