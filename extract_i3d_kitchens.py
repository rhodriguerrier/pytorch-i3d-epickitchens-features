import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from kitchens_dataset import EpicKitchensDataset
from pytorch_i3d import InceptionI3d
import numpy as np
import pickle
import argparse


class KitchenExtraction:
    def __init__(self, batch_size, labels_path, domain_id, train_domain_id, is_flow=False):
        self.domain_id = domain_id
        self.is_flow = is_flow
        self.batch_size = batch_size
        if is_flow:
            self.model = InceptionI3d(8, in_channels=2)
            self.model.load_state_dict(torch.load(f'./flow/{train_domain_id}_train.pt'))
        else:
            self.model = InceptionI3d(8, in_channels=3)
            self.model.load_state_dict(torch.load(f'./rgb/{train_domain_id}_train.pt'))
        self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.dataset = EpicKitchensDataset(labels_path=labels_path, is_flow=is_flow)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)


    def extract(self):
        self.model.eval()
        total_features = torch.tensor([])
        total_narration_ids = []
        for (labels, inputs, narration_ids) in self.dataloader:
            inputs = Variable(inputs.cuda())
            features = self.model.extract_features(torch.tensor(inputs).float())
            total_features = torch.cat((total_features, torch.reshape(features, (features.size()[0], features.size()[1]))))
            total_narration_ids.extend(narration_ids)
        return total_features, total_narration_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--train_domain_id", action="store", dest="train_domain_id", default="D2")
    parser.add_argument("--domain_id", action="store", dest="domain_id", default="D2")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="64")
    parser.add_argument("--flow", action="store_true", dest="is_flow")
    parser.set_defaults(is_flow=False)
    args = parser.parse_args()
    model = KitchenExtraction(
        batch_size=int(args.batch_size),
        labels_path=f"./epic_kitchens_data/label_lookup/{args.domain_id}_train.pkl",
        domain_id=args.domain_id,
        train_domain_id=args.train_domain_id,
        is_flow=args.is_flow
    )
    features, narration_ids = model.extract()
    pickle_dict = {
        "features": {
            "RGB": features.detach().numpy()
        },
        "narration_ids": narration_ids
    }
    with open(f"./{args.train_domain_id}-{args.domain_id}_train.pkl", "wb") as handle:
        pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
