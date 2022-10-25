import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from kitchens_dataset import EpicKitchensDataset
from pytorch_i3d import InceptionI3d
import videotransforms
import numpy as np
import pickle
import argparse
import logging


class KitchenExtraction:
    def __init__(self, batch_size, labels_path, domain_id, train_domain_id, is_flow=False):
        self.domain_id = domain_id
        self.is_flow = is_flow
        self.batch_size = batch_size
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
        self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.transforms = transforms.Compose([videotransforms.CenterCrop(224)])
        self.dataset = EpicKitchensDataset(labels_path=labels_path, is_flow=is_flow, transforms=self.transforms)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)


    def extract(self):
        self.model.eval()
        total_features = torch.tensor([])
        total_narration_ids = []
        counter = 0
        for (labels, inputs, narration_ids) in self.dataloader:
            logging.info(f"Batch number: {counter}")
            counter += 1
            inputs = torch.tensor(inputs).float()
            inputs = Variable(inputs.cuda())
            features = self.model.module.extract_features(inputs)
            ft_reshape = torch.reshape(features, (features.size()[0], features.size()[1])).detach().cpu()
            total_features = torch.cat((total_features, ft_reshape))
            total_narration_ids.extend(narration_ids)
        return total_features, total_narration_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--train_domain_id", action="store", dest="train_domain_id", default="D1")
    parser.add_argument("--domain_id", action="store", dest="domain_id", default="D2")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="10")
    parser.add_argument("--flow", action="store_true", dest="is_flow")
    parser.set_defaults(is_flow=False)
    args = parser.parse_args()
    if args.is_flow:
        logging_filename = f"extract_{args.train_domain_id}_{args.domain_id}_flow_logs.log"
    else:
        logging_filename = f"extract_{args.train_domain_id}_{args.domain_id}_rgb_logs.log"
    logging.basicConfig(
        filename=logging_filename,
        filemode='a',
        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG
    )
    model = KitchenExtraction(
        batch_size=int(args.batch_size),
        labels_path=f"/user/work/rg16964/label_lookup/{args.domain_id}_train.pkl",
        domain_id=args.domain_id,
        train_domain_id=args.train_domain_id,
        is_flow=args.is_flow
    )
    features, narration_ids = model.extract()
    logging.info("Finished extraction")
    pickle_dict = {
        "features": {
            "RGB": features.detach().cpu().numpy()
        },
        "narration_ids": narration_ids
    }
    with open(f"./{args.train_domain_id}-{args.domain_id}_train_features.pkl", "wb") as handle:
        pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
