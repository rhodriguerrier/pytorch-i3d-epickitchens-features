import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from kitchens_dataset import EpicKitchensDataset
from pytorch_i3d import InceptionI3d
import numpy as np
import pickle
import argparse


class EpicKitchensI3D:
    def __init__(self, epochs, init_lr, batch_size, train_labels_path, domain_id, is_flow=False):
        self.epochs = epochs
        self.domain_id = domain_id
        self.lr = init_lr
        self.is_flow = is_flow
        self.batch_size = batch_size
        if is_flow:
            self.model = InceptionI3d(400, in_channels=2)
            self.model.load_state_dict(torch.load('models/flow_imagenet.pt'))
        else:
            self.model = InceptionI3d(400, in_channels=3)
            self.model.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        self.model.replace_logits(8)
        self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0000001)
        self.lr_sched = optim.lr_scheduler.MultiStepLR(self.optim, [300, 1000])
        self.num_steps_per_update = 4
        self.train_dataset = EpicKitchensDataset(labels_path=train_labels_path, is_flow=is_flow)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.ce_loss = nn.CrossEntropyLoss()


    def train(self):
        for epoch in self.epochs:
            print(f"Epoch: {epoch}")
            sum_loss = 0
            counter = 0
            self.optim.zero_grad()
            for (train_labels, train_inputs, narration_ids) in self.train_dataloader:
                counter += 1
                train_inputs = Variable(train_inputs.cuda())
                train_labels = Variable(train_labels.cuda())
                output_logits = self.model(train_inputs)
                train_ce_loss = self.ce_loss(
                    torch.reshape(output, (output.size()[0], output.size()[1])),
                    train_labels.long()
                )
                loss = train_ce_loss / self.num_steps_per_update
                sum_loss += loss
                loss.backward()
                if counter == self.num_steps_per_update:
                    counter = 0
                    self.optim.step()
                    self.optim.zero_grad()
                    self.lr_sched.step()
                    if epoch % 10 == 0:
                        print(f"Total Loss: {sun_loss / 10}")
                        sum_loss = 0
        print("Epochs done, saving model...")
        if self.is_flow:
            torch.save(self.model.state_dict(), f"./flow_{self.domain_id}_train.pt")
        else:
            torch.save(self.model.state_dict(), f"./rgb_{self.domain_id}_train.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epic Kitchens Feature Extraction")
    parser.add_argument("--domain_id", action="store", dest="domain_id", default="D2")
    parser.add_argument("--epochs", action="store", dest="epochs", default="100")
    parser.add_argument("--batch_size", action="store", dest="batch_size", default="2")
    parser.add_argument("--flow", action="store_true", dest="is_flow")
    parser.set_defaults(is_flow=False)
    args = parser.parse_args()
    model = EpicKitchensI3D(
        epochs=int(args.epochs),
        init_lr=0.1,
        batch_size=int(args.batch_size),
        train_labels_path=f"/user/work/rg16964/label_lookup/{args.domain_id}_train.pkl",
        domain_id=args.domain_id,
        is_flow=args.is_flow
    )
    model.train()
