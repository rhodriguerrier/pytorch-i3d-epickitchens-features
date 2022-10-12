import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from kitchens_dataset import EpicKitchensDataset
import numpy as np


class EpicKitchensI3D:
    def __init__(self, epochs, init_lr, is_flow=False, batch_size, train_labels_path):
        self.epochs = epochs
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
        self.optim = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
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
            for (train_labels, train_inputs) in self.train_dataloader:
                counter += 1
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

    def test(self):
        self.model.eval()
        total_features = torch.tensor([])
        for (labels, inputs) in self.train_dataloader:
            features = self.model.extract_features(inputs)
            total_features = torch.cat((total_features, torch.reshape(features, (features.size()[0], features.size()[1]))))
            # Add in narration_id creation alongside features
        return total_features

