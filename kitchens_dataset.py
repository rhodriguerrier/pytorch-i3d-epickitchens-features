import torch
from torch.utils.data import Dataset, DataLoader
from random import randint
import pickle
import numpy as np
from PIL import Image


class RgbFlowKitchensDataset(Dataset):
    def __init__(self, labels_path):
        labels_df = load_pickle_data(labels_path)
        self.labels = []
        self.rgb_inputs = []
        self.flow_inputs = []
        for index, row in labels_df.iterrows():
            rgb_seg_img_names, flow_seg_img_names = sample_train_segment(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"]
            )
            self.labels.append(row["verb_class"])
            self.rgb_inputs.append(load_rgb_frames(rgb_seg_img_names))
            self.flow_inputs.append(load_flow_frames(flow_seg_img_names))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index], self.rgb_inputs[index], self.flow_inputs[index]


def load_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


def load_flow_frames(flow_filenames):
    frames = []
    for group in flow_filenames:
        img_u = np.array(Image.open(group[0]))
        img_v = np.array(Image.open(group[1]))
        frames.append(np.array([img_u, img_v]))
    return np.asarray(frames).transpose([1, 0, 2, 3])


def load_rgb_frames(rgb_filenames):
    num_channels = 3
    frames = []
    for filename in rgb_filenames:
        frames.append(np.array(Image.open(rgb_filenames[0])))
    return np.asarray(frames).transpose([3, 0, 1, 2])


def sample_train_segment(temporal_window, start_frame, end_frame, domain_num, part_id):
    half_frame = int(temporal_window/2)
    step = 2
    rgb_seg_img_names = []
    flow_seg_img_names = []
    segment_start = int(start_frame) + (step*half_frame)
    segment_end = int(end_frame) + 1 - (step*half_frame)
    # Write a comment to explain this weirdness
    if segment_start >= segment_end:
        segment_start = int(start_frame)
        segment_end = int(end_frame)
    if segment_start <= half_frame*step+1:
        segment_start <= half_frame*step+2
    centre_frame = randint(segment_start, segment_end)
    for i in range(centre_frame-(step*half_frame), centre_frame+(step*half_frame), step):
        rgb_seg_img_names.append(f"./epic_kitchens_data/rgb/{domain_num}/{part_id}/frame_{str(i).zfill(10)}.jpg")
        flow_seg_img_names.append([
            f"./epic_kitchens_data/flow/{domain_num}/{part_id}/u/frame_{str(int(i/2)).zfill(10)}.jpg",
            f"./epic_kitchens_data/flow/{domain_num}/{part_id}/v/frame_{str(int(i/2)).zfill(10)}.jpg"
        ])
    return rgb_seg_img_names, flow_seg_img_names


if __name__ == "__main__":
    train_dataset = RgbFlowKitchensDataset(labels_path="./epic_kitchens_data/label_lookup/D2_train.pkl")
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    for (labels, rgb_inputs, flow_inputs) in train_dataloader:
        print(labels)
        print(labels.shape)
        break
