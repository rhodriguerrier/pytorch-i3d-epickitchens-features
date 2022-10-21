import torch
from torch.utils.data import Dataset, DataLoader
from random import randint
import pickle
import numpy as np
from PIL import Image


class EpicKitchensDataset(Dataset):
    def __init__(self, labels_path, is_flow=False, transforms=None):
        labels_df = load_pickle_data(labels_path)
        self.transforms = transforms
        self.labels = []
        self.narration_ids = []
        self.is_flow = is_flow
        self.input_names = []
        for index, row in labels_df.iterrows():
            seg_img_names = sample_train_segment(
                16,
                row["start_frame"],
                row["stop_frame"],
                row["participant_id"],
                row["video_id"],
                is_flow=is_flow
            )
            self.labels.append(row["verb_class"])
            self.narration_ids.append(f"{row['video_id']}_{row['uid']}")
            self.input_names.append(seg_img_names)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        print(index)
        if self.is_flow:
            imgs = load_flow_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = flow_video_to_tensor(imgs)
        else:
            imgs = load_rgb_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = rgb_video_to_tensor(imgs)
        return self.labels[index], imgs, self.narration_ids[index]


def load_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data

def rgb_video_to_tensor(frames):
    return frames.transpose([3, 0, 1, 2])


def flow_video_to_tensor(frames):
    return frames.transpose([1, 0, 2, 3])


def load_flow_frames(flow_filenames):
    for i, group in enumerate(flow_filenames):
        img_u = np.array(Image.open(group[0]))
        img_v = np.array(Image.open(group[1]))
        if i == 0:
            frames = np.array([[img_u, img_v]])
        else:
            frames = np.concatenate((frames, np.array([[img_u, img_v]])), axis=0)
    return (((frames / 255) * 2) - 1)


def load_rgb_frames(rgb_filenames):
    for i, filename in enumerate(rgb_filenames):
        img_matrix = np.array(Image.open(filename))
        if i == 0:
            frames = np.array([img_matrix])
        else:
            frames = np.concatenate((frames, np.array([img_matrix])), axis=0)
    return (((frames / 255) * 2) - 1)


def sample_train_segment(temporal_window, start_frame, end_frame, part_id, video_id, is_flow):
    half_frame = int(temporal_window/2)
    step = 2
    seg_img_names = []
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
        if is_flow:
            seg_img_names.append([
                f"/user/home/rg16964/epic_kitchens_data/flow/{part_id}/{video_id}/u/frame_{str(int(i/2)).zfill(10)}.jpg",
                f"/user/home/rg16964/epic_kitchens_data/flow/{part_id}/{video_id}/v/frame_{str(int(i/2)).zfill(10)}.jpg"
            ])
        else:
            seg_img_names.append(f"/user/home/rg16964/epic_kitchens_data/rgb/{part_id}/{video_id}/frame_{str(i).zfill(10)}.jpg")
    return seg_img_names


if __name__ == "__main__":
    train_dataset = EpicKitchensDataset(labels_path="/user/home/rg16964/label_lookup/D2_train.pkl", is_flow=False)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    for (labels, rgb_inputs) in train_dataloader:
        print(labels)
        print(labels.shape)
        break
