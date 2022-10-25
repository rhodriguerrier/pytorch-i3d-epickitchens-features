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
        if self.is_flow:
            imgs = load_flow_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = video_to_tensor(imgs)
        else:
            imgs = load_rgb_frames(self.input_names[index])
            imgs = self.transforms(imgs)
            imgs = video_to_tensor(imgs)
        return self.labels[index], imgs, self.narration_ids[index]


class SequentialClassKitchens(Dataset):
    def __init__(self, labels_path, class_num, temporal_window=16, is_flow=False):
        labels_df = load_pickle_data(labels_path)
        filtered_df = labels_df[labels_df["verb_class"] == class_num]
        self.labels = []
        self.input_names = []
        self.frame_start_numbers = []
        self.is_flow = is_flow
        for index, row in filtered_df.iterrows():
            clip_length = row["stop_frame"] - row["start_frame"]
            num_windows, total_window_len = get_num_seq_windows(clip_length, temporal_window)
            frame_start_numbers = get_start_frame_numbers(
                row["start_frame"],
                row["stop_frame"],
                total_window_len,
                num_windows
            )
            self.labels.append(f"{row['verb_class']}.{row['uid']}")
            self.input_names.append(
                sample_test_sequential_frames(
                    frame_start_numbers,
                    temporal_window,
                    self.is_flow,
                    row["participant_id"],
                    row["video_id"]
                )
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        seq_window_imgs = []
        for frame_paths in self.input_names[index]:
            if self.is_flow:
                imgs = load_flow_frames(frame_paths)
            else:
                imgs = load_rgb_frames(frame_paths)
            imgs = video_to_tensor(imgs)
            seq_window_imgs.append(imgs)
        return self.labels[index], seq_window_imgs


def sample_test_sequential_frames(start_frames, temporal_window_len, is_flow, part_id, video_id):
    seq_frame_names = []
    for start_frame in start_frames:
        if is_flow:
            seq_frame_names.append([
                [
                    f"/user/work/rg16964/epic_kitchens_data/flow/{part_id}/{video_id}/u/frame_{str(int((start_frame+i)/2)).zfill(10)}.jpg",
                    f"/user/work/rg16964/epic_kitchens_data/flow/{part_id}/{video_id}/v/frame_{str(int((start_frame+i)/2)).zfill(10)}.jpg"
                ] for i in range(temporal_window_len)
            ])
        else:
            seq_frame_names.append(
                [f"/user/work/rg16964/epic_kitchens_data/rgb/{part_id}/{video_id}/frame_{str((start_frame+i)).zfill(10)}.jpg" for i in range(temporal_window_len)]
            )
    return seq_frame_names


def get_num_seq_windows(clip_length, window_len):
    num_windows = round(clip_length/window_len)
    window_total_len = (num_windows*window_len) - (2*(num_windows-1))
    if window_total_len > clip_length:
        num_windows -= 1
        window_total_len = (num_windows*window_len) - (2*(num_windows-1))
    return num_windows, window_total_len


def get_start_frame_numbers(start_frame, stop_frame, total_window_len, num_windows):
    centre_frame = int(start_frame + ((stop_frame - start_frame)/2))
    half_window_len = int(total_window_len / 2)
    frame_start_numbers = []
    for i in range(num_windows):
        frame_start_numbers.append((centre_frame-half_window_len) + (16*i) - (2*i))
    return frame_start_numbers


def load_pickle_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


def video_to_tensor(frames):
    return frames.transpose([3, 0, 1, 2])


def load_flow_frames(flow_filenames):
    for i, group in enumerate(flow_filenames):
        img_u = np.array(Image.open(group[0]))
        img_v = np.array(Image.open(group[1]))
        if i == 0:
            frames = np.array([[img_u, img_v]])
        else:
            frames = np.concatenate((frames, np.array([[img_u, img_v]])), axis=0)
    frames = frames.transpose([0, 2, 3, 1])
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
                f"/user/work/rg16964/epic_kitchens_data/flow/{part_id}/{video_id}/u/frame_{str(int(i/2)).zfill(10)}.jpg",
                f"/user/work/rg16964/epic_kitchens_data/flow/{part_id}/{video_id}/v/frame_{str(int(i/2)).zfill(10)}.jpg"
            ])
        else:
            seg_img_names.append(f"/user/work/rg16964/epic_kitchens_data/rgb/{part_id}/{video_id}/frame_{str(i).zfill(10)}.jpg")
    return seg_img_names


if __name__ == "__main__":
    train_dataset = EpicKitchensDataset(labels_path="/user/work/rg16964/label_lookup/D2_train.pkl", is_flow=False)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    for (labels, rgb_inputs) in train_dataloader:
        print(labels)
        print(labels.shape)
        break
