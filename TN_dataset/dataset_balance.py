import json
import math
import os

import numpy as np
from torch.utils.data import Dataset

#from text import text_to_sequence
from text import text_to_sequence, text_to_sequence_phone_vn, text_to_sequence_phone_vn_mfa
from utils.tools import pad_1D, pad_2D
import pdb
import random


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.num_sample = train_config["optimizer"]["num_sample"] # can chia het cho 4
        self.basename, self.speaker, self.text, self.raw_text, self.metadata = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.num_speaker = len(self.speaker_map)
        self.index_speaker = 0
        self.speaker_list = list(self.metadata.keys())

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sample_list = []
        for i in range(self.num_sample):
            speaker = self.speaker_list[self.index_speaker % self.num_speaker]
            self.index_speaker += 1
            # speaker = random.choice(speaker_list)
            # basename_list = list(self.metadata[speaker].keys())
            # basename = random.choice(basename_list)
            # random_sample = np.random.random_sample()
            # if random_sample > 0.99:
            #     index = self.metadata[speaker]['index']
            #     len_files = self.metadata[speaker]['len_files']
            #     basename = list(self.metadata[speaker]['files'])[index % len_files] 
            #     self.metadata[speaker]['index'] += 1
            # else:
                # basename_list = list(self.metadata[speaker]['files'].keys())
                # basename = random.choice(basename_list)
            basename = self.metadata[speaker]['list_files'].pop(random.randrange(len(self.metadata[speaker]['list_files']))) 
            if len(self.metadata[speaker]['list_files']) == 0:
                self.metadata[speaker]['list_files'] = list(self.metadata[speaker]['files'].keys())
            #########
            speaker_id = self.speaker_map[speaker]

            raw_text = self.metadata[speaker]['files'][basename]["raw_text"]
            text = self.metadata[speaker]['files'][basename]["text"]
            # phone = np.array(text_to_sequence(self.text[idx], self.cleaners)) 
            phone = np.array(text_to_sequence_phone_vn_mfa(text, self.cleaners)) # Luu y: khi huan luyen single thi su dung ham nay
            mel_path = os.path.join(
                self.preprocessed_path,
                "mel",
                "{}-mel-{}.npy".format(speaker, basename),
            )
            mel = np.load(mel_path)
            pitch_path = os.path.join(
                self.preprocessed_path,
                "pitch",
                "{}-pitch-{}.npy".format(speaker, basename),
            )
            pitch = np.load(pitch_path)
            energy_path = os.path.join(
                self.preprocessed_path,
                "energy",
                "{}-energy-{}.npy".format(speaker, basename),
            )
            energy = np.load(energy_path)
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{}-duration-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)

            sample = {
                "id": basename,
                "speaker": speaker_id,
                "text": phone,
                "raw_text": raw_text,
                "mel": mel,
                "pitch": pitch,
                "energy": energy,
                "duration": duration,
            }
            sample_list.append(sample)
        # random.shuffle(sample_list) 
        return sample_list

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            metadata = {}
            name = []
            speaker = []
            text = []
            raw_text = []
            try:
                for line in f.readlines():
                    n, s, t, r = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
                    if s not in metadata:
                        metadata[s] = {}
                        metadata[s]['files'] = {}
                    key = n
                    value = {
                            "text": t,
                            "raw_text": r
                            }
                    metadata[s]['files'][key] = value
                for speaker_tmp in list(metadata.keys()):
                    len_files = len(metadata[speaker_tmp]['files'].keys()) 
                    metadata[speaker_tmp]['index'] = 0
                    metadata[speaker_tmp]['len_files'] = len_files
                    metadata[speaker_tmp]['list_files'] = list(metadata[speaker_tmp]['files'].keys())
            except Exception as e:
                print(line)
                exit()
            return name, speaker, text, raw_text, metadata

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids, #0
            raw_texts, #1
            speakers, #2
            texts, #3
            text_lens, #4
            max(text_lens), #5
            mels, #6
            mel_lens, #7
            max(mel_lens), #8
            pitches, #9
            energies, #10
            durations, #11
        )

    def collate_fn(self, data):
        data = data[0]
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)
        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.num_sample) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.num_sample)]
        idx_arr = idx_arr.reshape((-1, self.num_sample)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output

if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "VNTTS_MultiSpeaker"
    # path = "VNTTS"
    preprocess_config = yaml.load(
        open("./config/{0}/preprocess.yaml".format(path), "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/{0}/train.yaml".format(path), "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            pdb.set_trace()
            print(batch[0])
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
