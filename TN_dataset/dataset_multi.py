import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FastSpeech2'))
sys.path.append("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2")
import json
import math

import numpy as np
from torch.utils.data import Dataset

#from text import text_to_sequence
from text import text_to_sequence, text_to_sequence_phone_vn, text_to_sequence_phone_vn_mfa
from utils.tools import pad_1D, pad_2D
import pdb

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.language, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        
        with open(os.path.join(self.preprocessed_path, "languages.json")) as f:
            self.language_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        lang = self.language[idx]
        lang_id = self.language_map[lang]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        # pdb.set_trace()
        # print(raw_text)
        # print(phone)
        # print("*"*20)

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
            "language": lang_id,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            language = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, l, s, t, r = line.strip("\n").split("|")
                name.append(n)
                language.append(l)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, language, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        languages = [data[idx]["language"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        languages = np.array(languages)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids, #0
            raw_texts, #1
            languages, #2
            speakers, #3
            texts, #4
            text_lens, #5
            max(text_lens), #6
            mels, #7
            mel_lens, #8
            max(mel_lens), #9
            pitches, #10
            energies, #11
            durations, #12
        )

    def collate_fn(self, data):
        data_size = len(data)
        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.language, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        # phone = np.array(text_to_sequence_phone_vn_mfa(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            language = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, l, s, t, r = line.strip("\n").split("|")
                name.append(n)
                language.append(l)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, language, speaker, text, raw_text

    def collate_fn(self, data):
        # TODO: can xac dinh language # 9/12/2021
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "LibriTTS_StyleSpeech_multilingual"
    # path = "VNTTS"
    preprocess_config = yaml.load(
        open("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2/config/{0}/preprocess.yaml".format(path), "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2/config/{0}/train.yaml".format(path), "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 1,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            pdb.set_trace()
            # if batch[4].shape != batch[10].shape:
            #     print("phone-pitch: ", batch[4].shape, batch[10].shape)
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
            # print(batch)
            # if batch[4].shape != batch[10].shape:
            #     print("phone-pitch: ", batch[4].shape, batch[10].shape)
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
