import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FastSpeech2'))
sys.path.append("/home/s2220411/Code/FastSpeech2_multilingual")
import json
import math

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D
import random
import pdb

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.num_sample = self.batch_size
        # self.num_sample = train_config["optimizer"]["num_sample"] # can chia het cho 6

        self.basename, self.language, self.speaker, self.text, self.raw_text, self.metadata, self.map_speaker_lang = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        
        with open(os.path.join(self.preprocessed_path, "languages.json")) as f:
            self.language_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        self.num_speaker = len(self.speaker_map)
        self.num_lang = len(self.language_map)
        self.speaker_list = list(self.speaker_map.keys())
        self.language_list = list(self.language_map.keys())
        self.index_lang = 0
        self.index_speaker = 0

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        sample_list = [] 
        # for i in range(self.num_sample):
        while True:
            if random.uniform(0, 1) > 0.15:
                speaker = self.speaker_list[self.index_speaker % self.num_speaker]
                self.index_speaker += 1
            else:
                speaker = random.choice(self.speaker_list)
            try:
                lang = self.map_speaker_lang[speaker]
            except Exception as e:
                continue
            lang_id = self.language_map[lang]
            speaker_id = self.speaker_map[speaker]

            basename = self.metadata[lang][speaker]['list_files'].pop(random.randrange(len(self.metadata[lang][speaker]['list_files'])))
            if len(self.metadata[lang][speaker]['list_files']) == 0:
                self.metadata[lang][speaker]['list_files'] = list(self.metadata[lang][speaker]['files'].keys())
            raw_text = self.metadata[lang][speaker]['files'][basename]["raw_text"]
            text = self.metadata[lang][speaker]['files'][basename]["text"]
            text_phone = text_to_sequence(text, self.cleaners)
            phone = np.array(text_phone)
            phone_lang = np.array([lang_id+1]*len(text_phone))

            mel_path = os.path.join(
                self.preprocessed_path,
                "mel",
                "{0}-{1}-mel-{2}.npy".format(lang, speaker, basename),
            )
            mel = np.load(mel_path)
            pitch_path = os.path.join(
                self.preprocessed_path,
                "pitch",
                "{0}-{1}-pitch-{2}.npy".format(lang, speaker, basename),
            )
            pitch = np.load(pitch_path)
            energy_path = os.path.join(
                self.preprocessed_path,
                "energy",
                "{0}-{1}-energy-{2}.npy".format(lang, speaker, basename),
            )
            energy = np.load(energy_path)
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration",
                "{0}-{1}-duration-{2}.npy".format(lang, speaker, basename),
            )
            duration = np.load(duration_path)

            sample = {
                "id": basename,
                "language": lang_id, # truoc mat thi nghiem voi lang_id, sau do se phai chay voi phone_lang,
                "speaker": speaker_id,
                "text": phone,
                "raw_text": raw_text,
                "mel": mel,
                "pitch": pitch,
                "energy": energy,
                "duration": duration,
            }
            sample_list.append(sample)
            if len(sample_list) == self.num_sample:
                break
        return sample_list

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            metadata = {}
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
                if l not in metadata:
                    metadata[l] = {}
                if s not in metadata[l]:
                    metadata[l][s] = {}
                    metadata[l][s]['files'] = {}
                key = n
                value = {
                        "text": t,
                        "raw_text": r
                    }
                metadata[l][s]['files'][key] = value
            map_speaker_lang = {}
            for lang_tmp in list(metadata.keys()):
                for speaker_tmp in list(metadata[lang_tmp].keys()):
                    len_files = len(metadata[lang_tmp][speaker_tmp]['files'].keys())
                    metadata[lang_tmp][speaker_tmp]['index'] = 0
                    metadata[lang_tmp][speaker_tmp]['len_files'] = len_files
                    metadata[lang_tmp][speaker_tmp]['list_files'] = list(metadata[lang_tmp][speaker_tmp]['files'].keys())
                    map_speaker_lang[speaker_tmp] = lang_tmp
            return name, language, speaker, text, raw_text, metadata, map_speaker_lang

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
        # languages = pad_1D(languages)
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
    # path = "LibriTTS_StyleSpeech_multilingual"
    # path = "VNTTS"

    path = "LibriTTS_StyleSpeech_multilingual_diffusion_style_6L"
    preprocess_config = yaml.load(
        open("../config/config_kaga/{0}/preprocess.yaml".format(path), "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("../config/config_kaga/{0}/train.yaml".format(path), "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
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
        pdb.set_trace()
        for batch in batchs:
            # print(batch)
            if batch[4].shape != batch[10].shape:
                print("phone-pitch: ", batch[4].shape, batch[10].shape)
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
