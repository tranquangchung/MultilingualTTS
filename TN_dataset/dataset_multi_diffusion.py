import json
import math
import os
import sys
sys.path.append("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2_multilingual")

import numpy as np
from torch.utils.data import Dataset

#from text import text_to_sequence
from text import text_to_sequence, text_to_sequence_phone_vn, text_to_sequence_phone_vn_mfa
from utils.tools import pad_1D, pad_2D
from utils.pitch_tools import norm_interp_f0, get_lf0_cwt
import pdb

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.preprocess_config = preprocess_config
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
        # pitch stats
        self.pitch_type = preprocess_config["preprocessing"]["pitch"]["pitch_type"]
        self.f0_mean = float(preprocess_config["preprocessing"]["pitch"]["f0_mean"])
        self.f0_std = float(preprocess_config["preprocessing"]["pitch"]["f0_std"])

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
        f0_path = os.path.join(
            self.preprocessed_path,
            "f0",
            "{}-f0-{}.npy".format(speaker, basename),
        )
        f0 = np.load(f0_path)
        f0, uv = norm_interp_f0(f0, self.preprocess_config["preprocessing"]["pitch"])
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
        mel2ph_path = os.path.join(
            self.preprocessed_path,
            "mel2ph",
            "{}-mel2ph-{}.npy".format(speaker, basename),
        )
        mel2ph = np.load(mel2ph_path)

        cwt_spec = f0_mean = f0_std = f0_ph = None
        if self.pitch_type == "cwt":
            cwt_spec_path = os.path.join(
                self.preprocessed_path,
                "cwt_spec",
                "{}-cwt_spec-{}.npy".format(speaker, basename),
            )
            cwt_spec = np.load(cwt_spec_path)
            f0cwt_mean_std_path = os.path.join(
                self.preprocessed_path,
                "f0cwt_mean_std",
                "{}-f0cwt_mean_std-{}.npy".format(speaker, basename),
            )
            f0cwt_mean_std = np.load(f0cwt_mean_std_path)
            f0_mean, f0_std = float(f0cwt_mean_std[0]), float(f0cwt_mean_std[1])
        elif self.pitch_type == "ph":
            f0_phlevel_sum = torch.zeros(phone.shape).float().scatter_add(
                0, torch.from_numpy(mel2ph).long() - 1, torch.from_numpy(f0).float())
            f0_phlevel_num = torch.zeros(phone.shape).float().scatter_add(
                0, torch.from_numpy(mel2ph).long() - 1, torch.ones(f0.shape)).clamp_min(1)
            f0_ph = (f0_phlevel_sum / f0_phlevel_num).numpy()

        sample = {
            "id": basename,
            "language": lang_id,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "f0": f0,
            "f0_ph": f0_ph,
            "uv": uv,
            "cwt_spec": cwt_spec,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "energy": energy,
            "duration": duration,
            "mel2ph": mel2ph,
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
        languages = [data[idx]["language"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        f0s = [data[idx]["f0"] for idx in idxs]
        uvs = [data[idx]["uv"] for idx in idxs]
        cwt_specs = f0_means = f0_stds = f0_phs = None
        if self.pitch_type == "cwt":
            cwt_specs = [data[idx]["cwt_spec"] for idx in idxs]
            f0_means = [data[idx]["f0_mean"] for idx in idxs]
            f0_stds = [data[idx]["f0_std"] for idx in idxs]
            cwt_specs = pad_2D(cwt_specs)
            f0_means = np.array(f0_means)
            f0_stds = np.array(f0_stds)
        elif self.pitch_type == "ph":
            f0s = [data[idx]["f0_ph"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        mel2phs = [data[idx]["mel2ph"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        languages = np.array(languages)
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        f0s = pad_1D(f0s)
        uvs = pad_1D(uvs)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        mel2phs = pad_1D(mel2phs)

        return (
            ids,#0
            raw_texts,#1
            languages,#2
            speakers,#3
            texts,#4
            text_lens,#5
            max(text_lens),#6
            mels,#7
            mel_lens,#8
            max(mel_lens),#9
            pitches,#10
            f0s,#11
            uvs,#12
            cwt_specs,#13
            f0_means,#14
            f0_stds,#15
            energies,#16
            durations,#17
            mel2phs,#18
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


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.tools import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = "LibriTTS_StyleSpeech_multilingual_diffusion_test"
    # path = "VNTTS"
    preprocess_config = yaml.load(
        open("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2_multilingual/config/{0}/preprocess.yaml".format(path), "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("/home/chungtran/Code/C_tutorial/L2/.TTS/FastSpeech2_multilingual/config/{0}/train.yaml".format(path), "r"), Loader=yaml.FullLoader
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
            # if batch[4].shape != batch[10].shape:
            #     print("phone-pitch: ", batch[4].shape, batch[10].shape)
            print(batch)
            pdb.set_trace()
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
