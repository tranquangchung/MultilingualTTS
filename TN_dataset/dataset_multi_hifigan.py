import json
import math
import os

import tgt
import numpy as np
from torch.utils.data import Dataset

#from text import text_to_sequence
from text import text_to_sequence, text_to_sequence_phone_vn, text_to_sequence_phone_vn_mfa
from utils.tools import pad_1D, pad_2D, get_alignment, mel_spectrogram
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import librosa
import torch
import torch.utils.data
import random
import audio as Audio
import pdb


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.raw_path = preprocess_config["path"]["raw_path"]
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
        self.segment_size = 8192
        self.MAX_WAV_VALUE = 32768.0

        self.STFT = Audio.stft.TacotronSTFT_hifigan(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
        self.STFT_Loss = Audio.stft.TacotronSTFT_hifigan(
            preprocess_config["preprocessing"]["stft"]["filter_length"],
            preprocess_config["preprocessing"]["stft"]["hop_length"],
            preprocess_config["preprocessing"]["stft"]["win_length"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            preprocess_config["preprocessing"]["audio"]["sampling_rate"],
            preprocess_config["preprocessing"]["mel"]["mel_fmin"],
            None,
        )

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
        # print(wav_path)
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
        
        tg_path = self.get_tg_path(self.preprocessed_path, basename, lang, speaker)
        textgrid = tgt.io.read_textgrid(tg_path)
        _, _, start, end = get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        filename = os.path.join(self.raw_path, lang, speaker, "{}.wav".format(basename))
        audio, sampling_rate = self.load_wav(filename)
        # audio = normalize(audio) * 0.95
        # print(filename)
        # print(audio)
        audio = audio[int(sampling_rate * start) : int(sampling_rate * end)].astype(np.float32)


        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        audio_start = 0
        if audio.size(1) >= self.segment_size:
            max_audio_start_short = audio.size(1) - self.segment_size # for short audio
            max_audio_start_long = self.segment_size # for long audio
            max_audio_start = min(max_audio_start_short, max_audio_start_long) # get min value
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start+self.segment_size]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        
        # print(audio.shape)
        # print(basename)
        # print(filename)
        # print("*"*20)
        
        ## Melspectrogram of Hifigan
        mel_hifigan, energy_hifigan = mel_spectrogram(audio, n_fft=1024, num_mels=80,sampling_rate=22050, hop_size=256, 
                                        win_size=1024, fmin=0, fmax=8000,center=False)
        mel_hifigan_loss, energy_hifigan_loss = mel_spectrogram(audio, n_fft=1024, num_mels=80,sampling_rate=22050, hop_size=256, 
                                        win_size=1024, fmin=0, fmax=None, center=False)

        ## Melspectrogram of Fastspeech2
        # audio = audio.cpu().detach().numpy()
        # mel_hifigan, energy_hifigan = Audio.tools.get_mel_from_wav(audio.squeeze(), self.STFT)
        # mel_hifigan_loss, energy_hifigan = Audio.tools.get_mel_from_wav(audio.squeeze(), self.STFT_Loss)

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
            "audio_hifigan": audio.cpu().detach().numpy(),
            "mel_hifigan": mel_hifigan.cpu().detach().numpy(),
            "mel_hifigan_loss": mel_hifigan_loss.cpu().detach().numpy(),
            "audio_start_stop": [audio_start, self.segment_size]
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
        
        audio_hifigans = [data[idx]["audio_hifigan"] for idx in idxs]
        mel_hifigans = [data[idx]["mel_hifigan"] for idx in idxs]
        mel_hifigan_losses = [data[idx]["mel_hifigan_loss"] for idx in idxs]
        audio_start_stops = [data[idx]["audio_start_stop"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        languages = np.array(languages)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        audio_hifigans = np.array(audio_hifigans)
        mel_hifigans = np.array(mel_hifigans)
        mel_hifigan_losses = np.array(mel_hifigan_losses)
        audio_start_stops = np.array(audio_start_stops)


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
            audio_hifigans, #13
            mel_hifigans, #14
            mel_hifigan_losses, #15
            audio_start_stops, #16
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

    def load_wav(self, full_path):
        # sampling_rate, data = read(full_path)
        wav, sampling_rate = librosa.load(full_path)
        np.clip(wav, -1, 1, out=wav)
        return wav, sampling_rate

    def get_tg_path(self, path, basename, lang, speaker):
        tg_path = ""
        if lang == "Vie":
            tg_path = os.path.join(
                path, "Multi_TextGrid", lang, "TextGrid", speaker, "{}.TextGrid".format(basename.replace("_", "-"))
            )
            # print(tg_path)
        if lang == "Chi" or lang == "Japan":
            tg_path = os.path.join(
                path, "Multi_TextGrid", lang, "TextGrid", speaker, "{}.TextGrid".format(basename)
            )
        elif lang == "Eng": # danh cho Libris va VTCK
            if len(basename.split("_")) == 4: # Libris
                tmp_sp = basename.split("_")[1]
                tg_path = os.path.join(
                    path, "Multi_TextGrid", lang, "TextGrid", speaker, tmp_sp, "{}.TextGrid".format(basename)
                )
            elif len(basename.split("_")) == 2: # VTCK
                tg_path = os.path.join(
                    path, "Multi_TextGrid", lang, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
        if os.path.exists(tg_path):
            return tg_path
        else:
            print(tg_path)
            return None

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
        batch_size=train_config["optimizer"]["batch_size"] * 1,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    # n_batch = 0
    # for batchs in train_loader:
    #     for batch in batchs:
    #         # if batch[4].shape != batch[10].shape:
    #         #     print("phone-pitch: ", batch[4].shape, batch[10].shape)
    #         # print(batch)
    #         # to_device(batch, device)
    #         n_batch += 1
    # print(
    #     "Training set  with size {} is composed of {} batches.".format(
    #         len(train_dataset), n_batch
    #     )
    # )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            # pdb.set_trace()
            # print()
            # print(batch)
            # if batch[4].shape != batch[10].shape:
            #     print("phone-pitch: ", batch[4].shape, batch[10].shape)
            # to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
