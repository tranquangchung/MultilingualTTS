import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import glob
import audio as Audio
import pdb
from pathlib import Path
from joblib import Parallel, delayed

number_worker = 20
class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def write_metadata(self):
        data_dir = self.in_dir
        out_dir = self.out_dir
        metadata = os.path.join(out_dir, 'metadata.csv')
        if not os.path.exists(metadata):
            wav_fname_list = [str(f) for f in list(Path(data_dir).rglob('*/*/[!.]*.wav'))] #ignore hidden file
            lines = []
            for wav_fname in wav_fname_list:
                print(wav_fname)
                basename = wav_fname.split('/')[-1].replace('.wav', '')
                language = wav_fname.split('/')[-3]
                sid = language + "_" + wav_fname.split('/')[-2] #speaker id
                txt_fname = wav_fname.replace('.wav', '.txt')
                if not os.path.exists(txt_fname):
                    txt_fname = wav_fname.replace('.wav', '.lab')
                with open(txt_fname, 'r') as f:
                    text = f.readline().strip()
                    f.close()
                lines.append('{}|{}|{}|{}'.format(basename, language, sid, text))
            with open(metadata, 'wt') as f:
                f.writelines('\n'.join(lines))
                f.close()

    def build_from_path_paralel(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        languages = {}
        speakers = {}
        index_spk = 0
        info_list = []
        user_list = []
        lang_list = []
        max_seq_len = -float('inf')
        mel_min = np.ones(80) * float('inf')
        mel_max = np.ones(80) * -float('inf')
        with open(os.path.join(self.out_dir, 'metadata.csv'), encoding='utf-8') as f:
            for line in f:
                tmp = line.strip().split('|') # basename, lang, user_id, raw_text
                info_list.append(tmp)
                user_list.append(tmp[2])
                lang_list.append(tmp[1])
        ########################
        # results = []
        # for info in info_list:
        #     print(info)
        #     tmp = self.process_utterance_paralel(info)
        #     results.append(tmp)

        ########################
        results = Parallel(n_jobs=number_worker, verbose=1)(
            delayed(self.process_utterance_paralel)(info) for info in info_list
        )
        ########################
        for ret in results:
            if ret is not None:
                info, pitch, energy, n, m_min, m_max = ret
                out.append(info)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                mel_min = np.minimum(mel_min, m_min)
                mel_max = np.maximum(mel_max, m_max)
                if n > max_seq_len:
                    max_seq_len = n
                n_frames += n
        user_unique = sorted(list(set(user_list)))
        for index_spk, speaker in enumerate(user_unique):
            speakers[speaker] = index_spk
        lang_unique = sorted(list(set(lang_list)))
        for index_lang, lang in enumerate(lang_unique):
            languages[lang] = index_lang

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
            pitch_mean = 0
            pitch_std = 1
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))
        
        with open(os.path.join(self.out_dir, "languages.json"), "w") as f:
            f.write(json.dumps(languages))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
                "spec_min": mel_min.tolist(),
                "spec_max": mel_max.tolist(),
                "max_seq_len": max_seq_len,
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]
        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[:]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def get_tg_path(self, info):
        basename, lang, speaker, raw_text = info
        tg_path = ""
        if lang in [
            "vietnamese", "polish", "dutch", "french",
            "german", "indonesian", "italian", "korean",
            "portuguese", "spanish"
        ]:
            tg_path = os.path.join(
                self.out_dir, "Multi_TextGrid", lang, speaker, "{}.TextGrid".format(basename.replace("_", "-"))
            )
            # print(tg_path)
        if lang in ["chinese", "japanese"]:
            tg_path = os.path.join(
                self.out_dir, "Multi_TextGrid", lang, speaker, "{}.TextGrid".format(basename)
            )
        if lang == "russian":
            tg_path = os.path.join(
                self.out_dir, "Multi_TextGrid", lang, speaker, "{0}-{1}.TextGrid".format(speaker,basename.replace("_", "-"))
            )
        if lang == "english": # danh cho Libris va VTCK
            if len(basename.split("_")) == 4: # Libris
                tmp_sp = basename.split("_")[1]
                tg_path = os.path.join(
                    self.out_dir, "Multi_TextGrid", lang, speaker, tmp_sp, "{}.TextGrid".format(basename)
                )
            elif len(basename.split("_")) == 2: # VTCK
                tg_path = os.path.join(
                    self.out_dir, "Multi_TextGrid", lang, speaker, "{}.TextGrid".format(basename)
                )
            elif speaker == "LJSpeech": # LJSpeech
                tg_path = os.path.join(
                    self.out_dir, "Multi_TextGrid", lang, speaker, "{}.TextGrid".format(basename)
                )
        if os.path.exists(tg_path):
            return tg_path
        else:
            # print(tg_path)
            return None

    def add_prefix2phone(self, phone, lang):
        prefix = ""
        _silences = ["sp", "spn", "sil"]
        dictionary_prefix = {
            "chinese": "cn_",
            "dutch": "du_",
            "english": "en_",
            "french": "fr_",
            "german": "ge_",
            "indonesian": "in_",
            "italian": "it_",
            "japanese": "", # "jp_"
            "korean": "ko_",
            "polish": "po_",
            "portuguese": "por_",
            "russian": "ru_",
            "spanish": "sp_",
            "vietnamese": "vn_"
        }
        prefix = dictionary_prefix[lang]
        prefix_phone = []
        for p in phone:
            if p not in _silences:
                prefix_phone.append(prefix+p)
            else:
                prefix_phone.append(p)
        return prefix_phone

    def process_utterance_paralel(self, info):
        basename, lang, speaker, raw_text = info
        speaker = speaker.split("_", 1)[1]
        info = [basename, lang, speaker, raw_text]
        wav_path = os.path.join(self.in_dir, lang, speaker, "{}.wav".format(basename))
        # text_path = os.path.join(self.in_dir, lang, speaker, "{}.txt".format(basename))
        tg_path = self.get_tg_path(info)
        # Get alignments
        if tg_path:
            textgrid = tgt.io.read_textgrid(tg_path)
        else:
            return None
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        phone = self.add_prefix2phone(phone, lang)

        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        raw_text = raw_text.strip()
        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]
        # if mel_spectrogram.shape[1] >= 1000:
        #     print("spectrogram >= 1000")
        #     return None
        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]
            pitch_has_nan = np.isnan(pitch).any()
            if pitch_has_nan:
                print(info)
                return None

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]
            energy_has_nan = np.isnan(energy).any()
            if energy_has_nan:
                print(info)
                return None

        # Save files
        speaker = lang+"_"+speaker
        dur_filename = "{0}-{1}-duration-{2}.npy".format(lang, speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{0}-{1}-pitch-{2}.npy".format(lang, speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{0}-{1}-energy-{2}.npy".format(lang, speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{0}-{1}-mel-{2}.npy".format(lang, speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, lang, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            np.min(mel_spectrogram, axis=1),
            np.max(mel_spectrogram, axis=1),
        )


    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        ##### Method 1: For Sequence #####
        # max_value = np.finfo(np.float64).min
        # min_value = np.finfo(np.float64).max
        # for filename in os.listdir(in_dir):
        #     filename = os.path.join(in_dir, filename)
        #     values = (np.load(filename) - mean) / std
        #     np.save(filename, values)
        #
        #     max_value = max(max_value, max(values))
        #     min_value = min(min_value, min(values))
        #
        # return min_value, max_value

        ##### Method 2: #######
        # filenames = os.listdir(in_dir)
        # min_value_list = []
        # max_value_list = []
        # for filename in filenames:
        #     min_value, max_value = self.process_normalize_paralel([in_dir, filename, mean, std])
        #     min_value_list.append(min_value)
        #     max_value_list.append(max_value)
        # return min(min_value_list), max(max_value_list)

        ##### Method 3: For Parallel #####
        filenames = os.listdir(in_dir)
        results = Parallel(n_jobs=number_worker, verbose=1)(
            delayed(self.process_normalize_paralel)([in_dir, filename, mean, std]) for filename in filenames
        )
        results = np.array(results)
        min_value = results[:,0].min()
        max_value = results[:,1].max()
        return min_value, max_value

    def process_normalize_paralel(self, info):
        in_dir = info[0]
        filename = info[1]
        mean = info[2]
        std = info[3]
        filename = os.path.join(in_dir, filename)
        values = (np.load(filename) - mean) / std
        np.save(filename, values)
        return min(values), max(values)


