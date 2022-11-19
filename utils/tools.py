import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import re
import random
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import librosa
import torch
import torch.utils.data
import audio as Audio
import pdb
matplotlib.use("Agg")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device_diffusion(data, device):
    if len(data) == 19:
        (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            f0s,
            uvs,
            cwt_specs,
            f0_means,
            f0_stds,
            energies,
            durations,
            mel2phs,
        ) = data

        languages = torch.from_numpy(languages).long().to(device)
        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device) if mels is not None else mels
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).long().to(device)
        f0s = torch.from_numpy(f0s).float().to(device)
        uvs = torch.from_numpy(uvs).float().to(device)
        cwt_specs = torch.from_numpy(cwt_specs).float().to(device) if cwt_specs is not None else cwt_specs
        f0_means = torch.from_numpy(f0_means).float().to(device) if f0_means is not None else f0_means
        f0_stds = torch.from_numpy(f0_stds).float().to(device) if f0_stds is not None else f0_stds
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        mel2phs = torch.from_numpy(mel2phs).long().to(device)

        pitch_data = {
            "pitch": pitches,
            "f0": f0s,
            "uv": uvs,
            "cwt_spec": cwt_specs,
            "f0_mean": f0_means,
            "f0_std": f0_stds,
        }

        return [
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitch_data,
            energies,
            durations,
            mel2phs,
        ]

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)

def to_device_14_languages(data, device):
    pass

def to_device(data, device):
    if len(data) == 17: #hifigan
        (
            ids, #0
            raw_texts, #1
            languages, #2
            speakers, #3
            texts, #4
            src_lens, #5
            max_src_len, #6
            mels, #7
            mel_lens, #8
            max_mel_len, #9
            pitches, #10
            energies, #11
            durations, #12
            audio_hifigans, #13
            mel_hifigans, #14
            mel_hifigan_losses, #15
            audio_start_stops, #16
        ) = data

        languages = torch.from_numpy(languages).long().to(device)
        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)
        audio_hifigans = torch.from_numpy(audio_hifigans).float().to(device)
        mel_hifigans = torch.from_numpy(mel_hifigans).float().to(device)
        mel_hifigan_losses = torch.from_numpy(mel_hifigan_losses).float().to(device)
        audio_start_stops = audio_start_stops

        return (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
            audio_hifigans,
            mel_hifigans,
            mel_hifigan_losses,
            audio_start_stops,
        )

    if len(data) == 14: # wav2vec2
        (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
            signals,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        languages = torch.from_numpy(languages).long().to(device)
        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        signals = torch.from_numpy(signals).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
            signals,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 13:
        (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data
        languages = torch.from_numpy(languages).long().to(device)
        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 7:
        (
            ids, #0
            raw_texts, #1
            languages, #2
            speakers, #3
            texts, #4
            src_lens, #5
            max_src_len, #6
        ) = data

        languages = torch.from_numpy(languages).long().to(device)
        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (
            ids,
            raw_texts,
            languages,
            speakers,
            texts,
            src_lens,
            max_src_len,
        )


    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, model=None, sampling_rate=22050, tag="", hifigan_train=False, hifigan_val=False, discriminator=False
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        if discriminator:
            logger.add_scalar("Loss/loss_disc_all", losses[6], step)
        if hifigan_train:
            logger.add_scalar("Loss/loss_disc_all", losses[6], step)
            logger.add_scalar("Loss/loss_gen_all", losses[7], step)
        if hifigan_val:
            logger.add_scalar("Loss/GAN_MEL_loss", losses[6], step)

    if model is not None:
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.add_histogram(tag, value.data.cpu().numpy(), step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )

def log_denoiser(
    logger, step=None, losses=None, fig=None, audio=None, model=None, sampling_rate=22050, tag="", hifigan_train=False, hifigan_val=False, discriminator=False
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        if discriminator:
            logger.add_scalar("Loss/loss_disc_all", losses[6], step)
        if hifigan_train:
            logger.add_scalar("Loss/loss_disc_all", losses[6], step)
            logger.add_scalar("Loss/loss_gen_all", losses[7], step)
        if hifigan_val:
            logger.add_scalar("Loss/GAN_MEL_loss", losses[6], step)

    if model is not None:
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.add_histogram(tag, value.data.cpu().numpy(), step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def log1(
    logger, step=None, losses=None, fig=None, audio=None, model=None, sampling_rate=22050, tag="", hifigan_train=False, hifigan_val=False, discriminator=False, generator=False
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        if discriminator:
            logger.add_scalar("Loss/loss_gen_all", losses[6], step)
            logger.add_scalar("Loss/loss_disc_all", losses[7], step)
        if hifigan_train:
            logger.add_scalar("Loss/loss_disc_all", losses[6], step)
            logger.add_scalar("Loss/loss_gen_all", losses[7], step)
        if hifigan_val:
            logger.add_scalar("Loss/GAN_MEL_loss", losses[6], step)

    if model is not None:
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.add_histogram(tag, value.data.cpu().numpy(), step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )


def log_style(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/style_loss", losses[6], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )

def log_gan(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)
        logger.add_scalar("Loss/G_query_t", losses[6], step)
        logger.add_scalar("Loss/G_query_s", losses[7], step)
        logger.add_scalar("Loss/D_t", losses[8], step)
        logger.add_scalar("Loss/D_s", losses[9], step)
        logger.add_scalar("Loss/cls_loss", losses[10], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)

def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[11][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_one_sample_denoiser(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    mel_target = targets[6][0, :].detach().transpose(0, 1)
    mel_prediction = predictions[0, :].detach().transpose(0, 1)
    pitch = targets[9][0, :].detach().cpu().numpy()
    energy = targets[10][0, :].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]


    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename

def synth_one_sample_multilingual(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[7][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[12][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[10][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[10][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[11][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[11][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename

def synth_one_sample_multilingual_wav2vec2(targets, predictions, vocoder, model_config, preprocess_config):

    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[8][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[13][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[11][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[11][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[12][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[12][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):

    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer
    mel_predictions = predictions[1].transpose(1, 2)
    #############
    shape1, shape2, shape3 = mel_predictions.shape
    shape = mel_predictions.shape
    # noise = torch.normal(mean=0, std=0.5, size=shape)
    # noise = noise.to('cuda')
    # mel_predictions += noise
    #############

    # _stft = Audio.stft.TacotronSTFT(
    #         preprocess_config["preprocessing"]["stft"]["filter_length"],
    #         preprocess_config["preprocessing"]["stft"]["hop_length"],
    #         preprocess_config["preprocessing"]["stft"]["win_length"],
    #         preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
    #         preprocess_config["preprocessing"]["audio"]["sampling_rate"],
    #         preprocess_config["preprocessing"]["mel"]["mel_fmin"],
    #         preprocess_config["preprocessing"]["mel"]["mel_fmax"],
    #     )
    # audio_file="/data/processed/speech/tts/prepare_align_TTS/VN_VANANH/99-0001.wav"
    # wav, sample_rate = librosa.load(audio_file, sr=None)
    # if sample_rate != 22050:
    #     wav = librosa.resample(wav, sample_rate, 22050)
    # mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    # mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    # mel_spectrogram = mel_spectrogram[:,:,:shape3]
    # mel_spectrogram = torch.from_numpy(mel_spectrogram)
    # mel_spectrogram = mel_spectrogram.to('cuda')
    # mel_predictions = 0.2*mel_predictions + 1*mel_spectrogram
    # mel_predictions += mel_spectrogram
    # mel_predictions = mel_spectrogram

    #############
    # mel_path="/data/processed/speech/tts/VNTTS_VANANH_normal_sil_mien_bac_expand_number_ESPEAK_norm/mel/VN_VANANH-mel-99-0026.npy" 
    # mel_spectrogram = np.load(mel_path)
    # mel_spectrogram = np.expand_dims(mel_spectrogram.T, axis=0)
    # mel_spectrogram = mel_spectrogram[:,:,:shape3]
    # mel_spectrogram = torch.from_numpy(mel_spectrogram)
    # mel_spectrogram = mel_spectrogram.to('cuda')
    # mel_predictions = 0.5*mel_predictions + 0.5*mel_spectrogram
    #############
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig

def plot_mel_hifigan(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig



def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

# full context label to accent label from ttslearn
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))

def pp_symbols(labels, drop_unvoiced_vowels=True):
    PP = []
    accent = []
    N = len(labels)

    for n in range(len(labels)):
        lab_curr = labels[n]


        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == 'sil':
            assert n== 0 or n == N-1
            if n == N-1:
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("")
                elif e3 == 1:
                    PP.append("")
            continue
        elif p3 == "pau":
            PP.append("sp")
            accent.append('0')
            continue
        else:
            PP.append(p3)
        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        lab_next = labels[n + 1]
        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", lab_next)
        # アクセント境界
        if a3 == 1 and a2_next == 1:
            accent.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            accent.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            accent.append("[")
        else:
            accent.append('0')
    return PP, accent

def random_list(batch_size):
    def _check_duplicate(list1, list2):
        duplicate = False
        for i, (item1, item2) in enumerate(zip(list1, list2)):
            if item1 == item2:
                duplicate = True
                return duplicate
        return duplicate
    positive_list = list(range(batch_size))
    negative_list = list(range(batch_size))
    random.shuffle(negative_list)
    while _check_duplicate(positive_list, negative_list):
        random.shuffle(negative_list)
    return positive_list, negative_list

# def add_prefix2phone(phone, lang):
#     prefix = ""
#     _silences = ["sp", "spn", "sil"]
#     if lang == "Vie":
#         prefix = "vn_"
#     elif lang == "Eng":
#         prefix = "eng_"
#     elif lang == "Japan":
#         prefix = ""
#     elif lang == "Chi":
#         prefix = "cn_"
#     prefix_phone = []
#     for p in phone:
#         if p not in _silences:
#             prefix_phone.append(prefix+p)
#         else:
#             prefix_phone.append(p)
#     return prefix_phone

def add_prefix2phone(phone, lang):
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

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))
    # device='cpu'
    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    magnitudes = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], magnitudes)
    spec = spectral_normalize_torch(spec)
    energy = torch.norm(magnitudes, dim=1)
    spec = torch.squeeze(spec, 0)
    energy = torch.squeeze(energy, 0)
    # spec = torch.squeeze(spec, 0).numpy().astype(np.float32)
    # energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return spec, energy

def get_alignment(tier):
    sampling_rate = 22050
    hop_length = 256
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
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, durations, start_time, end_time

def add_noise(mels, _stft, path_noise=None):
    random_sample = np.random.random_sample()
    shape1, shape2, shape3 = mels.shape
    shape = mels.shape
    noise_amplitude = [0.1, 0.2, 0.3, 0.4, 0.5]
    amp = random.choice(noise_amplitude)
    # if random_sample > 0.5:
    if False:
        noise = torch.normal(mean=0, std=amp, size=shape)
        noise = noise.to('cuda')
        mels += noise
    else:
        noise = torch.normal(mean=0, std=amp, size=shape)
        noise = noise.cpu().detach().numpy()
        wav, sample_rate = librosa.load(path_noise, sr=None)
        if sample_rate != 22050:
            wav = librosa.resample(wav, sample_rate, 22050)
        mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
        mel_spectrogram = mel_spectrogram.T
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
        shape_mel1, shape_mel2, shape_mel3 = mel_spectrogram.shape
        shape_min = min(shape2, shape_mel2)
        noise[:,:shape_min,:] = mel_spectrogram[:, :shape_min, :]
        noise = torch.from_numpy(noise)
        noise = noise.to('cuda')
        mels = 1*mels + amp*noise
    return mels

def dur_to_mel2ph(dur, dur_padding=None, alpha=1.0):
    """
    Example (no batch dim version):
        1. dur = [2,2,3]
        2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
        3. token_mask = [[1,1,0,0,0,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,1,1]]
        4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                        [0,0,2,2,0,0,0],
                                        [0,0,0,0,3,3,3]]
        5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]
    :param dur: Batch of durations of each frame (B, T_txt)
    :param dur_padding: Batch of padding of each frame (B, T_txt)
    :param alpha: duration rescale coefficient
    :return:
        mel2ph (B, T_speech)
    """
    assert alpha > 0
    dur = torch.round(dur.float() * alpha).long()
    if dur_padding is not None:
        dur = dur * (1 - dur_padding.long())
    token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
    dur_cumsum = torch.cumsum(dur, 1)
    dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

    pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
    token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
    mel2ph = (token_idx * token_mask.long()).sum(1)
    return mel2ph


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur

def get_noise_schedule_list(schedule_mode, timesteps, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    else:
        raise NotImplementedError
    return schedule_list

def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        global window
        if window is None:
            window = create_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)

def openjtalk2julius(p3, prefix="jp_"):
    if p3 in ['A','I','U',"E", "O"]:
        return prefix + p3.lower()
    if p3 == 'cl':
        return 'q'
    if p3 == 'pau':
        return 'sp'
    if p3 == 'sil':
        return 'sil'
    return prefix + p3.lower()