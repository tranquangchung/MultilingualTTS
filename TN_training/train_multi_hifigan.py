import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import (get_model, get_vocoder, get_param_num, \
        get_model_fastSpeech2_StyleEncoder, \
        get_model_fastSpeech2_StyleEncoder_MultiLanguage, \
        get_model_fastSpeech2_MultiSpeakers_MultiLangs, \
        get_model_fastSpeech2_StyleEncoder_HifiGan_MultiLanguage )
from model.HifiGan import feature_loss, discriminator_loss, generator_loss
from utils.tools import to_device, log, synth_one_sample, synth_one_sample_multilingual
from utils.tools import mel_spectrogram
from model.HifiGan import feature_loss, generator_loss
from model import FastSpeech2Loss_MultiLingual
from dataset_multi_hifigan import Dataset
import torch.nn.functional as F
from evaluate import evaluate, evaluate_multilingual_hifigan
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    # model, optimizer = get_model(args, configs, device, train=True)
    model, generator, mpd, msd, optimizer, scheduler_g, scheduler_d, optim_g, optim_d = get_model_fastSpeech2_StyleEncoder_HifiGan_MultiLanguage(args, configs, device, train=True)
    model = nn.DataParallel(model)
    generator = nn.DataParallel(generator)
    mpd = nn.DataParallel(mpd)
    msd = nn.DataParallel(msd)

    num_param = get_param_num(model)
    Loss = FastSpeech2Loss_MultiLingual(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)
    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()
    
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                # print("*"*20)
                # print(len(batch[0]))
                if len(batch[0]) != batch_size:
                    break
                batch = to_device(batch, device)
                #Forward to Generator
                audio_hifigans, mel_hifigans, mel_hifigan_losses, audio_start_stops = batch[13:]
                
                # Training G, D of hifigan
                ## First. Discriminator
                y_g_hat = generator(mel_hifigans)
                y_g_hat_mel, _ = mel_spectrogram(y_g_hat.squeeze(1), n_fft=1024, num_mels=80,sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=None, center=False)

                optim_d.zero_grad()
                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = mpd(audio_hifigans, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(audio_hifigans, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                optim_d.step()
                # print("***** Train D ******")
                # print(mpd.module.discriminators[4].convs[4].weight[:,:,0,0])
                # print("***** Train D ******")

                ## Second. Generator
                optim_g.zero_grad()
                loss_mel = F.l1_loss(mel_hifigan_losses, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(audio_hifigans, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(audio_hifigans, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                loss_gen_all.backward()
                optim_g.step()

                # Training Model of FS2 using MAE and Discriminator

                # Forward
                intput_fs2 = [
                        batch[2], batch[3], batch[4], batch[5], batch[6], 
                        batch[7], batch[8], batch[9], batch[10], batch[11],
                        batch[12], 1.0, 1.0, 1.0, y_g_hat.squeeze().detach() 
                        ]
                output = model(*(intput_fs2))  # melspectrogram
                mel_fs2 = output[0].transpose(1,2)
                post_mel_fs2 = output[1].transpose(1,2)
                # y_g_fs2_hat = generator(mel_fs2)
                with torch.no_grad():
                    y_g_pfs2_hat = generator(post_mel_fs2)
                tmp = []
                for i in range(audio_start_stops.shape[0]):
                    tmp1 = y_g_pfs2_hat[i][0][audio_start_stops[i][0]: audio_start_stops[i][0]+audio_start_stops[i][1]]
                    tmp.append(tmp1)
                y_g_pfs2_hat = torch.stack(tmp)
                y_g_pfs2_hat = y_g_pfs2_hat.unsqueeze(1)

                # MPD
                y_df_hat_r_fs2, y_df_hat_g_fs2, _, _ = mpd(audio_hifigans, y_g_pfs2_hat)
                loss_disc_f_fs2, losses_disc_f_r_fs2, losses_disc_f_g_fs2 = discriminator_loss(y_df_hat_r_fs2, y_df_hat_g_fs2)

                # MSD
                y_ds_hat_r_fs2, y_ds_hat_g_fs2, _, _ = msd(audio_hifigans, y_g_pfs2_hat)
                loss_disc_s_fs2, losses_disc_s_r_fs2, losses_disc_s_g_fs2 = discriminator_loss(y_ds_hat_r_fs2, y_ds_hat_g_fs2)

                # loss_disc_all = loss_disc_s + loss_disc_f

                # Cal Loss
                losses = Loss(batch[:13], output)
                total_loss = losses[0] + loss_disc_s_fs2 + loss_disc_f_fs2

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    losses.append(loss_disc_all.item())
                    losses.append(loss_gen_all.item())
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f} D_L: {:.4f} G_L: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, hifigan_train=True)
                    log(train_logger, step, model=model)
                if step % synth_step == 0   :
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample_multilingual(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate_multilingual_hifigan(model, generator, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "generator": generator.module.state_dict(),
                            "mpd": mpd.module.state_dict(),
                            "msd": msd.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
