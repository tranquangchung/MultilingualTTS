import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num, get_model_fastSpeech2_StyleEncoder, get_model_fastSpeech2_StyleEncoder_Discriminator
from utils.tools import to_device, log, synth_one_sample, log_gan
from model import FastSpeech2Loss
from dataset import Dataset
from model.Discriminators import Discriminator


from evaluate import evaluate
import pdb
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    # model, optimizer = get_model(args, configs, device, train=True)
    model, G_optim = get_model_fastSpeech2_StyleEncoder_Discriminator(args, configs, device, train=True)
    discriminator = Discriminator(configs).cuda()
    # model = nn.DataParallel(model)
    num_param = get_param_num(model)

    model_without_ddp = model
    discriminator_without_ddp = discriminator  

    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    adversarial_loss = discriminator_without_ddp.get_criterion()
    
    print("Number of FastSpeech2 Parameters:", num_param)
    num_param_D = get_param_num(discriminator)
    print("Number of Discriminator Parameters:", num_param_D)
    # Optimizer
    D_optim = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=[0.9, 0.98], eps=1e-9)

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
                B = len(batch[0])
                if B != batch_size:
                    continue
                batch = to_device(batch, device)
            
                #### Generator ####
                G_optim.zero_grad()
                # Forward
                output = model(*(batch[2:]))
                src_output = output[-2]
                D = batch[-1]
                src_target, _, _ = model_without_ddp.variance_adaptor.length_regulator(src_output, D)
                
                # # Cal Loss
                # losses = Loss(batch, output[:-1]) # ko lay style_vector
                # total_loss = losses[0]
                total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss = Loss(batch, output[:-2]) 

                #### META LEARNING ####
                # B = torch.tensor(batch[2].shape).item() # mel_target.shape[0]
                perm_idx = torch.randperm(B)
                style_vector = output[-1]
                text = batch[3]
                text_len = batch[4]
                q_text, q_text_len = text[perm_idx], text_len[perm_idx]
                # Generate query speech
                q_mel_output, q_mel_postnet, q_src_embedded, _, _, q_log_duration_output, \
                    q_d_rounded, q_src_masks, q_mel_masks, q_src_lens, q_mel_lens = model_without_ddp.inference(style_vector, q_text, q_text_len)
                q_duration = q_d_rounded.masked_fill(q_src_masks, 0).long()
                q_src, _, _ = model_without_ddp.variance_adaptor.length_regulator(q_src_embedded, q_duration)
                print("q_src", q_src.shape)

                # Adverserial loss 
                sid = batch[2] #speaker
                t_val, s_val, _= discriminator(q_mel_output, q_src, None, sid, q_mel_masks)
                G_GAN_query_t = adversarial_loss(t_val, is_real=True)
                G_GAN_query_s = adversarial_loss(s_val, is_real=True)
                alpha = 10.0
                G_Loss = alpha*postnet_mel_loss + alpha*mel_loss + pitch_loss + energy_loss + duration_loss +\
                        G_GAN_query_t + G_GAN_query_s
                G_Loss.backward()
                
                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                G_optim.step_and_update_lr()

                ##### Discriminator #####
                D_optim.zero_grad()
                # Real
                mel_target = batch[6]
                mel_mask = output[7]
                print("src_target: ", src_target.shape)
                real_t_pred, real_s_pred, cls_loss = discriminator(
                                    mel_target, src_target.detach(), style_vector.detach(), sid, mask=mel_mask)
                # Fake
                fake_t_pred, fake_s_pred, _ = discriminator(
                                    q_mel_output.detach(), q_src.detach(), None, sid, mask=q_mel_masks)
                D_t_loss = adversarial_loss(real_t_pred, is_real=True) + adversarial_loss(fake_t_pred, is_real=False)
                D_s_loss = adversarial_loss(real_s_pred, is_real=True) + adversarial_loss(fake_s_pred, is_real=False)
                D_loss = D_t_loss + D_s_loss + cls_loss
                D_loss.backward()
                D_optim.step()
                
                # Print Log
                if step % log_step == 0:
                    # losses = [l.item() for l in losses]
                    losses = [total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss, G_GAN_query_t, G_GAN_query_s, D_t_loss, D_s_loss, cls_loss]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    # message2 = "Generator Loss: Total Loss: {:.4f}, Mel: {:.4f}, Mel PostNet: {:.4f}, F0: {:.4f}, E: {:.4f}, D: {:.4f}, G_t: {:.4f}, G_s: {:.4f} -- Discriminator Loss: D_t: {:.4f}, D_s: {:.4f}, cls_loss: {:.4f}".format(
                    #         total_loss, mel_loss, postnet_mel_loss, pitch_loss, energy_loss, duration_loss, G_GAN_query_t, G_GAN_query_s, D_t_loss, D_s_loss, cls_loss)

                    message2 = "Generator Loss: Total Loss: {:.4f}, Mel: {:.4f}, Mel PostNet: {:.4f}, F0: {:.4f}, E: {:.4f}, D: {:.4f}, G_t: {:.4f}, G_s: {:.4f} -- Discriminator Loss: D_t: {:.4f}, D_s: {:.4f}, cls_loss: {:.4f}".format(*losses)

                    with open(os.path.join(train_log_path, "log_gan.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log_gan(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log_gan(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log_gan(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log_gan(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log_gan.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "discriminator": discriminator.state_dict(),
                            'G_optim': G_optim._optimizer.state_dict(),
                            'D_optim': D_optim.state_dict(),
                            'step': step,
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
