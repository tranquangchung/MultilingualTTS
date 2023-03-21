from speechbrain.pretrained import SpeakerRecognition
import os
from statistics import mean

import pdb
path_gt = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/synthesis_audio_interspeech2023_Unseen/vietnamese_fastspeech2_diffusion_Style_5_p281_p281_021_mic1.flac_english_UNSEEN_18_groundtruth.wav"
path_syn = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/synthesis_audio_interspeech2023_Unseen/vietnamese_fastspeech2_diffusion_Style_5_p281_p281_021_mic1.flac_english_UNSEEN_18.wav"
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb",
                                               run_opts={"device":"cuda"})
# score, prediction = verification.verify_batch(
#   path_gt, path_syn
# )
# print(score)
# print(prediction)
#
path_dir = "/home/ldap-users/s2220411/Code/FastSpeech2_multilingual/visualization/audio/synthesis_audio_interspeech2023_Recognition/"
path_dir_import_file = os.path.join(path_dir, "importfile.txt")
sim_score = {
  "baseline_fastspeech2": [],
  "fastspeech2_diffusion": [],
  "fastspeech2_diffusion_Style": [],
}
predict_score = {
  "baseline_fastspeech2": [],
  "fastspeech2_diffusion": [],
  "fastspeech2_diffusion_Style": [],
}

with open(path_dir_import_file) as fin:
  lines = fin.readlines()
  for index, line in enumerate(lines):
    lang, model_tts, duration, grountruth_text, infer_wav, gt_wav = line.strip().split("|")
    if duration in ["3", "5"]:
      gt_path = os.path.join(path_dir, gt_wav)
      infer_path = os.path.join(path_dir, infer_wav)
      score, prediction = verification.verify_files(gt_path, infer_path)
      sim_score[model_tts].append(score[0].item())
      predict_score[model_tts].append(prediction[0].item())
      print(model_tts, score, prediction)

for model_tts in sim_score:
  value_sim = mean(sim_score[model_tts])
  value_acc = mean(predict_score[model_tts])
  print(model_tts, ": ", "Acc: ", round(value_acc, 3), "Sim: ", round(value_sim, 3))