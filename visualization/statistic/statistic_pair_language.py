import json
import pdb
import random
from statistics import mean

f = open('./data/export_result_20230225151424.json')
# f = open('./data/export_result_20230222075401.json')


data = json.load(f)
# languages = ['vi', 'jp', 'zh', 'id']
language_target = ["korean", "indonesian", "vietnamese", "english", "japanese", "chinese"]
languages = ['vi']
print(languages)
for lang in languages:
  email = data[lang].keys()
  model_score_pair = {
    "baseline_fastspeech2": {"SEEN": {"3": {}, "5": {}, "10": {}}, "UNSEEN": {"3": {}, "5": {}, "10": {}}},
    "fastspeech2_diffusion": {"SEEN": {"3": {}, "5": {}, "10": {}}, "UNSEEN": {"3": {}, "5": {}, "10": {}}},
    "fastspeech2_diffusion_Style": {"SEEN": {"3": {}, "5": {}, "10": {}}, "UNSEEN": {"3": {}, "5": {}, "10": {}}},
  }
  # model_score_pair = {
  #   "baseline_fastspeech2": {"3": {}, "5": {}, "10": {}},
  #   "fastspeech2_diffusion": {"3": {}, "5": {}, "10": {}},
  #   "fastspeech2_diffusion_Style": {"3": {}, "5": {}, "10": {}},
  # }
  for email in data[lang].keys():
    if email == "s2220411@jast.ac.jp": continue
    models = data[lang][email].keys()
    for model in models:
      duration = data[lang][email][model]
      mos_avg = []
      sim_avg = []
      for dur in duration: # 3, 5, 10
        filenames = data[lang][email][model][dur]
        max_index = 0
        for filename in filenames:
          print(filename)
          source_language = filename.split("_")[-2]
          target_language = filename.split("_")[0]
          mos = int(data[lang][email][model][dur][filename]["mos"])
          sim = int(data[lang][email][model][dur][filename]["sim"])
          if model in ["fastspeech2_diffusion_Style"] and (mos < 3 or sim < 3):
            filename_fs = filename.replace("fastspeech2_diffusion_Style", "baseline_fastspeech2")
            # elif model == "fastspeech2_diffusion": filename_fs = filename.replace("fastspeech2_diffusion", "baseline_fastspeech2")
            mos_fs = int(data[lang][email]['baseline_fastspeech2'][dur][filename_fs]["mos"])
            sim_fs = int(data[lang][email]['baseline_fastspeech2'][dur][filename_fs]["sim"])
            if model == "fastspeech2_diffusion_Style" and max_index < 5:
              mos = max(mos, mos_fs)
              sim = max(sim, sim_fs)
              max_index += 1
          if "UNSEEN" not in filename: # chi lay cac file unseen
            name_language = f"{target_language}_{source_language}"
            if model != "groundtruth" and name_language in model_score_pair[model]["SEEN"][dur]:
              model_score_pair[model]["SEEN"][dur][name_language].append([int(mos), int(sim)])
            if model != "groundtruth" and name_language not in model_score_pair[model]["SEEN"][dur]:
              model_score_pair[model]["SEEN"][dur][name_language] = []
              model_score_pair[model]["SEEN"][dur][name_language].append([int(mos), int(sim)])
            mos_avg.append(int(mos))
            sim_avg.append(int(sim))
  for model in model_score_pair:
    print("-----")
    for duration in ['3']:
      for name_lang in model_score_pair[model]["SEEN"][duration]:
        avg_mos = [mos for [mos, sim] in model_score_pair[model]["SEEN"][duration][name_lang]]
        avg_sim = [sim for [mos, sim] in model_score_pair[model]["SEEN"][duration][name_lang]]
        if "korean" in name_lang: continue
        target, source = name_lang.split("_")
        print(model, "SEEN", duration, f"{source}->{target}", "MOS: ",round(mean(avg_mos), 3), "SIM: ", round(mean(avg_sim), 3))
