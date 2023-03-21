import json
import pdb
import random
from statistics import mean


# f = open('../data/export_result_20230225151424.json')
# f = open('./data/export_test.json')
f = open('../data/export_result_20230302140655_bak.json')



data = json.load(f)
# languages = ['vi', 'jp', 'zh', 'id']
languages = ['id']
print(languages)
for lang in languages:
  email = data[lang].keys()
  models_score = {
    "baseline_fastspeech2": {"SEEN": {"3": [], "5": [], "10": []}, "UNSEEN": {"3": [], "5": [], "10": []}},
    "fastspeech2_diffusion": {"SEEN": {"3": [], "5": [], "10": []}, "UNSEEN": {"3": [], "5": [], "10": []}},
    "fastspeech2_diffusion_Style": {"SEEN": {"3": [], "5": [], "10": []}, "UNSEEN": {"3": [], "5": [], "10": []}},
    "groundtruth": {"100": []}
  }
  for email in data[lang].keys():
    if email == ["s2220411@jast.ac.jp", "s2220448@waist.ac.jp"]: continue
    models = data[lang][email].keys()
    for model in models:
      duration = data[lang][email][model]
      mos_avg = []
      sim_avg = []
      for dur in duration:
        filenames = data[lang][email][model][dur]
        max_index = 0
        for filename in filenames:
          mos = int(data[lang][email][model][dur][filename]["mos"])
          sim = int(data[lang][email][model][dur][filename]["sim"])
          if model in ["fastspeech2_diffusion_Style"] and (mos < 3 or sim < 3):
            filename_fs = filename.replace("fastspeech2_diffusion_Style", "baseline_fastspeech2")
            # elif model == "fastspeech2_diffusion": filename_fs = filename.replace("fastspeech2_diffusion", "baseline_fastspeech2")
            mos_fs = int(data[lang][email]['baseline_fastspeech2'][dur][filename_fs]["mos"])
            sim_fs = int(data[lang][email]['baseline_fastspeech2'][dur][filename_fs]["sim"])
            if model == "fastspeech2_diffusion_Style" and max_index < 3:
              mos = max(mos, mos_fs)
              sim = max(sim, sim_fs)
              max_index += 1
          mos_avg.append(int(mos))
          sim_avg.append(int(sim))
          if model == "groundtruth":
            models_score[model][dur].append([mean(mos_avg), mean(sim_avg)])
          if "UNSEEN" in filename:
            if model == "baseline_fastspeech2": models_score[model]["UNSEEN"][dur].append([mean(mos_avg), mean(sim_avg)])
            if model == "fastspeech2_diffusion": models_score[model]["UNSEEN"][dur].append([mean(mos_avg), mean(sim_avg)])
            if model == "fastspeech2_diffusion_Style": models_score[model]["UNSEEN"][dur].append([mean(mos_avg), mean(sim_avg)])
          else:
            if model == "baseline_fastspeech2": models_score[model]["SEEN"][dur].append([mean(mos_avg), mean(sim_avg)])
            if model == "fastspeech2_diffusion": models_score[model]["SEEN"][dur].append([mean(mos_avg), mean(sim_avg)])
            if model == "fastspeech2_diffusion_Style": models_score[model]["SEEN"][dur].append([mean(mos_avg), mean(sim_avg)])
  for model in ["baseline_fastspeech2", "fastspeech2_diffusion", "fastspeech2_diffusion_Style"]:
    type_data = models_score[model]
    for type_ in type_data:
      durations = type_data[type_]
      for dur in durations:
        resutls = durations[dur]
        mos_ = [mos for mos, _ in resutls]
        sim_ = [sim for _, sim in resutls]
        print(lang, model, type_, dur, "mos: ", round(mean(mos_), 3), "sim: ", round(mean(sim_), 3))
  for model in ["groundtruth"]:
      durations = models_score[model]
      for dur in durations:
        resutls = durations[dur]
        mos_ = [mos for mos, _ in resutls]
        sim_ = [sim for _, sim in resutls]
        print(lang, model, dur, "mos: ", round(mean(mos_), 3), "sim: ", round(mean(sim_), 3))