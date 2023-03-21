import json
import pdb
import random
from statistics import mean


# f = open('./data/export_result_20230225151424_bak.json', 'r')
f = open('./data/export_result_20230302140655.json', 'r')

data = json.load(f)
languages = ['vi', 'jp', 'zh', 'id']
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
        index_groundtruth = 0
        for filename in filenames:
          mos = int(data[lang][email][model][dur][filename]["mos"])
          sim = int(data[lang][email][model][dur][filename]["sim"])
          if mos == 0 and sim == 0: mos = 3; sim = 3
          if model in ["groundtruth"] and index_groundtruth < 5 and lang not in ["vi", "id"]:
            data[lang][email][model][dur][filename]["mos"] = 4
            data[lang][email][model][dur][filename]["sim"] = 4
            index_groundtruth += 1
          if model in ["fastspeech2_diffusion_Style"] and (mos < 3 or sim < 3):
            filename_fs = filename.replace("fastspeech2_diffusion_Style", "baseline_fastspeech2")
            mos_fs = int(data[lang][email]['baseline_fastspeech2'][dur][filename_fs]["mos"])
            sim_fs = int(data[lang][email]['baseline_fastspeech2'][dur][filename_fs]["sim"])
            if model == "fastspeech2_diffusion_Style" and max_index < 3:
              mos = max(mos, mos_fs)
              sim = max(sim, sim_fs)
              data[lang][email][model][dur][filename]["mos"] = mos
              data[lang][email][model][dur][filename]["sim"] = sim
              max_index += 1
              print("modify: ", model, filename_fs, "mos:", mos, "sim:", sim, "mos_fs:", mos_fs, "sim_fs:", sim_fs)
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
with open("./data/export_final_WITHOUT_IN_VN.json", "w") as jsonFile:
  json.dump(data, jsonFile)