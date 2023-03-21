#python3 -W ignore step1_synthesize_multi_language.py \
# --restore_step 310000 \
# --mode single --name "ESPEAK" \
# -p ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
# -m ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/model.yaml \
# -t ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/train.yaml

#python3 -W ignore step1_synthesize_multi_language.py \
#  --restore_step 770000 \
#  --mode single --name "ESPEAK" \
#  -p ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/preprocess.yaml \
#  -m ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/model.yaml \
#  -t ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/train.yaml \
#  --model shallow

#python3 -W ignore step1_synthesize_multi_language.py \
#--restore_step 610000 \
#--mode single --name "ESPEAK" \
#-p ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/preprocess.yaml \
#-m ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/model.yaml \
#-t ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/train.yaml \
#--model shallow
#python3 -W ignore step1_synthesize_multi_language_vn.py
python3 -W ignore mcd_generate_data.py
