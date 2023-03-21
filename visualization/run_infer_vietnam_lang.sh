#python3 -W ignore synthesize_vietnam_language.py \
# --restore_step 310000 \
# --mode single --name "journal" \
# -p ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
# -m ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/model.yaml \
# -t ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/train.yaml

python3 -W ignore synthesize_vietnam_language.py \
--restore_step 710000 \
--mode single --name "ESPEAK" \
-p ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/preprocess.yaml \
-m ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/model.yaml \
-t ../config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/train.yaml \
--model shallow