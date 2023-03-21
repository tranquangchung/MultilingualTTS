# The first version
#python3 -W ignore synthesize_multi_language.py \
# --restore_step 310000 \
# --mode single --name "ESPEAK_test" \
# -p config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
# -m config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/model.yaml \
# -t config/config_pcgpu/LibriTTS_StyleSpeech_multilingual/train.yaml \
# --ref_audio audio_ref/0012_hjeu_hjeu-183-0001.wav

# the second version
#python3 -W ignore synthesize_multi_language_diffusion.py \
# --restore_step 770000 \
# --mode single --name "Sakti_Indo" \
# -p config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/preprocess.yaml \
# -m config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/model.yaml \
# -t config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_testV1/train.yaml \
# --model shallow \
# --ref_audio audio_ref/sakti.wav

#python3 -W ignore synthesize_multi_language_diffusion.py \
# --restore_step 610000 \
# --mode single --name "Zhang_DS_610k_ipad" \
# -p config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/preprocess.yaml \
# -m config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/model.yaml \
# -t config/config_pcgpu/LibriTTS_StyleSpeech_multilingual_diffusion_style/train.yaml \
# --model shallow \
# --ref_audio audio_ref/Zhang_ipad.wav