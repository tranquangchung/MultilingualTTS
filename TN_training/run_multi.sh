# python3 train_multi.py \
#   -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual/train.yaml

# python3 train_multi_StyleLoss.py \
#   -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual/train.yaml

# python3 train_multi_wav2vec2.py \
#   --restore_step 350000 \
#   -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual/train.yaml
# CUDA_VISIBLE_DEVICES=1
# python3 -W ignore train_multi_hifigan.py \
#   -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual/train.yaml

# CUDA_VISIBLE_DEVICES=4 python3 -W ignore train_multi.py \
#   -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual/train.yaml

CUDA_VISIBLE_DEVICES=4 python3 -W ignore train_multi_diffusion.py \
  -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
  -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
  -t config/LibriTTS_StyleSpeech_multilingual/train.yaml
