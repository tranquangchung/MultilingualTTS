# python3 train_multi.py \
#   -p config/LibriTTS_StyleSpeech_multilingual_test/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual_test/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual_test/train.yaml

# python3 train_multi_StyleLoss.py \
#   -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual/train.yaml

# python3 create_data_hifigan.py \
#   -p config/LibriTTS_StyleSpeech_multilingual_test/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual_test/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual_test/train.yaml

python3 create_data_hifigan.py \
  -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
  -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
  -t config/LibriTTS_StyleSpeech_multilingual/train.yaml
