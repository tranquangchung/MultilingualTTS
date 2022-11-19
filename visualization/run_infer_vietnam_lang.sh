python3 -W ignore synthesize_vietnam_language.py \
 --restore_step 310000 \
 --mode single --name "journal" \
 -p ../config/config_sakti/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
 -m ../config/config_sakti/LibriTTS_StyleSpeech_multilingual/model.yaml \
 -t ../config/config_sakti/LibriTTS_StyleSpeech_multilingual/train.yaml