#python3 synthesize.py --text "They later tested positive for Covid-19. From their hospital beds, they found out 12 of their pets had been killed by authorities over fears the animals could spread the
#virus." --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml

# python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
#python3 train.py -p config/VNTTS/preprocess.yaml -m config/VNTTS/model.yaml -t config/VNTTS/train.yaml
# python3 train.py -p config/VNTTS_LIEN/preprocess.yaml -m config/VNTTS_LIEN/model.yaml -t config/VNTTS_LIEN/train.yaml
# python3 train.py -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
# python3 train.py -p config/LibriTTS_StyleSpeech/preprocess.yaml -m config/LibriTTS_StyleSpeech/model.yaml -t config/LibriTTS_StyleSpeech/train.yaml --restore_step 100000
# python3 train_meta.py -p config/LibriTTS_StyleSpeech/preprocess.yaml -m config/LibriTTS_StyleSpeech/model.yaml -t config/LibriTTS_StyleSpeech/train.yaml
# python3 train.py -p config/LibriTTS_StyleSpeech_test/preprocess.yaml -m config/LibriTTS_StyleSpeech_test/model.yaml -t config/LibriTTS_StyleSpeech_test/train.yaml
# python3 train.py -p config/VNTTS_VANANH/preprocess.yaml -m config/VNTTS_VANANH/model.yaml -t config/VNTTS_VANANH/train.yaml
# python3 train.py -p config/VNTTS_DOANDUYLINH/preprocess.yaml -m config/VNTTS_DOANDUYLINH/model.yaml -t config/VNTTS_DOANDUYLINH/train.yaml
# python3 train.py -p config/VNTTS_Nga/preprocess.yaml -m config/VNTTS_Nga/model.yaml -t config/VNTTS_Nga/train.yaml
# python3 train.py -p config/VNTTS_Hoangnhan/preprocess.yaml -m config/VNTTS_Hoangnhan/model.yaml -t config/VNTTS_Hoangnhan/train.yaml
# python3 train.py -p config/LibriTTS_MultiSpeaker_VN/preprocess.yaml -m config/LibriTTS_MultiSpeaker_VN/model.yaml -t config/LibriTTS_MultiSpeaker_VN/train.yaml
# python3 train.py -p config/VNTTS/preprocess.yaml -m config/VNTTS/model.yaml -t config/VNTTS/train.yaml
# python3 train.py -p config/VNTTS_MultiSpeaker/preprocess.yaml -m config/VNTTS_MultiSpeaker/model.yaml -t config/VNTTS_MultiSpeaker/train.yaml
# python3 test.py # test.py is soft-link of train.py
 python3 train.py
