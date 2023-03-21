python3 synthesize.py \
  --text "sáu nguyên lãnh đạo bình thuận bị kỷ luật do sai phạm trong quản lý" \
  --restore_step 40000 \
  --mode single  --speaker_id 5 \
  -p config/VNTTS_MultiSpeaker/preprocess.yaml \
  -m config/VNTTS_MultiSpeaker/model.yaml \
  -t config/VNTTS_MultiSpeaker/train.yaml \
  --ref_audio /data/raw/speech/S_synthesize/Multi_Lang/VN/train_adaptation/Adaptation_paper/lien_1m/0015_lien_lien-104-0010.wav 
