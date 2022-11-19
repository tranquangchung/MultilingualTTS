#python3 synthesize.py --text "đó là đài tiếng nói việt nam , phát thanh từ thủ đô nước cộng hoà xã hội chủ nghĩa việt nam ." \
#  --restore_step 90000 --mode single -p config/VNTTS/preprocess.yaml  \"n d"n tỉnh , đã thông tin nhanh về tình hình phát triển kinh tế xã hội , kết quả thực hiện nghị quyết đại hội đảng bộ tỉnh lần
#  -m config/VNTTS/model.yaml \
#python3 synthesize.py --text "đồng chí phó bí thư tỉnh uỷ chủ tịch uỷ ban nhân dân tỉnh , đã thông tin nhanh về tình hình phát triển kinh tế xã hội , kết quả thực hiện nghị quyết đại hội đảng bộ tỉnh lần thứ mười chín , kết quả công tác đối ngoại ," \
#  --restore_step`n tộc`n . 190000 --mode single -p config/VNTTS/preprocess.yaml  \
#  -m config/VNTTS/model.yaml \
#  -t config/VNTTS/train.yaml \

# python3 synthesize.py --text "chủ tịch , bí thư tỉnh ủy , nên trả lời tin nhắn của nhân dân ." \
# python3 synthesize.py --text "tuyên ngôn độc lập văn kiện có ý nghĩa lịch sử sống còn với vận mệnh dân tộc" \
#   --restore_step 320000 --mode single -p config/VNTTS/preprocess.yaml  \
#   -m config/VNTTS/model.yaml \
#   -t config/VNTTS/train.yaml \
# python3 synthesize.py --text "However, existing methods either require to fine-tune the model or achieve low adaptation quality without fine-tuning."  --speaker_id 101 --restore_step 800000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
# python3 synthesize.py --text "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"  \
#   --speaker_id 215 --restore_step 400000 \
#   --mode single \
#   -p config/LibriTTS_StyleSpeech/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech/model.yaml \
#   -t config/LibriTTS_StyleSpeech/train.yaml
# python3 synthesize_stylespeech.py --text "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"  \
#   --speaker_id 0 --restore_step 500000 \
#   --ref_audio "/project/AI-team/exp/chungtran/tts/StyleSpeech/dataset/wav22050/5062/5062_294697_000036_000000.wav" \
#   --mode single \
#   -p config/LibriTTS_StyleSpeech/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech/model.yaml \
#   -t config/LibriTTS_StyleSpeech/train.yaml

# python3 synthesize_multi_language.py --text "みなさん、こんにちは。" --restore_step 10100 --mode single --language "Japan" --name "test1" \
#   -p config/LibriTTS_StyleSpeech_multilingual_test/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual_test/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual_test/train.yaml \
#   --ref_audio ./data_demo/Japan/jvs001/jvs001_BASIC5000_0025.wav

# python3 synthesize_multi_language.py --text "xin chào tất cả mọi người" --restore_step 10100 --mode single --language "Vie" --name "test1" \
#   -p config/LibriTTS_StyleSpeech_multilingual_test/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual_test/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual_test/train.yaml \
#   --ref_audio ./data_demo/Japan/jvs001/jvs001_BASIC5000_0025.wav

# python3 synthesize_multi_language.py --text "大家好" --restore_step 10100 --mode single --language "Chi" --name "test1" \
#   -p config/LibriTTS_StyleSpeech_multilingual_test/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual_test/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual_test/train.yaml \
#   --ref_audio ./data_demo/Japan/jvs001/jvs001_BASIC5000_0025.wav

# python3 synthesize_multi_language.py \
#   --restore_step 500000 \
#   --mode single --name "ESPEAK" \
#   -p config/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
#   -m config/LibriTTS_StyleSpeech_multilingual/model.yaml \
#   -t config/LibriTTS_StyleSpeech_multilingual/train.yaml \
#   --ref_audio /project/AI-team/exp/chungtran/tts/StyleSpeech/dataset/wav22050_multilingual/Vie/0012_hjeu/0012_hjeu_hjeu-100-0001.wav 
