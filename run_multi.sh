#CUDA_VISIBLE_DEVICES=0,1 python3 train_multi.py \
# -p config/config_sakti/LibriTTS_StyleSpeech_multilingual/preprocess.yaml \
# -m config/config_sakti/LibriTTS_StyleSpeech_multilingual/model.yaml \
# -t config/config_sakti/LibriTTS_StyleSpeech_multilingual/train.yaml
 CUDA_VISIBLE_DEVICES=0 python3 train_multi.py \
   -p config/config_sakti/LibriTTS_StyleSpeech_multilingual_test/preprocess.yaml \
   -m config/config_sakti/LibriTTS_StyleSpeech_multilingual_test/model.yaml \
   -t config/config_sakti/LibriTTS_StyleSpeech_multilingual_test/train.yaml


#CUDA_VISIBLE_DEVICES=3 python3 -W ignore TN_training/train_multi_diffusion.py --model naive\

#source /home/s2220411/.bashrc
#conda activate research_tts
#path_full="/home/s2220411/Code/FastSpeech2_multilingual"
#name="LibriTTS_StyleSpeech_multilingual"
#path_script="/home/s2220411/Code/FastSpeech2_multilingual/train_multi.py"
#CUDA_VISIBLE_DEVICES=0,1 python3 $path_script \
# -p $path_full/config/config_kaga/$name/preprocess.yaml \
# -m $path_full/config/config_kaga/$name/model.yaml \
# -t $path_full/config/config_kaga/$name/train.yaml