source /home/s2220411/.bashrc
conda activate research_tts
python_file="/home/s2220411/Code/FastSpeech2_multilingual/preprocess_multi.py"
#config_file="/home/s2220411/Code/FastSpeech2_multilingual/config/config_kaga/LibriTTS_StyleSpeech_multilingual/preprocess.yaml"
#config_file="/home/s2220411/Code/FastSpeech2_multilingual/config/config_kaga/LibriTTS_StyleSpeech_multilingual_diffusion/preprocess.yaml"
config_file="/home/s2220411/Code/FastSpeech2_multilingual/config/config_kaga/LibriTTS_StyleSpeech_singlespeaker1_diffusion/preprocess.yaml"
python $python_file $config_file