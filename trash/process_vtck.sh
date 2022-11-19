path_audio="/data/raw/speech/VCTK/wav48_silence_trimmed/*/*mic2.flac"
path_output="/data/raw/speech/VCTK/wav22050_silence_trimmed_chungtq_2"
path_txt="/data/raw/speech/VCTK/txt"
for entry in $path_audio
do
  echo "$entry"
  filename=$(basename $entry)
  audio_name=`echo $filename | cut -d "." -f 1`
  username=`echo $filename | cut -d "_" -f 1`
  filename2save=`echo "${audio_name/_mic2/""}"`
  mkdir -p $path_output/$username
  sox $entry -r 22050 $path_output/$username/$filename2save.wav
  # sox $entry /tmp/temp1.wav silence 1 0.1 1% reverse
  # sox /tmp/temp1.wav /tmp/temp2.wav silence 1 0.1 1% reverse
  # sox /tmp/temp2.wav -r 22050 $path_output/$username/$filename2save.wav
  cp $path_txt/$username/$filename2save.txt $path_output/$username/ 
done
