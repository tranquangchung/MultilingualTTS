import tgt
import numpy as np
import librosa
import pdb

sampling_rate = 22050
hop_length = 256

def get_alignment(tier):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, durations, start_time, end_time

# Get alignments
tg_path="/home/chungtran/Code/C_tutorial/L2/.TTS/montreal-forced-aligner_vn/accoutis_model_VN_Multispeaker/0000_ducnguyentrung/0000-ducnguyentrung-089f41-950-200CauRieng-.TextGrid"
textgrid = tgt.io.read_textgrid(tg_path)
phone, duration, start, end = get_alignment(
    textgrid.get_tier_by_name("phones")
)
text = "{" + " ".join(phone) + "}"

# Read and trim wav files
wav_path="/data/raw/speech/S_synthesize/Multi_Lang/VN/train_clean_VN_processed_22050/0000_ducnguyentrung/0000_ducnguyentrung_089f41_950_200CauRieng_.wav"
wav, _ = librosa.load(wav_path)
wav = wav[
int(sampling_rate * start) : int(sampling_rate * end)
].astype(np.float32)
pdb.set_trace()
print()
