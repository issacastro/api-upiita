import librosa
import os
def resamplig(file):
    SAMPLE_RATE=16000
    y, sr = librosa.load(file)
    os.remove(file)
    data = librosa.resample(y, sr, SAMPLE_RATE)
    data = librosa.to_mono(data)
    librosa.output.write_wav(file, data, SAMPLE_RATE)
