import numpy as np
import librosa

def mel(y, args):
    max_time_len = int((args.sr*args.msc)/args.hs)
    fea = librosa.feature.melspectrogram(y, sr=args.sr, n_fft=args.ws, 
                hop_length=args.hs, n_mels=args.mel)
    return np.log(1 + args.et * fea[:,:max_time_len]).reshape(1, args.mel, max_time_len)

