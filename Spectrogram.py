
import librosa
import numpy as np
import librosa.display
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt

y, sr = librosa.load('/Users/danielochoa/Library/Mobile Documents/com~apple~CloudDocs/TSU/Classes/Spring 2020/Senior Project 2/Gunshot-sound-classification-using-deep-learning-master-1/shotgun.mp3')

counts = np.float32(range(0,len(y)))

f = [np.float32(i/sr) for i in counts]

plt.plot(f,y)
plt.xlabel('Time')
plt.ylabel('Amp')

melspectrogram(y=y, sr=sr)

D = np.abs(librosa.stft(y))**2

S = librosa.feature.melspectrogram(S=D)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax = 8000)

plt.figure(figsize=(12, 4))

data = librosa.power_to_db(S, ref = np.max)

librosa.display.specshow(data,y_axis = 'mel', x_axis = 'time', fmax = 8000)
                           
                          
plt.colorbar(format='%+2.0f dB')
plt.title('Mel scaled spectrogram')
plt.tight_layout()
plt.show()
