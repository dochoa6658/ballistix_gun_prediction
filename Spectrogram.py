
import librosa
import numpy as np
import librosa.display
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt

string01 = '/Users/danielochoa/Music/iTunes/iTunes Media/Music/www.FesliyanStudios.com/www.FesliyanStudios.com/www.FesliyanStudios.com.mp3'
y, sr = librosa.load(string01)

counts = np.float32(range(0,len(y)))
print('sample rate', sr)
print('time series', y)

print('counts[4]', counts[len(counts)-1])
print('counts[4]/sr', counts[len(counts)-1]/sr)

f = [np.float32(i/sr) for i in counts]
#print('/n')
#print(f)

plt.plot(f,y)
plt.xlabel('Time')
plt.ylabel('Amp')

melspectrogram(y=y, sr=sr)

# Discrete Fourier Transforms (DFT)
D = np.abs(librosa.stft(y))**2

S = librosa.feature.melspectrogram(S=D)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax = 8000)
print(S)

plt.figure(figsize=(12, 4))

data = librosa.power_to_db(S, ref = np.max)

librosa.display.specshow(data,y_axis = 'mel', x_axis = 'time', fmax = 8000)
                           
                          
plt.colorbar(format='%+2.0f dB')
plt.title('Mel scaled spectrogram')
plt.tight_layout()
plt.show()
