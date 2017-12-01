2
# coding: utf-8

# In[43]:

import csv
import os
import librosa
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import time


# In[41]:

count=1

for t in os.listdir('audio/'):
    for folder in os.listdir('audio/'+t):
        #folder = '.'
        folderPath = 'audio/'+t+'/' + folder
        for fileName in os.listdir(folderPath):
            # if os.path.exists('train/spectrograms/' +folder + '/'+ fileName.split('.')[0] +'.jpg'):
            #     print("skipping ", count)
            #     count +=1
            #     continue

            if not fileName.endswith('.wav'):
                continue

            if not os.path.exists(folderPath):
                    os.makedirs(folderPath)

            y, sr = librosa.load(folderPath+'/'+fileName)

            jumpSize = len(y)/10;

            ind=0

            for i in range(0, len(y), jumpSize):

                # Let's make and display a mel-scaled power (energy-squared) spectrogram
                S = librosa.feature.melspectrogram(y[i:i+jumpSize], sr=sr, n_mels=128)

                # Convert to log scale (dB). We'll use the peak power as reference.
                log_S = librosa.logamplitude(S, ref_power=np.max)

                # Make a new figure
                fig = plt.figure(figsize=(6.4, 4.8))
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)

                # Display the spectrogram on a mel scale
                # sample rate and hop length parameters are used to render the time axis
                librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

                # Make the figure layout compact

                #plt.show()
                if not os.path.exists('spectrograms/' + t + '/' + folder):
                        os.makedirs('spectrograms/'+ t + '/' + folder)
                plt.savefig('spectrograms/' + t + '/' +folder + '/'+ fileName.split('.')[0]+'-'+str(ind)+'.jpg')
                plt.close()

                ind += 1


            print count

            count += 1
            #print count
