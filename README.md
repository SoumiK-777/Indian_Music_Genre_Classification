
# Indian Music Genre Classification

Through this project, I primarily hope to address the issue of incorrect genre classification for Indian music on well-known streaming services. There has been a lot of work done on classifying western music, but not much on classifying Indian music.

## Dataset Details

This dataset consists of 4 major Genres of Indian music, and contains around 100 songs (6 Hr in duration approx.) in each genre.

Genres are classified as shown below:

- **Bhajan**: Bhajan refers to any devotional song with religious theme or spiritual ideas, specifically among Indian religions, in any of the languages from the Indian subcontinent.
- **Bhojpuri**: Bhojpuri music includes a broad array of Bhojpuri language performances in distinct style, both traditional and modern. This from of music is mostly created in Indian states of Bihar, Uttar Pradesh and Jharkhand and other countries like Nepal, Suriname, Guyana, Netherlands, Mauritius and other Caribbean Islands.
- **Bollywood (Romantic)**: Music from Bollywood movies and albums that often express the feelings of love, longing, and romance.
- **Bollywood (Rap)**: Also known as Hip-Hop music, are new kind of Bollywood music which are inspired from the west. Rapping is a musical form of vocal delivery that incorporates rhyme, rhythmic speech, and street vernacular.

## Feature Extraction

Features which are useful for audio classification

- **Waveform**: A Waveform is a graphical representation of the shape and form of a signal moving in a gaseous, liquid, or solid medium. For sound, the term describes a depiction of the pattern of sound pressure variation (or amplitude) in the time domain.
![App Screenshot](https://i.stack.imgur.com/umKrW.png)

- **Spectrogram**: A Spectrogram is a visual way of representing the signal strength, or loudness, of a signal over time at various frequencies present in a particular waveform in time domain. 
![App Screenshot](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Spectrogram-19thC.png/640px-Spectrogram-19thC.png)

- **Mel-Spectrogram**: A Spectrogram with the Mel Scale as its y axis. The Mel Scale is the result of some non-linear transformation of the frequency scale. They are used to provide our models with sound information similar to what a human would perceive.
![App Screenshot](https://miro.medium.com/v2/resize:fit:1182/1*OOTqBsjpuXyfYJVdPxWtBA.png)

- **MFCC**: Mel-Frequency Cepstral Coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral (result of computing the inverse Fourier transform (IFT) of the logarithm of the estimated signal spectrum) representation of the audio clip (a nonlinear **spectrum-of-a-spectrum**). They are used in music classification because they capture the spectral characteristics of a sound that are most relevant for human perception of music.
![App Screenshot](https://i.stack.imgur.com/q8YfI.png)

## Models Trained

- **Artificial Neural Network (ANN)**: Trained a deep nueral network with extracted MFCCs and achieved an accuracy of **71.5%**. The model consists of Flatten, Dense and Dropout layers with L2 regularization.
- **Convolutional Neural Networks (CNN)**: Trained a CNN model with extracted MFCCs and achieved an accuracy of **81%**. The model architecture consists of Flatten, Conv2D, MaxPool2D, Dense and Dropout layers with Batch Normalization.
Finally, the best performing CNN model was selected which achieved better accuracy than other models. The best performing model was saved and integrated 
with Flask server.

## Getting Started
