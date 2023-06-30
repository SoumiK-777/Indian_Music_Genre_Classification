from flask import Flask, render_template, request, redirect, url_for,session
import tensorflow as tf
import numpy as np
import os
import math
import librosa
from pydub import AudioSegment

app = Flask(__name__)
app.config['SECRET_KEY'] = "ABCD@1234"

DIR_NAME="./static/uploads"
N_MFCC=13
N_FFT=2048
HOP_LENGTH=512
SAMPLE_RATE = 22050
TRACK_DURATION = 60 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS=10
LABELS=['Bhajan', 'Bollywood Romantic', 'Bhojpuri', 'Bollywood Rap']

model = tf.keras.models.load_model('cnn_model.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    session['savepath'] = None
    if request.method=="POST":
        session['savepath'] = os.path.join(DIR_NAME, "temp.mp3")
        request.files['music_file'].save(session['savepath'])
        return redirect(url_for('predict'))
    return render_template('index.html')

@app.route('/predict')
def predict():
    try:
        input=[]
        samples_per_segment = int(SAMPLES_PER_TRACK / NUM_SEGMENTS)
        num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / HOP_LENGTH)
        savepath = session['savepath']
        output = os.path.join(DIR_NAME, "temp.wav")
        sound = AudioSegment.from_mp3(savepath)
        sound.export(output, format="wav")
        signal, sample_rate = librosa.load(output, sr=SAMPLE_RATE, duration=TRACK_DURATION, offset=30)
        for d in range(NUM_SEGMENTS):
            start = samples_per_segment * d
            finish = start + samples_per_segment
            mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
            mfcc = mfcc.T
            if len(mfcc) == num_mfcc_vectors_per_segment:
                input.append(mfcc.tolist())

        prediction = model.predict(tf.expand_dims(input, axis=3))
        prediction_index = prediction.argmax(axis=1)[0]
        predicted_label = LABELS[prediction_index]
    except:
        return redirect(url_for('index'))

    return render_template('predict.html', predicted_label=predicted_label)

if __name__ == "__main__":
    app.run(debug=True)