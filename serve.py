import os
import ast
import pydub
import uvicorn
import tempfile

import numpy as np
import soundfile as sf

from scipy.io.wavfile import write
from typing import Any, Union
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel
from arizona_spotting.models import Wav2KWS
from arizona_spotting.learners import Wav2KWSLearner

TEMP_DIR = tempfile.mkdtemp()

app = FastAPI()

model = Wav2KWS(
        num_classes=2,
        model_type='binary',
        encoder_hidden_dim=768,
        out_channels=112,
        pretrained_model='wav2vec-base-en'
    )

learner = Wav2KWSLearner(model=model)
learner.load_model(model_path='models/wav2kws_model_v21_300601.pt')

# output = learner.inference(input='data/VISC_v0.2/test/active/1556466372400.wav')


class AudioFrame(BaseModel):
    value: Any
    sr: int=16000

@app.post("/api/infer/")
async def pinfer(audio_frame: AudioFrame):
    now = datetime.now()
    path = TEMP_DIR + '/temp.wav'
    value = np.array(audio_frame.value)
    write(path, audio_frame.sr, value.astype(np.int16))

    # value, sr = sf.read(path)
    # try:
    #     value = ast.literal_eval(value)
    # except:
    #     pass

    # value = np.array(value, dtype=np.float64)
    output = learner.inference(input=path)
    idx_label = output.get('idx')
    score = output.get('score')

    inference_time = datetime.now() - now
    print(f"- response: 'idx': {str(idx_label)}, 'score': {str(score)}, 'time': {inference_time}")
    
    return {"idx": str(idx_label), "score": str(score), "time": inference_time}


@app.get("/")
async def root():
    return {"message": "Denver-Spotting !!! v0.0.1"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
