# Copyright (c) 2021, phucpx@ftech.ai

import av
import os
import sys
import ast
import time
import queue
import pydub
import logging
import tempfile
import requests
import threading
import numpy as np
import pandas as pd
import urllib.request
import soundfile as sf
import streamlit as st
import logging.handlers

from typing import List
from datetime import datetime
from collections import deque
from itertools import groupby
from operator import itemgetter
from streamlit.proto.Markdown_pb2 import Markdown
from streamlit_webrtc import (
    AudioProcessorBase,
    ClientSettings,
    WebRtcMode,
    webrtc_streamer,
)

# logger = logging.getLogger(__name__)

### Parameters config:

THRESHOLD = 0.5
MIN_FRAMES = 2
PADDING_NUMBER = 10
WINDOW_SIZE =  1 # second
WINDOW_STEP = 0.25 # second
SAMPLE_RATE = 16000
WIDTH_PREDICTION = 10
WIDTH_LINE_CHART = 10
TEMP_DIR = tempfile.mkdtemp()
SERVE_MODEL_WAV2KWS_URL = 'http://192.168.1.11:8081/api/infer/'
SERVE_MODEL_KWT_URL = 'http://192.168.20.109:5000/kws'



# TODO: GET BEEP SOUND

beep_file = 'data/beep-01a.wav'
beep_ = pydub.AudioSegment.from_wav(beep_file)
beep_ = beep_.set_channels(1).set_frame_rate(SAMPLE_RATE)

s_beep = beep_ + beep_
s_beep = s_beep[:1000]
s_beep.export('./data/beep_1s.wav', format='wav')
wav_beep = s_beep.get_array_of_samples()


def main():
    st.title("Real Time Keywords Spotting")
    st.markdown(
        """
        _Copyright (c) 2021 Voice NLP_

        """
    )
    st.text("\n")
    st.text("\n")
    st.text("\n")
    st.header(
    """

    This demo app: 

    """
    )
    
    st.markdown(
        """
        This demo is using ``Wav2KWS`` or ``KWT`` architectures.
        """
    )
    st.text("\n")

    sound_only_page = "Sound only (sendonly)"
    with_video_page = "With video (sendrecv)"
    app_mode = st.selectbox("Choose the app mode", [sound_only_page, with_video_page])

    wav2kws_model = 'Wav2KWS Architecture'
    kwt_model = 'KWT Architecture'
    architecture_mode = st.selectbox("Choose the architecture mode", [wav2kws_model, kwt_model])

    if app_mode == sound_only_page and architecture_mode == wav2kws_model:
        try:
            app_kws(
                url=SERVE_MODEL_WAV2KWS_URL,
                sample_rate=SAMPLE_RATE,
                window_size=WINDOW_SIZE,
                window_step=WINDOW_STEP
            )
        except:
            st.markdown(f"Model ``Wav2KWS`` not served! :sunglasses:")
    elif app_mode == sound_only_page and architecture_mode == kwt_model:
        try:
            app_kws(
                url=SERVE_MODEL_KWT_URL,
                sample_rate=SAMPLE_RATE,
                window_size=WINDOW_SIZE,
                window_step=WINDOW_STEP
            )
        except:
            st.markdown(f"Model ``KWT`` not served! :sunglasses:")
    elif app_mode == with_video_page and architecture_mode == wav2kws_model:
        try:
            app_kws_with_video(
                url=SERVE_MODEL_WAV2KWS_URL,
                sample_rate=SAMPLE_RATE,
                window_size=WINDOW_SIZE,
                window_step=WINDOW_STEP
            )
        except:
            st.markdown(f"Model ``Wav2KWS`` not served! :sunglasses:")
    else:
        try:
            app_kws_with_video(
                url=SERVE_MODEL_KWT_URL,
                sample_rate=SAMPLE_RATE,
                window_size=WINDOW_SIZE,
                window_step=WINDOW_STEP
            )
        except:
            st.markdown(f"Model ``KWT`` not served! :sunglasses:")
    


def app_kws(url: str, sample_rate: int=16000, window_size: int=1, window_step: float=0.25):

    webrtc_ctx = webrtc_streamer(
        key="keyword-spotting",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": False, "audio": True},
        ),
    )

    status_indicator = st.empty()

    placeholder_chart = st.empty()
    placeholder_length = st.empty()
    
    text_output = st.empty()

    placeholder_raw_txt = st.empty()
    placeholder_raw_audio= st.empty()
    placeholder_process_txt = st.empty()
    placeholder_processed_audio = st.empty()

    all_sound = pydub.AudioSegment.empty()

    count = 0
    active_ids = []
    text_out_list = []

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")

    while True:
        if webrtc_ctx.audio_receiver:
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue
            
            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound = sound.set_channels(1).set_frame_rate(sample_rate)
                
                all_sound += sound
                
            idx_pred = 0
            if len(all_sound) - count * window_step * 1000 >= window_size * 1000:
                now = datetime.now()
                sound_chunk = all_sound[count*1000*window_step: count*1000*window_step + 1000]

                print('='*100)
                print(f"COUNT: {count}")
                print(f"ALL-SOUND: {len(all_sound)} - {all_sound.duration_seconds}")
                print(f"SOUND-CHUNK:  {sound_chunk.duration_seconds} [{count*1000*window_step}: {count*1000*window_step + 1000}]")
                print('='*100)

                url_buffer = {'value': sound_chunk.get_array_of_samples().tolist()}
                out = requests.post(url=url, json=url_buffer)

                pred = ast.literal_eval(out.text)

                score = pred.get('score', 0.0)
                idx_pred = pred.get('idx', 0)
                if float(score) < THRESHOLD and int(idx_pred) == 1:
                    idx_pred = '0'
                
                text_out_list.append(int(idx_pred))
                
                if len(text_out_list) >= WIDTH_PREDICTION:
                    text_output.markdown(
                        f"**Digital pred-log:**  \t  {'   '.join(str(i) for i in text_out_list[-WIDTH_PREDICTION:])}")
                else:
                    text_output.markdown(
                        f"**Digital pred-log:**  \t  {'   '.join(str(i) for i in text_out_list)}")

                print(f"Time process: {datetime.now() - now}")

                placeholder_raw_txt.markdown(f"RAW AUDIO: ")
                lit_audio(placeholder_raw_audio, all_sound[: count*1000*window_step + 1000])
                placeholder_process_txt.markdown(f"PROCESSED AUDIO: ")

                if int(idx_pred) == 1:
                    sound_chunk.export(f'./out{str(count)}.wav', format='wav')
                    active_ids.append(count)
                
                lit_processed_audio(placeholder_processed_audio, all_sound[: count*1000*window_step + 1000], active_ids)

                # TODO: Update variable: count
                count += 1

            # TODO: View line chart
            LENGTH_FRAMES = len(text_out_list)
            if len(text_out_list) >= WIDTH_LINE_CHART:
                index = [i for i in range(LENGTH_FRAMES - WIDTH_LINE_CHART, LENGTH_FRAMES)]
                seconds = [str(window_step * i) for i in index]
                predictions = text_out_list[-WIDTH_LINE_CHART:]
            else:
                seconds = [str(window_step * i) for i in range(LENGTH_FRAMES)]
                predictions = text_out_list

            data = pd.DataFrame({
                'second': seconds,
                'predict': predictions
            })

            data = data.rename(columns={'second': 'index'}).set_index('index')
            placeholder_chart.line_chart(data=data)

            placeholder_length.write(f"The numbers of Frames:  {count}")

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break

def lit_audio(place_holder_raw_audio, sound):
    try:
        temp_path = os.path.join(TEMP_DIR, f'raw_audio.wav')
        sound.export(temp_path, format='wav')

        audio_file = open(temp_path, 'rb')
        audio_bytes = audio_file.read()
        place_holder_raw_audio.audio(audio_bytes, format='audio/wav')
    except:
        st.markdown(
            "Error in lit_audio!"
        )


def lit_processed_audio(placeholder_processed_audio, sound, active_idx):
    if len(active_idx) > 0:
        print(f"\n\t>>> TODO: ---------- LIT PROCESS AUDIO ----------")
        temp_path = os.path.join(TEMP_DIR, f'processed_audio.wav')

        ranges = find_consecutive_values(data=active_idx)
        array_sound = np.array(sound.get_array_of_samples())
        array_sound = np.expand_dims(array_sound, axis=0)

        print(f"-> ACTIVE INDEXES: {active_idx} {ranges} | ARRAY_SOUND: {np.shape(array_sound)}")

        ## TODO: replace bip

        for r, irange in enumerate(ranges):
            
            start = int(irange[0] * SAMPLE_RATE * WINDOW_STEP)
            end = int(irange[1] * SAMPLE_RATE * WINDOW_STEP + 16000)
            
            if array_sound.shape[1] > end:
                start = start - PADDING_NUMBER
                end = end + PADDING_NUMBER

            print(f"- STart: {start} | ENd: {end}")    
            beep_r = np.tile(wav_beep, (1, irange[1] + 2 - irange[0]))
            beep_r = beep_r[0, :end - start]
            print(f"- BEEP_R: {np.shape(beep_r)} | ARRAY_SOUND CHUNK: {np.shape(array_sound[0, start: end])}")
            beep_r = np.expand_dims(np.array(beep_r), axis=0)

            array_sound[0, start: end] = beep_r[0, :]

        processed_sound = sound._spawn(array_sound)
        ## TODO: export to file 
        processed_sound.export(temp_path, format='wav')

        ## TODO: Show in st
        audio_file = open(temp_path, 'rb')
        audio_bytes = audio_file.read()
        placeholder_processed_audio.audio(audio_bytes, format='audio/wav')


def find_consecutive_values(data):
    ranges =[]

    for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        if group[-1] - group[0] >= MIN_FRAMES - 1:
            ranges.append((group[0],group[-1]))

    return ranges


def app_kws_with_video(
    url: str, sample_rate: int=16000, window_size: int=1, window_step: float=0.25
):
    class AudioProcessor(AudioProcessorBase):
        frames_lock: threading.Lock
        frames: deque

        def __init__(self) -> None:
            self.frames_lock = threading.Lock()
            self.frames = deque([])

        async def recv_queued(self, frames: List[av.AudioFrame]) -> av.AudioFrame:
            with self.frames_lock:
                self.frames.extend(frames)

            # Return empty frames to be silent.
            new_frames = []
            for frame in frames:
                input_array = frame.to_ndarray()
                new_frame = av.AudioFrame.from_ndarray(
                    np.zeros(input_array.shape, dtype=input_array.dtype),
                    layout=frame.layout.name,
                )
                new_frame.sample_rate = frame.sample_rate
                new_frames.append(new_frame)

            return new_frames

    webrtc_ctx = webrtc_streamer(
        key="keyword-spotting-w-video",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        client_settings=ClientSettings(
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": True},
        ),
    )

    status_indicator = st.empty()

    placeholder_chart = st.empty()
    placeholder_length = st.empty()
    
    text_output = st.empty()

    placeholder_raw_txt = st.empty()
    placeholder_raw_audio= st.empty()
    placeholder_process_txt = st.empty()
    placeholder_processed_audio = st.empty()

    count = 0
    active_ids = []
    text_out_list = []
    all_sound = pydub.AudioSegment.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")

    while True:
        if webrtc_ctx.audio_processor:
            ## Get frames
            audio_frames = []
            with webrtc_ctx.audio_processor.frames_lock:
                while len(webrtc_ctx.audio_processor.frames) > 0:
                    frame = webrtc_ctx.audio_processor.frames.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue
            
            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound = sound.set_channels(1).set_frame_rate(sample_rate)
                
                all_sound += sound
                
            idx_pred = 0
            if len(all_sound) - count * window_step * 1000 >= window_size * 1000:
                now = datetime.now()
                sound_chunk = all_sound[count*1000*window_step: count*1000*window_step + 1000]

                print('='*100)
                print(f"COUNT: {count}")
                print(f"ALL-SOUND: {len(all_sound)} - {all_sound.duration_seconds}")
                print(f"SOUND-CHUNK:  {sound_chunk.duration_seconds} [{count*1000*window_step}: {count*1000*window_step + 1000}]")
                print('='*100)

                url_buffer = {'value': sound_chunk.get_array_of_samples().tolist()}
                out = requests.post(url=url, json=url_buffer)

                pred = ast.literal_eval(out.text)

                score = pred.get('score', 0.0)
                idx_pred = pred.get('idx', 0)
                if float(score) < THRESHOLD and int(idx_pred) == 1:
                    idx_pred = '0'
                
                text_out_list.append(int(idx_pred))
                
                if len(text_out_list) >= WIDTH_PREDICTION:
                    text_output.markdown(
                        f"**Digital pred-log:**  \t  {'   '.join(str(i) for i in text_out_list[-WIDTH_PREDICTION:])}")
                else:
                    text_output.markdown(
                        f"**Digital pred-log:**  \t  {'   '.join(str(i) for i in text_out_list)}")

                print(f"Time process: {datetime.now() - now}")

                placeholder_raw_txt.markdown(f"RAW AUDIO: ")
                lit_audio(placeholder_raw_audio, all_sound[: count*1000*window_step + 1000])
                placeholder_process_txt.markdown(f"PROCESSED AUDIO: ")

                if int(idx_pred) == 1:
                    sound_chunk.export(f'./out{str(count)}.wav', format='wav')
                    active_ids.append(count)
                
                lit_processed_audio(placeholder_processed_audio, all_sound[: count*1000*window_step + 1000], active_ids)

                # TODO: Update variable: count
                count += 1

            # TODO: View line chart
            LENGTH_FRAMES = len(text_out_list)
            if len(text_out_list) >= WIDTH_LINE_CHART:
                index = [i for i in range(LENGTH_FRAMES - WIDTH_LINE_CHART, LENGTH_FRAMES)]
                seconds = [str(window_step * i) for i in index]
                predictions = text_out_list[-WIDTH_LINE_CHART:]
            else:
                seconds = [str(window_step * i) for i in range(LENGTH_FRAMES)]
                predictions = text_out_list

            data = pd.DataFrame({
                'second': seconds,
                'predict': predictions
            })

            data = data.rename(columns={'second': 'index'}).set_index('index')
            placeholder_chart.line_chart(data=data)

            placeholder_length.write(f"The numbers of Frames:  {count}")

        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
