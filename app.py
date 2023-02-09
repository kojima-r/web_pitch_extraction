import streamlit as st
st.title("Pitch Extraction")

from streamlit_webrtc import webrtc_streamer, WebRtcMode

import torch
import torchcrepe

import time
from matplotlib import pyplot as plt
import numpy as np
import librosa

webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )

status_indicator = st.empty()
pitch_indicator = st.empty()
pitch = np.empty(0)

while True:
    if webrtc_ctx.state.playing:
        try:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            time.sleep(0.1)
            status_indicator.write("Queue is empty. Abort.")
            break
        status_indicator.write("Running.")

        audio_chunk = np.empty(0)
        for audio_frame in audio_frames:
            audio_chunk = np.append(audio_chunk, audio_frame.to_ndarray())
            sr = audio_frame.sample_rate
        audio_chunk = audio_chunk.astype(np.float32) / np.iinfo(np.int16).max
        audio_chunk = audio_chunk.reshape(-1,len(audio_frame.layout.channels))
        audio_chunk = np.sum(audio_chunk, axis=1)/2

        tmp = librosa.pyin(audio_chunk,
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                            )[0]
        f0 = np.zeros_like(tmp)
        f0[tmp > 0] = tmp[tmp > 0]
        pitch = np.append(pitch,f0)[-1000:]

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_ylim(0, 1000)
        ax.plot(pitch)
        pitch_indicator.pyplot(fig)
    else:
        status_indicator.write("AudioReciver is not set. Abort.")
        break