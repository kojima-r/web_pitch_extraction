import streamlit as st
st.title("Pitch Extraction")

from streamlit_webrtc import webrtc_streamer, WebRtcMode

import torch
import torchcrepe
import torchaudio.transforms as T

import time
from matplotlib import pyplot as plt
import numpy as np
import librosa
import cv2


webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        media_stream_constraints={"video": False, "audio": True},
    )

status_indicator = st.empty()
values = st.slider('Select a range of values',  min_value=0, max_value=1000, step=1, value=(0,200))

if "fig" not in st.session_state:
    st.session_state.fig = plt.figure()
fig = st.session_state.fig
pitch_indicator = st.pyplot(fig)

if "pitch" not in st.session_state:
    st.session_state.pitch = np.empty(1000)
pitch = st.session_state.pitch

if "spec" not in st.session_state:
    st.session_state.spec = np.empty((128,1000))
spec = st.session_state.spec

spectrogram = None
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
        audio_chunk = np.sum(audio_chunk, axis=1)/len(audio_frame.layout.channels)
        audio = torch.tensor(np.copy(audio_chunk))[None].cuda()

        # tmp = librosa.pyin(audio_chunk,
        #                 fmin=librosa.note_to_hz('C2'),
        #                 fmax=librosa.note_to_hz('C7'),
        #                     )[0]
        tmp = torchcrepe.predict(audio,
                                audio_frame.sample_rate,
                                hop_length = int(audio_frame.sample_rate / 200.),
                                fmin = 50,
                                fmax = 1000,
                                model = 'tiny',
                                device = 'cuda:0'
                                )[0].cpu().detach().numpy()

        f0 = np.zeros_like(tmp)
        f0[tmp > 0] = tmp[tmp > 0]
        pitch = np.append(pitch,f0)[-1000:]

        if spectrogram == None:
            spectrogram = T.MelSpectrogram(
                            sample_rate=audio_frame.sample_rate,
                            ).cuda()
        spec_tmp = spectrogram(audio)[0].cpu().detach().numpy()
        spec_tmp = cv2.resize(spec_tmp, dsize=(tmp.shape[0], 128))
        spec = np.append(spec,spec_tmp,axis=1)[:,-1000:]

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(spec)
        ax.set_ylim(values[0], values[1])
        ax.plot(pitch)
        print(spec.shape,tmp.shape)


        pitch_indicator.pyplot(fig)

        
        st.session_state.pitch = pitch
        st.session_state.spec = spec
        st.session_state.fig = fig

        
    else:
        status_indicator.write("AudioReciver is not set. Abort.")
        break