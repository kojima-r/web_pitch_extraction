import streamlit as st
st.title("Pitch Extraction")

from streamlit_webrtc import webrtc_streamer, WebRtcMode

import torch
import torchcrepe
import torchaudio.transforms as T

import librosa
import time
import queue
from matplotlib import pyplot as plt
import mpl_toolkits.axes_grid1
import pandas as pd
import numpy as np
import cv2

import logging
import os
from twilio.rest import Client
logger = logging.getLogger(__name__)

# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )

N_FFT = 1024
RESAMPLE_RATE = 16000
PLOT_LENGTH = 1000

def get_fig(pitch, spec, min_value, max_value):
    fig = plt.figure()
    ax = fig.add_subplot()
    aximg = ax.imshow(
            cv2.resize(spec[0:int(max_value*(N_FFT//2 + 1)/(RESAMPLE_RATE//2)),:],
            dsize=(PLOT_LENGTH, max_value)),
            vmin=0, vmax=100,
        )
    ax.set_ylim(min_value, max_value)
    ax.plot(pitch)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(aximg, cax=cax)
    return fig

status_indicator = st.empty()

option = st.selectbox(
    'Select a Pitch Extraction Model',
     ('torchcrepe', 'librosa.pyin'))
values = st.slider('Select a range of values',  min_value=0, max_value=1000, step=1, value=(0,600))

if "pitch" not in st.session_state:
    st.session_state.pitch = np.zeros(PLOT_LENGTH)
pitch = st.session_state.pitch
if "spec" not in st.session_state:
    st.session_state.spec = np.zeros((N_FFT//2 + 1,PLOT_LENGTH))
spec = st.session_state.spec
if "fig" not in st.session_state:
    st.session_state.fig = get_fig(pitch, spec, values[0], values[1])
fig = st.session_state.fig
pitch_indicator = st.pyplot(fig)

pitch_csv = st.download_button(
    "Download pitch as CSV",
    data=pd.DataFrame(pitch).to_csv().encode('utf-8'),
    file_name="pitch.csv",
    mime='text/csv',
)

spectrogram = None
resampler = None

device = torch.device('cuda:0' if torch.cuda.is_available() and False else 'cpu')
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
        audio = torch.tensor(np.copy(audio_chunk))[None].to(device)

        if option == 'torchcrepe':
            pitch_tmp,periodicity = torchcrepe.predict(
                    audio,
                    audio_frame.sample_rate,
                    hop_length = int(audio_frame.sample_rate / 200.),
                    fmin = 50,
                    fmax = 1000,
                    model = 'tiny',
                    device = device,
                    return_periodicity = True,
                )
            periodicity = torchcrepe.filter.median(periodicity, win_length = 3)
            periodicity = torchcrepe.threshold.Silence(-60.)(
                    periodicity,
                    audio,
                    audio_frame.sample_rate,
                    hop_length = int(audio_frame.sample_rate / 200.),
                )
            pitch_tmp = torchcrepe.threshold.At(.21)(pitch_tmp, periodicity)
            pitch_tmp = torchcrepe.filter.mean(pitch_tmp, win_length = 3)
            pitch_tmp = pitch_tmp.cpu().detach().numpy()[0]
        else:
            pitch_tmp = librosa.pyin(
                    audio_chunk,
                    sr = audio_frame.sample_rate,
                    fmin = 50,
                    fmax = 1000,
                )[0]
        f0 = np.zeros_like(pitch_tmp)
        f0[pitch_tmp > 0] = pitch_tmp[pitch_tmp > 0]
        pitch = np.append(pitch,f0)[-PLOT_LENGTH:]

        if spectrogram == None:
            spectrogram = T.Spectrogram(
                    n_fft = N_FFT,
                ).to(device)
        if resampler == None:
            resampler = T.Resample(
                    audio_frame.sample_rate,
                    RESAMPLE_RATE,
                ).to(device)
        spec_tmp = spectrogram(resampler(audio)).cpu().detach().numpy()[0]
        spec_tmp = cv2.resize(spec_tmp, dsize=(pitch_tmp.shape[0], N_FFT//2 + 1))
        spec = np.append(spec,spec_tmp,axis=1)[:,-PLOT_LENGTH:]
        
        fig = get_fig(pitch, spec, values[0], values[1])
        pitch_indicator.pyplot(fig)

        st.session_state.pitch = pitch
        st.session_state.spec = spec
        st.session_state.fig = fig
    else:
        status_indicator.write("AudioReciver is not set. Abort.")
        break