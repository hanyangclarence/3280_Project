import tkinter as tk
from tkinter import messagebox, ttk
import os
import numpy as np
import pyaudio
import threading
import time
import io

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import ReadWrite
import math

import librosa
import webrtcvad
import noisereduce as nr
import speech_recognition as sr
from datetime import datetime

import ReadWrite
from base_ui import BaseUI


class BasicSoundRecorder(BaseUI):
    def __init__(self, master, save_dir, chunk_size, channels, sampling_rate):
        super().__init__(master)

        # Initialize PyAudio parameters
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.channels = 2
        self.rate = sampling_rate
        self.bytes_per_sample = 2



