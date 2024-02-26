import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import os

import numpy as np
import pyaudio
import threading
import time
import wave
import librosa
import soundfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import io
from PIL import Image, ImageTk
import ReadWrite

import tempfile
import webrtcvad
import noisereduce as nr
import soundfile as sf
# import progressbar

import speech_recognition as sr
# from pydub import AudioSegment
# from pydub.silence import split_on_silence
from datetime import datetime


class SoundRecorderApp:
    def __init__(self, master, save_dir, chunk_size, channels, sampling_rate):
        self.master = master
        master.title('Sound Recorder')

        # Initialize PyAudio parameters
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.channels = channels
        self.rate = sampling_rate
        self.frame_duration_ms = 30  # Define the duration of each frame in milliseconds

        # Recording state
        self.recording = False

        # PyAudio instance
        self.p = pyaudio.PyAudio()

        # About loading an audio, and storing it into memory for playing or manipulation
        self.selected_filename = None
        self.audio_array: np.ndarray = None
        self.audio_sampling_rate = None
        self.playing_stream = None
        self.frames = []  # To store frames of the audio file for playback
        self.current_frame = 0  # To keep track of the current frame during playback
        self.is_paused = False  # Playback pause state

        # Setup all UI Elements
        self._setup_gui(master)

        # set folder that stores recorded audios
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # set up about audio trimming
        self.start_frame = None
        self.end_frame = None

        # Load existing recordings
        self.load_all_recordings()

        self.recording_thread = None
        self.playing_thread = None

    def start_recording(self):
        self.recording = True
        self.record_button.config(state=tk.DISABLED)
        self.play_pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.visualization_frame.pack_forget()
        self.change_plot_style_button.pack(anchor='ne')
        self.canvas_widget.get_tk_widget().pack()
        self.recording_thread = threading.Thread(target=self._record)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def _record(self):
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        print("Recording started")

        frames = []
        count = 0
        while self.recording:
            count +=1
            data = stream.read(self.chunk_size)
            frames.append(data)
            if count % 5 ==0:
                count=0
                self.realtime_plot_waveform(frames)
        stream.stop_stream()
        stream.close()

        filename = f"recording_{int(time.time())}.wav"
        filepath = os.path.join(self.save_dir, filename)

        # DONE: Need to implement this part by ourself
        '''wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()'''
        ReadWrite.write_wav(frames, filepath, self.rate, self.channels, self.p.get_sample_size(self.format))

        print("Recording stopped")
        self.load_all_recordings()

    def stop_recording(self):
        self.recording = False
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.ax.clear()
        self.ax.axis('off')
        self.ax.axhline(y=0, color='black', alpha=0)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack_forget()
        self.change_plot_style_button.pack_forget()
        self.visualization_frame.pack(fill='both', expand=True)

    def load_all_recordings(self):
        self.recordings_listbox.delete(0, tk.END)
        for file in os.listdir(self.save_dir):
            if file.endswith(".wav"):
                self.recordings_listbox.insert(tk.END, file)

    def toggle_play_pause(self, event=None):
        if self.audio_array is None:
            return
        if self.is_paused:
            self.is_paused = False
            self.play_pause_button.config(text="Pause")
            # Resume playback in a new thread to avoid blocking the GUI
            self.playing_thread=threading.Thread(target=self.play_frames_from_current)
            self.playing_thread.daemon=True
            self.playing_thread.start()
        else:
            self.is_paused = True
            self.play_pause_button.config(text="Play")

    def play_frames_from_current(self):
        while not self.is_paused and self.current_frame < self.end_frame:
            data = self.frames[self.current_frame]
            self.playing_stream.write(data)
            self.current_frame += 1
            self.update_progress_bar()
        if not self.is_paused and self.audio_array is not None:
            # Playing reach the end
            self.setup_replay()

    def setup_replay(self):
        self.current_frame = self.start_frame
        self.is_paused = True

        self.play_pause_button.config(state=tk.NORMAL)
        self.play_pause_button.config(text="Play")

        self.update_progress_bar()

    def realtime_plot_waveform(self,frames):
        leng = 500
        int_frames = np.zeros((leng, ), dtype=np.int16)
        for i in range(min(leng,len(frames))):
            int_frames[leng-i-1] = np.frombuffer(frames[len(frames)-i-1], dtype=np.int16)[0]
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_ylim(-5000, 5000)
        if self.plot_style == 1:
            self.ax.plot(int_frames, color='gray')
        # self.ax.axhline(y=0, color='blue',xmin=0.05,xmax=0.95)
        # self.ax.vlines(range(len(int_frames)), 0, int_frames, color='blue', alpha=1,linewidths=1)
        else:
            self.ax.vlines(range(len(int_frames)), -abs(int_frames)/2, abs(int_frames)/2, color='gray', alpha=1, linewidths=1)
        self.canvas_widget.draw()

    def change_plot_style(self):
        if self.plot_style == 1:
            self.plot_style = 0
        else:
            self.plot_style = 1

    def _setup_gui(self, master):
        # Set the initial size of the window
        master.geometry('1000x600')  # Width x Height in pixels

        # Create the upper and lower frames
        self.upper_frame = tk.Frame(master, height=250, width=1000)
        self.upper_frame.pack(side="top", fill="both", expand=True)
        self.upper_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        self.lower_frame = tk.Frame(master, height=350, width=1000)
        self.lower_frame.pack(side="bottom", fill="both", expand=True)
        self.lower_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        # Subdivide the upper frame into left and right parts
        self.left_frame = tk.Frame(self.upper_frame, borderwidth=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.upper_frame, borderwidth=2, relief="groove")
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Setup widget
        # Listbox in the left frame
        self.recordings_listbox = tk.Listbox(self.left_frame)
        self.recordings_listbox.pack(expand=True, fill='both', side="left")
        # Bind the selection event to the listbox
        # This allows enabling and disabling "play" button when selecting items in the list
        self.last_selected_index = -1
        self.recordings_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)

        # real-time visualization
        self.ax = None
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.fig.patch.set_facecolor('gray')
        self.fig.patch.set_alpha(0.12)
        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.lower_frame)

        # button to change the style of real-time visualization
        self.plot_style = 1
        self.change_plot_style_button = tk.Button(self.lower_frame, text="Change Style", command=self.change_plot_style)

        # Buttons in the right frame
        self.record_button = tk.Button(self.right_frame, text="Record", command=self.start_recording)
        self.record_button.pack(fill='x')

        self.stop_button = tk.Button(self.right_frame, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(fill='x')

        self.play_pause_button = tk.Button(self.right_frame, text="Play", command=self.toggle_play_pause, state=tk.DISABLED)
        self.play_pause_button.pack(fill='x')
        self.master.bind("<space>", self.toggle_play_pause)

        self.save_button = tk.Button(self.right_frame, text='Save', command=self.save_trimmed_audio, state=tk.DISABLED)
        self.save_button.pack(fill='x')

        self.noise_reduction_button = tk.Button(self.right_frame, text="Reduce Noise", command=self.remove_background_noise, state=tk.DISABLED)
        self.noise_reduction_button.pack(fill='x')

        self.audio_to_text_button = tk.Button(self.right_frame, text="Convert to Text", command=self.convert_audio_to_text, state=tk.DISABLED)
        self.audio_to_text_button.pack(fill='x')

        # Waveform visualization at the lower frame
        self.audio_visualize_image = None
        self.photo_image = None
        self.visualization_frame = tk.Label(self.lower_frame)
        self.visualization_frame.pack(fill='both', expand=True)
        # Bind the visualization image to a click event to collect the click position
        self.visualization_frame.bind("<Button-1>", self.on_left_mouse_click_image)
        self.visualization_frame.bind("<Button-3>", self.on_right_mouse_click_image)

        # Add a progress bar
        self.progress_bar = ttk.Progressbar(self.lower_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(side='bottom', fill='x', pady=10)
        self.progress_bar['maximum'] = 1000
        self.progress_bar.bind("<Button-1>", self.on_left_mouse_click_progressbar)

    def write_to_text_file(self, text):
        """ Write a text to a file. """
        # 生成一个带时间戳的文件名
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"outputText_{timestamp}.txt"

        # 确保 save_dir 是目录路径
        if os.path.isdir(self.save_dir):
            file_path = os.path.join(self.save_dir, filename)
        else:
            # 如果 save_dir 是文件路径，直接使用它，但要确保文件名不重复
            save_dir, ext = os.path.splitext(self.save_dir)
            file_path = f"{save_dir}_{timestamp}{ext}"

        print(f"Writing text to file: {file_path}")
        with open(file_path, "w") as f:
            f.write(text)

    def convert_audio_to_text(self):
        # obj = AudioToText(self.audio_array, self.save_dir, self.language)
        # text = obj.get_text()
        # obj.write_to_text_file(text)

        audio_data_int = np.int16(self.audio_array * 32767)
        audio_data = sr.AudioData(audio_data_int.tobytes(), self.rate, 2)

        recognizer = sr.Recognizer()
        try:
            text = recognizer.recognize_sphinx(audio_data, language='en-US')
            self.write_to_text_file(text)
            print("识别结果：", text)
        except sr.UnknownValueError:
            print("无法识别的音频")
        except sr.RequestError as e:
            print(f"从服务请求结果时出错：{e}")

    def remove_background_noise(self):
        if self.audio_array is None or self.audio_sampling_rate is None:
            print("No audio loaded for noise reduction.")
            return

        # Check if sample rate is supported by WebRTC VAD
        supported_rates = [8000, 16000, 32000, 48000]
        if self.audio_sampling_rate not in supported_rates:
            print(f"Original sample rate ({self.audio_sampling_rate} Hz) is not supported by WebRTC VAD. Resampling...")
            # Choose a supported sample rate for resampling. Here we choose 16000 Hz as a default.
            target_rate = 16000
            # Resample the audio to the target rate
            self.audio_array = librosa.resample(self.audio_array, orig_sr=self.audio_sampling_rate, target_sr=target_rate)
            self.audio_sampling_rate = target_rate
            print(f"Audio has been resampled to {target_rate} Hz.")

        # Proceed with VAD and noise reduction as before
        # Convert audio from float to int16, ensuring compatibility with WebRTC VAD
        audio_data_int16 = (self.audio_array * 32768).astype(np.int16)

        # Initialize WebRTC VAD
        vad = webrtcvad.Vad(3)

        # Calculate frame size for VAD (30ms frame size)
        frame_size = int(self.audio_sampling_rate * 0.03)
        frames = [audio_data_int16[i:i + frame_size] for i in range(0, len(audio_data_int16), frame_size)]

        vad_mask = []
        for frame in frames:
            # Pad the last frame if needed
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)), 'constant', constant_values=0)
            is_speech = vad.is_speech(frame.tobytes(), self.audio_sampling_rate)
            vad_mask.extend([is_speech] * frame_size)

        vad_mask = np.array(vad_mask[:len(self.audio_array)])
        filtered_audio_data = self.audio_array * vad_mask

        # Apply noise reduction
        reduced_noise_audio = nr.reduce_noise(y=filtered_audio_data, sr=self.audio_sampling_rate)

        self.audio_array = reduced_noise_audio

        # Update GUI components as necessary
        self.plot_waveform()
        self.update_visualize_image()
        self.update_progress_bar()

        # Save the processed audio with a unique filename
        output_filename = os.path.join(self.save_dir, f"processed_audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
        sf.write(output_filename, reduced_noise_audio, self.audio_sampling_rate)

        # 更新录音列表以显示新文件
        self.load_all_recordings()
        print(f"Noise-reduced audio saved as {output_filename}.")

    def on_listbox_select(self, event):
        widget = event.widget
        current_selection = widget.curselection()

        # Check if there's a current selection
        if current_selection:
            current_index = int(current_selection[0])
            if current_index != self.last_selected_index:
                # An audio is selected, enable certain buttons
                self.last_selected_index = current_index

                # cleanup previous loaded audio and load the new audio into playable
                self.cleanup_selected_audio()
                self.load_selected_audio()

                self.play_pause_button.config(state=tk.NORMAL)
                self.play_pause_button.config(text="Play")
                self.record_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.DISABLED)
                self.save_button.config(state=tk.DISABLED)
                self.noise_reduction_button.config(state=tk.NORMAL)
                self.noise_reduction_button.config(text="Reduce Noise")
                self.audio_to_text_button.config(state=tk.NORMAL)
                self.audio_to_text_button.config(text="Convert to Text")

                return

            else:  # If the current selection is the same as the last selection
                # Clear the listbox to remove the selection sign
                widget.selection_clear(current_index)

        self.play_pause_button.config(state=tk.DISABLED)
        self.play_pause_button.config(text="Play")
        self.record_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)

        # remove the loaded audio
        self.cleanup_selected_audio()

        self.last_selected_index = -1  # Reset the last selected index

    def load_selected_audio(self):
        # load the audio selected in the listbox into streams and an ndarray
        if not self.recordings_listbox.curselection():
            messagebox.showerror("Error", "Please select a recording to play.")
            return

        selected_index = self.recordings_listbox.curselection()[0]
        selected_filename = self.recordings_listbox.get(selected_index)

        print(f'Load selected file: {selected_filename}')

        self.selected_filename = selected_filename
        filepath = os.path.join(self.save_dir, selected_filename)

        # DONE: load the audio into ndarray with our own function
        #waveform, sr = librosa.load(filepath, sr=None)
        frames, sr, channels, bps = ReadWrite.read_wav(filepath)
        self.audio_sampling_rate = sr
        self.audio_array = ReadWrite.frames_to_waveform(frames)

        # load the audio into playable stream
        wf = wave.open(filepath, 'rb')
        self.playing_stream = self.p.open(
            format=self.p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )

        # Reset playback state
        self.frames = []
        self.current_frame = 0
        self.is_paused = True

        data = wf.readframes(self.chunk_size)
        while data:
            self.frames.append(data)
            data = wf.readframes(self.chunk_size)
        wf.close()

        self.start_frame = 0
        self.end_frame = len(self.frames)

        self.plot_waveform()
        self.update_visualize_image()
        self.update_progress_bar()

    def cleanup_selected_audio(self):
        if self.audio_array is None:
            self.progress_bar['value'] = 0
            return

        self.is_paused = True

        if self.playing_stream is not None:
            self.playing_stream.stop_stream()
            self.playing_stream.close()
        self.current_frame = 0
        self.frames = []

        print('Selected file cleaned up')

        self.audio_sampling_rate = None
        self.audio_array = None
        self.start_frame = None
        self.end_frame = None
        self.selected_filename = None

        # Clean up the visualization image
        self.audio_visualize_image = None
        self.photo_image = None
        self.visualization_frame.configure(image=None)

        # Clean up progress bar
        self.update_progress_bar()

    def plot_waveform(self):
        # Visualize the waveform as an ndarray
        t = np.arange(len(self.audio_array)) / self.audio_sampling_rate  # Time axis

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 3))
        # Plot the waveform
        ax.plot(t, self.audio_array, linewidth=1)

        ax.axis('off')
        ax.set_xlim(min(t), max(t))
        ax.set_ylim(min(self.audio_array), max(self.audio_array))
        # Remove padding around the plot
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert buffer to PIL Image
        image = Image.open(buf)

        # Convert PIL Image to NumPy array
        image_array = np.array(image)   # shape (400, 1000), dtype = uint8

        # Close the buffer
        buf.close()

        self.audio_visualize_image = image_array

        print(f'Draw visualization image of shape: {self.audio_visualize_image.shape}')

    def update_visualize_image(self):
        img_start_idx = int(self.start_frame / len(self.frames) * 1000)
        img_end_idx = int(self.end_frame / len(self.frames) * 1000)
        mask = np.ones_like(self.audio_visualize_image, dtype=np.float32)
        mask[:, :img_start_idx] = 0.5
        mask[:, img_end_idx:] = 0.5

        # Convert to PIL Image
        image = Image.fromarray((self.audio_visualize_image * mask).astype(np.uint8))

        # Convert to Tkinter PhotoImage
        self.photo_image = ImageTk.PhotoImage(image)

        # Update the image to UI
        self.visualization_frame.configure(image=self.photo_image)

    def update_progress_bar(self):
        current_pos = int(self.current_frame / len(self.frames) * 1000) if len(self.frames) > 0 else 0
        self.progress_bar['value'] = current_pos

    def on_left_mouse_click_progressbar(self, event):
        if self.audio_array is None:  # If no audio is loaded
            return
        clicked_frame = int(event.x / 1000 * len(self.frames))
        clicked_frame = max(self.start_frame, clicked_frame)
        clicked_frame = min(self.end_frame, clicked_frame)
        self.current_frame = clicked_frame
        self.update_progress_bar()
        print(f'current frame update: {self.current_frame}')

    def on_left_mouse_click_image(self, event):
        if self.audio_array is None:  # If no audio is loaded
            return
        cut_frame = int(event.x / 1000 * len(self.frames))
        self.start_frame = min(cut_frame, self.end_frame)
        self.update_visualize_image()

        # update the progressbar accordingly
        self.current_frame = max(self.current_frame, self.start_frame)
        self.update_progress_bar()

        print(f'trim: start: {self.start_frame}, end: {self.end_frame}')

        # If the audio is trimmed, allow saving
        if self.start_frame != 0 or self.end_frame != len(self.frames):
            if self.start_frame < self.end_frame:
                self.save_button.config(state=tk.NORMAL)
                return
        self.save_button.config(state=tk.DISABLED)

    def on_right_mouse_click_image(self, event):
        if self.audio_array is None:  # If no audio is loaded
            return
        cut_frame = int(event.x / 1000 * len(self.frames))
        self.end_frame = max(cut_frame, self.start_frame)
        self.update_visualize_image()

        # update the progressbar accordingly
        self.current_frame = min(self.current_frame, self.end_frame)
        self.update_progress_bar()

        print(f'trim: start: {self.start_frame}, end: {self.end_frame}')

        # If the audio is trimmed, allow saving
        if self.start_frame != 0 or self.end_frame != len(self.frames):
            if self.start_frame < self.end_frame:
                self.save_button.config(state=tk.NORMAL)
                return
        self.save_button.config(state=tk.DISABLED)

    def save_trimmed_audio(self):
        old_filename = self.selected_filename.split('.')[0]
        new_filename = old_filename + '_trimmed.wav'
        save_path = os.path.join(self.save_dir, new_filename)

        # if the filename already exists
        audio_id = 0
        while os.path.exists(save_path):
            audio_id += 1
            new_filename = old_filename + f'_trimmed({audio_id}).wav'
            save_path = os.path.join(self.save_dir, new_filename)

        print(f'The trimmed audio is saved into: {save_path}')

        wav_start_idx = int(self.start_frame / len(self.frames) * self.audio_array.shape[0])
        wav_end_idx = int(self.end_frame / len(self.frames) * self.audio_array.shape[0])
        # DONE: replace this with our own function
        #soundfile.write(save_path, self.audio_array[wav_start_idx:wav_end_idx], samplerate=self.audio_sampling_rate)
        waveform = ReadWrite.waveform_to_frames(self.audio_array[wav_start_idx:wav_end_idx], sample_rate=self.audio_sampling_rate)
        ReadWrite.write_wav(waveform, save_path, self.audio_sampling_rate)

        # Update the listbox
        self.load_all_recordings()


