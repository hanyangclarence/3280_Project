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

import librosa
import webrtcvad
import noisereduce as nr
import speech_recognition as sr
from datetime import datetime

import ReadWrite
from scipy.signal import resample



class SoundRecorderApp:
    def __init__(self, master, save_dir, chunk_size, channels, sampling_rate):
        self.master = master
        master.title('Sound Recorder')

        # Initialize PyAudio parameters
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        self.channels = channels
        self.rate = sampling_rate

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
        # for changing speed
        self.playing_frames = []
        self.playing_current_frame = 0
        self.playing_end_frame = None

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
            self.playing_thread = threading.Thread(target=self.play_frames_from_current)
            self.playing_thread.daemon = True
            self.playing_thread.start()
        else:
            self.is_paused = True
            self.play_pause_button.config(text="Play")

    # def play_frames_from_current(self):
    #         speed = self.speed_scale.get()
    #         if speed != 1.0:
    #             self.playing_frames = self.change_speed(speed)
    #             self.playing_current_frame = round(self.current_frame / speed)
    #             self.playing_end_frame = len(self.playing_frames)
    #         else:
    #             self.playing_frames = self.frames
    #             self.playing_current_frame = self.current_frame
    #             self.playing_end_frame = self.end_frame
    #         while not self.is_paused and self.playing_current_frame < self.playing_end_frame:
    #             data = self.playing_frames[self.playing_current_frame]
    #             self.playing_stream.write(data)
    #             self.playing_current_frame += 1
    #             self.current_frame = round(self.playing_current_frame * speed)
    #             self.update_progress_bar()
    #         if not self.is_paused and self.audio_array is not None:
    #             # Playing reach the end
    #             self.setup_replay()

    def setup_replay(self):
        self.current_frame = self.start_frame
        self.playing_current_frame = int(self.current_frame / len(self.frames) * len(self.playing_frames))
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

    def play_frames_from_current(self):
        n_steps = self.n_steps.get()
        print(n_steps)
        speed = self.speed_scale.get()
        print(speed)
        if speed != 1.0:
            self.playing_frames = self.change_speed(speed)
            self.playing_current_frame = round(self.current_frame / speed)
            self.playing_end_frame = len(self.playing_frames)
        else:
            self.playing_frames = self.frames
            self.playing_current_frame = self.current_frame
            self.playing_end_frame = self.end_frame

        # while not self.is_paused and self.playing_current_frame < self.playing_end_frame:
        #     data = self.playing_frames[self.playing_current_frame]
        #     self.playing_stream.write(data)
        #     self.playing_current_frame += 1
        #     self.current_frame = round(self.playing_current_frame * speed)
        #     self.update_progress_bar()

        if n_steps != 0:
            self.playing_frames = self.change_pitch(self.playing_frames, n_steps)
            self.playing_current_frame = 0  # Start playing from the beginning of the audio
            self.playing_end_frame = len(self.playing_frames)

        # else:
        #     self.playing_frames = self.frames
        #     self.playing_current_frame = self.current_frame
        #     self.playing_end_frame = self.end_frame

        # while not self.is_paused and self.playing_current_frame < self.playing_end_frame:
        #     data = self.playing_frames[self.playing_current_frame]
        #     self.playing_stream.write(data)
        #     self.playing_current_frame += 1
        #     self.update_progress_bar()

        while not self.is_paused and self.playing_current_frame < self.playing_end_frame:
            data = self.playing_frames[self.playing_current_frame]
            self.playing_stream.write(data)
            self.playing_current_frame += 1
            self.current_frame = round(self.playing_current_frame * speed)
            self.update_progress_bar()

        if not self.is_paused and self.audio_array is not None:
            # Handling the end of playback
            self.setup_replay()

    # def change_pitch(self, frames, n_steps):
    #     arr = np.frombuffer(b''.join(frames), dtype=np.int16)
    #     y = arr.astype(np.float32)
    #     sr = self.audio_sampling_rate  # Correct the sampling rate here

    #     try:
    #         # Pitch shifting using librosa
    #         y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    #         # Convert back to int16
    #         y_shifted_int = y_shifted.astype(np.int16)

    #         # Splitting the shifted audio into frames
    #         bytes_arr = [y_shifted_int[i:i + self.chunk_size].tobytes() for i in range(0, len(y_shifted_int), self.chunk_size)]
    #         return bytes_arr
    #     except Exception as e:
    #         print(f"Error occurred during pitch shifting: {str(e)}")
    #         return frames  # Return original frames if an error occurs

    def change_pitch(self, frames, n_steps):
        arr = np.frombuffer(b''.join(frames), dtype=np.int16)
        y = arr.astype(np.float32)
        sr = self.audio_sampling_rate

        try:
            # Perform pitch shifting using FFT
            y_shifted = self.pitch_shift_fft(y, sr, n_steps)

            # Convert back to int16
            y_shifted_int = y_shifted.astype(np.int16)

            # Splitting the shifted audio into frames
            bytes_arr = [y_shifted_int[i:i + self.chunk_size].tobytes() for i in range(0, len(y_shifted_int), self.chunk_size)]
            return bytes_arr
        except Exception as e:
            print(f"Error occurred during pitch shifting: {str(e)}")
            return frames  # Return original frames if an error occurs

    def pitch_shift_fft(self, y, sr, n_steps):
        # Time-domain pitch shifting using FFT
        n = len(y)
        factor = 2 ** (1.0 * n_steps / 12.0)  # Frequency scaling factor
        y_shifted = np.interp(np.arange(0, n, factor), np.arange(n), y)

        return y_shifted


    def change_speed(self, speed):
        arr = np.frombuffer(b''.join(self.frames), dtype=np.int16)
        new_length = int(len(arr) / speed)
        new_arr = np.zeros(new_length, dtype=np.int16)
        win_size = self.chunk_size * 8
        hs = int(win_size * 0.5)
        ha = int(speed * hs)
        dmax = win_size // 16 # dmax cannot be too large, or 0.5x will produce noise.
        hanning_window = np.hanning(win_size)
        old_pos = 0
        new_pos = 0
        while old_pos < len(arr) - win_size and new_pos < len(new_arr) - win_size:
            for i in range(win_size):
                new_arr[new_pos+i] += arr[old_pos+i] * hanning_window[i]
            # find the position of the most similar frame within (old_pos+ha-dmax,old_pos+ha+dmax).
            # my attempt to compute cross-correlation without numpy, but failed.
            # maxval = -2147483648
            # maxidx = -1
            # for i in range(max(0,old_pos+ha-dmax),min(len(arr),old_pos+ha+dmax)):
            #     # metric of simiarity: cross-correlation. Refer to numpy.correlate(a, v, mode='valid')
            #     correlation = 0
            #     for j in range(win_size):
            #         correlation += arr[i+j] * arr[old_pos+hs+j] # may lead to value overflow. idk how to solve
            #     if correlation > maxval:
            #         maxval = correlation
            #         maxidx = i
            # old_pos = maxidx
            sp = max(0,old_pos+ha-dmax)
            ep = min(len(arr),old_pos+ha+dmax+win_size)
            old_pos = sp + np.argmax(np.correlate(arr[sp:ep],arr[old_pos+hs:old_pos+hs+win_size]))
            # old_pos += ha
            new_pos += hs
        tobytes = new_arr.tobytes()
        bytes_arr = [tobytes[i:i+4096] for i in range(0, len(tobytes), 4096)]
        return bytes_arr

    # def adjust_pitch(self):
    #     n_steps = self.n_steps.get()
    #     print(f"Adjusting pitch by {n_steps} steps.")

    #     if self.audio_array is None or self.audio_sampling_rate is None:
    #         print("No audio loaded for noise reduction.")
    #         return

    #     self.audio_array = librosa.effects.pitch_shift(y=self.audio_array, sr=self.audio_sampling_rate, n_steps=n_steps)
    #     # self.audio_array = 
    #     try:
    #         self.plot_waveform()
    #         self.update_visualize_image()
    #         self.update_progress_bar()

    #         if hasattr(self, 'selected_filename'):
    #             original_filename = self.selected_filename
    #         else:
    #             original_filename = f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    #         # Save the processed audio with a unique filename
    #         output_filename = os.path.join(self.save_dir, f"Pitch_shifted_{original_filename}")

    #         # sf.write(output_filename, reduced_noise_audio, self.audio_sampling_rate)
    #         pitch_shifted_audio = ReadWrite.waveform_to_frames(self.audio_array)
    #         ReadWrite.write_wav(pitch_shifted_audio, output_filename, self.audio_sampling_rate, 1, 2)
    #         self.load_all_recordings()
    #         print(f"Pitch-shifted audio saved as {output_filepath}.")
    #     except Exception as e:
    #         # Catching a general exception for demonstration; specify your exception
    #         print("Caught exception while trying to shift pitch:", e)



    def _setup_gui(self, master):
        master.geometry('1500x900')  # Width x Height

        # Create the upper and lower frames
        self.upper_frame = tk.Frame(master, height=400, width=1500)
        self.upper_frame.pack(side="top", fill="both", expand=True)
        self.upper_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        self.lower_frame = tk.Frame(master, height=500, width=1500)
        self.lower_frame.pack(side="bottom", fill="both", expand=True)
        self.lower_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        # Subdivide the upper frame into left and right parts
        self.left_frame = tk.Frame(self.upper_frame, borderwidth=2, relief="groove")
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.upper_frame, borderwidth=2, relief="groove")
        self.right_frame.pack(side="right", fill="both", expand=True)

        # setup widget
        # listbox in the left frame
        self.recordings_listbox = tk.Listbox(self.left_frame)
        self.recordings_listbox.pack(expand=True, fill='both', side="left")
        # bind the selection event to the listbox
        # this allows enabling and disabling "play" button when selecting items in the list
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

        # self.adjust_pitch_button = tk.Button(self.right_frame, text="Adjust Pitch", command=self.adjust_pitch, state=tk.DISABLED)
        # self.adjust_pitch_button.pack(fill='x')

        # 创建第一个滑动模块用于调整音调
        self.inner_frame_1 = tk.Frame(self.right_frame)
        self.inner_frame_1.pack(pady=20)

        # 添加一个标题标签来描述调整音调的滑动模块
        title_label_1 = tk.Label(self.inner_frame_1, text="Adjust Pitch", font=("Arial", 12, "bold"))
        title_label_1.pack()

        # 创建一个滑动条来选择音调调整的步长
        self.n_steps = tk.Scale(self.inner_frame_1, from_=-16, to=16, resolution=1, orient="horizontal", length=200)
        self.n_steps.pack()
        self.n_steps.set(0.0)

        # 创建第二个滑动模块用于调整速度
        self.inner_frame_2 = tk.Frame(self.right_frame)
        self.inner_frame_2.pack(pady=20)

        # 添加一个标题标签来描述调整速度的滑动模块
        title_label_2 = tk.Label(self.inner_frame_2, text="Adjust Speed", font=("Arial", 12, "bold"))
        title_label_2.pack()

        # 创建一个滑动条来选择速度调整的比例
        self.speed_scale = tk.Scale(self.inner_frame_2, from_=0.5, to=2, resolution=0.05, orient="horizontal", length=200)
        self.speed_scale.pack()
        self.speed_scale.set(1.0)

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
        self.progress_bar['maximum'] = 1500
        self.progress_bar.bind("<Button-1>", self.on_left_mouse_click_progressbar)

    def write_to_text_file(self, text):
        """ Write a text to a file. """
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"outputText_{timestamp}.txt"

        if os.path.isdir(self.save_dir):
            file_path = os.path.join(self.save_dir, filename)
        else:
            save_dir, ext = os.path.splitext(self.save_dir)
            file_path = f"{save_dir}_{timestamp}{ext}"

        print(f"Writing text to file: {file_path}")
        with open(file_path, "w") as f:
            f.write(text)

    def convert_audio_to_text(self):
        # Convert your audio array and sample rate to an AudioData object
        audio_data_int = np.int16(self.audio_array * 32767)
        audio_data = sr.AudioData(audio_data_int.tobytes(), self.rate, 2)

        # use the speech_recognition library's recognizer to recognize this AudioData object
        recognizer = sr.Recognizer()
        try:
            # sphinx
            text = recognizer.recognize_sphinx(audio_data, language='en-US')
            self.write_to_text_file(text)
            print("Identify Results:", text)
        except sr.UnknownValueError:
            print("Unrecognized audio")
        except sr.RequestError as e:
            print(f"An error occurred while requesting results from the service: {e}")

    def remove_background_noise(self):

        if self.audio_array is None or self.audio_sampling_rate is None:
            print("No audio loaded for noise reduction.")
            return

        # Check if the original sample rate is supported by WebRTC VAD. If not, resample the audio.
        supported_rates = [8000, 16000, 32000, 48000]
        if self.audio_sampling_rate not in supported_rates:
            print(f"Original sample rate ({self.audio_sampling_rate} Hz) is not supported by WebRTC VAD. Resampling...")
            # Choose a supported sample rate for resampling. Here we choose 16000 Hz as a default.
            target_rate = 16000
            # Resample the audio to the target rate
            self.audio_array = librosa.resample(self.audio_array, orig_sr=self.audio_sampling_rate, target_sr=target_rate)
            self.audio_sampling_rate = target_rate
            print(f"Audio has been resampled to {target_rate} Hz.")

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
        try:
            self.plot_waveform()
            self.update_visualize_image()
            self.update_progress_bar()

            if hasattr(self, 'selected_filename'):
                original_filename = self.selected_filename
            else:
                original_filename = f"audio_{datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Generate a new file name, prefixed with 'Remove_'
            new_filename = f"Remove_{original_filename}"
            # Save the processed audio with a unique filename
            output_filename = os.path.join(self.save_dir, f"Remove_{original_filename}")

            # sf.write(output_filename, reduced_noise_audio, self.audio_sampling_rate)
            reduced_noise_audio = ReadWrite.waveform_to_frames(reduced_noise_audio)
            ReadWrite.write_wav(reduced_noise_audio, output_filename, self.audio_sampling_rate, 1, 2)

            self.load_all_recordings()
            print(f"Noise-reduced audio saved as {output_filepath}.")
        except ValueError as e:
            print("Caught ValueError while trying to plot waveform:", e)

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
                # self.adjust_pitch_button.config(state=tk.NORMAL)
                # self.adjust_pitch_button.config(text="Adjust Pitch")
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
        # waveform, sr = librosa.load(filepath, sr=None)
        self.frames, sr, channels, bps = ReadWrite.read_wav(filepath)
        self.audio_sampling_rate = sr
        self.audio_array = ReadWrite.frames_to_waveform(self.frames)

        # load the audio into playable stream
        # wf = wave.open(filepath, 'rb')
        self.playing_stream = self.p.open(
            format=self.p.get_format_from_width(bps),
            channels=channels,
            rate=sr,
            output=True
        )
        #
        # # Reset playback state
        # self.frames = []
        # self.current_frame = 0
        # self.is_paused = True
        #
        # data = wf.readframes(self.chunk_size)
        # while data:
        #     self.frames.append(data)
        #     data = wf.readframes(self.chunk_size)
        # wf.close()

        self.start_frame = 0
        self.end_frame = len(self.frames)

        self.current_frame = 0
        self.is_paused = True

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
        fig, ax = plt.subplots(figsize=(15, 4))
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
        image_array = np.array(image)   # shape (400, 1500), dtype = uint8

        # Close the buffer
        buf.close()

        self.audio_visualize_image = image_array

        print(f'Draw visualization image of shape: {self.audio_visualize_image.shape}')

    def update_visualize_image(self):
        img_start_idx = int(self.start_frame / len(self.frames) * 1500)
        img_end_idx = int(self.end_frame / len(self.frames) * 1500)
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
        current_pos = int(self.playing_current_frame / len(self.playing_frames) * 1500) if len(self.playing_frames) > 0 else 0
        self.progress_bar['value'] = current_pos

    def on_left_mouse_click_progressbar(self, event):
        if self.audio_array is None:  # If no audio is loaded
            return
        clicked_frame = int(event.x / 1500 * len(self.frames))
        clicked_frame = max(self.start_frame, clicked_frame)
        clicked_frame = min(self.end_frame, clicked_frame)
        self.current_frame = clicked_frame
        self.playing_current_frame = int(self.current_frame / len(self.frames) * len(self.playing_frames))
        self.update_progress_bar()
        print(f'current frame update: {self.current_frame}')

    def on_left_mouse_click_image(self, event):
        if self.audio_array is None:  # If no audio is loaded
            return
        cut_frame = int(event.x / 1500 * len(self.frames))
        self.start_frame = min(cut_frame, self.end_frame)
        self.update_visualize_image()

        # update the progressbar accordingly
        self.current_frame = max(self.current_frame, self.start_frame)
        self.playing_current_frame = int(self.current_frame / len(self.frames) * len(self.playing_frames))
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
        cut_frame = int(event.x / 1500 * len(self.frames))
        self.end_frame = max(cut_frame, self.start_frame)
        self.update_visualize_image()

        # update the progressbar accordingly
        self.current_frame = min(self.current_frame, self.end_frame)
        self.playing_current_frame = int(self.current_frame / len(self.frames) * len(self.playing_frames))
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
        # soundfile.write(save_path, self.audio_array[wav_start_idx:wav_end_idx], samplerate=self.audio_sampling_rate)
        waveform = ReadWrite.waveform_to_frames(self.audio_array[wav_start_idx:wav_end_idx], sample_rate=self.audio_sampling_rate)
        ReadWrite.write_wav(waveform, save_path, rate=self.audio_sampling_rate, channels=self.channels)

        # Update the listbox
        self.load_all_recordings()


