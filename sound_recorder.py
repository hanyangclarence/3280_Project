import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
import pyaudio
import threading
import time
import io

import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import math

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
        self.speed_changing_mode = "OLA"

        # set folder that stores recorded audios
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # set up about audio trimming
        # These two are on original audio
        self.start_frame = None
        self.end_frame = None
        # These two are on speed changed audio
        self.playing_start_frame = None
        self.playing_end_frame = None

        self.recording_thread = None
        self.playing_thread = None

        # Load existing recordings
        self.load_all_recordings()

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
                self.realtime_visualization(frames)
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

    def setup_replay(self):
        self.current_frame = self.start_frame
        self.playing_current_frame = self.playing_start_frame
        self.is_paused = True

        self.play_pause_button.config(state=tk.NORMAL)
        self.play_pause_button.config(text="Play")

        self.update_progress_bar()

    def play_frames_from_current(self):
        n_steps = self.n_steps.get()
        print(n_steps)
        speed = self.speed_scale.get()
        print(speed)

        # Disable the bar during playing
        self.n_steps.config(state=tk.DISABLED)
        self.speed_scale.config(state=tk.DISABLED)

        if speed != 1.0:
            self.playing_frames = self.change_speed(speed, self.frames)
            self.playing_current_frame = round(self.current_frame / speed)
            self.playing_start_frame = int(self.start_frame / len(self.frames) * len(self.playing_frames))
            self.playing_end_frame = int(self.end_frame / len(self.frames) * len(self.playing_frames))
        else:
            self.playing_frames = self.frames
            self.playing_current_frame = self.current_frame
            self.playing_start_frame = self.start_frame
            self.playing_end_frame = self.end_frame

        if n_steps != 0:
            original_length = len(self.playing_frames)
            self.playing_frames = self.change_pitch(self.playing_frames, n_steps)
            self.playing_current_frame = int(self.playing_current_frame / original_length * len(self.playing_frames))
            self.playing_start_frame = int(self.playing_start_frame / original_length * len(self.playing_frames))
            self.playing_end_frame = int(self.playing_end_frame / original_length * len(self.playing_frames))

        while not self.is_paused and self.playing_current_frame < self.playing_end_frame:
            data = self.playing_frames[self.playing_current_frame]
            self.playing_stream.write(data)
            self.playing_current_frame += 1
            self.current_frame = round(self.playing_current_frame * speed)
            self.update_progress_bar()

        # Enable the bar when paused or ended
        self.n_steps.config(state=tk.NORMAL)
        self.speed_scale.config(state=tk.NORMAL)

        if not self.is_paused and self.audio_array is not None:
            # Handling the end of playback
            self.setup_replay()

    def pitch_interp(self, y, sr, n_steps):
        raise NotImplementedError("Subclass must implement abstract method")

    def change_pitch(self, frames, n_steps):
        raise NotImplementedError("Subclass must implement abstract method")

    def change_speed(self, speed, frames):
        raise NotImplementedError("Subclass must implement abstract method")

    def realtime_visualization(self, frames):
        leng = 512
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_ylim(-5000, 5000)
        int_frames = np.zeros((leng,), dtype=np.int16)
        if self.plot_style == 3:
            for i in range(min(leng, len(frames[-1]))):
                int_frames[leng - i - 1] = np.frombuffer(frames[-1], dtype=np.int16)[-i * 2]
            self.ax.plot(int_frames, color='gray')
        elif self.plot_style == 4:
            leng = self.chunk_size
            int_frames = np.zeros((leng,), dtype=np.int16)
            for i in range(leng):
                int_frames[leng - i - 1] = np.frombuffer(frames[-1], dtype=np.int16)[-i]
            self.ax.axis('on')
            self.ax.set_ylim(0, 1500)
            # generate the hanning window
            hanning_window = [0] * leng
            for i in range(leng):
                hanning_window[i] = 0.5 - 0.5 * math.cos(2 * math.pi * i / (leng - 1))
            float_frames = int_frames.astype(np.float64)
            float_frames *= hanning_window
            spectrum = np.fft.fft(float_frames)
            magnitude_spectrum = np.abs(spectrum) / (leng/2)
            frequency_axis = np.linspace(0, self.rate / 8, leng // 8)
            self.ax.plot(frequency_axis, magnitude_spectrum[:leng // 8])
        else:
            for i in range(min(leng,len(frames))):
                int_frames[leng-i-1] = np.frombuffer(frames[len(frames)-i-1], dtype=np.int16)[0]
            if self.plot_style == 1:
                self.ax.plot(int_frames, color='gray')
            # self.ax.axhline(y=0, color='blue',xmin=0.05,xmax=0.95)
            else:
                self.ax.vlines(range(len(int_frames)), -abs(int_frames)/2, abs(int_frames)/2, alpha=1, linewidths=1)
                # self.ax.vlines(range(len(int_frames)), 0, int_frames, color='gray', alpha=1,linewidths=1)
        self.canvas_widget.draw()

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
        # waveform, sr = librosa.load(filepath, sr=None)
        self.frames, sr, channels, bps = ReadWrite.read_wav(filepath, frame_size=self.chunk_size)
        self.audio_sampling_rate = sr
        self.audio_array = ReadWrite.frames_to_waveform(self.frames)
        # By default, set self.playing_frames as self.frames, as default there is no speed change
        self.playing_frames = self.frames

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

        self.start_frame = self.playing_start_frame = 0
        self.end_frame = self.playing_end_frame = len(self.frames)

        self.current_frame = 0
        self.playing_current_frame = 0
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
        self.current_frame = -1
        self.playing_current_frame = -1
        self.frames = []
        self.playing_frames = []

        print('Selected file cleaned up')

        self.audio_sampling_rate = None
        self.audio_array = None
        self.start_frame = None
        self.end_frame = None
        self.playing_start_frame = None
        self.playing_end_frame = None
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
        cut_playing_frame = int(event.x / 1500 * len(self.playing_frames))
        self.playing_start_frame = min(cut_playing_frame, self.playing_end_frame)
        self.update_visualize_image()

        # update the progressbar accordingly
        self.current_frame = max(self.current_frame, self.start_frame)
        self.playing_current_frame = max(self.playing_current_frame, self.playing_start_frame)
        self.update_progress_bar()

        print(f'trim: start: {self.start_frame}, end: {self.end_frame}')

        # If the audio is trimmed, allow saving or replacing
        if self.playing_start_frame != 0 or self.playing_end_frame != len(self.playing_frames):
            if self.playing_start_frame < self.playing_end_frame:
                self.save_button.config(state=tk.NORMAL)
                self.replace_button.config(state=tk.NORMAL)
                return
        self.save_button.config(state=tk.DISABLED)

    def on_right_mouse_click_image(self, event):
        if self.audio_array is None:  # If no audio is loaded
            return
        cut_frame = int(event.x / 1500 * len(self.frames))
        self.end_frame = max(cut_frame, self.start_frame)
        cut_playing_frame = int(event.x / 1500 * len(self.playing_frames))
        self.playing_end_frame = max(cut_playing_frame, self.playing_start_frame)
        self.update_visualize_image()

        # update the progressbar accordingly
        self.current_frame = min(self.current_frame, self.end_frame)
        self.playing_current_frame = min(self.playing_current_frame, self.playing_end_frame)
        self.update_progress_bar()

        print(f'trim: start: {self.start_frame}, end: {self.end_frame}')

        # If the audio is trimmed, allow saving or replacing
        if self.playing_start_frame != 0 or self.playing_end_frame != len(self.playing_frames):
            if self.playing_start_frame < self.playing_end_frame:
                self.save_button.config(state=tk.NORMAL)
                self.replace_button.config(state=tk.NORMAL)
                return
        self.save_button.config(state=tk.DISABLED)



