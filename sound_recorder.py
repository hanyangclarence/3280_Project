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
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageTk


class SoundRecorderApp:
    def __init__(self, master, save_dir, chunk_size, channels, sampling_rate, progress_bar_color):
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
        self.audio_array: np.ndarray
        self.audio_sampling_rate: int
        self.playing_stream = None
        self.frames = []  # To store frames of the audio file for playback
        self.current_frame = 0  # To keep track of the current frame during playback
        self.is_paused = False  # Playback pause state

        # Setup all UI Elements
        self._setup_gui(master)

        # set folder that stores recorded audios
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        os.makedirs(save_dir, exist_ok=True)

        self.progress_bar_color = np.asarray(progress_bar_color, dtype=np.uint8)[None, None]  # (1, 1, 4)

        # Load existing recordings
        self.load_all_recordings()

    def start_recording(self):
        self.recording = True
        self.record_button.config(state=tk.DISABLED)
        self.play_pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self._record).start()

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

        while self.recording:
            data = stream.read(self.chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        filename = f"recording_{int(time.time())}.wav"
        filepath = os.path.join(self.save_dir, filename)

        # TODO: Need to implement this part by ourself
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("Recording stopped")
        self.load_all_recordings()

    def stop_recording(self):
        self.recording = False
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def load_all_recordings(self):
        self.recordings_listbox.delete(0, tk.END)
        for file in os.listdir(self.save_dir):
            if file.endswith(".wav"):
                self.recordings_listbox.insert(tk.END, file)

    def toggle_play_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.play_pause_button.config(text="Pause")
            # Resume playback in a new thread to avoid blocking the GUI
            threading.Thread(target=self.play_frames_from_current).start()
        else:
            self.is_paused = True
            self.play_pause_button.config(text="Play")

    def play_frames_from_current(self):
        while self.current_frame < len(self.frames) and not self.is_paused:
            data = self.frames[self.current_frame]
            self.playing_stream.write(data)
            self.current_frame += 1
            print(f'{self.current_frame}/{len(self.frames)}')
            self.update_progress_bar()
        if not self.is_paused:
            # Finished playing, set to starting point by default
            self.setup_replay()

    def setup_replay(self):
        self.current_frame = 0
        self.is_paused = True

        self.play_pause_button.config(state=tk.NORMAL)
        self.play_pause_button.config(text="Play")

        self.update_progress_bar()

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

        # Buttons in the right frame
        self.record_button = tk.Button(self.right_frame, text="Record", command=self.start_recording)
        self.record_button.pack(fill='x')

        self.stop_button = tk.Button(self.right_frame, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(fill='x')

        self.play_pause_button = tk.Button(self.right_frame, text="Play", command=self.toggle_play_pause, state=tk.DISABLED)
        self.play_pause_button.pack(fill='x')

        # Waveform visualization at the lower frame
        self.audio_visualize_image = None
        self.photo_image = None
        self.visualization_frame = tk.Label(self.lower_frame)
        self.visualization_frame.pack(fill='both', expand=True)

        # Add a progress bar
        self.progress_bar = ttk.Progressbar(self.lower_frame, orient='horizontal', mode='determinate')
        self.progress_bar.pack(side='bottom', fill='x', pady=10)
        self.progress_bar['maximum'] = 1000

    def on_listbox_select(self, event):
        widget = event.widget
        current_selection = widget.curselection()

        # Check if there's a current selection
        if current_selection:
            current_index = int(current_selection[0])
            if current_index != self.last_selected_index:
                # An audio is selected, enable certain buttons
                self.last_selected_index = current_index

                # load the audio into playable
                self.load_selected_audio()

                self.play_pause_button.config(state=tk.NORMAL)
                self.play_pause_button.config(text="Play")
                self.record_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.DISABLED)

                return

            else:  # If the current selection is the same as the last selection
                # Clear the listbox to remove the selection sign
                widget.selection_clear(current_index)

        self.play_pause_button.config(state=tk.DISABLED)
        self.play_pause_button.config(text="Play")
        self.record_button.config(state=tk.NORMAL)

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

        filepath = os.path.join(self.save_dir, selected_filename)

        # TODO: load the audio into ndarray with our own function
        waveform, sr = librosa.load(filepath)
        self.audio_array = waveform
        self.audio_sampling_rate = sr

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

        self.plot_waveform()
        self.update_visualize_image()
        self.update_progress_bar()

    def cleanup_selected_audio(self):
        if self.playing_stream is not None:
            self.playing_stream.stop_stream()
            self.playing_stream.close()
        self.current_frame = 0
        self.frames = []

        print('Selected file cleaned up')

        self.audio_sampling_rate = None
        self.audio_array = None

        # Clean up the visualization image
        self.audio_visualize_image = None
        self.photo_image = None
        self.visualization_frame.configure(image=None)

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
        # Convert to PIL Image
        image = Image.fromarray(self.audio_visualize_image)

        # Convert to Tkinter PhotoImage
        self.photo_image = ImageTk.PhotoImage(image)

        # Update the image to UI
        self.visualization_frame.configure(image=self.photo_image)

    def update_progress_bar(self):
        current_pos = int(self.current_frame / len(self.frames) * 1000)
        self.progress_bar['value'] = current_pos
























