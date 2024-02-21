import tkinter as tk
from tkinter import messagebox
import os
import pyaudio
import threading
import time
import wave


class SoundRecorderApp:
    def __init__(self, master, save_dir, chunk, channels, sampling_rate):
        self.master = master
        master.title('Sound Recorder')

        # Initialize PyAudio parameters
        self.chunk = chunk
        self.format = pyaudio.paInt16
        self.channels = channels
        self.rate = sampling_rate

        # Recording state
        self.recording = False

        # PyAudio instance
        self.p = pyaudio.PyAudio()

        # About pausing and replaying
        self.playing_stream = None
        self.frames = []  # To store frames of the audio file for playback
        self.current_frame = 0  # To keep track of the current frame during playback
        self.is_paused = False  # Playback pause state

        # Setup all UI Elements
        self._setup_gui(master)

        # set folder that stores recorded audios
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Load existing recordings
        self.load_recordings()

    def start_recording(self):
        self.recording = True
        self.record_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        threading.Thread(target=self._record).start()

    def _record(self):
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        print("Recording started")

        frames = []

        while self.recording:
            data = stream.read(self.chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        filename = f"recording_{int(time.time())}.wav"
        filepath = os.path.join(self.save_dir, filename)
        wf = wave.open(filepath, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

        print("Recording stopped")
        self.load_recordings()

    def stop_recording(self):
        self.recording = False
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def play_last_recording(self):
        if not self.recordings_listbox.curselection():
            messagebox.showerror("Error", "Please select a recording to play.")
            return
        selected_index = self.recordings_listbox.curselection()[0]
        selected_filename = self.recordings_listbox.get(selected_index)

        filepath = os.path.join(self.save_dir, selected_filename)
        wf = wave.open(filepath, 'rb')

        self.playing_stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                          channels=wf.getnchannels(),
                                          rate=wf.getframerate(),
                                          output=True)
        # Reset playback state
        self.frames = []
        self.current_frame = 0
        self.is_paused = False

        data = wf.readframes(self.chunk)
        while data:
            self.frames.append(data)
            data = wf.readframes(self.chunk)

        wf.close()
        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        # Start playback
        threading.Thread(target=self.play_frames_from_current).start()

    def load_recordings(self):
        self.recordings_listbox.delete(0, tk.END)
        for file in os.listdir(self.save_dir):
            if file.endswith(".wav"):
                self.recordings_listbox.insert(tk.END, file)

    def toggle_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.pause_button.config(text="Pause")
            # Resume playback in a new thread to avoid blocking the GUI
            threading.Thread(target=self.play_frames_from_current).start()
        else:
            self.is_paused = True
            self.pause_button.config(text="Resume")

    def play_frames_from_current(self):
        while self.current_frame < len(self.frames) and not self.is_paused:
            data = self.frames[self.current_frame]
            self.playing_stream.write(data)
            self.current_frame += 1
        if not self.is_paused:
            # Playback finished
            self.cleanup_after_playback()

    def cleanup_after_playback(self):
        self.playing_stream.stop_stream()
        self.playing_stream.close()
        self.current_frame = 0
        self.frames = []
        self.play_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED, text="Pause")

    def _setup_gui(self, master):
        # Set the initial size of the window
        master.geometry('600x400')  # Width x Height in pixels

        # Create frames
        self.left_frame = tk.Frame(master)
        self.right_frame = tk.Frame(master)

        # Position frames
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Configure grid layout
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(0, weight=1)

        # Setup widget
        # Listbox in the left frame
        self.recordings_listbox = tk.Listbox(self.left_frame)
        self.recordings_listbox.pack(expand=True, fill='both')

        # Buttons in the right frame
        self.record_button = tk.Button(self.right_frame, text="Record", command=self.start_recording)
        self.record_button.pack(fill='x')

        self.stop_button = tk.Button(self.right_frame, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(fill='x')

        self.play_button = tk.Button(self.right_frame, text="Play", command=self.play_last_recording, state=tk.DISABLED)
        self.play_button.pack(fill='x')

        self.pause_button = tk.Button(self.right_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(fill='x')

        # Bind the selection event to the listbox
        # This allows enabling and disabling "play" button when selecting items in the list
        self.last_selected_index = -1
        self.recordings_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)

    def on_listbox_select(self, event):
        # Get the listbox widget
        widget = event.widget
        current_selection = widget.curselection()

        # Check if there's a current selection
        if current_selection:
            current_index = int(current_selection[0])

            # Check if the current selection is the same as the last selection
            if current_index == self.last_selected_index:
                # Deselect the item and disable the "Play" button
                widget.selection_clear(current_index)
                self.play_button.config(state=tk.DISABLED)
                self.last_selected_index = -1  # Reset the last selected index
            else:
                # Update the last selected index and enable the "Play" button
                self.last_selected_index = current_index
                self.play_button.config(state=tk.NORMAL)
        else:
            # No current selection, ensure the "Play" button is disabled
            self.play_button.config(state=tk.DISABLED)
            self.last_selected_index = -1  # Reset the last selected index


