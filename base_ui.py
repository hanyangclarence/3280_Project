import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class BaseUI:
    def __init__(self, master):
        self.master = master
        master.title('Sound Recorder')

        master.geometry('1500x900')  # Width x Height

        # Create the upper and lower frames
        self.upper_frame = tk.Frame(master, height=400, width=1500)
        self.upper_frame.pack(side="top", fill="both", expand=True)
        self.upper_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents

        self.lower_frame = tk.Frame(master, height=400, width=1000)
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
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
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

        self.replace_button = tk.Button(self.right_frame, text='Replace', command=self.start_replace_recording, state=tk.DISABLED)
        self.replace_button.pack(fill='x')

        self.noise_reduction_button = tk.Button(self.right_frame, text="Reduce Noise", command=self.remove_background_noise, state=tk.DISABLED)
        self.noise_reduction_button.pack(fill='x')

        self.audio_to_text_button = tk.Button(self.right_frame, text="Convert to Text", command=self.convert_audio_to_text, state=tk.DISABLED)
        self.audio_to_text_button.pack(fill='x')

        self.inner_frame_1 = tk.Frame(self.right_frame)
        self.inner_frame_1.pack(pady=10)

        title_label_1 = tk.Label(self.inner_frame_1, text="Adjust Pitch", font=("Arial", 12, "bold"))
        title_label_1.pack()

        self.n_steps = tk.Scale(self.inner_frame_1, from_=-10, to=10, orient=tk.HORIZONTAL, length=200, resolution=1.0)
        self.n_steps.pack()
        self.n_steps.set(0.0)

        self.pitch_changing_mode = "INTERP"
        label_mode = tk.Label(self.inner_frame_1, text="Mode ")
        label_mode.pack(anchor="se")
        self.pitch_changing_mode_button = tk.Button(self.inner_frame_1, text=self.pitch_changing_mode, width=6, command=self.pitch_mode)
        self.pitch_changing_mode_button.pack(anchor="se")

        self.inner_frame_2 = tk.Frame(self.right_frame)
        self.inner_frame_2.pack(pady=10)

        title_label_2 = tk.Label(self.inner_frame_2, text="Adjust Speed", font=("Arial", 12, "bold"))
        title_label_2.pack()

        self.speed_scale = tk.Scale(self.inner_frame_2, from_=0.5, to=2, resolution=0.05, orient="horizontal",
                                    length=200)
        self.speed_scale.pack()
        self.speed_scale.set(1.0)

        label_mode = tk.Label(self.inner_frame_2, text="Mode ")
        label_mode.pack(anchor="se")
        self.speed_changing_mode_button = tk.Button(self.inner_frame_2, text="OLA", width=6, command=self.speed_mode)
        self.speed_changing_mode_button.pack(anchor="se")

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

    def on_listbox_select(self, event):
        raise NotImplementedError("Subclass must implement abstract method")

    def change_plot_style(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def start_recording(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def stop_recording(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def toggle_play_pause(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def save_trimmed_audio(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def start_replace_recording(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def remove_background_noise(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def convert_audio_to_text(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def pitch_mode(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def speed_mode(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def on_left_mouse_click_image(self, event):
        raise NotImplementedError("Subclass must implement abstract method")

    def on_right_mouse_click_image(self, event):
        raise NotImplementedError("Subclass must implement abstract method")

    def on_left_mouse_click_progressbar(self, event):
        raise NotImplementedError("Subclass must implement abstract method")

