import tkinter as tk
from tkinter import messagebox
import os
import numpy as np
import threading
import math

import librosa
import webrtcvad
import noisereduce as nr
import speech_recognition as sr
from datetime import datetime

import ReadWrite
from sound_recorder import BasicSoundRecorder


class FullSoundRecorder(BasicSoundRecorder):
    def __init__(self, master, save_dir, chunk_size, channels, sampling_rate):
        super().__init__(master, save_dir, chunk_size, channels, sampling_rate)

    def start_replace_recording(self):
        self.recording = True
        self.record_button.config(state=tk.DISABLED)
        self.play_pause_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.visualization_frame.pack_forget()
        self.change_plot_style_button.pack(anchor='ne')
        self.canvas_widget.get_tk_widget().pack()
        self.recording_thread = threading.Thread(target=self._record_for_replace)
        self.recording_thread.daemon = True
        self.recording_thread.start()

    def _record_for_replace(self):
        # Clean up progressbar
        self.progress_bar['value'] = 0

        # Close the stream for playing audio
        if self.playing_stream is not None:
            self.playing_stream.stop_stream()
            self.playing_stream.close()

        # Reopen a stream for recording
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.audio.sampleRate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        count = 0
        frames = []
        total_n_frame = self.end_frame - self.start_frame
        framesize = len(self.frames[0])
        total_data_size = total_n_frame * framesize
        print("self.audio.channels "+str(self.audio.channels))
        print("self.channels "+str(self.channels))
        if self.audio.channels==2 and self.channels==1:
            print(" self.audio.channels==2 and self.channels==1")
            total_data_size = total_data_size//2
        elif self.audio.channels==1 and self.channels==2:
            print(" self.audio.channels==1 and self.channels==2")
            total_data_size *= 2
        print(f"Recording started to replace {self.start_frame} - {self.end_frame}")

        data = b''
        while len(data) < total_data_size:
            count += 1
            d = stream.read(self.chunk_size)
            data += d
            frames.append(d)
            if count % 5 == 0:
                self.realtime_visualization(frames)
        data = data[0:total_data_size]
        stream.stop_stream()
        stream.close()
        print("Recording stopped")
        #frames = ReadWrite.change_frame_size_and_channels(frames, framesize)    
        if self.audio.channels !=self.channels:
            tempAudio = ReadWrite.Audio()
            tempAudio.loadData(data, self.audio_sampling_rate, self.channels, self.bytes_per_sample)    # change to self.channel channels first
            data = tempAudio.getData(self.audio_sampling_rate, self.audio.channels, self.bytes_per_sample)

        # Setup stop recording
        self.recording = False
        self.setup_buttons_for_recording_state()
        self.ax.clear()
        self.ax.axis('off')
        self.ax.axhline(y=0, color='black', alpha=0)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack_forget()
        self.change_plot_style_button.pack_forget()
        self.visualization_frame.pack(fill='both', expand=True)

        # Replace the corresponding frames in self.frames
        for i in range(total_n_frame):
            self.frames[self.start_frame + i] = data[i*framesize:(i+1)*framesize]
        #keeps original framesize

        old_filename = self.selected_filename.split('.')[0]
        new_filename = old_filename + '_replaced.wav'
        save_path = os.path.join(self.save_dir, new_filename)

        # if the filename already exists
        audio_id = 0
        while os.path.exists(save_path):
            audio_id += 1
            new_filename = old_filename + f'_replaced({audio_id}).wav'
            save_path = os.path.join(self.save_dir, new_filename)

        #ReadWrite.write_wav(self.frames, save_path, rate=self.audio_sampling_rate, channels=2)
        self.audio.loadFrames(self.frames, self.audio_sampling_rate, self.channels, self.bytes_per_sample)
        self.audio.write(save_path)

        # Update listbox
        self.load_all_recordings()

        # Reload the replaced audio
        self.load_selected_audio(filename=new_filename)
        self.setup_buttons_for_playable_state()

    def pitch_interp(self, frames, n_steps):
        # Time-domain pitch shifting using FFT
        n = len(frames)
        factor = 2 ** (1.0 * n_steps / 12.0)  # Frequenccy scaling factor
        frames_shifted = np.interp(np.arange(0, n, factor), np.arange(n), frames)

        return frames_shifted

    def speed_shift(self, frames, factor):
        indices = np.round( np.arange(0, len(frames), factor) )
        indices = np.clip(indices, 0, len(frames) - 1).astype(int)
        return frames[ indices.astype(int) ]

    def change_pitch(self, frames, n_steps):
        arr = np.frombuffer(b''.join(frames), dtype=np.int16)
        y = arr.astype(np.float32)
        if self.pitch_changing_mode == "INTERP":
            # our speed changing method
            # frames = self.change_speed(1/(2 ** (1.0 * n_steps / 12.0)),frames)
            # librosa's speed changing method
            y = librosa.effects.time_stretch(y,rate=1/(2 ** (1.0 * n_steps / 12.0)))
            sr = self.audio_sampling_rate
            try:
                # Perform pitch shifting using FFT
                y_shifted = self.pitch_interp(y, n_steps)

                # Convert back to int16
                y_shifted_int = y_shifted.astype(np.int16)

                # Splitting the shifted audio into frames
                bytes_arr = [y_shifted_int[i:i + self.chunk_size].tobytes() for i in range(0, len(y_shifted_int), self.chunk_size)]
                return bytes_arr
            except Exception as e:
                print(f"Error occurred during pitch shifting: {str(e)}")
                return frames  # Return original frames if an error occurs
        elif self.pitch_changing_mode == "LIBROSA":
            sr = self.audio_sampling_rate  # Correct the sampling rate here
            try:
                # Pitch shifting using librosa
                y_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

                # Convert back to int16
                y_shifted_int = y_shifted.astype(np.int16)

                # Splitting the shifted audio into frames
                bytes_arr = [y_shifted_int[i:i + self.chunk_size].tobytes() for i in range(0, len(y_shifted_int), self.chunk_size)]
                return bytes_arr
            except Exception as e:
                print(f"Error occurred during pitch shifting: {str(e)}")
                return frames  # Return original frames if an error occurs
        else:
            factor = 2 ** (1.0 * n_steps / 12.0)
            window_size = 2 ** 13
            h = 2 ** 11
            stretched = self.audio_stretch(y, 1.0 / factor, window_size, h)
            frames_array = self.speed_shift(stretched[window_size:], factor)
            frames_array = np.frombuffer(b''.join(frames_array), dtype=np.int16)
            bytes_arr = [frames_array[i:i + self.chunk_size].tobytes() for i in range(0, len(frames_array), self.chunk_size)]
            return bytes_arr
        
    def audio_stretch(self, frames, stretch_factor, window_len, hop_distance):
        # Initialize phase array and window function
        initial_phase = np.zeros(window_len)
        window = np.hanning(window_len)
        stretched_frames = np.zeros(int(len(frames) / stretch_factor + window_len), dtype=np.complex128)

        for idx in np.arange(0, len(frames) - (window_len + hop_distance), int(round(hop_distance * stretch_factor))):
            # Extract two overlapping segments
            segment_one = frames[int(idx): int(idx + window_len)]
            segment_two = frames[int(idx + hop_distance): int(idx + window_len + hop_distance)]

            # FFT and phase manipulation
            fft_one = np.fft.fft(window * segment_one)
            fft_two = np.fft.fft(window * segment_two)
            safe_divisor = 1e-10  # To avoid division by zero
            initial_phase += np.angle(fft_two / (fft_one + safe_divisor)) % (2 * np.pi)
            rephased_segment = np.fft.ifft(np.abs(fft_two) * np.exp(1j * initial_phase))

            # Accumulate the processed segment
            result_idx = int(idx / stretch_factor)
            stretched_frames[result_idx: result_idx + window_len] += window * rephased_segment

        # Normalize to 16-bit range
        stretched_frames = (2**(16-4) * stretched_frames / np.max(np.abs(stretched_frames)))

        return np.real(stretched_frames).astype('int16')


    def pitch_mode(self):
        modes = ["INTERP", "LIBROSA", "FFT"]
        current_index = modes.index(self.pitch_changing_mode)
        next_index = (current_index + 1) % len(modes)
        self.pitch_changing_mode = modes[next_index]
        self.pitch_changing_mode_button.config(text=modes[next_index])

    def change_plot_style(self):
        if self.plot_style == 4:
            self.plot_style = 1
        else:
            self.plot_style += 1

    def change_speed(self, speed, frames):
        arr = np.frombuffer(b''.join(frames), dtype=np.int16)
        new_length = int(len(arr) / speed)
        new_arr = np.zeros(new_length, dtype=np.float64)
        win_size = self.chunk_size * 8
        if self.speed_changing_mode == "PV-TSM":
            y = arr.astype(np.float32)
            # see the procedure of PV-TSM in self.stretch()
            new_arr = self.stretch(y, speed, win_size, win_size//4)
        else:
            # OLA and WSOLA
            hs = win_size // 2
            ha = int(speed * hs)
            dmax = win_size // 8 # dmax cannot be too large, or 0.5x will produce noise.
            # calculate the Hann window
            hanning_window = [0] * win_size
            for i in range(win_size):
                hanning_window[i] = 0.5 - 0.5 * math.cos(2 * math.pi * i /(win_size - 1))
            old_pos = 0
            new_pos = 0
            delta = 0
            while old_pos + delta < len(arr) - win_size and new_pos < new_length - win_size:
                if self.speed_changing_mode == "OLA":
                    for i in range(win_size):
                        new_arr[new_pos+i] += arr[old_pos+i] * hanning_window[i]
                else:
                    for i in range(win_size):
                        new_arr[new_pos + i] += arr[old_pos + i + delta] * hanning_window[i]
                if self.speed_changing_mode == "WSOLA":
                    # enhanced part: WSOLA, find the position of the most similar frame within (old_pos+ha-dmax,old_pos+ha+dmax).
                    sp = max(0, old_pos + ha - dmax)
                    ep = min(len(arr), old_pos + ha + dmax + win_size)
                    # metric of similarity: cross-correlation
                    correlations = np.correlate(arr[sp:ep], arr[old_pos + delta + hs:old_pos + delta + hs + win_size])
                    max_idx, max_value = 0, correlations[0]
                    for i, value in enumerate(correlations):
                        if value > max_value:
                            max_idx, max_value = i, value
                    delta = max_idx - min(dmax, old_pos + ha)
                # update new_pos and old_pos
                new_pos += hs
                old_pos += ha
        new_arr = new_arr.astype(np.int16)
        tobytes = new_arr.tobytes()
        bytes_arr = [tobytes[i:i+4096] for i in range(0, len(tobytes), 4096)]
        return bytes_arr

    def speed_mode(self):
        if self.speed_changing_mode == "OLA":
            self.speed_changing_mode = "WSOLA"
            self.speed_changing_mode_button.config(text="WSOLA")
        elif self.speed_changing_mode == "WSOLA":
            self.speed_changing_mode = "PV-TSM"
            self.speed_changing_mode_button.config(text="PV-TSM")
        else:
            self.speed_changing_mode = "OLA"
            self.speed_changing_mode_button.config(text="OLA")

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
            tk.messagebox.showinfo(title='Text', message='The text of the audio: ' + text) 
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
        try:
            reduced_noise_audio = nr.reduce_noise(y=filtered_audio_data, sr=self.audio_sampling_rate)
        except Exception as e:
            print(f'Error in noise reduction')
            print(e)
            return

        old_filename = self.selected_filename.split('.')[0]
        new_filename = old_filename + '_denoised.wav'
        save_path = os.path.join(self.save_dir, new_filename)
        # filename already exists
        audio_id = 0
        while os.path.exists(save_path):
            audio_id += 1
            new_filename = old_filename + f'_denoised({audio_id}).wav'
            save_path = os.path.join(self.save_dir, new_filename)

        # sf.write(output_filename, reduced_noise_audio, self.audio_sampling_rate)
        '''reduced_noise_audio = ReadWrite.waveform_to_frames(reduced_noise_audio)
        ReadWrite.write_wav(reduced_noise_audio, save_path, self.audio_sampling_rate, 2, 2)'''
        self.audio.loadWaveForm(reduced_noise_audio, self.audio_sampling_rate, self.channels, self.bytes_per_sample)
        self.audio.write(save_path)
        print(f"Noise-reduced audio saved as {save_path}.")

        # Update listbox
        self.load_all_recordings()

        # Reload the replaced audio
        self.load_selected_audio(filename=new_filename)
        self.setup_buttons_for_playable_state()

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
        '''waveform = ReadWrite.waveform_to_frames(self.audio_array[wav_start_idx:wav_end_idx], sample_rate=self.audio_sampling_rate)
        ReadWrite.write_wav(waveform, save_path, rate=self.audio_sampling_rate, channels=self.channels)'''
        self.audio.loadWaveForm(self.audio_array[wav_start_idx:wav_end_idx], self.audio_sampling_rate, self.channels, self.bytes_per_sample)
        self.audio.write(save_path)

        # Update the listbox
        self.load_all_recordings()

        # Reload the replaced audio
        self.load_selected_audio(filename=new_filename)
        self.setup_buttons_for_playable_state()


