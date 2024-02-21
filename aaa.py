import numpy as np
import matplotlib.pyplot as plt
aaaa=1
audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
# Generate time axis
time = np.arange(0, len(audio_data)) / sample_rate

# Plot the audio data
plt.plot(time, audio_data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Audio Signal')
plt.grid(True)
plt.show()