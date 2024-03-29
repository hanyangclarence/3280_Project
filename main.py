import tkinter as tk
from advanced_features import FullSoundRecorder
from configs import configs


def on_closing():
    if app.recording is True:
        app.stop_recording()
    root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = FullSoundRecorder(root, **configs)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
