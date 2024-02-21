import tkinter as tk

from sound_recorder import SoundRecorderApp
from configs import configs

if __name__ == '__main__':
    root = tk.Tk()
    app = SoundRecorderApp(root, **configs)
    root.mainloop()
