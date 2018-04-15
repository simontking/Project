# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/	

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import urllib
import json

import pandas as pd
import numpy as np

import liveCorrelate
import findlag

labeltext = tk.StringVar()
global_latency = tk.DoubleVar()
LARGE_FONT = ("Verdana", 12)
style.use("ggplot")

f = Figure(figsize=(10, 6), dpi=100)
ax1 = f.add_subplot(311)
ax2 = f.add_subplot(312)
ax3 = f.add_subplot(313)
ax1.set_title('Correlation')
ax2.set_title('Recorded Data')
ax3.set_title('mic, aligned')
with open('corrdata', 'wb') as file:
    np.save(file, np.zeros(1))
with open('micdata', 'wb') as file:
    np.save(file, np.zeros(1))
with open('scdata', 'wb') as file:
    np.save(file, np.zeros(1))
with open('sampledata', 'wb') as file:
    np.save(file, np.zeros(1))


def animate(i):
    acor1 = np.load('corrdata')
    firstSample = np.load('micdata')
    secondSample = np.load('scdata')
    sample = np.load('sampledata')

    ax1.clear()
    ax1.plot(np.arange(0, len(acor1)), acor1)

    # ax1.plot(np.arange(0,len(acor2)),acor2)
    t0 = np.arange(0, len(sample)) / 44100;
    t1 = np.arange(0, len(firstSample)) / 44100;
    t2 = np.arange(0, len(secondSample)) / 44100;
    ax2.clear()
    ax2.plot(t1, firstSample)
    ax2.plot(t2, secondSample)
    ax2.plot(t0, sample)

    I = np.argmax(acor1) - (len(sample) - 1)
    Dataal1 = firstSample[I:]
    t1al = np.arange(0, len(Dataal1)) / 44100
    # I = np.argmax(acor2)- (len(sample)-1)
    # Dataal2 = secondSample[I:]
    # t2al = np.arange(0,len(Dataal2))/44100
    ax3.clear()
    ax3.plot(t1al, Dataal1)
    # ax3.plot(t2al,Dataal2)
    ax3.plot(t0, sample)


def toaButtonCallback():
    liveCorrelate.simple_transceiver()


def measureButtonCallback():
    timediff, distance = liveCorrelate.measure(global_latency.get())
    print(timediff, "s, ", distance, "m")


class AudioLocate(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Audio Locate")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, Graph_Page):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text=("""Start Page
        Put in some config options
        and that"""), font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Graphs",
                             command=lambda: controller.show_frame(Graph_Page))
        button1.pack()

        button2 = ttk.Button(self, text="Quit",
                             command=quit)
        button2.pack()


class Graph_Page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        latencyLabel = tk.Label(self, text="Latency:", font=LARGE_FONT)
        latencyLabel.pack()
        latencyValue = tk.Label(self, textvariable=labeltext)
        latencyValue.pack()
        label = tk.Label(self, text="Correlation Results", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                             command=lambda: controller.show_frame(StartPage))
        button1.pack()
        toaButton = ttk.Button(self, text="Time of Arrival", command=toaButtonCallback)
        toaButton.pack()

        measureTDOAButton = ttk.Button(self, text="Time Diff of Arrival", command=measureButtonCallback)
        measureTDOAButton.pack()

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = AudioLocate()
ani = animation.FuncAnimation(f, animate, interval=1000)
app.mainloop()
