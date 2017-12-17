import tkinter as gui
from tkinter import messagebox
import liveCorrelate
import findlag
import numpy as np


top = gui.Tk()
top.geometry("400x100")

global_latency = gui.DoubleVar()
labeltext = gui.StringVar()
textlabel = gui.Label(top, textvariable=labeltext)
textlabel.pack()

def latencyButtonCallback():
    respLatency = messagebox.askokcancel("AudioLocate","Place microphone at speaker")
    if(respLatency):
        global_latency.set(liveCorrelate.get_latency())
        labeltext.set(str(global_latency.get()))
        
def measureButtonCallback():
    #respmeasure = messagebox.askokcancel("AudioLocate","Place microphone away from speaker")
    #if(respmeasure):
        lat = global_latency.get()
        print(lat)
        timediff, distance = liveCorrelate.measure(global_latency.get())
        print(timediff,"s, ", distance, "m")
        tmp = np.load('micdata')
        print(tmp)
        
latencyButton = gui.Button(top, text="test latency", command=latencyButtonCallback)
latencyButton.pack()

measurelatencyButton = gui.Button(top, text="test measure", command=measureButtonCallback)
measurelatencyButton.pack()

top.mainloop()