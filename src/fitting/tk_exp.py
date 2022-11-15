import numpy as np
import tkinter as Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import pandas as pd
from scipy import interpolate
from os.path import join
from scipy.signal import hilbert

from lmfit.models import ExponentialModel
import argparse

from glob import glob


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)

class plotting(Tk.Frame):

    def __init__(self, parent, data_path, no_of_oscillators):
        self.index = 0
        self.length = len(data_path)
        self.data_path = data_path
        self.no_of_oscillators = no_of_oscillators
        self.initialize(self.index, parent)

    

    def initialize(self, index, parent):
        self.t, self.magnetizations, self.h_trans = self.read_data(self.data_path[index], self.no_of_oscillators)

        self.exp_model = ExponentialModel()
        self.params = self.exp_model.guess(self.h_trans[0], x=self.t)
        self.result_exp = self.exp_model.fit(self.h_trans[0], self.params, x=self.t)

        Tk.Frame.__init__(self, parent)
        self.fig = plt.Figure(dpi=200)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(bottom=0.25)

        self.ax.plot(self.t, self.h_trans[0], 'r')
        self.ax.plot(self.t, self.result_exp.best_fit, 'b')

        self.ax_time1 = self.fig.add_axes([0.2, 0.1, 0.6, 0.03])
        self.ax_time2 = self.fig.add_axes([0.2, 0.05, 0.6, 0.03])
        self.s_time1 = Slider(self.ax_time1, 'Lower Bound', 0, self.t[-1], valinit=0)
        self.s_time2 = Slider(self.ax_time2, 'Upper Bound', 0, self.t[-1], valinit=0)

        self.button_quit = Tk.Button(master=root, text='Quit', command=self.quit)
        self.button_quit.pack(anchor = "s", side = "left")

        self.button_next = Tk.Button(master=root, text='Next', command=self.next)
        self.button_next.pack(anchor = "s", side = "left")

        self.button_fit = Tk.Button(master=root, text='Fit', command=self.fit)
        self.button_fit.pack(anchor = "s", side = "left")

    def next(self):
        for widget in root.winfo_children():
            widget.destroy()

        self.index += 1

        if self.index == self.length:
            self.quit()
        else:
            self.initialize(self.index, root)
            self.s_time1.on_changed(self.update)
            self.s_time2.on_changed(self.update)
            root.mainloop()

    def fit(self):
        self.ax.lines.pop(1)
        pos1 = self.s_time1.val
        pos2 = self.s_time2.val

        self.t_pos1 = find_nearest(self.t, pos1)
        self.t_pos2 = find_nearest(self.t, pos2)

        t_bounded = self.t[self.t_pos1:self.t_pos2]
        m_bounded = self.h_trans[0][self.t_pos1:self.t_pos2]

        self.params = self.exp_model.guess(m_bounded, x=t_bounded)
        self.result_exp = self.exp_model.fit(m_bounded, self.params, x=t_bounded)

        self.ax.plot(t_bounded, self.result_exp.best_fit, 'b')

        self.fig.canvas.draw_idle()


    def update(self, val):
        self.pos1 = self.s_time1.val
        self.pos2 = self.s_time2.val

        self.t_pos1 = find_nearest(self.t, self.pos1)
        self.t_pos2 = find_nearest(self.t, self.pos2)

        if self.t_pos1 > self.t_pos2:
            self.t_pos1 = self.t_pos2

        # t_bounded = self.t[self.t_pos1:self.t_pos2]
        m_bounded = self.h_trans[0][self.t_pos1:self.t_pos2]
        self.ax.axis([self.t[self.t_pos1], self.t[self.t_pos2], 0, 1.1*m_bounded.max()])

    
        self.s_time1.ax.set_xlim(0, self.t[self.t_pos2])
        self.s_time2.ax.set_xlim(self.t[self.t_pos1], self.t[-1])
        self.fig.canvas.draw_idle()
        return

    def read_data(self, folder_path, no_of_oscillators):
        df = pd.read_csv(join(folder_path, 'table.txt'),sep='\t')
        t = df['# t (s)'].to_numpy()
        magnetizations = []
        for i in range(1, no_of_oscillators + 1):
            magnetizations.append(df['m.region{}y ()'.format(i)].to_numpy())

        mag_interpolated = []
        t_new = np.arange(t[0], t[-1], t[1] - t[0])

        for i in range(no_of_oscillators):
            f = interpolate.interp1d(t, magnetizations[i])
            mag_interpolated.append(f(t_new))


        hilbert_transformed = []
        for i in range(no_of_oscillators):
            envelop = np.abs(hilbert(mag_interpolated[i]))
            hilbert_transformed.append(envelop)

        return t_new, mag_interpolated, hilbert_transformed



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--no_of_oscillators', type=int)
    args = parser.parse_args()

    data_path = glob(join(args.data_path, '*/'))
    no_of_oscillators = 2

    root = Tk.Tk()
    root.wm_title('Plotting')
    app = plotting(root, data_path, no_of_oscillators)
    app.s_time1.on_changed(app.update)
    app.s_time2.on_changed(app.update)
    root.mainloop()

