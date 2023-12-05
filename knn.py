import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from scipy.spatial import distance

class KNN:
    def __init__(self, master):
        self.master = master
        self.canvas = tk.Canvas(self.master, width=600, height=600)
        self.canvas.pack()
        self.data = None
        self.norm_data = None
        self.k = tk.IntVar(value=1)
        self.metric = tk.StringVar(value='euclidean')
        self.vote = tk.StringVar(value='simple')
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        self.points = []
        self.last_point = None
        self.neighbors = []

        self.k_slider = tk.Scale(self.master, from_=1, to=20, orient='horizontal', variable=self.k)
        self.k_slider.pack()
        self.metric_menu = tk.OptionMenu(self.master, self.metric, 'euclidean', 'manhattan')
        self.metric_menu.pack()
        self.vote_menu = tk.OptionMenu(self.master, self.vote, 'simple', 'weighted')
        self.vote_menu.pack()
        self.load_button = tk.Button(self.master, text='Load data', command=self.load_data)
        self.load_button.pack()

    def load_data(self):
        filename = filedialog.askopenfilename()
        self.data = pd.read_csv(filename, header=None)
        self.norm_data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        self.draw_points()

    def draw_points(self):
        self.canvas.delete('all')
        for i in range(len(self.norm_data)):
            x = self.norm_data.iloc[i, 0] * 550 + 25
            y = self.norm_data.iloc[i, 1] * 550 + 25
            color = self.colors[self.data.iloc[i, 2]]
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color)
        self.canvas.bind('<Button-1>', self.classify_point)

    def classify_point(self, event):
        x = (event.x - 25) / 550
        y = (event.y - 25) / 550
        point = pd.Series([x, y])
        if self.metric.get() == 'euclidean':
            dists = self.norm_data.iloc[:, :2].apply(lambda row: distance.euclidean(row, point), axis=1)
        else:
            dists = self.norm_data.iloc[:, :2].apply(lambda row: distance.cityblock(row, point), axis=1)
        nearest = dists.nsmallest(self.k.get())
        if self.vote.get() == 'simple':
            votes = self.data.loc[nearest.index, 2].value_counts()
        else:
            weights = 1 / nearest**2
            votes = self.data.loc[nearest.index, 2].groupby(self.data.loc[nearest.index, 2]).apply(lambda x: (x * weights).sum())
        category = int(votes.idxmax())
        print(category)
        color = self.colors[category]
        if self.last_point is not None:
            self.canvas.delete(self.last_point)
        self.last_point = self.canvas.create_rectangle(event.x-5, event.y-5, event.x+5, event.y+5, fill=color)
        for neighbor in self.neighbors:
            self.canvas.delete(neighbor)
        self.neighbors = []
        for i in nearest.index:
            nx = self.norm_data.iloc[i, 0] * 550 + 25
            ny = self.norm_data.iloc[i, 1] * 550 + 25
            self.neighbors.append(self.canvas.create_oval(nx-7, ny-7, nx+7, ny+7, outline='black'))
            self.neighbors.append(self.canvas.create_text(nx, ny-10, text=f'{nearest[i]:.2f}', fill='black'))

root = tk.Tk()
knn = KNN(root)
root.mainloop()
