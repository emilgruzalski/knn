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
        self.data = None  # Data loaded from CSV file
        self.norm_data = None  # Normalized data
        self.k = tk.IntVar(value=1)  # Variable to store the value of the k parameter
        self.metric = tk.StringVar(value='euclidean')  # Variable to store the metric type (euclidean or manhattan)
        self.vote = tk.StringVar(value='simple')  # Variable to store the type of voting (simple or weighted)
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']  # Colors of points on the plane
        self.points = []  # List of points on the plane
        self.last_point = None  # Last added point
        self.neighbors = []  # List of neighbors for a point, used to display them on the plane

        # Creating a slider to set the value of k
        self.k_slider = tk.Scale(self.master, from_=1, to=20, orient='horizontal', variable=self.k)
        self.k_slider.pack()
        
        # Creating a menu to choose the metric
        self.metric_menu = tk.OptionMenu(self.master, self.metric, 'euclidean', 'manhattan')
        self.metric_menu.pack()
        
        # Creating a menu to choose the type of voting
        self.vote_menu = tk.OptionMenu(self.master, self.vote, 'simple', 'weighted')
        self.vote_menu.pack()
        
        # Creating a button to load data
        self.load_button = tk.Button(self.master, text='Load Data', command=self.load_data)
        self.load_button.pack()

    def load_data(self):
        # Choosing a file with data using a dialog window
        filename = filedialog.askopenfilename()
        
        # Loading data from a CSV file using the pandas library
        self.data = pd.read_csv(filename, header=None)
        
        # Normalizing data to the range [0, 1]
        self.norm_data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        
        # Drawing points on the plane
        self.draw_points()

    def draw_points(self):
        # Clearing the canvas
        self.canvas.delete('all')
        
        # Drawing points on the plane
        for i in range(len(self.norm_data)):
            x = self.norm_data.iloc[i, 0] * 550 + 25
            y = self.norm_data.iloc[i, 1] * 550 + 25
            color = self.colors[self.data.iloc[i, 2]]
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color)
        
        # Assigning a function to the left mouse button click event
        self.canvas.bind('<Button-1>', self.classify_point)

    def classify_point(self, event):
        # Determining the coordinates of the point based on the cursor position
        x = (event.x - 25) / 550
        y = (event.y - 25) / 550
        
        # Creating a data series for the point
        point = pd.Series([x, y])
        
        # Calculating the distance between the point and the other data points
        if self.metric.get() == 'euclidean':
            dists = self.norm_data.iloc[:, :2].apply(lambda row: distance.euclidean(row, point), axis=1)
        else:
            dists = self.norm_data.iloc[:, :2].apply(lambda row: distance.cityblock(row, point), axis=1)
        
        # Selecting the k nearest neighbors
        nearest = dists.nsmallest(self.k.get())
        
        # Voting for the class of the point depending on the selected voting method
        if self.vote.get() == 'simple':
            votes = self.data.loc[nearest.index, 2].value_counts()
        else:
            weights = 1 / nearest**2
            votes = self.data.loc[nearest.index, 2].groupby(self.data.loc[nearest.index, 2]).apply(lambda x: (x * weights).sum())
        
        # Selecting the category based on the voting
        category = int(votes.idxmax())
        
        # Determining the color based on the selected category
        color = self.colors[category]
        
        # Deleting the previous point
        if self.last_point is not None:
            self.canvas.delete(self.last_point)
        
        # Drawing the new point
        self.last_point = self.canvas.create_rectangle(event.x-5, event.y-5, event.x+5, event.y+5, fill=color)
        
        # Deleting the previous neighbors
        for neighbor in self.neighbors:
            self.canvas.delete(neighbor)
        
        # Clearing the list of neighbors
        self.neighbors = []
        
        # Drawing circles around neighbors and their distances
        for i in nearest.index:
            nx = self.norm_data.iloc[i, 0] * 550 + 25
            ny = self.norm_data.iloc[i, 1] * 550 + 25
            self.neighbors.append(self.canvas.create_oval(nx-7, ny-7, nx+7, ny+7, outline='black'))
            self.neighbors.append(self.canvas.create_text(nx, ny-10, text=f'{nearest[i]:.2f}', fill='black'))

# Initializing the main Tkinter window
root = tk.Tk()

# Creating an instance of the KNN class
knn = KNN(root)

# Running the main Tkinter event loop
root.mainloop()
