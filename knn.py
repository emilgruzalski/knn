import tkinter as tk
import math
from tkinter import filedialog

class KNNClassifier:
    def __init__(self, k, distance_metric, voting_type):
        self.k = k
        self.distance_metric = distance_metric
        self.voting_type = voting_type
        self.training_data = []

    def load_training_data(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                x, y, category = map(float, line.strip().split(','))
                self.training_data.append((x, y, category))

    def normalize_data(self):
        min_x = min(data[0] for data in self.training_data)
        max_x = max(data[0] for data in self.training_data)
        min_y = min(data[1] for data in self.training_data)
        max_y = max(data[1] for data in self.training_data)

        for i in range(len(self.training_data)):
            x = (self.training_data[i][0] - min_x) / (max_x - min_x)
            y = (self.training_data[i][1] - min_y) / (max_y - min_y)
            self.training_data[i] = (x, y, self.training_data[i][2])

    def classify_point(self, point):
        distances = [(self.calculate_distance(point, data[:2]), data[2]) for data in self.training_data]
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]

        if self.voting_type == 'simple':
            votes = [neighbor[1] for neighbor in neighbors]
            majority_vote = max(set(votes), key=votes.count)
            return majority_vote
        elif self.voting_type == 'weighted':
            total_weighted_votes = sum(1 / (distance[0] ** 2) * distance[1] for distance in neighbors)
            return round(total_weighted_votes)

    def calculate_distance(self, point1, point2):
        if self.distance_metric == 'euclidean':
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        elif self.distance_metric == 'manhattan':
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

class KNNApp:
    def __init__(self, master):
        self.master = master
        self.master.title("KNN Classifier")
        self.canvas_width = 500
        self.canvas_height = 500
        self.canvas = tk.Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()
        self.knn_classifier = None
        self.clicked_points = []

        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self.master, text="Load Training Data", command=self.load_training_data)
        self.load_button.pack()

        self.k_label = tk.Label(self.master, text="Select k (1-20):")
        self.k_label.pack()
        self.k_entry = tk.Entry(self.master)
        self.k_entry.pack()

        self.metric_label = tk.Label(self.master, text="Select Distance Metric:")
        self.metric_label.pack()
        self.metric_var = tk.StringVar()
        self.metric_var.set('euclidean')
        self.metric_dropdown = tk.OptionMenu(self.master, self.metric_var, 'euclidean', 'manhattan')
        self.metric_dropdown.pack()

        self.voting_label = tk.Label(self.master, text="Select Voting Type:")
        self.voting_label.pack()
        self.voting_var = tk.StringVar()
        self.voting_var.set('simple')
        self.voting_dropdown = tk.OptionMenu(self.master, self.voting_var, 'simple', 'weighted')
        self.voting_dropdown.pack()

        self.classify_button = tk.Button(self.master, text="Classify", command=self.classify_point)
        self.classify_button.pack()

        self.canvas.bind("<Button-1>", self.click_event)

    def load_training_data(self):
        filename = filedialog.askopenfilename(title="Select Training Data File", filetypes=[("Text Files", "*.txt")])
        if filename:
            self.knn_classifier = KNNClassifier(1, 'euclidean', 'simple')
            self.knn_classifier.load_training_data(filename)
            self.knn_classifier.normalize_data()
            self.display_training_data()

    def display_training_data(self):
        self.canvas.delete("all")
        categories = set(data[2] for data in self.knn_classifier.training_data)
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for data in self.knn_classifier.training_data:
            x, y, category = data
            color = colors[int(category)]
            self.canvas.create_oval(x * self.canvas_width - 5, y * self.canvas_height - 5,
                                    x * self.canvas_width + 5, y * self.canvas_height + 5, fill=color)

    def click_event(self, event):
        x = event.x / self.canvas_width
        y = event.y / self.canvas_height
        category = self.knn_classifier.classify_point((x, y))
        self.clicked_points.append((x, y, category))
        self.display_clicked_points()

    def display_clicked_points(self):
        self.canvas.delete("lines")
        for point in self.clicked_points:
            x, y, category = point
            color = ['red', 'blue', 'green', 'yellow', 'purple', 'orange'][int(category)]
            self.canvas.create_rectangle(x * self.canvas_width - 5, y * self.canvas_height - 5,
                                         x * self.canvas_width + 5, y * self.canvas_height + 5, fill=color)

            if len(self.clicked_points) > 1:
                prev_x, prev_y, _ = self.clicked_points[-2]
                self.canvas.create_line(prev_x * self.canvas_width, prev_y * self.canvas_height,
                                        x * self.canvas_width, y * self.canvas_height, fill=color, tags="lines")

    def classify_point(self):
        if self.knn_classifier:
            k = int(self.k_entry.get())
            distance_metric = self.metric_var.get()
            voting_type = self.voting_var.get()
            self.knn_classifier.k = k
            self.knn_classifier.distance_metric = distance_metric
            self.knn_classifier.voting_type = voting_type
            self.display_clicked_points()

if __name__ == "__main__":
    root = tk.Tk()
    app = KNNApp(root)
    root.mainloop()
