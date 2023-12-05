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
        self.data = None  # Dane wczytane z pliku CSV
        self.norm_data = None  # Znormalizowane dane
        self.k = tk.IntVar(value=1)  # Zmienna przechowująca wartość parametru k
        self.metric = tk.StringVar(value='euclidean')  # Zmienna przechowująca rodzaj metryki (euclidean lub manhattan)
        self.vote = tk.StringVar(value='simple')  # Zmienna przechowująca rodzaj głosowania (simple lub weighted)
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']  # Kolory punktów na płaszczyźnie
        self.points = []  # Lista punktów na płaszczyźnie
        self.last_point = None  # Ostatnio dodany punkt
        self.neighbors = []  # Lista sąsiadów punktu, używana do ich wyświetlania na płaszczyźnie

        # Tworzenie suwaka do ustawiania wartości k
        self.k_slider = tk.Scale(self.master, from_=1, to=20, orient='horizontal', variable=self.k)
        self.k_slider.pack()
        
        # Tworzenie menu do wyboru metryki
        self.metric_menu = tk.OptionMenu(self.master, self.metric, 'euclidean', 'manhattan')
        self.metric_menu.pack()
        
        # Tworzenie menu do wyboru rodzaju głosowania
        self.vote_menu = tk.OptionMenu(self.master, self.vote, 'simple', 'weighted')
        self.vote_menu.pack()
        
        # Tworzenie przycisku do wczytywania danych
        self.load_button = tk.Button(self.master, text='Wczytaj dane', command=self.load_data)
        self.load_button.pack()

    def load_data(self):
        # Wybieranie pliku z danymi za pomocą okna dialogowego
        filename = filedialog.askopenfilename()
        
        # Wczytywanie danych z pliku CSV przy użyciu biblioteki pandas
        self.data = pd.read_csv(filename, header=None)
        
        # Normalizacja danych na przedział [0, 1]
        self.norm_data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        
        # Rysowanie punktów na płaszczyźnie
        self.draw_points()

    def draw_points(self):
        # Czyszczenie płaszczyzny
        self.canvas.delete('all')
        
        # Rysowanie punktów na płaszczyźnie
        for i in range(len(self.norm_data)):
            x = self.norm_data.iloc[i, 0] * 550 + 25
            y = self.norm_data.iloc[i, 1] * 550 + 25
            color = self.colors[self.data.iloc[i, 2]]
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color)
        
        # Przypisanie funkcji do zdarzenia kliknięcia lewym przyciskiem myszy
        self.canvas.bind('<Button-1>', self.classify_point)

    def classify_point(self, event):
        # Wyznaczenie współrzędnych punktu na podstawie pozycji kursora
        x = (event.x - 25) / 550
        y = (event.y - 25) / 550
        
        # Utworzenie serii danych dla punktu
        point = pd.Series([x, y])
        
        # Obliczenie odległości między punktem a pozostałymi danymi
        if self.metric.get() == 'euclidean':
            dists = self.norm_data.iloc[:, :2].apply(lambda row: distance.euclidean(row, point), axis=1)
        else:
            dists = self.norm_data.iloc[:, :2].apply(lambda row: distance.cityblock(row, point), axis=1)
        
        # Wybór k najbliższych sąsiadów
        nearest = dists.nsmallest(self.k.get())
        
        # Głosowanie na klasę punktu w zależności od wybranej metody głosowania
        if self.vote.get() == 'simple':
            votes = self.data.loc[nearest.index, 2].value_counts()
        else:
            weights = 1 / nearest**2
            votes = self.data.loc[nearest.index, 2].groupby(self.data.loc[nearest.index, 2]).apply(lambda x: (x * weights).sum())
        
        # Wybór kategorii na podstawie głosowania
        category = int(votes.idxmax())
        
        # Ustalenie koloru na podstawie wybranej kategorii
        color = self.colors[category]
        
        # Usunięcie poprzedniego punktu
        if self.last_point is not None:
            self.canvas.delete(self.last_point)
        
        # Rysowanie nowego punktu
        self.last_point = self.canvas.create_rectangle(event.x-5, event.y-5, event.x+5, event.y+5, fill=color)
        
        # Usunięcie poprzednich sąsiadów
        for neighbor in self.neighbors:
            self.canvas.delete(neighbor)
        
        # Wyczyszczenie listy sąsiadów
        self.neighbors = []
        
        # Rysowanie okręgów wokół sąsiadów i ich odległości
        for i in nearest.index:
            nx = self.norm_data.iloc[i, 0] * 550 + 25
            ny = self.norm_data.iloc[i, 1] * 550 + 25
            self.neighbors.append(self.canvas.create_oval(nx-7, ny-7, nx+7, ny+7, outline='black'))
            self.neighbors.append(self.canvas.create_text(nx, ny-10, text=f'{nearest[i]:.2f}', fill='black'))

# Inicjalizacja głównego okna Tkinter
root = tk.Tk()

# Utworzenie obiektu klasy KNN
knn = KNN(root)

# Uruchomienie pętli głównej programu Tkinter
root.mainloop()
