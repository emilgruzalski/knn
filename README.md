# KNN Visualization Tool

## Description
This is a visualization tool for the k-nearest neighbors (KNN) algorithm with a graphical user interface created using `tkinter`. It allows users to load data, normalize it, and classify new points based on the chosen metric and voting method.

## Features
- Load data from a CSV file
- Normalize data to the range [0, 1]
- Select the number of nearest neighbors (k)
- Choose the metric (Euclidean or Manhattan)
- Choose the voting method (simple or weighted)
- Visualize points and classify in real-time

## Requirements
- Python 3.x
- `tkinter`
- `pandas`
- `numpy`
- `scipy`

## Installation
Ensure you have Python installed along with the libraries listed above. If not, you can install them using the following command:
```bash 
pip install pandas numpy scipy
```

## Usage
To run the tool, clone the repository and execute the `knn.py` file:
```bash
git clone https://github.com/emilgruzalski/knn.git cd knn python knn.py
```

## License
This project is released under the MIT License.
