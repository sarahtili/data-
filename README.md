import numpy as np
import cv2
import plotly.express as px
import pandas as pd

# --- 1. Simuler des correspondances de keypoints entre deux images ---
num_points = 100
np.random.seed(42)

# Points 2D détectés dans l'image 1 et l'image 2 (coordonnées pixel)
kp1 = np.random.uniform(100, 500, (num_points, 2))  # Image 1
kp2 = kp1 + np.random.normal(0, 5, (num_points, 2))  # Image 2 (perturbée)

# Matrices de projection caméra (simplifiées)
P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # Caméra 1
P2 = np.array([[1, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]])  # Caméra 2 (décalée)

# --- 2. Reconstruction des points 3D avec triangulation ---
points_4d = cv2.triangulatePoints(P1, P2, kp1.T, kp2.T)
points_3d = points_4d[:3] / points_4d[3]  # Normalisation homogène

# --- 3. Visualisation du nuage de points 3D ---
df = pd.DataFrame({'X': points_3d[0], 'Y': points_3d[1], 'Z': points_3d[2]})
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color=df['Z'], title='3D Point Cloud Reconstruction')
fig.show()
