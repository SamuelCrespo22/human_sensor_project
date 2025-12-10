# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:56:51 2025

@authors: Miguel & Samuel
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from matplotlib.lines import Line2D
from scipy import stats
from scipy.signal import welch
from scipy.integrate import cumulative_trapezoid
from scipy.fft import fft
from sklearn.model_selection import KFold


FIGURES_PATH = "./figures"
DATASET_BASEPATH = "./FORTH_TRACE_DATASET"
NUMBER_OF_PARTICIPANTS = 15
NUMBER_OF_DEVICES = 5
BODY_SENSOR_POSITIONS = {
    1: "Left Wrist",
    2: "Right Wrist",
    3: "Chest",
    4: "Right Upper Leg",
    5: "Left Lower Leg"
}
CSV_COLUMNS = {
    1: "Device ID",
    2: "Accelerometer X",
    3: "Accelerometer Y",
    4: "Accelerometer Z",
    5: "Gyroscope X",
    6: "Gyroscope Y",
    7: "Gyroscope Z",
    8: "Magnetometer X",
    9: "Magnetometer Y",
    10: "Magnetometer Z",
    11: "Timestamp",
    12: "Activity Label"
}
IMU_SENSORS = ["Accelerometer", "Gyroscope", "Magnetometer"]
ACTIVITIES = {
    1: "Stand",
    2: "Sit",
    3: "Sit and Talk",
    4: "Walk",
    5: "Walk and Talk",
    6: "Climb Stair (up/down)",
    7: "Climb Stair (up/down) and talk",
    8: "Stand -> Sit",
    9: "Sit -> Stand",
    10: "Stand -> Sit and talk",
    11: "Sit -> Stand and talk",
    12: "Stand -> Walk",
    13: "Walk -> Stand",
    14: "Stand -> Climb Stairs (up/down), Stand -> Climb Stairs (up/down) and talk",
    15: "Climb Stairs (up/down) -> Walk",
    16: "Climb Stairs (up/down) and talk -> Walk and talk"
}

# ============================
# FUNÇÕES AUXILIARES
# ============================

def load_participant_data(participant_id: int, to_numpy: bool = True):
    participant_path = os.path.join(DATASET_BASEPATH, f"part{participant_id}")

    if not os.path.isdir(participant_path):
        raise ValueError(f"Directory does not exist: '{participant_path}'")

    df_list = []

    for device_id in range(1, NUMBER_OF_DEVICES + 1):
        csv_file_path = os.path.join(participant_path, f"part{participant_id}dev{device_id}.csv")

        if not os.path.isfile(csv_file_path):
            print(f"File not found. '{csv_file_path}'")
            continue

        df_list.append(pd.read_csv(csv_file_path, header=None, names=CSV_COLUMNS.values()))

    if not df_list:
        raise ValueError(f"No file was found for participant {participant_id}")

    participant_data_df = pd.concat(df_list, ignore_index=True)

    return participant_data_df.to_numpy(dtype=float) if to_numpy else participant_data_df

def load_all_participants_data():
    participants_data = []
    
    for i in range(NUMBER_OF_PARTICIPANTS):
        participant_data_df = load_participant_data(i, False)
        
        if participant_data_df is None:
            print(f"Could not load data from participant with id {i}")
            continue
            
        participants_data.append(participant_data_df)
    
    if not participants_data:
        raise ValueError("No CSV file found in the provided directory.")
    
    return pd.concat(participants_data, ignore_index=True)

def compute_modules(df: pd.DataFrame):
    # t⃗  =  (tₓ, tᵧ, t𝓏)
    # || t⃗ ||  =  √(tₓ² + tᵧ² + t𝓏²)
    df["Accelerometer Module"] = np.sqrt(df["Accelerometer X"]**2 + df["Accelerometer Y"]**2 + df["Accelerometer Z"]**2)
    df["Gyroscope Module"] = np.sqrt(df["Gyroscope X"]**2 + df["Gyroscope Y"]**2 + df["Gyroscope Z"]**2)
    df["Magnetometer Module"] = np.sqrt(df["Magnetometer X"]**2 + df["Magnetometer Y"]**2 + df["Magnetometer Z"]**2)
    return df

def iqr_outlier_count(data, whis: float = 1.5):
    data = data.dropna().astype(float)
    arr = data.to_numpy()
    
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower = q1 - whis * iqr
    upper = q3 + whis * iqr
    
    outliers = arr[(arr < lower) | (arr > upper)]
    return outliers.size

def plot_outliers_scatter(x, y, outlier_mask, title, save_path: str = None):
    plt.figure(figsize=(12, 4))
    plt.scatter(x, y, color='blue', label='Normal')
    plt.scatter(x[outlier_mask], y[outlier_mask], color='red', label='Outlier')
    plt.title(title)
    plt.ylabel('Magnitude')
    plt.xticks([])
    plt.legend()
    
    if save_path is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(
            FIGURES_PATH,
            f"{save_path}_{timestamp}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    
def plot_outliers_grid_subplots(
        grid_data,
        row_labels,
        col_labels,
        xlabel,
        ylabel,
        save_path: str = None
):    
    n_rows = len(grid_data)
    n_cols = len(grid_data[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            cell_data = grid_data[row_idx][col_idx]
            magnitudes_list = cell_data['magnitudes']
            outlier_masks_list = cell_data['outlier_masks']
            activity_ids = cell_data['activity_ids']

            for mag, mask, act_id in zip(magnitudes_list, outlier_masks_list, activity_ids):
                ax.scatter([act_id] * len(mag[~mask]), mag[~mask], color='blue', label='Normal' if act_id == activity_ids[0] else "")
                ax.scatter([act_id] * len(mag[mask]), mag[mask], color='red', label='Outlier' if act_id == activity_ids[0] else "")

            ax.set_title(f"{col_labels[col_idx]} - {row_labels[row_idx]}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xticks(activity_ids)
            ax.set_xticklabels(activity_ids, rotation=45, ha='right')
            if row_idx == 0 and col_idx == 0:
                ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_plot(fig, save_path)

    plt.show()

def kmeans_outlier_count(data,
                         number_of_clusters, 
                         closest_clusters, 
                         centroids, 
                         distance_metric: str = 'Euclidean',
                         k: int = 3
):
    if data is None:
        raise ValueError("Invalid input: 'data' is None.")
        
    if distance_metric not in ['Euclidean', 'Manhattan']:
        raise ValueError(f"Invalid distance metric: {distance_metric}")
    
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data, dtype=float)
        except Exception as e:
            raise TypeError(f"Could not convert data to np.ndarray: {e}")
    
    results = []
    
    for i in range(number_of_clusters):
        mask = (closest_clusters == i)
        data_points = data[mask]
        
        if data_points.size == 0:
            results.append({
                "cluster_id": i,
                "data_points": data_points,
                "centroid": centroids[i],
                "distances": np.array([]),
                # "threshold": None,
                "outlier_indices": np.array([])
            })
            continue
        
        # Manhattan distance (L1)
        if distance_metric == 'Manhattan':
            
            # d(p,q) = Σ(i=1 to n) |pᵢ - qᵢ|
            distances = np.sum(np.abs(data_points - centroids[i]), axis=1)
        
        # Euclidean distance (L2)
        elif distance_metric == 'Euclidean':
            
            # d(p,q) = √[Σ(i=1 to n) (pᵢ - qᵢ) ** 2] = ||p - q||
            distances = np.linalg.norm(data_points - centroids[i], axis=1)
        
        # Z-Score (Queremos apenas o limite superior)
        # threshold_low = distances.mean() - k * distances.std()
        threshold_high = distances.mean() + k * distances.std()
        outliers = np.where(distances > threshold_high)[0]
        # outliers = np.where((distances < threshold_low) | (distances > threshold_high))[0]
        
        # zscore_results = zscore(variable_samples=distances, k=k)
        # outliers = zscore_results['indexes']
        
        results.append({
            "cluster_id": i,
            "data_points": data_points,
            "centroid": centroids[i],
            "distances": distances,
            # "threshold": threshold,
            "outlier_indices": outliers
        })
    
    return results

def kmeans_outlier_cluster_count(
        closest_clusters, 
        n_clusters, 
        total_points,
        method: str = 'Z-Score'):
    '''
    → Conta os pontos por cada Cluster.
    → O(s) Cluster(s) com menor número de pontos será(ão) considerado(s) ruído.
    '''
    
    if method not in ['Z-Score', 'Fraction', 'Relative']:
        print(f"[ERROR] Invalid method '{method}' for noise cluster detection.")
        return
    
    cluster_counts = np.bincount(closest_clusters, minlength=n_clusters)
    
    # print(f"[INFO] Cluster Count")
    
    if method == 'Relative':
        # Option 1: Percentagem dos pontos totais
        # nᵢ < f * N
        # onde:
        # N = Número total de pontos
        # nᵢ = Número de pontos no cluster i
        # f = threshold (e.g., 0.01 => 1%)
        f = 0.01
        outlier_clusters = np.where(cluster_counts < f * total_points)[0]
    
    elif method == 'Fraction':
        # Option 2: Média ou mediana do tamanho do cluster
        # nᵢ < α * n̄
        # n̄ = N / k       (Média do tamanho do cluster)
        # α = threshold (e.g., 0.1–0.3)
        # Alternativa com a mediana:
        # nᵢ < α * median(n₁, n₂, ..., nₖ)
        alfa = 0.1
        outlier_clusters = np.where(cluster_counts < alfa * cluster_counts.mean())[0]
    
    elif method == 'Z-Score':
        
        # Opção 3.
        # zᵢ = (nᵢ - n̄) / σ
        # Cluster i é considerado noise se zᵢ < z_threshold (e.g., -1 ou -2)
        k = -1
        outlier_clusters = np.where(cluster_counts < cluster_counts.mean() - k * cluster_counts.std())[0]
        
    return {
        "method": method,
        "thresholds": f if method == 'Relative' else alfa if method == 'Fraction' else k,
        "noise_clusters": outlier_clusters
        }     

def plot_cluster_outliers(results, title, xlabel, ylabel):
    plt.figure(figsize=(8, 5))
    
    colors = plt.cm.tab20.colors
    
    for cluster in results:
        data_points = cluster["data_points"]
        centroid = cluster["centroid"]
        outliers = cluster["outlier_indices"]
        cid = cluster["cluster_id"]
        
        color = colors[cid % len(colors)]
        
        if data_points.shape[1] == 1:
            x, y = data_points[:, 0], np.zeros(len(data_points))
            cx, cy = centroid[0], 0
        else:
            x, y = data_points[:, 0], data_points[:, 1]
            cx, cy = centroid[0], centroid[1]
        
        plt.scatter(x, y, c=[color], alpha=0.6, label=f'Cluster {cid}')
        
        # Outliers
        if len(outliers) > 0:
            plt.scatter(
                x[outliers],
                y[outliers],
                c='red',
                marker='x',
                s=100,
                label=f'Outliers {cid}'
            )
        
        # Centroids
        plt.scatter(
            cx, cy,
            s=200,
            c=[color],
            edgecolors='black',
            marker='X'
        )
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_kmeans_results_3d(imu_magnitude_data, 
                           closest_clusters, centroids,
                           kmeans_outliers, n_clusters, 
                           title, xlabel, ylabel, zlabel,
                           save_path: str = None):

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    #palette = plt.cm.tab20.colors
    # filtered_palette = [tuple(c) for c in palette if not (c[0] > 0.6 and c[0] > c[1] + 0.1 and c[0] > c[2] + 0.1)]
    
    palette = [
        "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#98df8a", "#ffbb78",
        "#c5b0d5", "#f7b6d2", "#c49c94", "#9edae5", "#393b79", "#637939",
        "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#6b6ecf", "#b5cf6b"
    ]
    colors = [palette[label % len(palette)] for label in closest_clusters]

    # Normal Points
    ax.scatter(
        imu_magnitude_data[:, 0],
        imu_magnitude_data[:, 1],
        imu_magnitude_data[:, 2],
        c=colors,
        alpha=0.25,
        s=20,
        zorder=1,
        depthshade=False
    )

    # Display Outliers
    outliers_already_labeled = False
    for cluster_info in kmeans_outliers:
        outlier_indices = cluster_info["outlier_indices"]
        if len(outlier_indices) > 0:
            ax.scatter(
                imu_magnitude_data[outlier_indices, 0],
                imu_magnitude_data[outlier_indices, 1],
                imu_magnitude_data[outlier_indices, 2],
                c="red",
                s=90,
                label="Outliers" if not outliers_already_labeled else None,
                edgecolors="black",
                linewidths=0.8,
                depthshade=False,
                zorder=3
            )
            outliers_already_labeled = True

    # Display Centroids
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c="black",
        s=200,
        marker="X",
        edgecolors="yellow",
        label="Centroids",
        depthshade=False,
        zorder=4
    )

    # Axis and Title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel, labelpad=25)
    ax.set_title(title)

    # Legends (clusters + centroids + outliers)
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=palette[i % len(palette)],
               label=f"Cluster {i}", markersize=7)
        for i in range(n_clusters)
    ]
    
    handles.append(Line2D([0], [0], marker="X", color="w", label="Centroids",
                          markerfacecolor="black", markersize=9))
    handles.append(Line2D([0], [0], marker="o", color="w", label="Outliers",
                          markerfacecolor="red", markersize=7))

    ax.legend(handles=handles, loc="best")

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range]) / 2.0
    
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    
    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])
    
    if save_path: save_plot(fig, save_path)
    
    plt.show()

def plot_kmeans_results_3d_grid_subplots(
    imu_magnitude_data,
    closest_clusters,
    centroids,
    kmeans_outliers,
    n_clusters,
    title,
    xlabel,
    ylabel,
    zlabel,
    save_path: str = None
):
    print("[INFO] Starting 3D K-Means subplot visualization...")
    fig = plt.figure(figsize=(18, 16))
    rows, cols = 4, 4
    palette = [
        "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#98df8a", "#ffbb78",
        "#c5b0d5", "#f7b6d2", "#c49c94", "#9edae5"
    ]

    print("[INFO] Preparing outlier index mapping...")
    
    # Precompute outliers per cluster
    cluster_outliers = {i: [] for i in range(n_clusters)}
    for cluster_info in kmeans_outliers:
        idx = cluster_info["cluster_id"]
        cluster_outliers[idx] = cluster_info["outlier_indices"]
    
    print(f"[INFO] Creating subplots grid ({rows}x{cols}) for {n_clusters} clusters...")
    for cluster_idx in range(n_clusters):
        ax = fig.add_subplot(rows, cols, cluster_idx + 1, projection="3d")

        # Cluster points
        cluster_points = imu_magnitude_data[closest_clusters == cluster_idx]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=palette[cluster_idx % len(palette)],
            alpha=0.35,
            s=20,
            depthshade=False
        )

        # Outliers
        outlier_indices = cluster_outliers[cluster_idx]
        if len(outlier_indices) > 0:
            outlier_points = imu_magnitude_data[outlier_indices]
            ax.scatter(
                outlier_points[:, 0],
                outlier_points[:, 1],
                outlier_points[:, 2],
                c="red",
                s=15,
                edgecolors="black",
                linewidths=0.6,
                depthshade=False
            )

        # Centroid
        centroid = centroids[cluster_idx]
        ax.scatter(
            centroid[0],
            centroid[1],
            centroid[2],
            c="black",
            marker="X",
            s=100,
            edgecolors="yellow",
            depthshade=False
        )

        ax.set_title(f"Cluster {cluster_idx}", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)

        # Equal aspect
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        max_range = max(
            abs(x_limits[1] - x_limits[0]),
            abs(y_limits[1] - y_limits[0]),
            abs(z_limits[1] - z_limits[0])
        ) / 2.0
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
        ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
        ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

    print("[INFO] Finalizing layout and rendering figure...")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.15) 

    if save_path:
        save_plot(fig, save_path)
    
    print("[INFO] Displaying figure...")
    plt.show()
    print("[INFO] Plot generation complete.")

def save_plot(fig, save_path: str):
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(FIGURES_PATH, f"{save_path}_{timestamp}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        print(f"[INFO] Figure saved successfully: {save_path}\n")
    except Exception as e:
        print(f"[ERROR] There was an error saving the figure: {e}\n")

def linear_regression_predict(x_input, weights):
    x_input = np.hstack(([1], x_input))
    return float(np.dot(x_input, weights))

def cross_validation_error(X, Y, k_folds=5):
    kf = KFold(n_splits=k_folds, shuffle=False)
    errors = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]

        weights = linear_regression(X_train, Y_train)
        Y_pred = np.array([linear_regression_predict(x, weights) for x in X_val])
        
        # Slide 93
        # εᵢ = yᵢ − β₀ + β₁xᵢ,₁ + β₂xᵢ,₂ + ... + βₚxᵢ,ₚ = yᵢ - ŷᵢ
        errors.extend(Y_val - Y_pred)

    errors = np.array(errors)
    mse = np.mean(errors ** 2) # Mean Squared Error = (1/n) * Σ (yᵢ - ŷᵢ)²
    mae = np.mean(np.abs(errors)) # Mean Absolute Error = (1/n) * Σ |yᵢ - ŷᵢ|
    return errors, mse, mae

# ============================
# ROTINAS
# ============================

def plot_activities_boxplots_solution1(imu_sensor: str, save_fig: bool = False):
    plt_data = [
        all_participants_data_df.loc[
            all_participants_data_df["Activity Label"] == key, 
            f"{imu_sensor} Module"
        ].values for key in ACTIVITIES.keys()
    ]
    
    x = [id for id in ACTIVITIES.keys()]
    y = plt_data
    
    plt.boxplot(
        y, 
        tick_labels = x, 
        vert=False,
        flierprops=dict(marker='o', markersize=3, markerfacecolor='red', alpha=0.5)
    )
    plt.title(f'{imu_sensor} Magnitude per Activity')
    plt.xlabel(f'{imu_sensor} Magnitude')
    plt.ylabel('Activity')
    plt.tight_layout()
    
    if save_fig:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = os.path.join(
            FIGURES_PATH,
            f"Exercise-3.1-solution1/{imu_sensor}_Magnitude_per_Activity_{timestamp}.png"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
    plt.show()
    
def plot_activities_boxplots_solution2(imu_sensor: str, save_fig: bool = False):    
    for sensor_position_id in BODY_SENSOR_POSITIONS.keys():
        
        plt_data = []
        for activity_id in ACTIVITIES.keys():
            plt_data.append(all_participants_data_df.loc[
                (all_participants_data_df["Activity Label"] == activity_id) & 
                (all_participants_data_df["Device ID"] == sensor_position_id),
                f"{imu_sensor} Module"
            ])
        
        x = [id for id in ACTIVITIES.keys()]
        y = plt_data
        
        plt.boxplot(
            y, 
            tick_labels = x, 
            vert=False,
            flierprops=dict(marker='o', markersize=3, markerfacecolor='red', alpha=0.5)
        )
        plt.title(f'{imu_sensor} Magnitude in {BODY_SENSOR_POSITIONS[sensor_position_id]} per Activity')
        plt.xlabel('Magnitude')
        plt.ylabel('Activity')
        plt.tight_layout()
        
        if save_fig:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FIGURES_PATH,
                f"Exercise-3.1-solution2/{imu_sensor}_Magnitude_in_{BODY_SENSOR_POSITIONS[sensor_position_id]}_per_Activity_{timestamp}.png"
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        
        plt.show()
    
def zscore(variable_samples, k: float = 3.0):
    if variable_samples is None:
        return
    
    variable_samples = np.asarray(variable_samples)

    # Z = (X - µ) / σ (Slide 67)
    mean = np.nanmean(variable_samples)
    std_dev = np.nanstd(variable_samples)

    if std_dev == 0 or np.isnan(std_dev):
        return
    
    z_scores = (variable_samples - mean) / std_dev
    out_idx = np.where(np.abs(z_scores) > k)[0]
    out_vals = variable_samples[out_idx] if out_idx.size else np.array([])

    return {
        'indexes': out_idx, 
        'values': out_vals, 
        'z_scores': z_scores, 
        'count': int(out_idx.size),
        'mask': np.abs(z_scores) > k
    }

def kmeans(data, number_of_clusters, initialization: str = 'Forgy Method', distance_metric: str = 'Euclidean'):
    if data is None:
        raise ValueError("Invalid input: 'data' is None.")
        
    if initialization not in ['Forgy Method', 'Random Partition']:
        raise ValueError(f"Invalid initialization method: {initialization}")
        
    if distance_metric not in ['Euclidean', 'Manhattan']:
        raise ValueError(f"Invalid distance metric: {distance_metric}")
    
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data, dtype=float)
        except Exception as e:
            raise TypeError(f"Could not convert data to np.ndarray: {e}")
    
    n_samples, _ = data.shape
    
    # (Slides 74, 75 e 76 PDF)
    if initialization == 'Forgy Method':
        centroids = data[np.random.choice(n_samples, number_of_clusters, replace=False)]
    elif initialization == 'Random Partition':
        labels = np.random.randint(0, number_of_clusters, size=n_samples)
        centroids = np.array([
            data[labels == k].mean(axis=0) for k in range(number_of_clusters)
        ])

    while True:
        
        # Euclidean distance (L2)
        if distance_metric == 'Euclidean':
            
            # d(p,q) = √[Σ(i=1 to n) (pᵢ - qᵢ) ** 2] = ||p - q||
            distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        
        # Manhattan distance (L1)
        elif distance_metric == 'Manhattan':
            
            # d(p,q) = Σ(i=1 to n) |pᵢ - qᵢ|
            distances = np.abs(data[:, np.newaxis] - centroids).sum(axis=2)
        
        closest_clusters = np.argmin(distances, axis=1)
        new_centroids = []
        for k in range(number_of_clusters):
            
            cluster_points = data[closest_clusters == k]        
            if len(cluster_points) > 0:
                
                if distance_metric == 'Euclidean':
                    new_centroid = cluster_points.mean(axis=0)
                
                # K-Medians: Caso L1, atualizamos os centróides com a mediana em vez da média
                elif distance_metric == 'Manhattan':
                    new_centroid = np.median(cluster_points, axis=0)
            
            else: # Quando o cluster está vazio, reatribuímos aleatoriamente
                new_centroid = data[np.random.choice(n_samples)]
            
            new_centroids.append(new_centroid)
        
        new_centroids = np.array(new_centroids)
        
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return closest_clusters, centroids

def inject_outliers(data, x, k, z, random_state=None):
    if data is None:
        raise ValueError("Invalid input: 'data' is None.")    
    
    if not isinstance(data, np.ndarray):
        try:
            data = data.copy()
            data = np.array(data, dtype=float)
        except Exception as e:
            raise TypeError(f"Could not convert data to np.ndarray: {e}")
    
    rng = np.random.default_rng(random_state)
    
    mean, std = np.mean(data), np.std(data)
    
    # Thresholds [μ - k*σ, μ + k*σ]
    floor_threshold, ceiling_threshold = mean - k * std, mean + k * std
    
    outlier_mask = (data < floor_threshold) | (data > ceiling_threshold)
    outlier_count = np.sum(outlier_mask)
    outlier_density = (outlier_count / data.size) * 100
    
    print(f"Total Samples: {data.size}\n"
      f"Mean: {mean}\n"
      f"Std: {std}\n"
      f"Outliers Count: {outlier_count}\n"
      f"Outliers Density: {outlier_density:.2f}%")
    
    if outlier_density >= x:
        print(f"The density of outliers ({outlier_density:.2f}%) is already ≥ {x}%. No outliers injected.")
        return data
    
    num_outliers_to_inject = int(((x - outlier_density) / 100) * data.size)
    
    if num_outliers_to_inject <= 0:
        print("No additional outliers needed, as the density is already close to the target.")
        return data
    
    indexes_to_inject = rng.choice(
        np.where(~outlier_mask)[0], 
        size=num_outliers_to_inject,
        replace=False
    )
        
    for idx in indexes_to_inject:
        # p ⇽ μ + s * k * (σ + q)
        s = rng.choice([-1, 1])
        q = rng.uniform(0, z)
        data[idx] = mean + s * k * (std + q)
    
    outlier_mask = (data < floor_threshold) | (data > ceiling_threshold)
    outlier_count = np.sum(outlier_mask)
    outlier_density = (outlier_count / data.size) * 100
    
    print(f"\nk = {k} | z = {z}\n"
          f"Injected {num_outliers_to_inject} outliers\n"
          f"New Outlier Count: {outlier_count}\n"
          f"Target density: {x:.2f}% => Final Density: {outlier_density:.2f}%")
    
    return data, outlier_mask

def linear_regression(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    n_samples, p = X.shape
    
    # Adicionamos esta coluna cheia de 1s para obter também o valor β₀
    X = np.hstack([np.ones((n_samples, 1)), X])  # shape (n_samples, p+1)
    
    # Least Squares Method (Slides 92,93,94)
    # β = (XᵀX)⁻¹ XᵀY = Pseudo-inverse @ Y
    # O método pinv() usa SVD 'under the hood'.
    return np.linalg.pinv(X) @ Y

def ks_test(data):
    statistic, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f"Statistic: {statistic:.4f}")
    print(f"P-value: {p_value:.4f}")
    return p_value

def spectral_entropy(signal, fs):
    f, psd = welch(signal, fs=fs, nperseg=min(128, len(signal)//2)) # boa resolução espectral
    psd_norm = psd / np.sum(psd)  # normaliza para distribuição de probabilidade
    return stats.entropy(psd_norm, base=2)

def calc_df(signal, fs):
    N = len(signal)
    y_fft = fft(signal)
    mag_squared = np.abs(y_fft[:N//2])**2          # apenas metade positiva
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]       # frequências positivas
    indice_max = np.argmax(mag_squared)
    df = freqs[indice_max]                          # frequência dominante
    return df

def energy_calc(signal, fs):
    N = len(signal)
    y_fft = fft(signal)
    mag_squared = np.abs(y_fft[1:N//2 + 1])**2
    potency_sum = np.sum(mag_squared)
    energy = potency_sum / N
    return energy

# ============================
# MAIN
# ============================

def main():
    global all_participants_data_df
    
    # Dados comuns em grande parte dos exercícios.
    all_participants_data_df = compute_modules(load_all_participants_data())

    exercises_to_run = {
        "Exercise 2": False,
        "Exercise 3.1": False,
            "Exercise 3.1 Solution 1": False,
            "Exercise 3.1 Solution 2": False,
        "Exercise 3.2": False,
            "Exercise 3.2 Solution 1": False,
            "Exercise 3.2 Solution 2": False,
        "Exercise 3.3": False,
        "Exercise 3.4": False,
            "Exercise 3.4 Solution 1": False,
            "Exercise 3.4 Solution 2": False,
            "Exercise 3.4 Solution 3": False,
            "Exercise 3.4 Solution 4": False,
        "Exercise 3.5": False,
        "Exercise 3.6": False,
        "Exercise 3.7": False,
            "Exercise 3.7 Solution 1": False,
            "Exercise 3.7 Solution 2": False,
            "Exercise 3.7 Solution 3": False,
            "Exercise 3.7 Solution 4": False,
            "Exercise 3.7.1": False, # TODO (Bónus)
        "Exercise 3.8": False,
        "Exercise 3.9": False,
        "Exercise 3.10": True, # TODO
        "Exercise 3.11": False, # TODO
        "Exercise 4.1": False,
        "Exercise 4.2": False,
        "Exercise 4.3": False,
        "Exercise 4.4": False,
        "Exercise 4.5": False,
        "Exercise 4.6": False
    }
    
    if exercises_to_run["Exercise 2"]:
        '''
        Elabore uma rotina que carregue os dados relativos a um indivíduo e os devolva num Array NumPy. 
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 2 ====\n\n")
        # ====================================================================================
        
        participant_data_np = load_participant_data(participant_id=5, to_numpy=True)
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
        
    if exercises_to_run["Exercise 3.1"]:
        ''' 
        Elabore uma rotina que apresente simultaneamente o boxplot de cada atividade
        (coluna 12 - eixo horizontal) relativo a todos os sujeitos e a uma das seguintes 
        variáveis transformadas:
        
            módulo do vector de aceleração;
            módulo do vector de giroscópio;
            módulo do vector de magnetómetro;
        
        Sugere-se o uso da biblioteca matplotlib. 
        '''

        if exercises_to_run["Exercise 3.1 Solution 1"]:
            '''
            Solução 1.
            → Filtra as linhas por atividade.
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.1 Solution 1 ====\n\n")
            # ====================================================================================
            
            
            plot_activities_boxplots_solution1(imu_sensor="Gyroscope", save_fig=True)
            
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
            
        if exercises_to_run["Exercise 3.1 Solution 2"]:
            '''
            Solução 2.
            → Filtra as linhas por atividade e posição do sensor.
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.1 Solution 2 ====\n\n")
            # ====================================================================================
            
            
            plot_activities_boxplots_solution2(imu_sensor="Accelerometer", save_fig=True)
                
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
                
    if exercises_to_run["Exercise 3.2"]:
        '''
        Analise e comente a densidade de Outliers existentes no dataset transformado, isto é,
        nos módulos dos vectores aceleração, giroscópio e magnetómetro para cada atividade.
        Observe que a densidade é determinada recorrendo:
        (Nr. de Pts Outliers / Nr. Total de Pontos) * 100
        '''
    
        if exercises_to_run["Exercise 3.2 Solution 1"]:
            '''
            Solução 1.
            → Filtra as linhas por atividade.
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.2 Solution 1 ====\n\n")
            # ====================================================================================
            
            
            for imu_sensor in IMU_SENSORS:
                
                print(f"\n\nOutlier Analysis for {imu_sensor} Module:\n")
                
                for activity_id in ACTIVITIES.keys():
                    filtered_data = all_participants_data_df.loc[
                        all_participants_data_df["Activity Label"] == activity_id, 
                        f"{imu_sensor} Module"
                    ]
                    
                    outlier_count = iqr_outlier_count(filtered_data)
                    total_count = filtered_data.size
                    outlier_density = (outlier_count / total_count) * 100 if total_count > 0 else 0
                    
                    print(f"Activity {activity_id} ({ACTIVITIES[activity_id]}):\n"
                          f"Total Samples = {total_count}\n"
                          f"Outliers = {outlier_count}\n"
                          f"Density = {outlier_density:.2f}%\n")
                
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
    
        if exercises_to_run["Exercise 3.2 Solution 2"]:
            '''
            Solução 2.
            → Filtra as linhas por atividade e por posição do sensor.
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.2 Solution 2 ====\n\n")
            # ====================================================================================
            
            
            for imu_sensor in IMU_SENSORS:
                for sensor_position_id in BODY_SENSOR_POSITIONS.keys():
                    
                    print(f"\n\nOutlier Analysis for {imu_sensor} Module:\n"
                         f"\nSensor Position: {BODY_SENSOR_POSITIONS[sensor_position_id]}\n")
                    
                    for activity_id in ACTIVITIES.keys():
                        filtered_data = all_participants_data_df.loc[
                            (all_participants_data_df["Activity Label"] == activity_id) &
                            (all_participants_data_df["Device ID"] == sensor_position_id), 
                            f"{imu_sensor} Module"
                        ]
                        
                        outlier_count = iqr_outlier_count(filtered_data)
                        total_count = filtered_data.size
                        outlier_density = (outlier_count / total_count) * 100 if total_count > 0 else 0
                        
                        print(f"Activity {activity_id} ({ACTIVITIES[activity_id]}):\n"
                              f"Total Samples = {total_count}\n"
                              f"Outliers = {outlier_count}\n"
                              f"Density = {outlier_density:.2f}%\n")
                
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
    
    if exercises_to_run["Exercise 3.3"]:
        '''
        Escreva uma rotina que receba um Array de amostras de uma variável e identifique 
        os outliers usando o teste Z-Score para um k variável (parâmetro de entrada).
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 3.3 ====\n\n")
        # ====================================================================================
        
        
        participant_id = 7
        participant_data_df = load_participant_data(participant_id, False)
        zscore_outliers = zscore(participant_data_df["Accelerometer X"], k=3)    
        
        print(f"\t\t\t\tZ-Score Results\n\n"
              f"Outliers Indexes: {zscore_outliers['indexes']}\n"
              f"Outliers Values: {zscore_outliers['values']}\n"
              f"Z-Scores: {zscore_outliers['z_scores']}\n"
              f"Number of Outliers: {zscore_outliers['count']}\n"
              f"Mask: {zscore_outliers['mask']}")
        
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
        
    if exercises_to_run["Exercise 3.4"]:
        '''
        Usando o Z-score implementado assinale todos as amostras consideradas outliers nos
        módulos dos vectores de aceleração, giroscópio e magnetómetro. 
        Apresente plots em que estes pontos surgem a vermelho enquanto os restantes surgem a 
        azul. Use k=3, 3.5 e 4.
        '''
    
        if exercises_to_run["Exercise 3.4 Solution 1"]:
            '''
            Solução 1.
            → Sem qualquer tipo de filtragem de linhas. (3 * 3 = 9 gráficos)
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.4 Solution 1 ====\n\n")
            # ====================================================================================
            
            
            for k in [3, 3.5, 4]:
                for imu_sensor in ["Accelerometer", "Gyroscope", "Magnetometer"]:
                    magnitude = all_participants_data_df[f"{imu_sensor} Module"]
                    zscore_outliers = zscore(magnitude, k)
                    plot_outliers_scatter(
                        x=np.arange(len(magnitude)),
                        y=magnitude,
                        outlier_mask=zscore_outliers['mask'],
                        title=f'{imu_sensor} Magnitude - Outliers Analysis with Z-Score (k = {k})',
                        # save_path=f"Exercise-3.4-solution1/{imu_sensor}_Magnitude_Z-Score_Outliers"
                    )
                    
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
                
        if exercises_to_run["Exercise 3.4 Solution 2"]:
            '''
            Solução 2.
            → Filtra as linhas por posição do sensor (Device ID)
            → (3 * 5 * 3 = 45 gráficos).
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.4 Solution 2 ====\n\n")
            # ====================================================================================
             
            
            for k in [3, 3.5, 4]:
                for device_id in range(1, NUMBER_OF_DEVICES + 1):
                    for imu_sensor in ["Accelerometer", "Gyroscope", "Magnetometer"]:
                        magnitude = all_participants_data_df.loc[
                            (all_participants_data_df["Device ID"] == device_id), 
                            f"{imu_sensor} Module"
                        ]
                        zscore_outliers = zscore(magnitude, k)
                        plot_outliers_scatter(
                            x=np.arange(len(magnitude)),
                            y=magnitude,
                            outlier_mask=zscore_outliers['mask'],
                            title=(f"{imu_sensor} Magnitude in {BODY_SENSOR_POSITIONS[device_id]}"
                                   f" - Outliers Analysis with Z-Score (k = {k})"),
                            # save_path=(f"Exercise-3.4-solution2/{imu_sensor}_Magnitude"
                            #            f"_in_{BODY_SENSOR_POSITIONS[device_id]}_Z-Score_Outliers")
                        )
                        
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
        
        if exercises_to_run["Exercise 3.4 Solution 3"]:
            '''
            Solução 3.
            → Filtra as linhas por posição do sensor (Device ID) e por atividade.
            → (3 * 5 * 3 * 16 = 720 gráficos).
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.4 Solution 3 ====\n\n")
            # ====================================================================================
            
            
            for k in [3, 3.5, 4]:
                for imu_sensor in ["Accelerometer", "Gyroscope", "Magnetometer"]:
                    for device_id, device_name in BODY_SENSOR_POSITIONS.items():
                        for activity_id, activity_name in ACTIVITIES.items():
                            magnitude = all_participants_data_df.loc[
                                        (all_participants_data_df["Device ID"] == device_id) & 
                                        (all_participants_data_df["Activity Label"] == activity_id), 
                                        f"{imu_sensor} Module"
                                    ]
                            
                            zscore_outliers = zscore(magnitude, k=k)
                            plot_outliers_scatter(
                                x=np.arange(len(magnitude)),
                                y=magnitude,
                                outlier_mask=zscore_outliers['mask'],
                                title=(f"{imu_sensor} Magnitude in {device_name} for {activity_name}"
                                       f" - Outliers Analysis with Z-Score (k = {k})"),
                                # save_path=(f"Exercise-3.4-solution3/{imu_sensor}_Magnitude"
                                #            f"_in_{device_name}_for_activity_{activity_id}"
                                #            "_Z-Score_Outliers")
                            )

            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
            
        if exercises_to_run["Exercise 3.4 Solution 4"]:
            '''
            Solução 4.
            → Filtra as linhas por posição do sensor (Device ID) e por atividade.
            → Faz um Plot agrupado com todos os Sensores, Posições e Atividades.
            → 3 gráficos, um para cada k.
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.4 Solution 4 ====\n\n")
            # ====================================================================================
            
            
            for k in [3, 3.5, 4]:
                grid_data = []
            
                for device_id, device_name in BODY_SENSOR_POSITIONS.items():
                    row_cells = []
                    for imu_sensor in IMU_SENSORS:
                        activity_data = []
            
                        for activity_id, activity_name in ACTIVITIES.items():
                            magnitude = all_participants_data_df.loc[
                                (all_participants_data_df["Device ID"] == device_id) & 
                                (all_participants_data_df["Activity Label"] == activity_id), 
                                f"{imu_sensor} Module"
                            ]
            
                            zscore_outliers = zscore(magnitude, k=k)
                            activity_data.append({
                                "magnitude": magnitude,
                                "outlier_mask": zscore_outliers['mask'],
                                "activity_id": activity_id
                            })
            
                        magnitudes = [item["magnitude"] for item in activity_data]
                        outlier_masks = [item["outlier_mask"] for item in activity_data]
                        activity_ids = [item["activity_id"] for item in activity_data]
            
                        row_cells.append({
                            "magnitudes": magnitudes,
                            "outlier_masks": outlier_masks,
                            "activity_ids": activity_ids
                        })
            
                    grid_data.append(row_cells)
            
                plot_outliers_grid_subplots(
                    grid_data=grid_data,
                    row_labels=list(BODY_SENSOR_POSITIONS.values()),
                    col_labels=IMU_SENSORS,
                    xlabel='Activity ID',
                    ylabel='Magnitude',
                    # save_path="Exercise-3.4-solution4/Z-Score_Outliers-General-Overview"
                )
                
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
    
    if exercises_to_run["Exercise 3.6"]:
        '''
        Elabore uma rotina que implemente o algoritmo K-means para n (valor de entrada) clusters.
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 3.6 ====\n\n")
        # ====================================================================================
        
        
        synthetic_data = [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [8.0, 8.0],
            [1.0, 0.6],
            [9.0, 11.0],
            [8.0, 2.0],
            [10.0, 2.0],
            [9.0, 3.0],
        ]
        
        clusters, centroids = kmeans(
            synthetic_data,
            number_of_clusters=3,
            initialization='Forgy Method',
            distance_metric='Euclidean'
        )
        
        print(f"Clusters: {clusters}")
        for i, centroid in enumerate(centroids):
            print(f"Centroid {i}: {centroid}")
        
        results = kmeans_outlier_count(
            synthetic_data, number_of_clusters=3, closest_clusters=clusters, centroids=centroids
        )
        
        plot_cluster_outliers(
            results, 'Clusters Analysis', 'X axis', 'Y axis'
        )
        
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
    
    if exercises_to_run["Exercise 3.7"]:
        ''' 
        Determine os outliers no dataset transformado usando o k-means.
        Experimente k clusters igual ao número de labels e compare com os resultados obtidos em 3.4. 
        Ilustre graficamente os resultados usando plots 3D.
        '''
    
        if exercises_to_run["Exercise 3.7 Solution 1"]:
            '''
            Solução 1.
            → K-Means é aplicado sobre os módulos combinados dos vetores de Aceleração, Giroscópio e Magnetómetro.
            → 1 Cluster por atividade = 16 Clusters
            → Sem qualquer tipo de filtragem de linhas.
            → X -> Módulo Acelerómetro
            → Y -> Módulo Giroscópio
            → Z -> Módulo Magnetómetro
            (1 gráfico)
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.7 Solution 1 ====\n\n")
            # ====================================================================================
    
            
            n_clusters = len(ACTIVITIES)
            kmeans_initialization = "Forgy Method"
            kmeans_distance_metric = 'Euclidean'

            imu_magnitude_data = all_participants_data_df[
                ['Accelerometer Module', 'Gyroscope Module', 'Magnetometer Module']
            ].values
            
            # (Slide 118) Z = (X - μ) / σ
            # Normalização por coluna: Zᵢ = (Xᵢ - μᵢ) / σᵢ
            imu_magnitude_data = (
                imu_magnitude_data - imu_magnitude_data.mean(axis=0)
            ) / imu_magnitude_data.std(axis=0)
            
            closest_clusters, centroids = kmeans(
                data=imu_magnitude_data, # (n_linhas, 3 cols)
                number_of_clusters=n_clusters,
                initialization=kmeans_initialization,
                distance_metric=kmeans_distance_metric
            )
            
            for cluster_id in range(n_clusters):
                cluster_points_indices = np.where(closest_clusters == cluster_id)[0]
                activity_labels_in_cluster = all_participants_data_df.iloc[
                    cluster_points_indices
                ]["Activity Label"].values
            
                unique_labels, counts = np.unique(activity_labels_in_cluster, return_counts=True)
                label_count_dict = dict(zip(unique_labels, counts))
            
                print(f"Cluster {cluster_id}:")
                total = len(cluster_points_indices)
            
                for label, count in sorted(label_count_dict.items(), key=lambda x: x[1]/total, reverse=True):
                    print(f"  Activity {label}: {count} samples | Density: {(count/total*100):.2f}%")
                print()
            
            noise_clusters = kmeans_outlier_cluster_count(
                closest_clusters=closest_clusters,
                n_clusters=n_clusters,
                total_points=imu_magnitude_data.shape[0],
                method='Z-Score'
            )

            print("Clusters considered as Noise (Outliers) based on Z-Score method:\n")
            for cluster_id in noise_clusters['noise_clusters']:
                print(f"Cluster ID: {cluster_id}")
            print()
            
            global imu_magnitude_data_test
            imu_magnitude_data_test = imu_magnitude_data[np.isin(closest_clusters, noise_clusters)]
            
            global closest_clusters_test
            closest_clusters_test = closest_clusters[np.isin(closest_clusters, noise_clusters)]
            
            global centroids_test
            centroids_test = centroids[np.isin(np.arange(len(centroids)), noise_clusters)]
            
            # Mostra os Clusters considerados Noisy
            # plot_kmeans_results_3d(
            #     imu_magnitude_data=imu_magnitude_data[np.isin(closest_clusters, noise_clusters)],
            #     closest_clusters=closest_clusters[np.isin(closest_clusters, noise_clusters)],
            #     centroids=centroids[np.isin(np.arange(len(centroids)), noise_clusters)],
            #     kmeans_outliers=[],
            #     n_clusters=len(noise_clusters),
            #     title='K-Means Clustering - Noise Clusters for IMU Sensors Magnitude',
            #     xlabel='Accelerometer Module',
            #     ylabel='Gyroscope Module',
            #     zlabel='Magnetometer Module',
            #     save_path="Exercise-3.7-solution1/IMU_Sensors_Magnitude_K-Means_Noise-Clusters",
            # )

            kmeans_outliers = kmeans_outlier_count(
                data=imu_magnitude_data,
                number_of_clusters=n_clusters,
                closest_clusters=closest_clusters,
                centroids=centroids,
                distance_metric=kmeans_distance_metric,
                k=3
            )
            
            for result in kmeans_outliers:
                print(f"Cluster: {result['cluster_id']} | "
                      f"Number of Outliers: {len(result['outlier_indices'])}")
            print()
            
            # plot_kmeans_results_3d(
            #     imu_magnitude_data=imu_magnitude_data,
            #     closest_clusters=closest_clusters,
            #     centroids=centroids,
            #     kmeans_outliers=kmeans_outliers,
            #     n_clusters=n_clusters,
            #     title='K-Means Clustering & Outliers for IMU Sensors Magnitude',
            #     xlabel='Accelerometer Module',
            #     ylabel='Gyroscope Module',
            #     zlabel='Magnetometer Module',
            #     # save_path="Exercise-3.7-solution1/IMU_Sensors_Magnitude_K-Means",
            # )
            
            # plot_kmeans_results_3d_grid_subplots(
            #     imu_magnitude_data=imu_magnitude_data,
            #     closest_clusters=closest_clusters,
            #     centroids=centroids,
            #     kmeans_outliers=kmeans_outliers,
            #     n_clusters=n_clusters,
            #     title='K-Means Clustering & Outliers for IMU Sensors Magnitude',
            #     xlabel='Accelerometer Module',
            #     ylabel='Gyroscope Module',
            #     zlabel='Magnetometer Module',
            #     # save_path="Exercise-3.7-solution1/IMU_Sensors_Magnitude_K-Means",
            # )
                    
            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================

        if exercises_to_run["Exercise 3.7 Solution 2"]:
            '''
            Solução 2.
            → K-Means é aplicado sobre os módulos combinados dos vetores de Aceleração, Giroscópio e Magnetómetro.
            → 1 Cluster por atividade = 16 Clusters
            → Filtra as linhas por posição do sensor (Device ID).
            → X -> Módulo Acelerómetro
            → Y -> Módulo Giroscópio
            → Z -> Módulo Magnetómetro
            (5 gráficos => um por posição do sensor)
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.7 Solution 2 ====\n\n")
            # ====================================================================================
            
            
            n_clusters = len(ACTIVITIES)
            kmeans_initialization = "Forgy Method"
            kmeans_distance_metric = 'Euclidean'
            
            for device_id, device_name in BODY_SENSOR_POSITIONS.items():
            
                imu_magnitude_data = all_participants_data_df.loc[
                    (all_participants_data_df["Device ID"] == device_id),
                    ['Accelerometer Module', 'Gyroscope Module', 'Magnetometer Module']
                ].values
                
                # (Slide 118) Z = (X - μ) / σ
                # Normalização por coluna: Zᵢ = (Xᵢ - μᵢ) / σᵢ
                imu_magnitude_data = (
                    imu_magnitude_data - imu_magnitude_data.mean(axis=0)
                ) / imu_magnitude_data.std(axis=0)
                
                closest_clusters, centroids = kmeans(
                    data=imu_magnitude_data, # (n_linhas, 3 cols)
                    number_of_clusters=n_clusters,
                    initialization=kmeans_initialization,
                    distance_metric=kmeans_distance_metric
                )
                
                kmeans_outliers_result = kmeans_outlier_count(
                    data=imu_magnitude_data,
                    number_of_clusters=n_clusters,
                    closest_clusters=closest_clusters,
                    centroids=centroids,
                    distance_metric=kmeans_distance_metric
                )
                
                print(device_name)
                for result in kmeans_outliers_result:
                    print(f"Cluster: {result['cluster_id']} | "
                          f"Number of Outliers: {len(result['outlier_indices'])}")
                print()
                
                plot_kmeans_results_3d(
                    imu_magnitude_data=imu_magnitude_data,
                    closest_clusters=closest_clusters,
                    centroids=centroids,
                    kmeans_outliers=kmeans_outliers_result,
                    n_clusters=n_clusters,
                    title=f'K-Means Clustering & Outliers for IMU Sensors Magnitude in {device_name}',
                    xlabel='Accelerometer Module',
                    ylabel='Gyroscope Module',
                    zlabel='Magnetometer Module',
                    # save_path=f"Exercise-3.7-solution2/IMU_Sensors_Magnitude_dev{device_id}_K-Means"
                )

            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
                    
        if exercises_to_run["Exercise 3.7 Solution 3"]:
            '''
            Solução 3.            
            → K-Means é aplicado individualmente sobre os vetores de Aceleração, Giroscópio e Magnetómetro.
            → 1 Cluster por atividade = 16 Clusters
            → Sem filtragem de linhas.
            → X -> componente X de Accelerometer / Gyroscope / Magnetometer 
            → Y -> componente Y de Accelerometer / Gyroscope / Magnetometer
            → Z -> componente Z de Accelerometer / Gyroscope / Magnetometer
            (3 gráficos => um por sensor IMU)
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.7 Solution 3 ====\n\n")
            # ====================================================================================
            
            
            n_clusters = len(ACTIVITIES)
            kmeans_initialization = "Forgy Method"
            kmeans_distance_metric = 'Euclidean'
            
            for imu_sensor in IMU_SENSORS:
                sensor_data = all_participants_data_df[
                    [f"{imu_sensor} X", f"{imu_sensor} Y", f"{imu_sensor} Z"]
                    ].to_numpy()
                
                # Normalizar
                sensor_data = (sensor_data - sensor_data.mean(axis=0)) / sensor_data.std(axis=0)
                
                closest_clusters, centroids = kmeans(
                    sensor_data, n_clusters, kmeans_initialization, kmeans_distance_metric
                )
                
                kmeans_outliers_result = kmeans_outlier_count(
                    data=sensor_data,
                    number_of_clusters=n_clusters,
                    closest_clusters=closest_clusters,
                    centroids=centroids,
                    distance_metric=kmeans_distance_metric
                )
                
                print(imu_sensor)
                for result in kmeans_outliers_result:
                    print(f"Cluster: {result['cluster_id']} | "
                          f"Number of Outliers: {len(result['outlier_indices'])}")
                print()
                
                plot_kmeans_results_3d(
                    imu_magnitude_data=sensor_data,
                    closest_clusters=closest_clusters,
                    centroids=centroids,
                    kmeans_outliers=kmeans_outliers_result,
                    n_clusters=n_clusters,
                    title=f'K-Means Clustering & Outliers for {imu_sensor}',
                    xlabel=f'{imu_sensor} X',
                    ylabel=f'{imu_sensor} Y',
                    zlabel=f'{imu_sensor} Z',
                    # save_path=f"Exercise-3.7-solution3/{imu_sensor}_K-Means",
                )

            
            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
            
        if exercises_to_run["Exercise 3.7 Solution 4"]:
            '''
            Solução 4.
            → K-Means é aplicado individualmente sobre os vetores de Aceleração, Giroscópio e Magnetómetro.
            → 1 Cluster por atividade = 16 Clusters
            → Linhas são filtradas por posição do sensor.
            → X -> componente X de Accelerometer / Gyroscope / Magnetometer 
            → Y -> componente Y de Accelerometer / Gyroscope / Magnetometer
            → Z -> componente Z de Accelerometer / Gyroscope / Magnetometer
            (3 * 5 = 15 gráficos => um por sensor IMU e posição)
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.7 Solution 4 ====\n\n")
            # ====================================================================================
            
            
            n_clusters = len(ACTIVITIES)
            kmeans_initialization = "Forgy Method"
            kmeans_distance_metric = 'Euclidean'
            
            for imu_sensor in IMU_SENSORS:
                for device_id, device_name in BODY_SENSOR_POSITIONS.items():                    
                    sensor_data = all_participants_data_df.loc[
                        (all_participants_data_df["Device ID"] == device_id),
                        [f"{imu_sensor} X", f"{imu_sensor} Y", f"{imu_sensor} Z"]
                    ].values
                    
                    # Normalizar
                    sensor_data = (sensor_data - sensor_data.mean(axis=0)) / sensor_data.std(axis=0)
                    
                    closest_clusters, centroids = kmeans(
                        sensor_data, n_clusters, kmeans_initialization, kmeans_distance_metric
                    )
                    
                    kmeans_outliers_result = kmeans_outlier_count(
                        data=sensor_data,
                        number_of_clusters=n_clusters,
                        closest_clusters=closest_clusters,
                        centroids=centroids,
                        distance_metric=kmeans_distance_metric
                    )
                    
                    print(imu_sensor, "-", device_name)
                    for result in kmeans_outliers_result:
                        print(f"Cluster: {result['cluster_id']} | "
                              f"Number of Outliers: {len(result['outlier_indices'])}")
                    print()
                    
                    plot_kmeans_results_3d(
                        imu_magnitude_data=sensor_data,
                        closest_clusters=closest_clusters,
                        centroids=centroids,
                        kmeans_outliers=kmeans_outliers_result,
                        n_clusters=n_clusters,
                        title=f'K-Means Clustering & Outliers for {imu_sensor} in {device_name}',
                        xlabel=f'{imu_sensor} X',
                        ylabel=f'{imu_sensor} Y',
                        zlabel=f'{imu_sensor} Z',
                        # save_path=f"Exercise-3.7-solution4/{imu_sensor}_dev{device_id}_K-Means",
                    )


            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================
        
        if exercises_to_run["Exercise 3.7.1"]:
            ''' 
            Bónus: Poderá realizar um estudo análogo usando o algoritmo DBSCAN
            (sugere-se que recorra à biblioteca sklearn)
            '''
            
            # ====================================================================================
            print("\n\n\t\t==== RUNNING EXERCISE 3.7.1 ====\n\n")
            # ====================================================================================
            
            
            # TODO: Implementar...


            # ====================================================================================
            print("\n\n\t\t\t======== DONE ========\n\n")
            # ====================================================================================

    if exercises_to_run["Exercise 3.8"]:
        ''' 
        Implemente uma rotina que injete ouliers com uma densidade igual
        ou superior a x% nas amostras de variável fornecida.
        
        Para o efeito deverá:
            
            → Calcular a densidade de outliers existente no Array fornecido
            com Nr. de Pts; Observe que a densidade é determinada recorrendo:
                
                Densidade = (Nr. de Pts Outliers / Nr. Total de Pontos) * 100
                
            em que:
                
                Nr. de Pts Outliers = # { Pontos ∉ [μ - k*σ, μ + k*σ] }
        
        Se a densidade d for inferior a x%, então deverá sortear (x-d)% dos pontos
        não outliers de forma aleatória e para cada ponto selecionado deverá transformá-lo tal que:
        
            p ⇽ μ + s * k * (σ + q)
        
        em que:
            
            μ e σ representam respectivamente, os valores médio e o desvio padrão da amostra
            k = Limite especificado no ponto 3.3 (k = [3, 3.5, 4])
            s = Varíavel escolhida de forma aleatória usando uma distribuição uniforme. ( s ∈ {-1, 1})
            q = Variável aleatória uniforme no intervalo q ∈ [0, z[
            z = Amplitude máxima do outlier relativamente a μ ± kσ
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 3.8 ====\n\n")
        # ====================================================================================
        
        participant_id = 7
        participant_data_df = load_participant_data(participant_id, False)
        data = participant_data_df["Accelerometer X"]
        inject_outliers(data, x=5, k=3, z=5)
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
    
    if exercises_to_run["Exercise 3.9"]:
        '''
        Elabore uma rotina que determine o modelo linear de ordem p.
        Para o efeito, a sua rotina deverá receber n amostras de treino
        de um vetor de dimensão p, isto é, (xᵢ,₁, xᵢ,₂, ..., xᵢ,ₚ) e a respetiva saída yᵢ.
        
        A sua rotina deverá determinar o melhor vetor de pesos β tal que:
        
                argminᵦ  Σᵢ₌₁ᵖ ( yᵢ − β₀ + β₁xᵢ,₁ + β₂xᵢ,₂ + ... + βₚxᵢ,ₚ )²  =  argminᵦ ‖ Y − Xβ ‖²
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 3.9 ====\n\n")
        # ====================================================================================
        
        
        X = np.array([ # p = 2
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0]
        ])
        Y = np.array([
            5.0, 
            6.0, 
            11.0, 
            12.0
        ])
    
        weights = linear_regression(X, Y)
        print("Weights: ", weights)
        
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================

    if exercises_to_run["Exercise 3.10"]:
        '''
        Determine o modelo linear para o módulo aceleração usando uma janela com p valores anteriores.
        Usando a rotina desenvolvida no ponto 3.8 injete 10% de outliers no módulo da aceleração.
        Elimine esses outliers e substitua-os pelos valores previstos pelo modelo linear.
        
        Analise o erro de predição apresentando:
            i) a distribuição do erro
            ii) exemplos de plots contendo o valor previsto e real.
        
        Notas: Estimar o erro de previsão com cross-validation.
        (1/n)  *  (∑ᵢ₌₁ⁿ  ERROᵢ)
        
        Determine o melhor p para o seu modelo.
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 3.10 ====\n\n")
        # ====================================================================================
        
    
        imu_sensor = "Accelerometer"
        magnitude = all_participants_data_df[f'{imu_sensor} Module'].values
        
        # Injetar outliers
        magnitude_with_outliers, outlier_mask = inject_outliers(magnitude, x=10, k=3, z=5)
        magnitude_cleaned = magnitude_with_outliers.copy()
    
        p_values = range(1, 11)
        best_p = None
        best_mse = np.inf
        errors_dict = {}
    
        for p in p_values:
            X, Y = [], []
            for i in range(p, len(magnitude_with_outliers)):
                X.append(magnitude_with_outliers[i - p:i])
                Y.append(magnitude_with_outliers[i])
            X = np.array(X)
            Y = np.array(Y)
    
            # Remover os outliers antes de fazer compute dos pesos
            valid_idx = np.where(~outlier_mask[p:])[0]
            weights = linear_regression(X[valid_idx], Y[valid_idx])
    
            # Substituir Outliers
            for idx in np.where(outlier_mask)[0]:
                if idx < p:
                    continue
                x_input = magnitude_with_outliers[idx - p:idx]
                magnitude_cleaned[idx] = linear_regression_predict(x_input, weights)
    
            # Cross-validation
            errors, mse, mae = cross_validation_error(X, Y)
            errors_dict[p] = errors
            print(f"p={p}: MSE={mse:.5f}, MAE={mae:.5f}")
            if mse < best_mse:
                best_mse = mse
                best_p = p
    
        print(f"\nBest p selected by cross-validation: {best_p} with MSE={best_mse:.5f}")
    
        X, Y = [], []
        for i in range(best_p, len(magnitude_cleaned)):
            X.append(magnitude_cleaned[i - best_p:i])
            Y.append(magnitude_cleaned[i])
        X = np.array(X)
        Y = np.array(Y)
    
        weights = linear_regression(X, Y)
        print(f"Linear Model Weights (best p={best_p}):\n\n{weights}")
    
        # Distribuição do erro
        plt.hist(errors_dict[best_p], bins=50)
        plt.title(f"Prediction Error Distribution (Best p={best_p})")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.show()
    
        # Previsão vs Real
        Y_pred = np.array([linear_regression_predict(x, weights) for x in X[:50]])
        plt.plot(Y[:50], label="Real")
        plt.plot(Y_pred, label="Predicted")
        plt.title(f"Predicted vs Real Values (Best p={best_p})")
        plt.legend()
        plt.show()
        
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
    
    if exercises_to_run["Exercise 3.11"]:
        '''
        Repita 3.10 usando uma janela de dimensão p centrada no instante a prever.
        Deverá usar não só os p/2 valores anteriores e seguintes da variável que pretende prever bem como
        das restantes variáveis disponíveis (módulos disponíveis).
        Compare com os resultados obtidos em 3.10.
        
        Fazer previsão dos pontos do acelerómetro, com base nos pontos anteriores dos acelerometro e nos oumodulos.
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 3.11 ====\n\n")
        # ====================================================================================
        
        
        modules = all_participants_data_df[
            ['Accelerometer Module', 'Gyroscope Module', 'Magnetometer Module']
        ].values
        
        # Injetar Outliers no Acelerómetro
        acc_module = modules[:, 0]
        acc_with_outliers, outlier_mask = inject_outliers(acc_module, x=10, k=3, z=5)
        modules_cleaned = modules.copy()
        modules_cleaned[:, 0] = acc_with_outliers
        
        # Tamanhos da janela => Número par para conseguir centrar
        p_values = range(2, 22, 2)
        best_p = None
        best_mse = np.inf
        errors_dict = {}
        
        for p in p_values:
            half_p = p // 2
            X, Y = [], []
        
            for i in range(half_p, len(modules_cleaned) - half_p):
                # p/2 anteriores e p/2 seguintes
                # reservamos o valor central, que será usado para previsão (centro i)
                window_before = modules_cleaned[i - half_p:i]
                window_after = modules_cleaned[i + 1:i + half_p + 1]
                window = np.vstack((window_before, window_after)).flatten()
        
                X.append(window)
                Y.append(modules_cleaned[i, 0])  # Acelerómetro
        
            X = np.array(X)
            Y = np.array(Y)
            
            # Remover os outliers antes de fazer compute dos pesos
            valid_idx = np.where(~outlier_mask[half_p:len(modules_cleaned) - half_p])[0]
            weights = linear_regression(X[valid_idx], Y[valid_idx])
            
            # Substituir os outliers
            for idx in np.where(outlier_mask)[0]:
                if idx < half_p or idx >= len(modules_cleaned) - half_p:
                    continue
                window_before = modules_cleaned[idx - half_p:idx]
                window_after = modules_cleaned[idx + 1:idx + half_p + 1]
                window = np.vstack((window_before, window_after)).flatten()
                modules_cleaned[idx, 0] = linear_regression_predict(window, weights)
        
            # Cross-validation
            errors, mse, mae = cross_validation_error(X, Y)
            errors_dict[p] = errors
            print(f"p={p}: MSE={mse:.5f}, MAE={mae:.5f}")
            if mse < best_mse:
                best_mse = mse
                best_p = p
        
        print(f"\nBest p selected by cross-validation: {best_p} with MSE={best_mse:.5f}")
        
        half_p = best_p // 2
        X, Y = [], []
        
        for i in range(half_p, len(modules_cleaned) - half_p):
            window_before = modules_cleaned[i - half_p:i]
            window_after = modules_cleaned[i + 1:i + half_p + 1]
            window = np.vstack((window_before, window_after)).flatten()
            X.append(window)
            Y.append(modules_cleaned[i, 0])
        
        X = np.array(X)
        Y = np.array(Y)
        
        weights = linear_regression(X, Y)
        print(f"Linear Model Weights (best p={best_p}):\n\n{weights}")
        
        # Distribuição do erro
        plt.hist(errors_dict[best_p], bins=50)
        plt.title(f"Prediction Error Distribution (Best p={best_p})")
        plt.xlabel("Error")
        plt.ylabel("Frequency")
        plt.show()
        
        # Previsão vs Real
        Y_pred = np.array([linear_regression_predict(x, weights) for x in X[:50]])
        plt.plot(Y[:50], label="Real")
        plt.plot(Y_pred, label="Predicted")
        plt.title(f"Predicted vs Real Values (Best p={best_p})")
        plt.legend()
        plt.show()

    
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
    
    if exercises_to_run["Exercise 4.1"]:
        '''
        Usando as variáveis aplicadas na alínea 3.1, determine a significância estatística
        dos seus valores médios nas diferentes atividades.
        
        Observe que poderá aferir a gaussianidade da distribuição usando, por exemplo, o teste 
        Kolmogorov-Smirnov (vide documentação do SciPy). Para rever a escolha de testes estatísticos
        sugere-se a referência:
            
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2881615/
            
        Comente.
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 4.1 ====\n\n")
        # ====================================================================================

        # Retirar dados da magnitude, uma de cada vez.
        results_activity = []
        for activity_key in ACTIVITIES.keys():
            results_device = {}
            for device_key in BODY_SENSOR_POSITIONS.keys():
                activity_data = all_participants_data_df.loc[
                    (all_participants_data_df["Activity Label"] == activity_key) &
                    (all_participants_data_df["Device ID"] == device_key), 
                    f"Accelerometer Module"
                ]
                results_device[device_key] = activity_data
            results_activity.append([activity_key, results_device]) # results_activity = [[activity, {device: [Accelerometer Module]}]]

        """
        Teste Kolmogorov-Smirnov para cada atividade em cada dispositivo.
        0.05 é o alpha -> nível de significância.
        - H0 -> Dados seguem uma distribuição Gaussiana (p > 0.05).
        - H1 -> Dados não seguem uma distribuição Gaussiana (p < 0.05).
        """
        
        print("-----------------------------")
        for activity in results_activity:
            for device in activity[1]:
                print(f"Activity: {activity[0]}, Device: {device}")
                p_val = ks_test(activity[1][device])

                if p_val < 0.05:
                    print("Result: Data doesn't follow a normal distribution (reject H0)")
                else:
                    print("Result: Data follows a normal distribution (fail to reject H0)")
                print("-----------------------------")

        """
        P_values são muito pequenos -> < 0.05,
        Logo, os dados rejeitam fortemente a hipótese de ser uma distribuição gaussiana.
        A distribuição é então não gaussiana -> Usar testes não-paramétricos.
        Como estamos a comparar mais do que 2 atividades -> usar Kruskal-Wallis test.
        """

        # A cada dispositivo -> juntar 16 magnitudes das 16 atividades e usar no teste.
        print("-----------------------------")
        for activity in results_activity:
            magnitudes = []
            print(f"Activity: {activity[0]}")
            for device in activity[1]:
                print(f"Device: {device}")
                magnitudes.append(activity[1][device]) # magnitudes = [[magnitudeact1],[magnitudeact2], ...]
            stat, p_value = stats.kruskal(*magnitudes)
            print(f"Accelerometer module - H: {stat}, p-value: {p_value}")

            if p_value < 0.05:
                print("Result: Significant difference between activities (reject H0)")
            else:
                print("Result: No significant difference between activities (fail to reject H0)")

            print("-----------------------------")

        # Resultado: existe significância entre magnitudes, p < 0.05.
        # Rejeitar a hipótese de as diferentes atividades terem a mesma magnitude.
        # Logo diferentes atividades têm diferentes distribuições de magnitude.

        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================      

    if exercises_to_run["Exercise 4.2"]:
        '''
        Desenvolva as rotinas necessárias à extração do feature set temporal e espectral
        sugerido no artigo: 
            
            https://pdfs.semanticscholar.org/8522/ce2bfce1ab65b133e411350478183e79fae7.pdf
            
        Para o efeito deverá:
            
            → Ler o artigo e identificar o conjunto de features temporais e espectrais 
            identificadas por estes autores.
            
            → Para cada feature deverá elaborar uma rotina para a respectiva extração.
            
            → Usando as rotinas elaboradas no item anterior, deverá escrever o código necessário
            para extrair o vetor de features em cada instante.
        
        Nota: Poderá usar as bibliotecas NumPy e SciPy.
        (Qualquer outra biblioteca deverá ser identificada.)
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 4.2 ====\n\n")
        # ====================================================================================
        
        # Para cada atividade e posição do sensor, extrair as features.
        # Considerar fs = 100Hz.
        # Janela temporal de 2 segundos.
        # 50% de sobreposição

        def extract_features(df, fs = 50, window_time = 2.0):

            all_window_features = []
            window_size = int(window_time * fs) # 200 amostras por janela.
            step = 100 # 50% de sobreposição dá 100 amostras.

            for i in range(0, len(df) - window_size, step):
                wind = df[i : i + window_size]
        
                feature_set = {
                    "acc_x_mean": np.mean(wind['Accelerometer X']),
                    "acc_y_mean": np.mean(wind['Accelerometer Y']),
                    "acc_z_mean": np.mean(wind['Accelerometer Z']),
                    "gyr_x_mean": np.mean(wind['Gyroscope X']),
                    "gyr_y_mean": np.mean(wind['Gyroscope Y']),
                    "gyr_z_mean": np.mean(wind['Gyroscope Z']),
                    "mag_x_mean": np.mean(wind['Magnetometer X']),
                    "mag_y_mean": np.mean(wind['Magnetometer Y']),
                    "mag_z_mean": np.mean(wind['Magnetometer Z']),

                    "acc_x_median": np.median(wind['Accelerometer X']),
                    "acc_y_median": np.median(wind['Accelerometer Y']),
                    "acc_z_median": np.median(wind['Accelerometer Z']),
                    "gyr_x_median": np.median(wind['Gyroscope X']),
                    "gyr_y_median": np.median(wind['Gyroscope Y']),
                    "gyr_z_median": np.median(wind['Gyroscope Z']),
                    "mag_x_median": np.median(wind['Magnetometer X']),
                    "mag_y_median": np.median(wind['Magnetometer Y']),
                    "mag_z_median": np.median(wind['Magnetometer Z']),

                    "acc_x_std": np.std(wind['Accelerometer X']),
                    "acc_y_std": np.std(wind['Accelerometer Y']),
                    "acc_z_std": np.std(wind['Accelerometer Z']),
                    "gyr_x_std": np.std(wind['Gyroscope X']),
                    "gyr_y_std": np.std(wind['Gyroscope Y']),
                    "gyr_z_std": np.std(wind['Gyroscope Z']),
                    "mag_x_std": np.std(wind['Magnetometer X']),
                    "mag_y_std": np.std(wind['Magnetometer Y']),
                    "mag_z_std": np.std(wind['Magnetometer Z']),

                    "acc_x_var": np.var(wind['Accelerometer X']),
                    "acc_y_var": np.var(wind['Accelerometer Y']),
                    "acc_z_var": np.var(wind['Accelerometer Z']),
                    "gyr_x_var": np.var(wind['Gyroscope X']),
                    "gyr_y_var": np.var(wind['Gyroscope Y']),
                    "gyr_z_var": np.var(wind['Gyroscope Z']),
                    "mag_x_var": np.var(wind['Magnetometer X']),
                    "mag_y_var": np.var(wind['Magnetometer Y']),
                    "mag_z_var": np.var(wind['Magnetometer Z']),

                    "acc_x_rms": np.sqrt(np.mean(wind['Accelerometer X']**2)),
                    "acc_y_rms": np.sqrt(np.mean(wind['Accelerometer Y']**2)),
                    "acc_z_rms": np.sqrt(np.mean(wind['Accelerometer Z']**2)),
                    "gyr_x_rms": np.sqrt(np.mean(wind['Gyroscope X']**2)),
                    "gyr_y_rms": np.sqrt(np.mean(wind['Gyroscope Y']**2)),
                    "gyr_z_rms": np.sqrt(np.mean(wind['Gyroscope Z']**2)),
                    "mag_x_rms": np.sqrt(np.mean(wind['Magnetometer X']**2)),
                    "mag_y_rms": np.sqrt(np.mean(wind['Magnetometer Y']**2)),
                    "mag_z_rms": np.sqrt(np.mean(wind['Magnetometer Z']**2)),

                    "acc_x_skew": stats.skew(wind['Accelerometer X']),
                    "acc_y_skew": stats.skew(wind['Accelerometer Y']),
                    "acc_z_skew": stats.skew(wind['Accelerometer Z']),
                    "gyr_x_skew": stats.skew(wind['Gyroscope X']),
                    "gyr_y_skew": stats.skew(wind['Gyroscope Y']),
                    "gyr_z_skew": stats.skew(wind['Gyroscope Z']),
                    "mag_x_skew": stats.skew(wind['Magnetometer X']),
                    "mag_y_skew": stats.skew(wind['Magnetometer Y']),
                    "mag_z_skew": stats.skew(wind['Magnetometer Z']),

                    "acc_x_kurtosis": stats.kurtosis(wind['Accelerometer X']),
                    "acc_y_kurtosis": stats.kurtosis(wind['Accelerometer Y']),
                    "acc_z_kurtosis": stats.kurtosis(wind['Accelerometer Z']),
                    "gyr_x_kurtosis": stats.kurtosis(wind['Gyroscope X']),
                    "gyr_y_kurtosis": stats.kurtosis(wind['Gyroscope Y']),
                    "gyr_z_kurtosis": stats.kurtosis(wind['Gyroscope Z']),
                    "mag_x_kurtosis": stats.kurtosis(wind['Magnetometer X']),
                    "mag_y_kurtosis": stats.kurtosis(wind['Magnetometer Y']),
                    "mag_z_kurtosis": stats.kurtosis(wind['Magnetometer Z']),

                    "acc_x_diff_mean": np.mean(np.diff(wind['Accelerometer X'])),
                    "acc_y_diff_mean": np.mean(np.diff(wind['Accelerometer Y'])),
                    "acc_z_diff_mean": np.mean(np.diff(wind['Accelerometer Z'])),
                    "gyr_x_diff_mean": np.mean(np.diff(wind['Gyroscope X'])),
                    "gyr_y_diff_mean": np.mean(np.diff(wind['Gyroscope Y'])),
                    "gyr_z_diff_mean": np.mean(np.diff(wind['Gyroscope Z'])),
                    "mag_x_diff_mean": np.mean(np.diff(wind['Magnetometer X'])),
                    "mag_y_diff_mean": np.mean(np.diff(wind['Magnetometer Y'])),
                    "mag_z_diff_mean": np.mean(np.diff(wind['Magnetometer Z'])),

                    "acc_x_iqr": stats.iqr(wind['Accelerometer X']),
                    "acc_y_iqr": stats.iqr(wind['Accelerometer Y']),
                    "acc_z_iqr": stats.iqr(wind['Accelerometer Z']),
                    "gyr_x_iqr": stats.iqr(wind['Gyroscope X']),
                    "gyr_y_iqr": stats.iqr(wind['Gyroscope Y']),
                    "gyr_z_iqr": stats.iqr(wind['Gyroscope Z']),
                    "mag_x_iqr": stats.iqr(wind['Magnetometer X']),
                    "mag_y_iqr": stats.iqr(wind['Magnetometer Y']),
                    "mag_z_iqr": stats.iqr(wind['Magnetometer Z']),

                    "acc_x_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Accelerometer X']))) == 2) / len(wind),
                    "acc_y_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Accelerometer Y']))) == 2) / len(wind),
                    "acc_z_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Accelerometer Z']))) == 2) / len(wind),
                    "gyr_x_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Gyroscope X']))) == 2) / len(wind),
                    "gyr_y_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Gyroscope Y']))) == 2) / len(wind),
                    "gyr_z_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Gyroscope Z']))) == 2) / len(wind),
                    "mag_x_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Magnetometer X']))) == 2) / len(wind),
                    "mag_y_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Magnetometer Y']))) == 2) / len(wind),
                    "mag_z_zero_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Magnetometer Z']))) == 2) / len(wind),

                    "acc_x_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Accelerometer X'] - np.mean(wind['Accelerometer X'])))) == 2) / window_size,
                    "acc_y_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Accelerometer Y'] - np.mean(wind['Accelerometer Y'])))) == 2) / window_size,
                    "acc_z_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Accelerometer Z'] - np.mean(wind['Accelerometer Z'])))) == 2) / window_size,
                    "gyr_x_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Gyroscope X'] - np.mean(wind['Gyroscope X'])))) == 2) / window_size,
                    "gyr_y_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Gyroscope Y'] - np.mean(wind['Gyroscope Y'])))) == 2) / window_size,
                    "gyr_z_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Gyroscope Z'] - np.mean(wind['Gyroscope Z'])))) == 2) / window_size,
                    "mag_x_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Magnetometer X'] - np.mean(wind['Magnetometer X'])))) == 2) / window_size,
                    "mag_y_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Magnetometer Y'] - np.mean(wind['Magnetometer Y'])))) == 2) / window_size,
                    "mag_z_mean_crossing_rate": np.sum(np.abs(np.diff(np.sign(wind['Magnetometer Z'] - np.mean(wind['Magnetometer Z'])))) == 2) / window_size,

                    "acc_x_y_pairwise_correlation": np.corrcoef(wind['Accelerometer X'], wind['Accelerometer Y'])[0, 1],
                    "acc_x_z_pairwise_correlation": np.corrcoef(wind['Accelerometer X'], wind['Accelerometer Z'])[0, 1],
                    "acc_y_z_pairwise_correlation": np.corrcoef(wind['Accelerometer Y'], wind['Accelerometer Z'])[0, 1],
                    "gyr_x_Y_pairwise_correlation": np.corrcoef(wind['Gyroscope X'], wind['Gyroscope Y'])[0, 1],
                    "gyr_x_z_pairwise_correlation": np.corrcoef(wind['Gyroscope X'], wind['Gyroscope Z'])[0, 1],
                    "gyr_y_z_pairwise_correlation": np.corrcoef(wind['Gyroscope Y'], wind['Gyroscope Z'])[0, 1],
                    "mag_x_y_pairwise_correlation": np.corrcoef(wind['Magnetometer X'], wind['Magnetometer Y'])[0, 1],
                    "mag_x_z_pairwise_correlation": np.corrcoef(wind['Magnetometer X'], wind['Magnetometer Z'])[0, 1],
                    "mag_y_z_pairwise_correlation": np.corrcoef(wind['Magnetometer Y'], wind['Magnetometer Z'])[0, 1],

                    "acc_x_spectral_entropy": spectral_entropy(wind['Accelerometer X'].values, fs),
                    "acc_y_spectral_entropy": spectral_entropy(wind['Accelerometer Y'].values, fs),
                    "acc_z_spectral_entropy": spectral_entropy(wind['Accelerometer Z'].values, fs),
                    "gyr_x_spectral_entropy": spectral_entropy(wind['Gyroscope X'].values, fs),
                    "gyr_y_spectral_entropy": spectral_entropy(wind['Gyroscope Y'].values, fs),
                    "gyr_z_spectral_entropy": spectral_entropy(wind['Gyroscope Z'].values, fs),
                    "mag_x_spectral_entropy": spectral_entropy(wind['Magnetometer X'].values, fs),
                    "mag_y_spectral_entropy": spectral_entropy(wind['Magnetometer Y'].values, fs),
                    "mag_z_spectral_entropy": spectral_entropy(wind['Magnetometer Z'].values, fs),
                }


                # Dominant Frequency

                # 1. ACELERÓMETRO
                feature_set["acc_x_dominant_frequency"] = calc_df(wind['Accelerometer X'].values, fs)
                feature_set["acc_y_dominant_frequency"] = calc_df(wind['Accelerometer Y'].values, fs)
                feature_set["acc_z_dominant_frequency"] = calc_df(wind['Accelerometer Z'].values, fs)

                # 2. GIROSCÓPIO (Onde o erro estava a ocorrer)
                feature_set["gyr_x_dominant_frequency"] = calc_df(wind['Gyroscope X'].values, fs)
                feature_set["gyr_y_dominant_frequency"] = calc_df(wind['Gyroscope Y'].values, fs)
                feature_set["gyr_z_dominant_frequency"] = calc_df(wind['Gyroscope Z'].values, fs)

                # 3. MAGNETÓMETRO
                feature_set["mag_x_dominant_frequency"] = calc_df(wind['Magnetometer X'].values, fs)
                feature_set["mag_y_dominant_frequency"] = calc_df(wind['Magnetometer Y'].values, fs)
                feature_set["mag_z_dominant_frequency"] = calc_df(wind['Magnetometer Z'].values, fs)
                
                # Features Módulo Acelerómetro.
                feature_set["acc_ai"] = np.mean(wind["Accelerometer Module"]) # AI - média do módulo.
                feature_set["acc_vi"] = np.var(wind["Accelerometer Module"]) # VI - variância do módulo.

                # SMA.
                feature_set["acc_sma"] = np.sum(np.abs(wind["Accelerometer X"]) + np.abs(wind["Accelerometer Y"]) + np.abs(wind["Accelerometer Z"])) / window_size

                # acc_window -> linhas = tamanho da janela; colunas são os valores do acelerómetro para x, y e z.
                acc_window = np.array([wind["Accelerometer X"], wind["Accelerometer Y"], wind["Accelerometer Z"]]).T

                # EVA.
                cov_matrix = np.cov(acc_window, rowvar=False)
        
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                ordered_eigenvalues = np.sort(eigenvalues)[::-1]
                eva_features = ordered_eigenvalues[:2]

                feature_set["acc_eva1"] = eva_features[0]
                feature_set["acc_eva2"] = eva_features[1]

                # CAGH. 
                G_acc = acc_window[:, 0] # G -> Gravity (direção do x)
                H_acc_mag = np.linalg.norm(acc_window[:, 1:], axis=1) # H -> Horizontal (direção do y e z)
                cagh = np.corrcoef(G_acc, H_acc_mag)[0, 1]

                feature_set["acc_cagh"] = cagh

                # AVH.
                vy = cumulative_trapezoid(wind["Accelerometer Y"], dx=1/fs, initial=0)
                vz = cumulative_trapezoid(wind["Accelerometer Z"], dx=1/fs, initial=0)
                feature_set["acc_avh"] = np.mean(np.sqrt(vy**2 + vz**2))

                # AVG.
                vx = cumulative_trapezoid(wind["Accelerometer X"], dx=1/fs, initial=0)
                feature_set["acc_avg"] = np.mean(vx)


                # Feature Módulo Giroscópio.
                rot_window = np.array([wind["Gyroscope X"], wind["Gyroscope Y"], wind["Gyroscope Z"]]).T

                # ARATG, ang -> ângulo.
                rot_rate_G = rot_window[:, 0]
        
                dt = 1.0 / fs # dt -> intervalo de tempo entre amostras
                G_accumulated_angle = cumulative_trapezoid(rot_rate_G, dx=dt, initial=0.0)
                feature_set["gyr_arATG"] = np.mean(G_accumulated_angle)

                # Energy
                acc_x_energy = energy_calc(wind["Accelerometer X"], fs)
                acc_y_energy = energy_calc(wind["Accelerometer Y"], fs)
                acc_z_energy = energy_calc(wind["Accelerometer Z"], fs)
                gyr_x_energy = energy_calc(wind["Gyroscope X"], fs)
                gyr_y_energy = energy_calc(wind["Gyroscope Y"], fs)
                gyr_z_energy = energy_calc(wind["Gyroscope Z"], fs)
                mag_x_energy = energy_calc(wind["Magnetometer X"], fs)
                mag_y_energy = energy_calc(wind["Magnetometer Y"], fs)
                mag_z_energy = energy_calc(wind["Magnetometer Z"], fs)

                feature_set["acc_x_energy"] = acc_x_energy
                feature_set["acc_y_energy"] = acc_y_energy
                feature_set["acc_z_energy"] = acc_z_energy
                feature_set["gyr_x_energy"] = gyr_x_energy
                feature_set["gyr_y_energy"] = gyr_y_energy
                feature_set["gyr_z_energy"] = gyr_z_energy
                feature_set["mag_x_energy"] = mag_x_energy
                feature_set["mag_y_energy"] = mag_y_energy
                feature_set["mag_z_energy"] = mag_z_energy
                
                feature_set["acc_aae"] = np.sqrt(acc_x_energy**2 + acc_y_energy**2 + acc_z_energy**2) # Averaged Acceleration Energy AAE
                feature_set["gyr_aae"] = np.sqrt(gyr_x_energy**2 + gyr_y_energy**2 + gyr_z_energy**2) # Averaged Rotation Energy ARE
                # feature_set["mag_aae"] = np.sqrt(mag_x_energy**2 + mag_y_energy**2 + mag_z_energy**2) # Averaged Magnitude Energy AME


                feature_set["Activity Label"] = wind["Activity Label"].iloc[0]
                feature_set["Device ID"] = wind["Device ID"].iloc[0]

                all_window_features.append(feature_set)

            return pd.DataFrame(all_window_features)

        all_features_extracted_df = pd.DataFrame()
        for activity_key in ACTIVITIES.keys():
            for device_key in BODY_SENSOR_POSITIONS.keys():
                new_df = all_participants_data_df.loc[
                    (all_participants_data_df["Activity Label"] == activity_key) & 
                    (all_participants_data_df["Device ID"] == device_key)
                ]
                features_extracted_df = extract_features(new_df)
                all_features_extracted_df = pd.concat([all_features_extracted_df, features_extracted_df], ignore_index=True)
                print("Activity:", activity_key, "\n", "Device ID:", device_key, "\n", features_extracted_df)
        print("Todas as features:", "\n", all_features_extracted_df)

        csv_path = 'feature_set.csv'
        all_features_extracted_df.to_csv(csv_path, index=False)
        print(f"Dataset comprimido salvo em: {csv_path}")
    
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
        
    if exercises_to_run["Exercise 4.3"]:
        '''
        Desenvolva o código necessário para implementar o PCA de um feature set.
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 4.3 ====\n\n")
        # ====================================================================================

        def pca(x):
            # x -> feature set -> numpy array
            
            # Z-score.
            x_mean = np.mean(x, axis=0)
            x_std = np.std(x, axis=0)
            x_std[x_std == 0] = 1e-8
            x_cent = (x - x_mean)/x_std

            # Calcular a matriz de covariância
            cov_matrix = np.cov(x_cent, rowvar=False)

            # Calcular os valores próprios e vetores próprios
            eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

            # Ordenar os valores próprios por ordem decrescente
            sorted_idx = np.argsort(eigen_values)[::-1]
            eigen_values = eigen_values[sorted_idx]
            eigen_vectors = eigen_vectors[:, sorted_idx]

            # d = Pw (SLIDE 197)
            d = x_cent.dot(eigen_vectors)

            # lambdas = somatório de d**2 (SLIDE 199)
            lambdas = np.sum(np.square(d), axis = 0)

            return eigen_vectors, d, lambdas
    
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
        
    if exercises_to_run["Exercise 4.4"]:
        '''
        Determine a importância de cada vetor na explicação da variabilidade do espaço de features.
        Note que deverá normalizar as features usando o z-score.
        Quantas variáveis deverá usar para explicar 75% do feature set?
        
        → Indique como poderia obter as features relativas a esta compressão e exemplifique
        para um instante à sua escolha.
        
        → Indique as vantagens e as limitações desta abordagem. 
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 4.4 ====\n\n")
        # ====================================================================================
        
        # Retirar feature set.
        pcs_loaded_df = pd.read_csv('feature_set.csv')
        labels = pcs_loaded_df[["Activity Label", "Device ID"]]
        features_extracted_df_no_labels = pcs_loaded_df.drop(columns=["Activity Label", "Device ID"])

        # PCA (já tem z-score na função).
        pca_vectors, pcs_matrix, lambdas = pca(features_extracted_df_no_labels.values)

        # Variância.
        explained_var = lambdas / np.sum(lambdas)
        acumulated_var = np.cumsum(explained_var)

        limit = 0.75
        n_components_75 = np.argmax(acumulated_var >= limit) + 1 # +1 devido ao index 0
        
        print("PCA")
        print(f"Variância Acumulada: {acumulated_var[n_components_75-1]:.2%} (com {n_components_75} Componentes).")

        # Obter matriz de valores com os PCs mais importantes.
        pcs_matrix = pcs_matrix[:, :n_components_75]
        pcs_matrix_df = pd.DataFrame(pcs_matrix, columns=[f'PC{i+1}' for i in range(n_components_75)])
        pcs_matrix_df_with_labels = pd.concat([labels, pcs_matrix_df], axis=1)

        # Para um instante.
        print(pcs_matrix_df.head(1))

        # Gráfico da Variância Acumulada
        print("=" * 80)
        plt.figure(figsize=(8, 5))
        plt.plot(acumulated_var, marker='o', linestyle='--')
        plt.hlines(limit, 0, len(acumulated_var), color='red', linestyle='-', label=f'{limit:.0%} Limite')
        plt.vlines(n_components_75 - 1, 0, limit, color='red', linestyle=':', label=f'{n_components_75} PCs')
        plt.xlabel('Número de Componentes Principais')
        plt.ylabel('Variância Acumulada Explicada')
        plt.title('Seleção do Número de Componentes (PCA)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
        
    if exercises_to_run["Exercise 4.5"]:
        '''
        Desenvolva o código necessário para implementar o Fisher feature Score e o ReliefF.
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 4.5 ====\n\n")
        # ====================================================================================

        def fisher_score(X, y):
            """
            Fisher - calcula um score para cada feature baseada no rácio discriminante de Fisher.
            Parâmetros:
            - X: numpy array (n_samples, n_features) - feature matrix
            - y: numpy array (n_samples,) - class labels
            Retorna:
            - scores: numpy array (n_features,) - Fisher scores para cada feature
            """

            classes = np.unique(y)
            n_classes = len(classes)
            n_samples, n_features = X.shape
            
            # Matriz dos resultados finais.
            scores = np.zeros(n_features)
            
            # Percorrer features
            for f in range(n_features):
                feature = X[:, f]

                # Média da feature.
                overall_mean = np.mean(feature)
                
                # Iniciar numerador e denominador para fazer o somatório.
                numerator = 0.0
                denominator = 0.0
                
                # Percorrer classes
                for c in classes:
                    class_mask = (y == c)
                    n_c = np.sum(class_mask)
                    if n_c == 0:
                        continue

                    # Média e variância da feature na classe.
                    mean_c = np.mean(feature[class_mask])
                    var_c_sq = np.var(feature[class_mask])
                    
                    # (SLIDE 270 - Fórmula de cima)
                    numerator += n_c * (mean_c - overall_mean) ** 2
                    denominator += n_c * var_c_sq
                
                if denominator == 0:
                    scores[f] = 0.0
                else:
                    scores[f] = numerator / denominator
            
            return scores

        def reliefF(X, y, n_neighbors=10):
            """
            ReliefF com correção da ponderação e tratamento de casos extremos.
            X: numpy array (n_samples, n_features)
            y: numpy array (n_samples,)
            n_neighbors
            Retorna:
                scores
            """

            n_samples, n_features = X.shape
            classes = np.unique(y)
            n_classes = len(classes)

            # Matriz final dos resultados.
            scores = np.zeros(n_features, dtype=float)
            
            # Pré-cálculo das contagens e probabilidades das classes
            class_counts = {c: np.sum(y == c) for c in classes}
            
            # Se houver apenas uma classe, o ReliefF não pode ser calculado.
            if n_classes <= 1:
                return scores 
                
            # Iterar todos os pontos.
            for i in range(n_samples):
                xi = X[i, :]
                yi = y[i]
                
                # 1. Distâncias para todos os outros pontos
                distances = np.linalg.norm(X - xi, axis=1)
                distances[i] = np.inf  # Ignorar a amostra xi
                
                # 2. Encontrar vizinhos da mesma classe (hits)
                same_class_mask = (y == yi)
                same_class_mask[i] = False
                hit_indices = np.argsort(distances[same_class_mask])[:n_neighbors]
                
                # Se não houver vizinhos da mesma classe, ignorar esta amostra
                if len(hit_indices) == 0:
                    continue
                    
                hits = X[same_class_mask][hit_indices]
                
                # 3. Contribuição do hit (Penalidade)
                hit_contrib = np.mean((xi - hits) ** 2, axis=0)

                # 4. Encontrar vizinhos de classes diferentes (misses) e ponderar
                miss_contrib = np.zeros(n_features)
                
                n_yi = class_counts[yi]
                prob_yi = n_yi / n_samples
                total_prob_miss = 1.0 - prob_yi
                
                # Se total_prob_miss for zero, algo está errado ou há apenas uma classe
                # (já tratado, mas como segurança)
                if total_prob_miss == 0:
                    continue 

                for c in classes:
                    if c == yi:
                        continue
                    
                    class_mask = (y == c)
                    
                    # Ponderação da classe miss: P(c) / (1 - P(yi))
                    prob_c = class_counts.get(c, 0) / n_samples
                    weight_c = prob_c / total_prob_miss 
                    
                    # Encontrar os vizinhos miss
                    miss_indices = np.argsort(distances[class_mask])[:n_neighbors]
                    
                    if len(miss_indices) > 0:
                        miss_samples = X[class_mask][miss_indices]
                        avg_diff_sq = np.mean((xi - miss_samples) ** 2, axis=0) 
                        miss_contrib += weight_c * avg_diff_sq
                
                # 5. Atualizar score
                scores += miss_contrib - hit_contrib
                
            # Normalizar pelo número de amostras
            scores /= n_samples
            return scores
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================
        
    if exercises_to_run["Exercise 4.6"]:
        '''
        Indentifique as 10 melhores features de acordo com o Fisher Score e o ReliefF.
        
        → Indique como poderia obter as features relativas a esta compressão e exemplifique
        para um instante à sua escolha.
        
        → Indique as vantagens e as limitações desta abordagem.
        '''
        
        # ====================================================================================
        print("\n\n\t\t==== RUNNING EXERCISE 4.6 ====\n\n")
        # ====================================================================================
        
        # Retirar feature set.
        features_loaded_df = pd.read_csv('feature_set.csv')
        X_np = features_loaded_df.drop(columns=["Activity Label", "Device ID"]).values  # features
        y = features_loaded_df["Activity Label"].values                                 # labels

        feature_names = features_loaded_df.drop(columns=["Activity Label", "Device ID"]).columns.tolist()

        # Fisher Score
        res_fish = fisher_score(X_np, y)
        print("Resultados teste Fisher (Top 10)")
        scores_series = pd.Series(res_fish, index=feature_names)
        scores_sorted = scores_series.sort_values(ascending=False)
        print(scores_sorted.head(10))
        print("-----------------------------")

        print("Data Frame com as 10 melhores features:")
        sorted_indices = np.argsort(res_fish)[::-1]
        top_10_indices = sorted_indices[:10]
        features_fisher_df = features_loaded_df.iloc[:, top_10_indices]
        features_fisher_df["Activity Label"] = features_loaded_df["Activity Label"]
        features_fisher_df["Device ID"] = features_loaded_df["Device ID"]
        print(features_fisher_df)
        print("-----------------------------")

        # ReliefF
        res_relief = reliefF(X_np, y)
        print("Resultados teste ReliefF (Top 10)")
        res_relief = pd.Series(res_relief, index=feature_names)
        res_relief = res_relief.sort_values(ascending=False)
        print(res_relief.head(10))
        print("-----------------------------")

        print("Data Frame com as 10 melhores features:")
        sorted_indices = np.argsort(res_relief)[::-1]
        top_10_indices = sorted_indices[:10]
        features_relief_df = features_loaded_df.iloc[:, top_10_indices]
        features_relief_df["Activity Label"] = features_loaded_df["Activity Label"]
        features_relief_df["Device ID"] = features_loaded_df["Device ID"]
        print(features_relief_df)
        print("-----------------------------")
        
        # ====================================================================================
        print("\n\n\t\t\t======== DONE ========\n\n")
        # ====================================================================================

if __name__ == "__main__":
    main()