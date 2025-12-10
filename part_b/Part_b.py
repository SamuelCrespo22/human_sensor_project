# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 09:30:00 2025

@authors: Miguel & Samuel
"""

from config import RANDOM_STATE, BEST_FEATURES
from utils import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample


# ========================
# Exercise Functions
# ========================


def ex_2_1(X, y, ex2=False):
    
    # Train-only
    print("\n=== Train-only ===")
    X_train, y_train = iris.data, iris.target

    if ex2:
        y_pred_random = random_classifier(X_train, y_train, X_train)
        res_train_only_random = show_accuracy_metrics(y_train, y_pred_random, "Random Classifier")
    
    y_pred_oner = oner_class_classifier(X_train, y_train, X_train)
    res_train_only_oner = show_accuracy_metrics(y_train, y_pred_oner, "OneR Classifier")

    # Train-test split 70-30
    print("\n=== Train-test split ===")
    X_train, X_test, y_train, y_test = train_test(X, y, test_size=0.3)

    if ex2:
        y_pred_random = random_classifier(X_train, y_train, X_test)
        res_tt_random = show_accuracy_metrics(y_test, y_pred_random, "Random Classifier")
    
    y_pred_oner = oner_class_classifier(X_train, y_train, X_test)
    res_tt_oner = show_accuracy_metrics(y_test, y_pred_oner, "OneR Classifier")

    # 10x10-fold cross-validation
    print("\n=== 10x10 CV ===")

    if ex2:
        kf_results = {'random': {'Confusion Matrix': [], 'Precision Score': [], 'Recall Score': [], 'F1 Score': []},
                'oner': {'Confusion Matrix': [], 'Precision Score': [], 'Recall Score': [], 'F1 Score': []}}
    else:
        kf_results = {'oner': {'Confusion Matrix': [], 'Precision Score': [], 'Recall Score': [], 'F1 Score': []}}
    
    for i in range(10):
        for X_train_cv, X_test_cv, y_train_cv, y_test_cv in k_fold_split(X.values, y.values, n_splits=10, random_state=RANDOM_STATE+i):

            if ex2:
                y_pred_random_cv = random_classifier(X_train_cv, y_train_cv, X_test_cv)
                kf_results['random']['Confusion Matrix'].append(confusion_matrix(y_test_cv, y_pred_random_cv))
                kf_results['random']['Precision Score'].append(precision_score(y_test_cv, y_pred_random_cv, average='macro'))
                kf_results['random']['Recall Score'].append(recall_score(y_test_cv, y_pred_random_cv, average='macro'))
                kf_results['random']['F1 Score'].append(f1_score(y_test_cv, y_pred_random_cv, average='macro'))

            y_pred_oner_cv = oner_class_classifier(X_train_cv, y_train_cv, X_test_cv)
            kf_results['oner']['Confusion Matrix'].append(confusion_matrix(y_test_cv, y_pred_oner_cv))
            kf_results['oner']['Precision Score'].append(precision_score(y_test_cv, y_pred_oner_cv, average='macro'))
            kf_results['oner']['Recall Score'].append(recall_score(y_test_cv, y_pred_oner_cv, average='macro'))
            kf_results['oner']['F1 Score'].append(f1_score(y_test_cv, y_pred_oner_cv, average='macro'))

    df_random = pd.DataFrame(kf_results['random']) if ex2 else None
    df_oner = pd.DataFrame(kf_results['oner'])

    print("\n=== Results 10x10 CV (Avg +/- Std) ===")
    for name, df in [('Random', df_random), ('OneR', df_oner)]:
        print(f"\n=== {name} ===")
        print(f"Precision: {df['Precision Score'].mean():.3f} (+/- {df['Precision Score'].std():.3f})")
        print(f"Recall:    {df['Recall Score'].mean():.3f} (+/- {df['Recall Score'].std():.3f})")
        print(f"F1 Score:  {df['F1 Score'].mean():.3f} (+/- {df['F1 Score'].std():.3f})")
        
        cm_list = df['Confusion Matrix'].tolist()
        cm_avg = np.mean(cm_list, axis=0)
        print("Confusion Matrix Average:\n", cm_avg)

    return ""

def ex_2_2_1(X, y):

    # kNN_1 (k=1)

    # Train-only
    print("\n=== Train-only ===")
    X_train, y_train = X, y
    y_pred_knn = knn_classifier(X_train, y_train, X_train, n_neighbors=1)
    res_train_only_knn_1 = show_accuracy_metrics(y_train, y_pred_knn, "KNN Classifier")

    # TT 70-30
    print("\n=== TT 70-30 ===")
    X_train, X_test, y_train, y_test = train_test(X, y, test_size=0.3)
    y_pred_knn = knn_classifier(X_train, y_train, X_test, n_neighbors=1)
    res_tt_knn_1 = show_accuracy_metrics(y_test, y_pred_knn, "KNN Classifier")

    # 10x10 CV
    print("\n=== 10x10 CV ===")
    knn_1_results = {
        'Confusion Matrix': [],
        'Precision Score': [],
        'Recall Score': [],
        'F1 Score': []
    }
    
    for i in range(10):
        for X_train_cv, X_test_cv, y_train_cv, y_test_cv in k_fold_split(X.values, y.values, n_splits=10, random_state=RANDOM_STATE+i):
            y_pred_knn_cv = knn_classifier(X_train_cv, y_train_cv, X_test_cv, n_neighbors=1)
            knn_1_results['Confusion Matrix'].append(confusion_matrix(y_test_cv, y_pred_knn_cv))
            knn_1_results['Precision Score'].append(precision_score(y_test_cv, y_pred_knn_cv, average='macro'))
            knn_1_results['Recall Score'].append(recall_score(y_test_cv, y_pred_knn_cv, average='macro'))
            knn_1_results['F1 Score'].append(f1_score(y_test_cv, y_pred_knn_cv, average='macro'))

    knn_1_summary = {
        'Confusion Matrix Mean': np.mean(knn_1_results['Confusion Matrix'], axis=0),
        'Confusion Matrix Std': np.std(knn_1_results['Confusion Matrix'], axis=0),
        'Precision Score Mean': np.mean(knn_1_results['Precision Score']),
        'Precision Score Std': np.std(knn_1_results['Precision Score']),
        'Recall Score Mean': np.mean(knn_1_results['Recall Score']),
        'Recall Score Std': np.std(knn_1_results['Recall Score']),
        'F1 Score Mean': np.mean(knn_1_results['F1 Score']),
        'F1 Score Std': np.std(knn_1_results['F1 Score'])
    }

    labels = np.unique(y) 
    print("\n=== 10x10 CV Results (Mean +/- Std) ===")
    print(f"Precision: {knn_1_summary['Precision Score Mean']:.3f} (+/- {knn_1_summary['Precision Score Std']:.3f})")
    print(f"Recall:    {knn_1_summary['Recall Score Mean']:.3f} (+/- {knn_1_summary['Recall Score Std']:.3f})")
    print(f"F1 Score:  {knn_1_summary['F1 Score Mean']:.3f} (+/- {knn_1_summary['F1 Score Std']:.3f})")
    print("\nConfusion Matrix (Mean):")
    cm_avg = knn_1_summary['Confusion Matrix Mean'].tolist()
    pretty_print_cm_matrix(cm_avg, labels, title="Confusion Matrix Average")
    print("\nConfusion Matrix (Std):")
    cm_std = knn_1_summary['Confusion Matrix Std'].tolist()
    pretty_print_cm_matrix(cm_std, labels, title="Confusion Matrix STD")
    
    return knn_1_summary

def ex_2_2_2(X, y):

    # kNN, k={1, 3, 5, 7, 9, 11, 13, 15}

    k_results_summary = []

    for k in range(1,16,2):
        print(f"\n=== K = {k} ===")

        # Train-only
        print("\n=== Train-only ===")
        X_train, y_train = X, y
        y_pred_knn = knn_classifier(X_train, y_train, X_train, n_neighbors=k)
        res_train_only_knn = show_accuracy_metrics(y_train, y_pred_knn, "KNN Classifier")

        # TVT 40-30-30
        print("\n=== TVT 40-30-30 ===")
        X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test(
            X_train, y_train, test_size=0.3, val_size=0.4286
        )
        y_pred_knn_tvt = knn_classifier(X_train, y_train, X_test, n_neighbors=k)
        res_tvt_knn = show_accuracy_metrics(y_test, y_pred_knn_tvt, "KNN Classifier")

        # 10x10 CV
        print("\n=== 10x10 CV ===")
        cv_metrics = {
            'Confusion Matrix': [],
            'Precision Score': [],
            'Recall Score': [],
            'F1 Score': []
        }
        
        for i in range(10):
            for X_train_cv, X_test_cv, y_train_cv, y_test_cv in k_fold_split(X.values, y.values, n_splits=10):
                y_pred_knn_cv = knn_classifier(X_train_cv, y_train_cv, X_test_cv, n_neighbors=k)
                cv_metrics['Confusion Matrix'].append(confusion_matrix(y_test_cv, y_pred_knn_cv))
                cv_metrics['Precision Score'].append(precision_score(y_test_cv, y_pred_knn_cv, average='macro'))
                cv_metrics['Recall Score'].append(recall_score(y_test_cv, y_pred_knn_cv, average='macro'))
                cv_metrics['F1 Score'].append(f1_score(y_test_cv, y_pred_knn_cv, average='macro'))

        stats = {
            'k': k,
            'Train_Only_Prec': res_train_only_knn['Precision Score'],
            'Train_Only_Rec':  res_train_only_knn['Recall Score'],
            'Train_Only_F1':   res_train_only_knn['F1 Score'],
            'Train_Only_CM':   res_train_only_knn['Confusion Matrix'],
            'TVT_Prec':  res_tvt_knn['Precision Score'],
            'TVT_Rec':   res_tvt_knn['Recall Score'],
            'TVT_F1':    res_tvt_knn['F1 Score'],
            'TVT_CM':    res_tvt_knn['Confusion Matrix'],
            'CV_Prec_Mean': np.mean(cv_metrics['Precision Score']),
            'CV_Prec_Std':  np.std(cv_metrics['Precision Score']),
            'CV_Rec_Mean':  np.mean(cv_metrics['Recall Score']),
            'CV_Rec_Std':   np.std(cv_metrics['Recall Score']),
            'CV_F1_Mean':   np.mean(cv_metrics['F1 Score']),
            'CV_F1_Std':    np.std(cv_metrics['F1 Score']),
            'CV_CM_Mean':   np.mean(cv_metrics['Confusion Matrix'], axis=0),
            'CV_CM_Std':    np.std(cv_metrics['Confusion Matrix'], axis=0)
        }

        k_results_summary.append(stats)

        labels = np.unique(y) 
        print(f"\nk={k}: 10x10 CV Summary:")
        print(f"  Precision: {stats['CV_Prec_Mean']:.3f} (+/- {stats['CV_Prec_Std']:.3f})")
        print(f"  Recall:    {stats['CV_Rec_Mean']:.3f} (+/- {stats['CV_Rec_Std']:.3f})")
        print(f"  F1 Score:  {stats['CV_F1_Mean']:.3f} (+/- {stats['CV_F1_Std']:.3f})")
        print("\nConfusion Matrix (Mean):")
        cm_avg = stats['CV_CM_Mean'].tolist()
        pretty_print_cm_matrix(cm_avg, labels, title="Confusion Matrix Average")
        print("\nConfusion Matrix (Std):")
        cm_std = stats['CV_CM_Std'].tolist()
        pretty_print_cm_matrix(cm_std, labels, title="Confusion Matrix STD")
    
    df_summary = pd.DataFrame([{k: v for k, v in d.items() if k != 'CV_CM_Mean'} for d in k_results_summary])
    print("\n", df_summary[['k', 'Train_Only_F1', 'TVT_F1', 'CV_F1_Mean', 'CV_F1_Std']])

    # Best k Based on F1 Score
    best_k_idx = df_summary['CV_F1_Mean'].idxmax()
    best_k = int(df_summary.loc[best_k_idx, 'k'])
    best_f1 = df_summary.loc[best_k_idx, 'CV_F1_Mean']
    
    print(f"\nBest k based on 10x10 CV F1 Score: k = {best_k} (F1 = {best_f1:.3f})")
    print(f"\nConfusion Matrix Mean for k = {best_k}:")
    cm_avg = k_results_summary[best_k_idx]['CV_CM_Mean'].tolist()
    pretty_print_cm_matrix(cm_avg, labels, title="Confusion Matrix Average")
    
    # Visualization
    print("\n=== Creating Plots ===")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: F1 Score Comparison (Train-only, TVT, CV)
    axes[0, 0].plot(df_summary['k'], df_summary['Train_Only_F1'], 'o-', label='Train-only', linewidth=2, markersize=8)
    axes[0, 0].plot(df_summary['k'], df_summary['TVT_F1'], 's-', label='TVT (40-30-30)', linewidth=2, markersize=8)
    axes[0, 0].errorbar(df_summary['k'], df_summary['CV_F1_Mean'], 
                        yerr=df_summary['CV_F1_Std'], fmt='d-', label='10x10 CV', 
                        linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[0, 0].set_xlabel('k (number of neighbors)', fontsize=11)
    axes[0, 0].set_ylabel('F1 Score', fontsize=11)
    axes[0, 0].set_title('F1 Score vs k', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(range(1, 16, 2))
    
    # Plot 2: CV metrics (Precision, Recall, F1)
    axes[0, 1].errorbar(df_summary['k'], df_summary['CV_Prec_Mean'], 
                        yerr=df_summary['CV_Prec_Std'], fmt='o-', label='Precision', 
                        linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[0, 1].errorbar(df_summary['k'], df_summary['CV_Rec_Mean'], 
                        yerr=df_summary['CV_Rec_Std'], fmt='s-', label='Recall', 
                        linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[0, 1].errorbar(df_summary['k'], df_summary['CV_F1_Mean'], 
                        yerr=df_summary['CV_F1_Std'], fmt='d-', label='F1', 
                        linewidth=2, markersize=8, capsize=5, capthick=2)
    axes[0, 1].set_xlabel('k (number of neighbors)', fontsize=11)
    axes[0, 1].set_ylabel('Score', fontsize=11)
    axes[0, 1].set_title('10x10 CV Metrics vs k', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(range(1, 16, 2))
    
    # Plot 3: Bias-Variance (Train vs CV performance)
    train_cv_gap = df_summary['Train_Only_F1'] - df_summary['CV_F1_Mean']
    axes[1, 0].plot(df_summary['k'], train_cv_gap, 'o-', linewidth=2, markersize=8, color='red')
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('k (number of neighbors)', fontsize=11)
    axes[1, 0].set_ylabel('Train F1 - CV F1 (Overfitting Gap)', fontsize=11)
    axes[1, 0].set_title('Bias-Variance Trade-off', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(range(1, 16, 2))
    
    # Plot 4: Standard deviation (model stability)
    axes[1, 1].plot(df_summary['k'], df_summary['CV_F1_Std'], 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 1].set_xlabel('k (number of neighbors)', fontsize=11)
    axes[1, 1].set_ylabel('F1 Standard Deviation', fontsize=11)
    axes[1, 1].set_title('Model Stability (10x10 CV)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(1, 16, 2))
    
    plt.tight_layout()
    plt.savefig('figures_b/knn_analysis_exercise_2_2_2.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'knn_analysis_exercise_2_2_2.png'")
    plt.show()

    return k_results_summary

def ex_2_3(X, y):

    # TVT 40-30-30, ReliefF, kNN.

    # ===========================================================
    # 2.3.1 - Data Splitting, Feature Ranking, Feature Selection
    # ===========================================================
    print("\n=== Data Splitting ===")
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test(X, y, test_size=0.3, val_size=0.4286)

    print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Feature Ranking
    print("\n=== Feature Ranking ===")
    n_total_features = X_train.shape[1]
    feature_ranking = ReliefF(n_features_to_select=n_total_features, n_neighbors=10, n_jobs=-1)
    feature_ranking.fit(X_train.values, y_train.values.ravel())

    feature_scores = pd.Series(
        feature_ranking.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    print("\nFeature importance scores (ReliefF):")
    for feat, score in feature_scores.items():
        print(f"  {feat}: {score:.4f}")
    
    ordered_features = feature_scores.index.tolist()

    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )

    # Feature Selection
    print("\n=== Feature Selection ===")
    f1_results = {}
    for num_features in range(1, len(ordered_features) + 1):
        selected_features = ordered_features[:num_features]
        res_temp_knn = knn_classifier(X_train_scaled[selected_features], y_train, X_val_scaled[selected_features], n_neighbors=1)
        f1_results[num_features] = f1_score(y_val, res_temp_knn)
    
    # Best Feature Set Based on 95% of Maximum F1-Score
    max_f1 = max(f1_results.values())
    threshold_95 = 0.95 * max_f1
    absolute_max = max(f1_results, key=f1_results.get)

    best_num_features = None
    for num_features in sorted(f1_results.keys()):
        if f1_results[num_features] >= threshold_95:
            best_num_features = num_features
            break

    best_features = ordered_features[:best_num_features]
    
    print(f"Maximum F1-Score: {max_f1:.4f} (with {absolute_max} features)")
    print(f"95% threshold: {threshold_95:.4f}")
    print(f"Selected features: {best_num_features} (F1={f1_results[best_num_features]:.4f})")
    print(f"\nChosen features: {best_features}")

    # ===========================================================
    # 2.3.2 - Features Graph
    # ===========================================================
    print("\n=== Features Graph ===")

    plt.figure(figsize=(12, 7))
    num_features_list = list(f1_results.keys())
    f1_values_list = list(f1_results.values())

    plt.plot(num_features_list, f1_values_list, marker='o', linestyle='-', linewidth=2, markersize=4)
    plt.axhline(y=threshold_95, color='orange', linestyle=':', label=f'95% threshold ({threshold_95:.3f})')
    plt.axvline(x=best_num_features, color='r', linestyle='--', linewidth=2, label=f'Selected: {best_num_features} features')
    plt.axvline(x=absolute_max, color='green', linestyle='--', linewidth=1, alpha=0.5, label=f'Maximum: {absolute_max} features')

    plt.title('Elbow Graph: F1-Score vs Number of Features (Validation Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Features (ordered by ReliefF)', fontsize=12)
    plt.ylabel('F1-Score (Validation)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('figures_b/elbow_graph_features.png', dpi=300)
    plt.show()

    # ===========================================================
    # 2.3.3 - Parameter Optimization
    # ===========================================================
    print("\n=== Parameter Optimization ===")

    all_parameters = {}
    X_train_selected = X_train_scaled[best_features]
    X_val_selected = X_val_scaled[best_features]

    for k in range(1, 16, 2):
        all_parameters[k] = {}

        y_pred_knn_tvt = knn_classifier(X_train_selected, y_train, X_val_selected, n_neighbors=k)
        res_temp_knn = show_accuracy_metrics(y_val, y_pred_knn_tvt, "KNN Classifier", show=False)
        
        all_parameters[k]['F1 Score'] = res_temp_knn['F1 Score']
        all_parameters[k]['Confusion Matrix'] = res_temp_knn['Confusion Matrix']
        all_parameters[k]['Precision Score'] = res_temp_knn['Precision Score']
        all_parameters[k]['Recall Score'] = res_temp_knn['Recall Score']

    best_k = max(all_parameters, key=lambda k: all_parameters[k]['F1 Score'])
    print(f"\nBest k: {best_k} with F1-score: {all_parameters[best_k]['F1 Score']:.4f}")

    # Parameter Graph
    plt.figure(figsize=(10, 6))
    k_values = list(all_parameters.keys())
    f1_k_values = list(all_parameters[k]['F1 Score'] for k in k_values)

    plt.plot(k_values, f1_k_values, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'Best k={best_k}')

    plt.title('Parameter Optimization: F1-Score vs k (Validation Set)', fontsize=14, fontweight='bold')
    plt.xlabel('k (number of neighbors)', fontsize=12)
    plt.ylabel('F1-Score (Validation)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('figures_b/parameter_optimization_k.png', dpi=300)
    plt.show()

    # ===========================================================
    # 2.3.4 - Analysis of All Feature Sets and All K
    # ===========================================================
    print("\n=== Analysis ===")
    # Bias-variance, underfitting-overfitting
    # Create comprehensive results table
    all_combinations = []
    for n_feat in range(1, len(ordered_features) + 1):
        sel_features = ordered_features[:n_feat]
        X_tr_sel = X_train[sel_features]
        X_val_sel = X_val[sel_features]
        
        for k in range(1, 16, 2):
            y_pred = knn_classifier(X_tr_sel, y_train, X_val_sel, n_neighbors=k)
            f1 = f1_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            rec = recall_score(y_val, y_pred)
            
            all_combinations.append({
                'n_features': n_feat,
                'k': k,
                'F1 Score': f1,
                'Precision Score': prec,
                'Recall Score': rec
            })
    
    df_combinations = pd.DataFrame(all_combinations)

    print("\nTop 10 combinations (by F1-score):")
    top_10 = df_combinations.nlargest(10, 'F1 Score')
    print(top_10.to_string(index=False))

    # Heatmap of F1-scores
    pivot_f1 = df_combinations.pivot(index='k', columns='n_features', values='F1 Score')
    
    plt.figure(figsize=(12, 8))
    im = plt.imshow(pivot_f1.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, label='F1-Score')
    plt.yticks(range(len(pivot_f1.index)), pivot_f1.index)
    plt.xticks(range(len(pivot_f1.columns)), pivot_f1.columns)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('k (neighbors)', fontsize=12)
    plt.title('F1-Score Heatmap: All Feature-Parameter Combinations', 
            fontsize=14, fontweight='bold')
    
    # Mark best combination
    best_row = df_combinations.loc[df_combinations['F1 Score'].idxmax()]
    best_feat_idx = list(pivot_f1.columns).index(best_row['n_features'])
    best_k_idx = list(pivot_f1.index).index(best_row['k'])
    plt.plot(best_feat_idx, best_k_idx, 'b*', markersize=20, 
            label=f"Best: {int(best_row['n_features'])} feat, k={int(best_row['k'])}")
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures_b/2_3_4_heatmap_combinations.png', dpi=300, bbox_inches='tight')
    print("\n  → Heatmap saved as '2_3_4_heatmap_combinations.png'")
    plt.show()

    # ===========================================================
    # 2.3.5 - Ideal Model Results
    # ===========================================================
    print("\n=== Final Model ===")
    X_train_full = pd.concat([X_train_selected, X_val_selected])
    y_train_full = pd.concat([y_train, y_val])
    X_test_selected = X_test[best_features]

    y_pred_final = knn_classifier(X_train_full, y_train_full, X_test_selected, n_neighbors=best_k)
    final_results = show_accuracy_metrics(y_test, y_pred_final, "Final kNN Model", show=True)

    print(f"\n=== FINAL RESULTS ON TEST SET ===")
    print(f"Number of features: {best_num_features}")
    print(f"Best k: {best_k}")
    print(f"Test F1-Score: {final_results['F1 Score']:.3f}")

    # Comparison
    val_f1 = all_parameters[best_k]['F1 Score']
    test_f1 = final_results['F1 Score']

    print(f"\n--- Comparison: Validation vs Test ---")
    print(f"  Validation F1: {val_f1:.4f}")
    print(f"  Test F1:       {test_f1:.4f}")
    print(f"  Difference:    {abs(val_f1 - test_f1):.4f}")

    if abs(val_f1 - test_f1) < 0.05:
        print(f"  → Model generalizes well (small difference)")
    elif test_f1 < val_f1:
        print(f"  → Test performance slightly lower (expected, unseen data)")
    else:
        print(f"  → Test performance higher (fortunate test split)")

    # Recall
    val_rec = all_parameters[best_k]['Recall Score']
    test_rec = final_results['Recall Score']
    print(f"  Validation Recall: {val_rec:.4f}")
    print(f"  Test Recall:       {test_rec:.4f}")
    print(f"  Difference:    {abs(val_rec - test_rec):.4f}")

    # Precision
    val_prec = all_parameters[best_k]['Precision Score']
    test_prec = final_results['Precision Score']
    print(f"  Validation Precision: {val_prec:.4f}")
    print(f"  Test Precision:       {test_prec:.4f}")
    print(f"  Difference:    {abs(val_prec - test_prec):.4f}")

    # Confusion Matrix
    labels = np.unique(y)
    val_cm = all_parameters[best_k]['Confusion Matrix']
    test_cm = final_results['Confusion Matrix']
    pretty_print_cm_matrix(val_cm, labels, title="Validation Confusion Matrix")
    pretty_print_cm_matrix(test_cm, labels, title="Test Confusion Matrix")

    # ===========================================================
    # 2.3.6 - Deployment Model
    # ===========================================================
    print("\n=== Deployment Model ===")
    X_train_deployment = pd.concat([X_train_full, X_test_selected])
    y_train_deployment = pd.concat([y_train_full, y_test])
    y_pred_deployment = knn_classifier(X_train_deployment, y_train_deployment, X_test_selected, n_neighbors=best_k)
    
    print(f"\nSelected Features ({best_num_features}):")
    for i, feat in enumerate(best_features, 1):
        print(f"  {i}. {feat} (importance: {feature_scores[feat]:.4f})")
    
    print(f"\nOptimal Parameters:")
    print(f"  • Algorithm: k-Nearest Neighbors (kNN)")
    print(f"  • k (neighbors): {best_k}")
    print(f"  • Distance metric: Euclidean (default)")
    print(f"  • Weights: Uniform (default)")
    
    print(f"\nExpected Performance:")
    print(f"  • F1-Score: {test_f1:.4f}")
    print(f"  • Precision: {final_results['Precision Score']:.4f}")
    print(f"  • Recall: {final_results['Recall Score']:.4f}")

    return 

def ex_2_4(X, y):

    n_total_features = X.shape[1]

    # 1.1. e 1.2. Aleatoriedade e estratificação.
    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=RANDOM_STATE)

    for i, (train_index, test_index) in enumerate(rskf.split(X.values, y.values)):

        # 1.3. Organizar em Tr e Te
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        
        # 2.1. (Nested CV, 5-fold)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for tr_s_idx, val_idx in inner_cv.split(X_train, y_train):
            
            # Organizar em TrS e Tv
            X_tr_s, X_val = X_train[tr_s_idx], X_train[val_idx]
            y_tr_s, y_val = y_train[tr_s_idx], y_train[val_idx]

            # 2.2. Feature Ranking (TrS)
            fs = ReliefF(n_features_to_select=n_total_features, n_neighbors=10)
            fs.fit(X_tr_s, y_tr_s)

            ranking = fs.top_features_

            # Guardar scores para gráfico.
            scores_v = []
            
            # 2.3.1.2 Avaliar conjuntos de features (1, 1+2, 1+2+3...)
            for n_feats in range(1, n_total_features+1):
                
                # Selecionar as Top-N features do ranking
                selected_cols = ranking[:n_feats]
                
                # Reduzir TrS e V
                X_tr_s_sel = X_tr_s[:, selected_cols]
                X_val_sel  = X_val[:, selected_cols]
                
                y_pred_val = knn_classifier(X_tr_s_sel, y_tr_s, X_val_sel, n_neighbors=1)
                score = f1_score(y_val, y_pred_val, average='weighted')
                
                scores_v.append(score)

            # 2.3.1.3 escolher os 95%
            best_score = max(scores_v)
            # Critério de 95% do máximo.
            threshold = 0.95 * best_score
            best_n_features = next(i for i, s in enumerate(scores_v) if s >= threshold) + 1

            # Gráfico do cotovelo
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, n_total_features+1), scores_v, marker='o', linestyle='-')
            plt.title('Elbow Graph: F1-Score vs Número de Features')
            plt.xlabel('Número de Features (ordenadas pelo ReliefF)')
            plt.ylabel('F1-Score (Validation)')
            plt.axvline(x=best_n_features, color='r', linestyle='--', label=f'Best k={best_n_features}')
            plt.grid(True)
            plt.xticks(range(1, n_total_features+1))
            plt.show()
            # --- FIM DO INNER LOOP ---
            # Agora sabemos que, para esta dobra, o melhor é usar 'best_n_features'.
    

    return ""

def ex_4(X, y, selected_features):
    
    X_sel = X[selected_features]
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test(
        X_sel, y, test_size=0.3, val_size=0.4286
    )

    # Scaling (Fit no Train, Transform no Val e Test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # hidden_layer_sizes=(N,) define a camada escondida com N neurónios
    hidden_neurons = (100,)
    max_iter = 1000  # Aumentar se não convergir
    
    scenarios = [
        {
            "name": "4.1 - LR Fixa",
            "params": {
                "learning_rate": "constant", 
                "learning_rate_init": 0.01, 
                "momentum": 0
            }
        },
        {
            "name": "4.2 - LR Variável (Adaptive)",
            "params": {
                "learning_rate": "adaptive", 
                "learning_rate_init": 0.01, 
                "momentum": 0
            }
        },
        {
            "name": "4.3 - LR Variável + Momentum",
            "params": {
                "learning_rate": "adaptive", 
                "learning_rate_init": 0.01, 
                "momentum": 0.9
            }
        }
    ]

    results = []

    for scen in scenarios:
        print(f"\n--- {scen['name']} ---")
        
        # Batch learning simulado com batch_size='auto' (minibatch) ou len(X_train)
        # O enunciado pede Batch Learning, mas SGD puro geralmente usa mini-batches.
        # Vamos usar solver='sgd' que é necessário para controlar LR e Momentum.
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_neurons,
            activation='relu', # ou 'logistic'
            solver='sgd',
            batch_size=200, # Ajustar para simular batch vs mini-batch
            random_state=RANDOM_STATE,
            max_iter=max_iter,
            early_stopping=True, # Usa conjunto de validação interno
            validation_fraction=0.1,
            **scen['params']
        )
        
        # Treino
        mlp.fit(X_train_scaled, y_train)
        
        # Previsão no conjunto de Teste (conforme enunciado 4 pede comparação geral)
        # Nota: O enunciado diz para usar TVT, idealmente validas no Val e testas no Test.
        # Aqui apresento resultados do Teste para o output final.
        y_pred = mlp.predict(X_test_scaled)
        
        # Métricas
        metrics_res = show_accuracy_metrics(y_test, y_pred, f"MLP {scen['name']}", show=True)
        results.append((scen['name'], metrics_res['F1 Score']))

        # Curva de perdas (Loss Curve) - Opcional mas útil
        plt.plot(mlp.loss_curve_, label=scen['name'])

    # Mostrar gráfico de convergência
    plt.title("Curva de Aprendizagem (Loss over epochs)")
    plt.xlabel("Iterações")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Comparação Final
    print("\n=== Resumo F1-Score (Test Set) ===")
    for name, f1 in results:
        print(f"{name}: {f1:.4f}")
    
    return ""


# ========================
# Main
# ========================


if __name__ == "__main__":
    iris = load_iris_data()
    df = load_ds_data()

    # Choose device 1 for now.
    df = df[df['Device ID'] == 1]
    X_ds = df.drop(['Activity Label', 'Device ID'], axis=1)
    y_ds = df['Activity Label']

    X = iris.data
    y = iris.target

    exercises_to_run = ["2.5"]

    if "2.1" in exercises_to_run:
        print("=== 2.1 ===")
        ex_2_1(X, y, ex2=True)

    if "2.2.1" in exercises_to_run:
        print("\n=== 2.2.1 ===")
        res_2_2_1 = ex_2_2_1(X, y)

    if "2.2.2" in exercises_to_run:
        print("\n=== 2.2.2 ===")
        res_2_2_2 = ex_2_2_2(X, y)

    if "2.3" in exercises_to_run:
        print("\n=== 2.3 ===")
        ex_2_3(X, y)

    if "2.4" in exercises_to_run:
        print("\n=== 2.4 ===")
        ex_2_4(X, y)

    if "2.5" in exercises_to_run:
        # Class Imbalance
        print("\n=== 2.5 ===")

        TARGET_CLASS_LIMITS = {
            'iris-setosa': 50,
            'iris-versicolor': 30,
            'iris-virginica': 10
        }

        X_undersampled = pd.DataFrame()
        y_undersampled = pd.Series(dtype='object')

        for class_name, limit in TARGET_CLASS_LIMITS.items():
            X_class = X[y == class_name]
            y_class = y[y == class_name]

            # Se o limite for maior ou igual ao que já existe, mantém tudo.
            # Caso contrário, aplica o undersampling.
            if len(X_class) <= limit:
                X_resampled = X_class
                y_resampled = y_class
            else:
                X_resampled, y_resampled = resample(
                    X_class, 
                    y_class, 
                    replace=False,           # Amostragem sem reposição (undersampling)
                    n_samples=limit,         # Número desejado de amostras
                    random_state=RANDOM_STATE
                )
            
            X_undersampled = pd.concat([X_undersampled, X_resampled])
            y_undersampled = pd.concat([y_undersampled, y_resampled])

        X = X_undersampled
        y = y_undersampled

        print("=== NOVO CONJUNTO DE DADOS (X, y) ===")
        print(f"Total de Amostras: {len(X)}")
        print("Distribuição das Classes:")
        print(y.value_counts())
        print("-" * 35)

        ex_2_3(X, y)

    if "3.1" in exercises_to_run:
        X, y = X_ds, y_ds
        print("=== 3.1 ===")
        ex_2_1(X, y)

    if "3.2.1" in exercises_to_run:
        print("\n=== 3.2.1 ===")
        res_3_2_1 = ex_2_2_1(X_ds, y_ds)

    if "3.2.2" in exercises_to_run:
        print("\n=== 3.2.2 ===")
        res_3_2_2 = ex_2_2_2(X_ds, y_ds)

    if "3.3" in exercises_to_run:
        print("\n=== 3.3 ===")
        ex_2_3(X_ds, y_ds)

    if "4" in exercises_to_run:
        # TVT, MLP 3 Layers, 
        print("\n=== 4 ===")

        best_features = BEST_FEATURES

        ex_4(X_ds, y_ds, best_features)