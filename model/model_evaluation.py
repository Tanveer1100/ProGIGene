import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, brier_score_loss, accuracy_score
)
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Inputs: adjust these if needed
# ----------------------------
y_true = risk_test_labels.values
y_prob = final_model.predict_proba(X_test_final_aligned)[:, 1]
threshold = best_thresh  # from F1 optimization
y_pred = (y_prob >= threshold).astype(int)

# ----------------------------
# Confusion Matrix
# ----------------------------
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# ----------------------------
# Performance Metrics
# ----------------------------
metrics = {
    'ROC AUC': roc_auc_score(y_true, y_prob),
    'PR AUC': average_precision_score(y_true, y_prob),
    'F1 Score': f1_score(y_true, y_pred),
    'Precision': precision_score(y_true, y_pred),
    'Recall (Sensitivity)': recall_score(y_true, y_pred),
    'Specificity': tn / (tn + fp),
    'Accuracy': accuracy_score(y_true, y_pred),
    'Brier Score': brier_score_loss(y_true, y_prob)
}

# ----------------------------
# Optional: Permutation Test
# ----------------------------
from sklearn.utils import shuffle
from copy import deepcopy

n_permutations = 100
perm_scores = []
for i in range(n_permutations):
    y_perm = shuffle(y_true, random_state=i)
    score = roc_auc_score(y_perm, y_prob)
    perm_scores.append(score)

perm_pval = np.mean(np.array(perm_scores) >= metrics['ROC AUC'])

# Add permutation result
metrics['Permutation p-value'] = perm_pval

# ----------------------------
# Display Results
# ----------------------------
metrics_df = pd.DataFrame(metrics, index=['Model Performance']).T
print("\n=== 📊 Final Model Performance Metrics ===\n")
print(metrics_df.round(4))

# ----------------------------
# Optional: Plot Confusion Matrix
# ----------------------------
conf_matrix = np.array([[tn, fp], [fn, tp]])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Pred: Low", "Pred: High"],
            yticklabels=["True: Low", "True: High"])
plt.title("Confusion Matrix")
plt.show()
