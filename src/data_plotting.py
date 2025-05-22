

from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
def plot_precision_recall_curve(title, y, y_scores, ax):

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y, y_scores)
    avg_precision = average_precision_score(y, y_scores)

    # Plotting
    ax.plot(recall, precision, label=f'{title} (AP = {avg_precision:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve (Cross-Validated) for {title}')
    ax.legend()
    ax.grid(True)