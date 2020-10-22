import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure


class MtbVisualizer:

    def print_confusion_matrix(self, y, y_pred, labels, save_filename=False):
        cm = confusion_matrix(y, y_pred, labels=labels, normalize='true')
        fig = plt.figure(figsize=(10, 10), dpi=60)
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + labels)
        ax.set_yticklabels([''] + labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        if save_filename:
            plt.savefig(save_filename)
            plt.close()
        else:
            plt.show()