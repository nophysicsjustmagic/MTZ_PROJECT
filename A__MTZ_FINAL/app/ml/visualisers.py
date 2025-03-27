import matplotlib
import matplotlib.pyplot as plt
import io
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)


class Visualizer:
    """Класс для построения и сохранения визуализаций."""

    @staticmethod
    def visualize_results(df_prepared, df_filtered, x_orig, y_orig, x_smooth, y_smooth,
                          history, eval_data):
        """
        Строит графики:
        - Исходная и сглаженная кривая импеданса
        - Распределение параметров до и после фильтрации
        - Графики обучения (если история доступна)
        - ROC-кривая
        Возвращает два буфера (с 4 подграфиками и с ROC-кривой).
        """
        X_test, y_test, y_pred_prob = eval_data

        # Первая фигура (4 подграфика)
        plt.figure(figsize=(16, 12))
        # (1) Исходная vs сглаженная кривая
        plt.subplot(2, 2, 1)
        plt.plot(x_orig, y_orig, 'o', markersize=3, label='Исходные данные')
        plt.plot(x_smooth, y_smooth, '-', linewidth=2, label='Сглаженная кривая')
        plt.xlabel('x')
        plt.ylabel('Нормированный импеданс')
        plt.title('Сплайн-интерполяция импеданса')
        plt.legend()

        # (2) Распределение до/после фильтрации
        plt.subplot(2, 2, 2)
        plt.scatter(df_prepared['resistance'], df_prepared['phase'],
                    c='gray', alpha=0.5, label='До фильтрации')
        plt.scatter(df_filtered['resistance'], df_filtered['phase'],
                    c='blue', alpha=0.8, label='После фильтрации')
        plt.xlabel('Нормированный resistance')
        plt.ylabel('Нормированный phase')
        plt.title('Фильтрация выбросов')
        plt.legend()

        # (3) График потерь
        plt.subplot(2, 2, 3)
        if history is not None and 'loss' in history:
            plt.plot(history['loss'], label='Обучение (loss)')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Валидация (loss)')
            plt.xlabel('Эпоха')
            plt.ylabel('Потеря')
            plt.title('График потерь')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'История обучения не доступна',
                     horizontalalignment='center', verticalalignment='center')

        # (4) График точности
        plt.subplot(2, 2, 4)
        if history is not None and 'accuracy' in history:
            plt.plot(history['accuracy'], label='Обучение (accuracy)')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Валидация (accuracy)')
            plt.xlabel('Эпоха')
            plt.ylabel('Точность')
            plt.title('График точности')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'История обучения не доступна',
                     horizontalalignment='center', verticalalignment='center')

        plt.tight_layout()
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png')
        buf1.seek(0)

        # Вторая фигура: ROC-кривая
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_val = roc_auc_score(y_test, y_pred_prob)
        plt.plot(fpr, tpr, label=f'ROC-кривая (AUC = {auc_val:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая')
        plt.legend()

        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        plt.close('all')

        return buf1, buf2