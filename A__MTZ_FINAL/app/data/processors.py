from scipy import interpolate, stats
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
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



import numpy as np
import pandas as pd

class DataProcessor:
    @staticmethod
    def _calculate_phase_resistance(df: pd.DataFrame):
        """Рассчитывает phase и resistance из Re(Z) и Im(Z) при необходимости"""
        if 'Re(Z)' in df.columns and 'Im(Z)' in df.columns:
            df['resistance'] = np.sqrt(df['Re(Z)'] ** 2 + df['Im(Z)'] ** 2)
            df['phase'] = np.degrees(np.arctan2(df['Im(Z)'], df['Re(Z)']))
            df.drop(['Re(Z)', 'Im(Z)'], axis=1, inplace=True)
        return df

    @staticmethod
    def _validate_physical_limits(df: pd.DataFrame):
        """Фильтрация по физическим ограничениям"""
        # Фильтрация некорректных значений
        mask = (
                (df['resistance'] > 0) &
                (df['phase'].between(-90, 90))  # Допустимый диапазон фаз
        )
        invalid_count = len(df) - mask.sum()
        if invalid_count > 0:
            print(f"Удалено {invalid_count} записей с физически некорректными значениями")
        return df[mask]

    @staticmethod
    def load_and_prepare_data(file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        print("Первые строки загруженных данных:")
        print(df.head())

        # Автоматическое определение типа данных
        df = DataProcessor._calculate_phase_resistance(df)

        if df.isnull().sum().sum() > 0:
            df.fillna(method='ffill', inplace=True)

        # Фильтрация физически некорректных значений
        df = DataProcessor._validate_physical_limits(df)

        # Нормализация
        for col in ['impedance', 'resistance', 'phase']:
            if col in df.columns:
                df[col] = (df[col] - df[col].mean()) / df[col].std()

        print("Статистика данных после нормализации:")
        print(df.describe())
        return df

    @staticmethod
    def filter_outliers(df: pd.DataFrame, cols=None) -> pd.DataFrame:
        """Автоматический выбор колонок для анализа"""
        if cols is None:
            cols = [c for c in ['impedance', 'resistance', 'phase'] if c in df.columns]

        iso = IsolationForest(contamination=0.05, random_state=42)
        df['anomaly'] = iso.fit_predict(df[cols])
        df_filtered = df[df['anomaly'] == 1].copy()
        print(f"Удалено {df.shape[0] - df_filtered.shape[0]} выбросов методом Isolation Forest")

        z_scores = np.abs(stats.zscore(df_filtered[cols]))
        mask = (z_scores < 3).all(axis=1)
        df_filtered = df_filtered[mask].copy()
        print(f"Осталось {df_filtered.shape[0]} точек после фильтрации по z-score")

        return df_filtered

    @staticmethod
    def smooth_and_interpolate(df: pd.DataFrame, col='impedance', smooth_param=5):
        df_sorted = df.sort_values('x')
        x = df_sorted['x'].values
        y = df_sorted[col].values

        spline = interpolate.UnivariateSpline(x, y, s=smooth_param)
        x_smooth = np.linspace(x.min(), x.max(), 500)
        y_smooth = spline(x_smooth)

        return x, y, x_smooth, y_smooth