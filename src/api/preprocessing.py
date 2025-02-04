from typing import List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    mean_fill_values,
    outlier_clip_ranges,
    kmeans_model,
    most_freq_ids,
    columns_to_drop_initial,
    columns_to_drop_correlated,
    label_encoders,
    numerical_scaler,
    categorical_cols_names,
    numerical_cols_names,
    CLASS_WEIGHTS,
)


def split_address(address: str) -> tuple[str, str]:
    """Разделяет адрес на название улицы и тип улицы.

    Args:
        address (str): Полный адрес строкой.

    Returns:
        tuple[str, str]: Кортеж, содержащий название улицы и тип улицы.
                         Возвращает ("missing", "missing")
                         если адрес не может быть распознан.
    """
    parts = address.split()
    if not parts:
        return "missing", "missing"
    street_parts = parts[1:]
    if not street_parts:
        return "missing", "missing"
    if len(street_parts) == 1:
        return "missing", street_parts[0]
    street_type = street_parts[-1]
    street_name_parts = street_parts[:-1]
    street_name = " ".join(street_name_parts)
    return street_name, street_type


def preprocess_input_data(input_features_list: List[dict]) -> pd.DataFrame:
    """Предварительная обработка входных данных для модели.

    Выполняет следующие шаги:
    1. Обработка пропущенных значений.
    2. Удаление неинформативных и коррелированных столбцов.
    3. Обработка выбросов в числовых признаках.
    4. Генерация новых признаков (feature engineering).
    5. Label Encoding категориальных признаков.
    6. Масштабирование числовых признаков.

    Args:
        input_features_list (List[dict]): Список словарей,
        представляющих входные признаки.

    Returns:
        pd.DataFrame: DataFrame с обработанными признаками.
    """
    input_df = pd.DataFrame(input_features_list)

    # Обработка пропущенных значений
    input_df.dropna(subset='sidewalk', inplace=True)
    input_df["guards"].fillna("None", inplace=True)
    input_df['spc_latin'].fillna('None', inplace=True)
    input_df['spc_common'].fillna('None', inplace=True)
    input_df['problems'].fillna('None', inplace=True)
    input_df['steward'].fillna('None', inplace=True)
    input_df['census tract'].fillna(value=mean_fill_values['census tract'],
                                    inplace=True)
    input_df['bbl'].fillna(value=mean_fill_values['bbl'], inplace=True)

    # Удаление столбцов
    input_df.drop(columns=columns_to_drop_initial,
                  axis=1, errors='ignore', inplace=True)
    input_df.drop(columns=columns_to_drop_correlated,
                  axis=1, errors='ignore', inplace=True)

    # Обработка выбросов
    for column in ["tree_dbh", 'x_sp', 'y_sp', 'bbl']:
        lower_bound = 0
        upper_bound = 0
        for ind, cls in enumerate(['Fair', 'Good', 'Poor']):
            lower_bound += (CLASS_WEIGHTS[ind] *
                            outlier_clip_ranges[column][cls]['lower'])
            upper_bound += (CLASS_WEIGHTS[ind] *
                            outlier_clip_ranges[column][cls]['upper'])
        input_df[column] = (input_df[column].
                            clip(lower=lower_bound, upper=upper_bound))

    # Feature Engineering
    input_df['created_month'] = pd.to_datetime(input_df['created_at']).dt.month
    input_df['cluster_label'] = kmeans_model.predict(
        input_df[['x_sp', 'y_sp']])

    street_name_series, street_type_series = (
        zip(*input_df['address'].apply(split_address)))

    input_df['street_name'] = pd.Series(street_name_series,
                                        index=input_df.index)

    input_df['street_type'] = pd.Series(street_type_series,
                                        index=input_df.index)

    input_df['block_id'] = input_df['block_id'].apply(
        lambda x: x if x in most_freq_ids else -1
    )

    input_df["census tract"] = input_df["census tract"].astype(np.int64)

    # Label Encoding
    for col in categorical_cols_names:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Масштабирование числовых признаков
    numerical_input_data = input_df[numerical_cols_names]
    input_df[numerical_cols_names] = (numerical_scaler.
                                      transform(numerical_input_data))

    input_df.drop(columns=["address"], axis=1, errors='ignore', inplace=True)
    return input_df


class TreeHealthDataset(Dataset):
    """Датасет для классификации состояния деревьев."""

    def __init__(self, encoded_categorical_data, numerical_tensor):
        """
        Args:
            encoded_categorical_data (dict): Словарь с категориальными данными
            numerical_tensor (torch.Tensor): Тензор с числовыми данными.
        """
        self.categorical_data = encoded_categorical_data
        self.numerical_data = numerical_tensor
        self.feature_names = list(encoded_categorical_data.keys())

    def __len__(self):
        """Возвращает длину датасета."""
        return self.__numerical_data_len()

    def __numerical_data_len(self):
        """Возвращает количество образцов в числовых данных."""
        return self.numerical_data.shape[0]

    def __getitem__(self, idx):
        """Возвращает элемент датасета по индексу.

        Args:
            idx (int): Индекс элемента.

        Returns:
            tuple[dict, torch.Tensor]: Кортеж, содержащий
            категориальные признаки (словарь)
            и числовой признак (тензор).
        """
        categorical_features = {
            name: data[idx] for name, data in self.categorical_data.items()
        }
        numerical_feature = self.numerical_data[idx]
        return categorical_features, numerical_feature


def get_input_dataset(df: pd.DataFrame) -> TreeHealthDataset:
    """Создает TreeHealthDataset из DataFrame.

    Args:
        df (pd.DataFrame): DataFrame с признаками для датасета.

    Returns:
        TreeHealthDataset: Готовый к использованию датасет.
    """
    encoded_categorical_data = {}
    for col in categorical_cols_names:
        encoded_categorical_data[col] = (
            torch.tensor(df[col].values, dtype=torch.long))

    numerical_tensor = torch.tensor(
        df[numerical_cols_names].copy().values, dtype=torch.float32
    )
    dataset = TreeHealthDataset(encoded_categorical_data, numerical_tensor)

    return dataset
