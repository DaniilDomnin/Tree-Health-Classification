import torch
import joblib
import numpy as np

from model import SimpleTreeHealthModel

BATCH_SIZE = 256
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EMBEDDING_DIMS_PATH = 'preprocessors/embedding_dims.joblib'
MODEL_PATH = '../../models/model.pth'
LABELS_PATH = 'preprocessors/labels.txt'
MEAN_FILL_VALUES_PATH = 'preprocessors/mean_fill_values.joblib'
OUTLIER_CLIP_RANGES_PATH = 'preprocessors/outlier_clip_ranges.joblib'
KMEANS_MODEL_PATH = 'preprocessors/kmeans_model.joblib'
MOST_FREQ_IDS_PATH = 'preprocessors/most_freq_ids.joblib'
COLUMNS_TO_DROP_INITIAL_PATH = 'preprocessors/columns_to_drop_initial.joblib'
COLUMNS_TO_DROP_CORRELATED_PATH = 'preprocessors/columns_to_drop_correlated.joblib'
LABEL_ENCODERS_PATH = 'preprocessors/label_encoders.joblib'
NUMERICAL_SCALER_PATH = 'preprocessors/numerical_scaler.joblib'
CATEGORICAL_COLS_NAMES_PATH = 'preprocessors/categorical_cols_names.joblib'
NUMERICAL_COLS_NAMES_PATH = 'preprocessors/numerical_cols_names.joblib'
CLASS_WEIGHTS = [0.20916101852560978, 0.03816782208495894, 0.7526711593894312]


def load_artifacts():
    """Загрузка артефактов модели из файлов.

    Returns:
        tuple: Кортеж загруженных артефактов в следующем порядке:
            (model_instance, label_list,
             mean_fill_values_data, outlier_clip_ranges_data,
             kmeans_model_data, most_freq_ids_data,
             columns_to_drop_initial_data, columns_to_drop_correlated_data,
             label_encoders_data, numerical_scaler_data,
             categorical_cols_names_data, numerical_cols_names_data,
             embedding_dims)

    Raises:
        Exception: Если возникает ошибка при загрузке любого из файлов.
    """
    try:
        embedding_dims = joblib.load(EMBEDDING_DIMS_PATH)
        model_instance = SimpleTreeHealthModel(embedding_dims, 3, 4)
        model_instance.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE))
        model_instance.to(DEVICE)
        model_instance.eval()
        with open(LABELS_PATH, "r") as f:
            label_list = np.array([line.strip() for line in f])
        if len(label_list) != 3:
            raise ValueError("Неверный файл labels.txt.")
        mean_fill_values_data = (
            joblib.load(MEAN_FILL_VALUES_PATH))

        outlier_clip_ranges_data = (
            joblib.load(OUTLIER_CLIP_RANGES_PATH))

        kmeans_model_data = (
            joblib.load(KMEANS_MODEL_PATH))

        most_freq_ids_data = (
            joblib.load(MOST_FREQ_IDS_PATH))

        columns_to_drop_initial_data = (
            joblib.load(COLUMNS_TO_DROP_INITIAL_PATH))

        columns_to_drop_correlated_data = (
            joblib.load(COLUMNS_TO_DROP_CORRELATED_PATH))

        label_encoders_data = (
            joblib.load(LABEL_ENCODERS_PATH))

        numerical_scaler_data = (
            joblib.load(NUMERICAL_SCALER_PATH))

        categorical_cols_names_data = (
            joblib.load(CATEGORICAL_COLS_NAMES_PATH))

        numerical_cols_names_data = (
            joblib.load(NUMERICAL_COLS_NAMES_PATH))

        return (
            model_instance,
            label_list,
            mean_fill_values_data,
            outlier_clip_ranges_data,
            kmeans_model_data,
            most_freq_ids_data,
            columns_to_drop_initial_data,
            columns_to_drop_correlated_data,
            label_encoders_data,
            numerical_scaler_data,
            categorical_cols_names_data,
            numerical_cols_names_data,
            embedding_dims,
        )
    except Exception as e:
        raise Exception(f"Ошибка загрузки файлов: {e}")


(
    model,
    labels,
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
    embedding_dims,
) = load_artifacts()
