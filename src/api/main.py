import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from typing import List

from schemas import MultipleInputData, PredictionResponse
from preprocessing import preprocess_input_data, get_input_dataset
from config import (
    BATCH_SIZE,
    DEVICE,
    labels,
    model
)
from torch.utils.data import DataLoader
import uvicorn

app = FastAPI(
    title="Tree Condition Classifier API",
    description="API для классификации состояния дерева (Good/Fair/Poor).",
)


@app.post(
    "/predict",
    response_model=List[PredictionResponse],
    summary="Predict tree condition for multiple inputs",
    description="""
    Принимает список наборов признаков
    и возвращает список предсказаний состояния деревьев.
    Каждый набор признаков должен соответствовать схеме MultipleInputData.
    """,
)
def predict_condition(input_data: MultipleInputData) \
        -> List[PredictionResponse]:
    """
    Предсказывает состояние дерева для списка входных данных.

    Args:
        input_data (MultipleInputData): Входные данные
        в формате MultipleInputData,
        содержащие список наборов признаков.

    Returns:
        List[PredictionResponse]: Список объектов PredictionResponse,
                                  содержащих предсказанный класс и
                                  уверенность для каждого входа.

    Raises:
        HTTPException: В случае ошибки во время обработки или предсказания,
                       возвращает HTTP 500 ошибку.
    """
    try:
        preprocessed_df = preprocess_input_data(input_data.instances)
        dataset = get_input_dataset(preprocessed_df)
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        all_predictions: List[int] = []
        all_probabilities: List[float] = []

        with torch.no_grad():
            for categorical_batch_test, numerical_batch_test in test_loader:
                categorical_batch_gpu = {
                    name: tensor.to(DEVICE)
                    for name, tensor in categorical_batch_test.items()
                }
                numerical_batch_gpu = numerical_batch_test.to(DEVICE)

                outputs_test = model(categorical_batch_gpu,
                                     numerical_batch_gpu)
                _, predicted_test = torch.max(outputs_test.data, 1)

                all_predictions.extend(predicted_test.cpu().numpy().tolist())

                probabilities = torch.softmax(outputs_test,
                                              dim=1).cpu().numpy()
                predicted_probabilities = np.take_along_axis(
                    probabilities,
                    predicted_test.cpu().numpy().reshape(-1, 1),
                    axis=1,
                ).reshape(-1)
                all_probabilities.extend(predicted_probabilities.tolist())

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Ошибка предсказания: {e}")

    predicted_class_names = labels[np.array(all_predictions)]

    prediction_responses: List[PredictionResponse] = []
    for i in range(len(predicted_class_names)):
        prediction_responses.append(
            PredictionResponse(
                predicted_class=predicted_class_names[i],
                confidence=all_probabilities[i],
            )
        )

    return prediction_responses


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
