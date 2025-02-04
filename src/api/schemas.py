from typing import List, Dict, Any

from pydantic import BaseModel, Field


class MultipleInputData(BaseModel):
    """
    Модель данных для множественного ввода признаков для предсказания.

    Предназначена для валидации входных данных, представляющих собой список
    наборов признаков, где каждый набор - это словарь.
    """
    instances: List[Dict[str, Any]] = Field(
        ...,
        description="Список наборов признаков, каждый из которых "
                    "представляет собой словарь"
                    " для одного экземпляра предсказания."
    )


class PredictionResponse(BaseModel):
    """
    Модель ответа предсказания.

    Описывает структуру ответа, содержащего предсказанный класс и
    уверенность предсказания.
    """
    predicted_class: str = Field(...,
                                 description="Предсказанное состояние дерева.")
    confidence: float = Field(...,
                              description="Уверенность модели"
                                          " в предсказанном классе.")
