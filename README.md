# Классификация состояния деревьев Нью-Йорка с использованием Deep Learning

## Описание проекта

Этот проект разработан для классификации состояния деревьев (Good/Fair/Poor) в Нью-Йорке на основе данных переписи уличных деревьев 2015 года. Цель проекта - построить модель машинного обучения, способную автоматически оценивать состояние дерева, используя различные характеристики, представленные в датасете.  Это может быть полезно для городских служб для более эффективного управления и обслуживания зеленых насаждений.


## Используемые технологии

*   **Python:** Основной язык программирования.
*   **Pandas:**  Для работы с табличными данными.
*   **Scikit-learn:** Для задач предобработки данных и оценки моделей.
*   **Jupyter Notebook:** Для EDA, разработки и обучения модели.
*   **dataprep:** Для анализа данных
*   **FastAPI:** Для создания API.
*   **PyTorch:** Фреймворк для построения и обучения Deep Learning моделей.
*   **CatBoost:** Сильное решение из коробки для сравнения

## Структура проекта

```
├── data
│   ├── processed           # Обработанные данные, готовые для обучения
│   │   └── processed_data.csv
│   └── row               # Исходные данные
│       └── 2015-street-tree-census-tree-data.csv
├── models              # Сохраненные модели машинного обучения
│   └── model.pth
├── notebooks           # Jupyter Notebooks для EDA и обучения
│   ├── EDA.ipynb
│   ├── training2.ipynb
│   └── training.ipynb
├── requirements.txt    # Список зависимостей Python
└── src
    └── api             # Код для FastAPI API
        ├── config.py
        ├── main.py         # Основной файл API
        ├── model.py        # Определение модели для инференса
        ├── preprocessing.py # Предобработка данных для API
        ├── preprocessors   # Сохраненные объекты предобработки (scaler, encoders и т.д.)
        │   ├── categorical_cols_names.joblib
        │   ├── columns_to_drop_correlated.joblib
        │   ├── columns_to_drop_initial.joblib
        │   ├── embedding_dims.joblib
        │   ├── kmeans_model.joblib
        │   ├── label_encoders.joblib
        │   ├── labels.txt
        │   ├── mean_fill_values.joblib
        │   ├── most_freq_ids.joblib
        │   ├── numerical_cols_names.joblib
        │   ├── numerical_scaler.joblib
        │   └── outlier_clip_ranges.joblib
        └── schemas.py      # Схемы данных для API
```

## Запуск проекта

### 1. Клонирование репозитория

```bash
git clone https://github.com/DaniilDomnin/Tree-Health-Classification.git
cd Tree-Health-Classification
```

### 2. Создание и активация виртуального окружения (рекомендуется)

```bash
python -m venv venv
source venv/bin/activate  # для Linux/macOS
venv\Scripts\activate  # для Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Изучение EDA и обучения модели (рекомендуется)

*   Запустите и изучите Jupyter Notebook `notebooks/EDA.ipynb` для понимания данных.
*   Запустите Jupyter Notebook `notebooks/training.ipynb` для понимания обучения модели.

### 5. Запуск FastAPI API

```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
API будет доступен по адресу `http://0.0.0.0:8000`.

## Использование API

### Эндпоинт `/predict/`

**Метод:** `POST`

**Описание:**  Принимает данные о деревьях в формате JSON и возвращает предсказанное состояние дерева (`Good`, `Fair`, или `Poor`) и степень уверенность в ответе [0, 1].

**Пример запроса (JSON):**

```json
{
  "instances": [
    {
      "block_id": 348711,
      "created_at": "2015-08-27T00:00:00.000",
      "tree_dbh": 3,
      "curb_loc": "OnCurb",
      "spc_latin": "Acer rubrum",
      "spc_common": "red maple",
      "steward": "None",
      "guards": "None",
      "sidewalk": "NoDamage",
      "user_type": "TreesCount Staff",
      "problems": "None",
      "root_stone": "No",
      "root_grate": "No",
      "root_other": "No",
      "trunk_wire": "No",
      "trnk_light": "No",
      "trnk_other": "No",
      "brch_light": "No",
      "brch_shoe": "No",
      "brch_other": "No",
      "address": "108-005 70 AVENUE",
      "postcode": 11375,
      "zip_city": "Forest Hills",
      "borough": "Queens",
      "st_senate": 16,
      "nta": "QN17",
      "nta_name": "Forest Hills",
      "x_sp": 1027431.148,
      "y_sp": 202756.7687,
      "census tract": 739,
      "bbl": 4022210001.0,
      "created_month": 8,
      "cluster_label": 86,
      "street_name": "70",
      "street_type": "AVENUE"
    },
    {
      "block_id": 315986,
      "created_at": "2015-09-03T00:00:00.000",
      "tree_dbh": 21,
      "curb_loc": "OnCurb",
      "spc_latin": "Quercus palustris",
      "spc_common": "pin oak",
      "steward": "None",
      "guards": "None",
      "sidewalk": "Damage",
      "user_type": "TreesCount Staff",
      "problems": "Stones",
      "root_stone": "Yes",
      "root_grate": "No",
      "root_other": "No",
      "trunk_wire": "No",
      "trnk_light": "No",
      "trnk_other": "No",
      "brch_light": "No",
      "brch_shoe": "No",
      "brch_other": "No",
      "address": "147-074 7 AVENUE",
      "postcode": 11357,
      "zip_city": "Whitestone",
      "borough": "Queens",
      "st_senate": 11,
      "nta": "QN49",
      "nta_name": "Whitestone",
      "x_sp": 1034455.701,
      "y_sp": 228644.8374,
      "census tract": 973,
      "bbl": 4044750045.0,
      "created_month": 9,
      "cluster_label": 136,
      "street_name": "7",
      "street_type": "AVENUE"
    },
    {
      "block_id": -1,
      "created_at": "2015-09-05T00:00:00.000",
      "tree_dbh": 3,
      "curb_loc": "OnCurb",
      "spc_latin": "Gleditsia triacanthos var. inermis",
      "spc_common": "honeylocust",
      "steward": "1or2",
      "guards": "None",
      "sidewalk": "Damage",
      "user_type": "Volunteer",
      "problems": "None",
      "root_stone": "No",
      "root_grate": "No",
      "root_other": "No",
      "trunk_wire": "No",
      "trnk_light": "No",
      "trnk_other": "No",
      "brch_light": "No",
      "brch_shoe": "No",
      "brch_other": "No",
      "address": "390 MORGAN AVENUE",
      "postcode": 11211,
      "zip_city": "Brooklyn",
      "borough": "Brooklyn",
      "st_senate": 18,
      "nta": "BK90",
      "nta_name": "East Williamsburg",
      "x_sp": 1001822.831,
      "y_sp": 200716.8913,
      "census tract": 449,
      "bbl": 3028870001.0,
      "created_month": 9,
      "cluster_label": 116,
      "street_name": "MORGAN",
      "street_type": "AVENUE"
    },
    {
      "block_id": -1,
      "created_at": "2015-09-05T00:00:00.000",
      "tree_dbh": 10,
      "curb_loc": "OnCurb",
      "spc_latin": "Gleditsia triacanthos var. inermis",
      "spc_common": "honeylocust",
      "steward": "None",
      "guards": "None",
      "sidewalk": "Damage",
      "user_type": "Volunteer",
      "problems": "Stones",
      "root_stone": "Yes",
      "root_grate": "No",
      "root_other": "No",
      "trunk_wire": "No",
      "trnk_light": "No",
      "trnk_other": "No",
      "brch_light": "No",
      "brch_shoe": "No",
      "brch_other": "No",
      "address": "1027 GRAND STREET",
      "postcode": 11211,
      "zip_city": "Brooklyn",
      "borough": "Brooklyn",
      "st_senate": 18,
      "nta": "BK90",
      "nta_name": "East Williamsburg",
      "x_sp": 1002420.358,
      "y_sp": 199244.2531,
      "census tract": 449,
      "bbl": 3029250001.0,
      "created_month": 9,
      "cluster_label": 116,
      "street_name": "GRAND",
      "street_type": "STREET"
    },
    {
      "block_id": 223043,
      "created_at": "2015-08-30T00:00:00.000",
      "tree_dbh": 21,
      "curb_loc": "OnCurb",
      "spc_latin": "Tilia americana",
      "spc_common": "American linden",
      "steward": "None",
      "guards": "None",
      "sidewalk": "Damage",
      "user_type": "Volunteer",
      "problems": "Stones",
      "root_stone": "Yes",
      "root_grate": "No",
      "root_other": "No",
      "trunk_wire": "No",
      "trnk_light": "No",
      "trnk_other": "No",
      "brch_light": "No",
      "brch_shoe": "No",
      "brch_other": "No",
      "address": "603 6 STREET",
      "postcode": 11215,
      "zip_city": "Brooklyn",
      "borough": "Brooklyn",
      "st_senate": 21,
      "nta": "BK37",
      "nta_name": "Park Slope-Gowanus",
      "x_sp": 990913.775,
      "y_sp": 182202.426,
      "census tract": 165,
      "bbl": 3010850052.0,
      "created_month": 8,
      "cluster_label": 52,
      "street_name": "6",
      "street_type": "STREET"
    }
  ]
}
```

**Пример ответа (JSON):**

```json
[
  {
    "predicted_class": "Fair",
    "confidence": 0.46222004294395447
  },
  {
    "predicted_class": "Good",
    "confidence": 0.4957438111305237
  },
  {
    "predicted_class": "Good",
    "confidence": 0.39959320425987244
  },
  {
    "predicted_class": "Good",
    "confidence": 0.7429365515708923
  },
  {
    "predicted_class": "Good",
    "confidence": 0.5547523498535156
  }
]
```

Вы можете протестировать API, используя инструменты вроде `curl` или `Postman`. Например, с помощью `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d @example_request.json http://0.0.0.0:8000/predict
```

*(Где `example_request.json` - файл с JSON запросом, аналогичным примеру выше)*

## Технические подробности и выбор архитектуры

Ключевой особенностью тренировочных данных является выраженный дисбаланс классов, поэтому требуются некоторые дополнения для эффективного обучения и правильного замера качества модели. В качестве метрик были выбраны balanced accuracy, weighted f1, ovo macro roc auc

Основными методами решения задач с табличными данными являются решения на основании деревьев, в частности, градиентный бустинг. Поэтому в качестве безлайна, с которым сравниваются DL решения, будет использован Catboost.Поскольку данные не имеют никакой специальной структуры, архитектура модели преимущественно будет состоят из линейных слоев.
Все категориальные признаки и некоторые числовые признаки, которые по смыслу являются категориальными, будут обработаны с помощью эмбеддинг слоев. Остальная архитектура будет построена по принципу: линейный слой -> нормализация -> функция активации -> дропаут.

Большие подробности о предобработке данных и обучении модели находятся в соответствующих ноутбуках.

## Результаты моделей
| Модель          | Метрика                 | Значение |
|-----------------|-------------------------|----------|
| CatBoost        | Balanced Accuracy       | 0.6703   |
| CatBoost        | Weighted F1-Score       | 0.7795   |
| CatBoost        | Macro AUC-ROC           | 0.8553   |
| DL Model        | Balanced Accuracy       | 0.5760   |
| DL Model        | Weighted F1-Score       | 0.7109   |
| DL Model        | Macro AUC-ROC           | 0.7658   |

## Возможные улучшения

Для улучшения текущего решения можно рассмотреть:
  *   Более тщательный и аккуратный подбор гиперпараметров
  *   Дальнейшее изучение данных и построение новых признаков
  *   Поиск новых и более совершенных способов борьбы с дисбалансом классов
  *   Пересмотр архитектуры модели в пользу более новых моделей, например, [TabNet](https://arxiv.org/pdf/1908.07442v5)
