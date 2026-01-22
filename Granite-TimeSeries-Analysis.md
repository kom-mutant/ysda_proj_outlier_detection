## Модели семейства IBM Granite time series. Обзор:

IBM Granite Time Series — семейство foundation models для обработки временных рядов, состоящее из трёх основных моделей: 
**FlowState**, 
**Tiny Time Mixer (TTM)**
и **Time Series Pulse (TSPulse)**. Все модели рассчитаны на GPU-free inference и работают на cpu, не требуют gpu, в отличие от PatchTST
 
У моделей семейства есть [ноутбуки с примерами инференса.](github.com/ibm-granite-community/granite-timeseries-cookbook/tree/main) для основных сценариев

У библиотеки есть [вики-страница](github.com/ibm-granite/granite-tsfm/wiki).

## Избранное: 

- 1. [Карточка ts-pulse на HuggingFace](https://huggingface.co/ibm-granite/granite-timeseries-tspulse-r1)
-- Здесь можно убедиться, что датасеты с нашего бенчмарка не участвовали в тренировке TSPulse.
- 2. [Пример использования TSPulse для задачи zero-shot prediction](github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/tspulse_anomaly_detection.ipynb)
-- Здесь важно взять реализацию Mode Prediction - выбор режима предсказания
- 3. [Видеоролик с NeurIPS 2024 с презентацией семейства моделей](https://slideslive.com/39031413/granite-time-series-foundation-models)
- 4. [Пример использования Mode](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/flowstate_getting_started.ipynb)

## Примеры использования от ibm-granite-community

**NB**: [Руководство для хозяйки](https://github.com/ibm-granite-community/granite-timeseries-cookbook/tree/main/recipes) поделено на 5 разделов: Classification, Imputation, Retail_Forecasting, Search и Time-series. Нам интересны разделы Retail_Forecasting и Time_Series.

- 1. [TTM. Zero-shot + Few-shot(5%) | Forecasing | Multivariate](https://colab.research.google.com/github/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb)
В этом примере показывается применение TTM для forecasting-а по одному каналу. Показан пример, как в библиотеке можно заворачивать модели в pipeline.
- 2. [Препроцессинг bike sharing dataset с экзогенными и категориальными фичами; файнтюнинг TTM ](https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Time_Series/Bike_Sharing_Finetuning_with_Exogenous.ipynb)
- 3. [fКак делать few-shot forecasting and fine-tuning с помощью TTM](https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Retail_Forecasting/M5_retail_sales_forecasting.ipynb)


## Характеристики и ограничения моделей.

| Модель | Параметры | Поддерживаемые задачи (которые нам интересны) | Спецификация |
|--------|-----------|----------------------|---------------|
| **TSPulse** | 1M |Anomaly Detection (AD), Imputation and Search, classification | Фокус на задаче anomaly detection, но может решать и другие |
| **FlowState** | 2.6M – 9.1M | Forecasting | Univariate, time-scale adjustable |
| **TTM** | 1M – 5M | Forecasting, Multivariate forecasting | Multivariate моделирование, поддержка экзогенных признаков, поддержка категориальных признаков |

### TSPulse-r1

- Натренирована с окном контекста длины 512
- Для задачи AD нужно собрать датасет как минимум размером 3-4 окна контекста (1536-2048)
- В zero-shot решает любую из своих задач
- Эта модель присопсоблена для файнтюна, который ловит паттерны в эмбеддингах для time-series: "TSPulse introduces TSLens for task-aware embedding extraction".
- Структура данных для файнтюна под задачу AD: пары (временной ряд, бинарная метка: аномалия/нормально), нужно как минимум несколько десятков примеров

### Flow-state

- можно сразу применять в режиме zero-shot для задачи forecasting
- тем не менее, можно сделать fine-tuning, но это не основной сценарий использования. Для fine-tuning нужно скормить пары (временной ряд до момента времени t; целевое значение)
- карточка на HuggingFace

### TTM-r2

- Работает в zero-shot режиме.
- Авторы рекомендуют файнтюнить на 5-10% тренировочного датасета
- Структура данных: таблица с timestamp, таргетом, экзогенными признаками и категориальными признаками
- [карточка на HuggingFace](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)