## Обзор на IBM Granite time series:

IBM Granite Time Series — семейство foundation models для обработки временных рядов, состоящее из трёх основных моделей: 
**FlowState**, 
**Tiny Time Mixer (TTM)**
и **Time Series Pulse (TSPulse)**. Все модели рассчитаны на GPU-free inference и работают на cpu, не требуют продвинутого железа для работы. Это отличает их от PatchTST и трансформерных аналогов - производительных, но требующих GPU для инференса моделей.
 
У моделей семейства есть [ноутбуки с примерами инференса.](github.com/ibm-granite-community/granite-timeseries-cookbook/tree/main) для основных сценариев

У библиотеки есть удобная [вики-страница](github.com/ibm-granite/granite-tsfm/wiki) со всеми релевантными блогами и видео-туториалами. Видео-туториалы пока не смотрел, блогпосты примерно повторяют кукбук.


## Общие характеристики

| Модель | Параметры | Поддерживаемые задачи (которые нам интересны) | Спецификация |
|--------|-----------|----------------------|---------------|
| **FlowState** | 2.6M – 9.1M | Forecasting | Univariate, time-scale adjustable |
| **TTM** | 1M – 5M | Forecasting, Multivariate forecasting | Multivariate моделирование, поддержка экзогенных признаков, поддержка категориальных признаков |
| **TSPulse** | 1M |Anomaly Detection (AD), Outlier Detection (OD), поиск паттернов | Фокус на задаче anomaly detection, но может решать и другие |

### Flow-state

- можно сразу применять в режиме zero-shot для задачи forecasting
- тем не менее, можно сделать fine-tuning, но это не основной сценарий использования. Для fine-tuning нужно скормить пары (временной ряд до момента времени t; целевое значение)

### TTM

- Работает в zero-shot режиме.
- Авторы рекомендуют файнтюнить на 5-10% тренировочного датасета
- Структура данных: таблица с timestamp, таргетом, экзогенными признаками и категориальными признаками.

### TSPulse

- В zero-shot решает любую из своих задач
- Эта модель присопсоблена для файнтюна, который ловит паттерны в эмбеддингах для time-series: "TSPulse introduces TSLens for task-aware embedding extraction".
- Структура данных для файнтюна под задачу AD: пары (временной ряд, бинарная метка: аномалия/нормально), нужно как минимум несколько десятков примеров

## Обзор релевантных примеров из кукбука

**NB**: [поваренная книга](https://github.com/ibm-granite-community/granite-timeseries-cookbook/tree/main/recipes) поделена на 5 разделов: Classification, Imputation, Retail_Forecasting, Search и Time-series. Нам интересны разделы Retail_Forecasting и Time_Series.

- 1. [Zero-shot применение TTM для форкастинга](https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Time_Series/Time_Series_Getting_Started.ipynb)
В этом примере показывается применение TTM для forecasting-а по одному каналу. Показан пример, как в библиотеке можно заворачивать модели в pipeline.
- 2. [Препроцессинг сложного датасета (bike sharing dataset) с экзогенными и категориальными фичами и файнтюнинг TTM ](https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Time_Series/Bike_Sharing_Finetuning_with_Exogenous.ipynb)
- 3. [fКак делать few-shot forecasting and fine-tuning с помощью TTM](https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Retail_Forecasting/M5_retail_sales_forecasting.ipynb)
- 4. [(на всякий случай) туториал, как работать с моделькой, развернутой IBM WatsonX SDK](https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Time_Series/Getting_Started_with_WatsonX_AI_SDK.ipynb)
