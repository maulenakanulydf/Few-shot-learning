
# Анализ тональности текста с использованием DistilBERT и Few-Shot Learning

Этот проект демонстрирует использование моделей на базе DistilBERT и кастомного Few-Shot классификатора для анализа тональности текста. Включает примеры применения предобученных моделей для классификации текста на разных языках (включая русский) и реализацию адаптивного Few-Shot классификатора с минимальным количеством примеров.

## Описание

Проект разделен на две основные части:
1. **Предобученные модели**:
   - `tabularisai/multilingual-sentiment-analysis` — многоязычная модель для анализа тональности.
   - `nlptown/bert-base-multilingual-uncased-sentiment` — модель для оценки тональности с выходом в виде звезд (1-5).
   - `seara/rubert-tiny2-russian-sentiment` — модель, оптимизированная для русского языка.
2. **Few-Shot классификатор**:
   - Реализация кастомной модели на основе `distilbert-base-multilingual-cased` для классификации текста с использованием небольшого набора примеров (Few-Shot Learning).

Дата последнего обновления знаний: март 2025 года (на основе текущей даты — 11 марта 2025).

## Установка

1. **Требования**:
   - Python 3.7+
   - Установленные библиотеки: `transformers`, `torch`, `sklearn`, `numpy`

2. **Установка зависимостей**:
   Выполните следующую команду в терминале:
   ```bash
   pip install transformers torch scikit-learn numpy
   ```

3. **Склонируйте репозиторий** (если применимо):
   ```bash
   git clone <URL_репозитория>
   cd <название_папки>
   ```

## Использование

### 1. Предобученные модели
Пример использования моделей из `transformers.pipeline`:

#### Многоязычная модель
```python
from transformers import pipeline

pipe_1 = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
sentence = "Фильм оказался скучным и затянутым"
result = pipe_1(sentence)
print(result)  # [{'label': 'Negative', 'score': 0.5545058846473694}]
```

#### Оценка в звездах
```python
pipe_2 = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
sentence = "Этот телефон работает, как и ожидалось, без сюрпризов"
result = pipe_2(sentence)
print(result)  # [{'label': '5 stars', 'score': 0.5823605060577393}]
```

#### Русскоязычная модель
```python
model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
result = model("я более менее отношусь к этому продукту")
print(result)  # [{'label': 'neutral', 'score': 0.6445968151092529}]
```

### 2. Few-Shot классификатор
Пример запуска кастомного Few-Shot классификатора:

```python
def main():
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FewShotClassifier(model_name)

    # Набор примеров для обучения
    support_set = {
        "positive": ["I love this product!", "This is amazing!", "Absolutely fantastic!"],
        "negative": ["I hate this.", "This is terrible.", "Awful experience."],
        "neutral": ["It's okay, not great.", "It's average.", "I feel indifferent."]
    }

    # Тексты для классификации
    query_texts = ["This is a great day!", "I dislike this service.", "более менее сервис"]

    # Запуск классификации
    few_shot_classification(model, tokenizer, support_set, query_texts)

if __name__ == "__main__":
    main()
```

Вывод:
```
Text: 'This is a great day!' -> Predicted Category: positive
Text: 'I dislike this service.' -> Predicted Category: negative
Text: 'более менее сервис' -> Predicted Category: neutral
```

## Структура проекта

- **Код предобученных моделей**: Примеры использования `pipeline` для анализа тональности.
- **Few-Shot классификатор**:
  - `TaskAdaptiveSemanticFeatureLearner`: Адаптивный слой для извлечения признаков.
  - `FewShotClassifier`: Классификатор с использованием прототипов.
  - `few_shot_classification`: Функция для классификации на основе косинусного сходства.

## Зависимости

- `transformers` — для работы с моделями BERT и токенизаторами.
- `torch` — для реализации нейронных сетей.
- `sklearn` — для вычисления косинусного сходства.
- `numpy` — для работы с массивами.

## Ограничения

- Few-Shot классификатор требует ручной настройки `support_set` для каждой задачи.
- Производительность зависит от качества и количества примеров в `support_set`.
- Модели могут быть чувствительны к языковым особенностям и контексту.

---
