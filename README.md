<div align="center">

<img src="https://github.com/user-attachments/assets/e2d77b56-376b-44fb-ab33-f9f6e179ac7d" width="60%"><br>
<img src="https://github.com/user-attachments/assets/17d03a48-af49-431b-8cbd-82e03ea8ba6d" width="60%"><br>
<img src="https://github.com/user-attachments/assets/17fcb4c8-88a7-42ee-bc9c-0b525dcfcd62" width="60%"><br>
<img src="https://github.com/user-attachments/assets/6b00d307-6209-4314-9681-67471a7e9af2" width="60%"><br>
<img src="https://github.com/user-attachments/assets/f5f3ec67-720f-4f81-9bae-afb020a11261" width="60%"><br>
<img src="https://github.com/user-attachments/assets/aa79d494-7db6-49b0-a400-cfb5e59ea007" width="60%"><br>
<img src="https://github.com/user-attachments/assets/759b32fb-022d-4932-a722-0e4b54b2f60f" width="60%"><br>
<img src="https://github.com/user-attachments/assets/74d3eb54-a34b-4639-b8a7-375818838812" width="60%"><br>
<img src="https://github.com/user-attachments/assets/38aa2b38-9c69-4b47-9117-a171f4499f38" width="60%"><br>
<img src="https://github.com/user-attachments/assets/452418ac-8855-4aa5-858c-814e2fd8c41d" width="60%"><br>
<img src="https://github.com/user-attachments/assets/ff15563d-78d7-4b4a-b7f9-6212b634b140" width="60%"><br>
<img src="https://github.com/user-attachments/assets/d6a201fc-0468-43c8-92bd-8674352539bb" width="60%"><br>
<img src="https://github.com/user-attachments/assets/506b96df-44e5-455f-8091-5035764208b7" width="60%"><br>
<img src="https://github.com/user-attachments/assets/61ccad47-483b-4ed2-b3fa-9bf1123905a9" width="60%"><br>
<img src="https://github.com/user-attachments/assets/a08967f2-0f1d-48de-8ff7-8bd530bb9b57" width="60%"><br>
<img src="https://github.com/user-attachments/assets/26320369-741f-4adb-a675-0686b965c517" width="60%"><br>
<img src="https://github.com/user-attachments/assets/7991a703-d7ad-4994-a23c-238dc6351e69" width="60%"><br>

</div>

# 🧹 broom - Room Matching Content Detector

> **Т-Банк AI Hackathon 2026  «Матчинг комнат»**
> Детекция комнат с недостатком контента до отправки в краудсорсинг

## Задача

Сервис Т-Путешествия сопоставляет предложения от поставщиков с мастер-комнатами. Когда ML-модель не уверена — задача уходит в краудсорсинг. Но часто описание комнаты настолько бедное, что даже люди не могут найти матч — деньги тратятся впустую.

**Цель:** заранее предсказать, что комната не сматчится (`target=1`), и не отправлять её в краудсорсинг.

**Метрика:** PR-AUC (Precision-Recall AUC).

## Решение

**Модель:** LightGBM классификатор поверх гибридного вектора признаков (эмбеддинги + табличные фичи).

### Пайплайн

```
supplier_room_name → нормализация → фичи → LightGBM → P(не сматчится)
```

### Признаки

| Группа | Описание |
|--------|----------|
| **Sentence Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim), cosine-нормализованные |
| **Текстовые** | word/char count, уникальность токенов, наличие цифр/латиницы/кириллицы, скобки, ключевые слова (bed, suite, deluxe, завтрак, …) |
| **Hotel-level** | кол-во комнат в отеле, кол-во уникальных названий, средняя длина названий |
| **Target encoding** | text prior (средний target по нормализованному названию), token prior (mean/max/min target по токенам) |
| **Centroid similarity** | косинусное расстояние до центроидов классов 0 и 1 в пространстве эмбеддингов |
| **Fuzzy within-hotel** | Jaccard-сходство с ближайшей комнатой того же отеля, target ближайшего соседа, флаги subset/superset (LOO на трейне) |

### Итерации модели

| Версия | Фичи | Результат |
|--------|-------|-----------|
| v1 | embeddings + текстовые + hotel-level | baseline PR-AUC |
| v2 | + target encoding + token priors + centroid sim | улучшение PR-AUC |
| **v3** | **+ fuzzy Jaccard within-hotel matching** | **лучший PR-AUC** |

## Интерпретация (SHAP)

Проведён SHAP-анализ (waterfall plots) для трёх кейсов:
- **Уверенный target=1** — модель уверена, что комната не сматчится
- **Уверенный target=0** — комната легко матчится
- **Пограничный** — модель сомневается

Ключевые наблюдения:
- `text_prior` и `fuzzy_target` — сильнейшие предикторы: если похожие названия исторически не матчились, новое тоже вряд ли сматчится
- `fuzzy_jaccard` — высокое сходство с уже сматченной комнатой в том же отеле резко снижает вероятность target=1
- Короткие названия с малым количеством ключевых слов (тип номера, кровати, вид) чаще оказываются несматченными
- Внутриотельная специфика существенна — один и тот же текст может иметь разный target в разных отелях

## Запуск

```bash
pip install pandas numpy scikit-learn lightgbm sentence-transformers shap matplotlib
jupyter notebook clear_ai_hackathon.ipynb
```
