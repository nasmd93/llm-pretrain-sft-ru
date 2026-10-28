# 🧠 LLM Pipeline: Pretrain & SFT для русского языка

[![Pretrain Model](https://img.shields.io/badge/🤗%20HuggingFace-Pretrain%20Model-yellow)](https://huggingface.co/nasmd93/ru-literature-pretrain-157M)
[![SFT Model](https://img.shields.io/badge/🤗%20HuggingFace-SFT%20Model-blue)](https://huggingface.co/nasmd93/qwen-sft-ru)

В данном проекте реализован полный цикл разработки и обучения больших языковых моделей (LLM) — от подготовки сырых данных до получения готового инструктивного ассистента (Instruct Model).

## 📌 Источники данных
- **Для Pretrain:** [JoannaBy/RussianNovels](https://github.com/JoannaBy/RussianNovels) — коллекция русскоязычных литературных произведений.
- **Для SFT:** [d0rj/alpaca-cleaned-ru](https://huggingface.co/datasets/d0rj/alpaca-cleaned-ru) — русскоязычный инструктивный датасет (~51k примеров).

## 📌 Этапы проекта

### 1. Pretrain (Обучение с нуля)
Цель: обучение базовой языковой модели (Base Model), понимающей структуру и стиль русского литературного языка.
- **Данные:** Корпус русской классики (107 произведений, >42 млн символов). Очищен от дубликатов, не-кириллицы и артефактов.
- **Токенизатор:** С нуля обучен собственный BPE-токенизатор (словарь 3000 токенов).
- **Архитектура:** Decoder-only Transformer на базе `LlamaConfig` (~157M параметров, GQA).
- **Результат:** Итоговый loss **3.49** (Perplexity ≈ 32.8). Модель выучила стиль, диалоговые тире и ассоциации «тема → персонаж».

### 2. Post-train SFT (Дообучение на инструкциях)
Цель: Научить модель следовать инструкциям пользователя.
- **Базовая модель:** [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
- **Оптимизация:** Использован `SFTTrainer` (TRL) с `assistant_only_loss=True`. Вручную запатчен `chat_template` (добавлены маркеры `{% generation %}`) для корректного маскирования лосса.
- **Результат:** Итоговый loss **1.31**. Модель преобразилась: от генерации хаотичного мусора (Base Model) до связных структурированных ответов на русском языке.

## 🛠 Стек технологий
- **Python 3.10**
- **PyTorch** (обучение, bf16, gradient checkpointing)
- **Hugging Face** (`transformers`, `datasets`, `tokenizers`)
- **TRL** (`SFTTrainer`)

## 📂 Структура репозитория
- `notebooks/01_pretrain.ipynb` — Единый ноутбук, содержащий EDA, очистку данных, обучение токенизатора, Pretrain и SFT-пайплайн.
- `models/tokenizer/` — Файлы обученного BPE-токенизатора.
- `requirements.txt` — Зависимости проекта.

> **Примечание:** Веса обученных моделей доступны на Hugging Face:
> - 🤗 [Pretrain Model (157M, Llama Architecture)](https://huggingface.co/nasmd93/ru-literature-pretrain-157M) — обучена с нуля на русской литературе.
> - 🤗 [SFT Model (Qwen2.5-0.5B, Russian Instructions)](https://huggingface.co/nasmd93/qwen-sft-ru) — дообученная версия Qwen для диалогов.
> 
> Вы можете протестировать SFT-модель прямо в браузере через встроенную Inference API на странице Hugging Face.

## 🚀 Как запустить
1. Установите зависимости: `pip install -r requirements.txt`
2. Скачайте корпус `JoannaBy/RussianNovels` и положите в `data/raw/`
3. Запустите `01_pretrain.ipynb` от начала до конца. Он последовательно выполнит предобучение и дообучение модели.


