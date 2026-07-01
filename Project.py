import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer
import evaluate
import gc
import torch

# ------------------------------------------------------------
# 0. Инициализация ресурсов NLTK
# ------------------------------------------------------------
nltk.download("punkt", quiet=True)

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------
print("Загрузка датасета CNN/DailyMail...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:3]")
texts = [x["article"] for x in dataset]
references = [x["highlights"] for x in dataset]

# ------------------------------------------------------------
# 2. Экстрактивное реферирование (Lead-3 Baseline)
# ------------------------------------------------------------
def extractive_summarize(text, n=3):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:n])

print("Выполнение экстрактивного реферирования...")
extractive_results = [extractive_summarize(t) for t in texts]

# ------------------------------------------------------------
# 3. Функция безопасного запуска абстрактивных моделей
# ------------------------------------------------------------
def run_model(model_name, texts):
    print(f"\nИнициализация модели: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "summarization",
        model=model_name,
        tokenizer=tokenizer,
        device=-1  # Запуск на CPU. Измените на 0 для активации GPU с CUDA
    )
    
    results = []
    for t in texts:
        # Устанавливаем лимит токенов в зависимости от архитектуры модели
        max_input_length = 16384 if "led" in model_name.lower() else 1024
        
        inputs = tokenizer(
            t,
            max_length=max_input_length,
            truncation=True,
            return_tensors="pt"
        )
        # Получаем очищенный от служебных символов текст в рамках лимита контекста
        safe_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        
        summary = pipe(
            safe_text,
            max_length=150,
            min_length=40,
            do_sample=False
        )[0]["summary_text"]
        results.append(summary)
        
    # Освобождение оперативной памяти
    del pipe
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results

# ------------------------------------------------------------
# 4. Выполнение генерации абстрактивных резюме
# ------------------------------------------------------------
bart_results = run_model("facebook/bart-large-cnn", texts)
pegasus_results = run_model("google/pegasus-xsum", texts)
longformer_results = run_model("allenai/led-base-16384", texts)

# ------------------------------------------------------------
# 5. Оценка качества (ROUGE Metrics)
# ------------------------------------------------------------
rouge = evaluate.load("rouge")

def compute_rouge(preds, refs):
    scores = rouge.compute(predictions=preds, references=refs)
    print(f"ROUGE-1: {scores['rouge1']:.4f}")
    print(f"ROUGE-2: {scores['rouge2']:.4f}")
    print(f"ROUGE-L: {scores['rougeL']:.4f}")

# ------------------------------------------------------------
# 6. Вывод демонстрационных результатов
# ------------------------------------------------------------
for i, t in enumerate(texts):
    print("\n" + "="*60)
    print(f"ДОКУМЕНТ №{i+1}")
    print("="*60)
    print("\nОригинальный текст (фрагмент):")
    print(t[:500], "...")
    print("\n--- Экстрактивный метод (Lead-3 Baseline) ---")
    print(extractive_results[i])
    print("\n--- Абстрактивный метод (BART) ---")
    print(bart_results[i])
    print("\n--- Абстрактивный метод (PEGASUS) ---")
    print(pegasus_results[i])
    print("\n--- Абстрактивный метод (Longformer/LED) ---")
    print(longformer_results[i])

# ------------------------------------------------------------
# 7. Вывод итоговых метрик
# ------------------------------------------------------------
print("\n\n" + "="*40)
print(" СРАВНИТЕЛЬНЫЙ АНАЛИЗ МЕТРИК ROUGE ")
print("="*40)
print("\nЭкстрактивный метод (Baseline) vs Эталон:")
compute_rouge(extractive_results, references)
print("\nBART vs Эталон:")
compute_rouge(bart_results, references)
print("\nPEGASUS vs Эталон:")
compute_rouge(pegasus_results, references)
print("\nLongformer/LED vs Эталон:")
compute_rouge(longformer_results, references)
print("\n[Процесс успешно завершен]")
