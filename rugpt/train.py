import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import os

def load_and_prepare_dataset(filepath: str = "koap_dataset.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = []
    for item in data:
        text = f"Вопрос: {item['instruction']}\n"
        if item.get('input'):
            text += f"Контекст: {item['input']}\n"
        text += f"Ответ: {item['output']}{tokenizer.eos_token}"
        texts.append(text)
    
    return texts

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

def train_model():
    MODEL_NAME = "ai-forever/rugpt3medium_based_on_gpt2"
    DATASET_PATH = "data/train_data.json"
    OUTPUT_DIR = "./rugpt3-koap-finetuned"
    
    global tokenizer
    
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    
    print("Настройка LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Подготовка датасета...")
    texts = load_and_prepare_dataset(DATASET_PATH)
    
    dataset_dict = {"text": texts}
    dataset = Dataset.from_dict(dataset_dict)
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {"labels": examples["input_ids"]},
        batched=True
    )
    
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"Размер тренировочного датасета: {len(train_dataset)}")
    print(f"Размер валидационного датасета: {len(eval_dataset)}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        save_total_limit=2,
        eval_strategy="steps",
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Начало обучения...")
    trainer.train()
    
    print("Сохранение модели...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Модель сохранена в {OUTPUT_DIR}")

def create_dataset_file():
    dataset = {}
    
    with open("koap_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("Датасет создан: koap_dataset.json")

if __name__ == "__main__":
    train_model()
    
    print("\nТестирование модели...")
    
    from transformers import pipeline
    
    model_path = "./rugpt3-koap-finetuned"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    test_prompts = [
        "Вопрос: Какая статья КоАП РФ регулирует превышение скорости?\nОтвет:",
        "Вопрос: Какой штраф за превышение скорости на 20-40 км/ч?\nОтвет:",
        "Вопрос: Что считается административным правонарушением?\nОтвет:"
    ]
    
    for prompt in test_prompts:
        print(f"\nПромпт: {prompt}")
        result = generator(
            prompt,
            max_new_tokens=100,
            temperature=0.2,
            do_sample=True,
            top_p=0.9
        )
        print(f"{result[0]['generated_text']}")