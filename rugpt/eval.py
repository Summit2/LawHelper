# quick_eval.py
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def quick_eval():
    model_path = "./rugpt3-koap-finetuned"
    
    # Загрузка модели
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    questions = [
        "Какая статья КоАП РФ регулирует превышение скорости?",
        "Какой штраф за превышение скорости на 20-40 км/ч?"
    ]
    
    print("Быстрая оценка модели:\n")
    
    for q in questions:
        prompt = f"Вопрос: {q}\nОтвет:"
        result = generator(prompt, max_new_tokens=100, do_sample=False)
        answer = result[0]['generated_text'].split("Ответ:")[-1].strip()
        print(f"Q: {q}")
        print(f"A: {answer}\n")

if __name__ == "__main__":
    quick_eval()