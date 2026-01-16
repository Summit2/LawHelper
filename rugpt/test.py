import json
import torch
import pickle
import os
import re
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import evaluate
from tqdm import tqdm

class MinimalRAG:
    def __init__(self, use_cpu: bool = False):
        print("Инициализация RAG моделей...")
        
        self.embedder = SentenceTransformer("cointegrated/rubert-tiny2")
        model_name = 'rugpt3-koap-finetuned'
        
        print(f"Загрузка генеративной модели: {model_name}")
        
        if torch.cuda.is_available() and not use_cpu:
            device = "cuda:0"
            torch_dtype = torch.float16
            print("Используется GPU для RAG")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Используется CPU для RAG")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.model.to(device)
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda:0" else -1
            )
            
        except Exception as e:
            print(f"Ошибка загрузки модели {model_name}: {e}")
            print("Попытка загрузить более легкую модель...")
            assert(False)
            model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(device)
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda:0" else -1
            )
        
        self.index = None
        self.chunks = []
        self.chunk_info = []
        print("RAG модели загружены успешно!")
    
    def load_text_file(self, filepath: str, chunk_size: int = 300):
        print(f"Загрузка файла: {filepath}")
        with open(filepath, 'r', encoding='windows-1251') as f:
            text = f.read()
        
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*', text)
        
        current_chunk = ""
        current_article = "Не указано"
        
        for sentence in sentences:
            if "Статья" in sentence[:50] or "ст." in sentence[:50]:
                current_article = sentence[:100]
            
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    self.chunks.append(current_chunk.strip())
                    self.chunk_info.append({
                        "source": filepath,
                        "article": current_article,
                        "length": len(current_chunk)
                    })
                current_chunk = sentence + " "
        
        if current_chunk:
            self.chunks.append(current_chunk.strip())
            self.chunk_info.append({
                "source": filepath,
                "article": current_article,
                "length": len(current_chunk)
            })
        
        print(f"Создано {len(self.chunks)} фрагментов")
    
    def build_index(self):
        if not self.chunks:
            print("Нет данных для индексации. Сначала загрузите файлы.")
            return
        
        print(f"Создание эмбеддингов для {len(self.chunks)} фрагментов...")
        embeddings = self.embedder.encode(
            self.chunks, 
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype('float32'))
        print(f"Индекс построен. Векторов: {self.index.ntotal}")
    
    def save_index(self, filename: str = "rag_index.pkl"):
        data = {
            'chunks': self.chunks,
            'chunk_info': self.chunk_info,
            'index': faiss.serialize_index(self.index) if self.index else None
        }
        
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Индекс сохранен в {filename}")
    
    def load_index(self, filename: str = "rag_index.pkl"):
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return False
        
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_info = data['chunk_info']
            if 'index' in data.keys():
                self.index = faiss.deserialize_index(data['index'])
        
        print(f"Индекс загружен из {filename}")
        print(f"Фрагментов: {len(self.chunks)}")
        return True
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None:
            print("Индекс не построен. Сначала создайте индекс.")
            return []
        
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx],
                    'distance': float(distances[0][i]),
                    'info': self.chunk_info[idx]
                })
        
        return results
    
    def answer(self, question: str, top_k_context: int = 3) -> str:
        if not self.chunks:
            return "Система не готова. Сначала загрузите данные КоАП."
        
        context_chunks = self.search(question, top_k=top_k_context)
        
        if not context_chunks:
            return "Не найдено релевантной информации в КоАП РФ."
        
        context = "\n".join([f"Фрагмент {i+1} (статья: {chunk['info'].get('article', 'Не указана')}):\n{chunk['text']}" 
                           for i, chunk in enumerate(context_chunks)])
        
        prompt = f"""Ты - юридический ассистент, специализирующийся на Кодексе об административных правонарушениях РФ (КоАП РФ).

Тебе предоставлены фрагменты из КоАП РФ:

{context}

На основании предоставленных фрагментов КоАП РФ ответь на вопрос: {question}

Требования к ответу:
1. Будь максимально точным и используй только информацию из предоставленных фрагментов
2. Укажи, если информация неполная
3. Если ответа нет в предоставленных фрагментах, так и скажи
4. Формулируй ответ официальным юридическим языком

Ответ:"""
        
        try:
            result = self.generator(
                prompt,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
            
            generated_text = result[0]['generated_text']
            
            if "Ответ:" in generated_text:
                answer = generated_text.split("Ответ:")[-1].strip()
            else:
                answer = generated_text.strip().split('\n')[-1].strip()
            
            return answer
            
        except Exception as e:
            return f"Ошибка при генерации ответа: {str(e)}"

def calculate_rouge(predictions, references):
    rouge = evaluate.load('rouge')
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )
    return results

def calculate_bleu(predictions, references):
    bleu = evaluate.load('bleu')
    results = bleu.compute(
        predictions=predictions,
        references=references
    )
    return results

def calculate_exact_match(predictions, references):
    exact_match = evaluate.load("exact_match")
    results = exact_match.compute(
        predictions=predictions,
        references=references,
        ignore_case=True,
        ignore_punctuation=True
    )
    return results

def load_test_data(filepath: str = "data/test_data.json"):
    if not os.path.exists(filepath):
        print(f"Файл {filepath} не найден")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    return test_data

def load_koap_data(rag_system: MinimalRAG, data_dir: str = "data/koap_data"):
    if not os.path.exists(data_dir):
        print(f"Директория {data_dir} не найдена.")
        return False
    
    txt_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"Не найдено .txt файлов в директории {data_dir}")
        return False
    
    for txt_file in txt_files:
        filepath = os.path.join(data_dir, txt_file)
        rag_system.load_text_file(filepath)
    
    return True

def evaluate_with_rag(model_path: str = "./rugpt3-koap-finetuned", 
                     use_rag: bool = True,
                     rag_index_path: str = "rag_index.pkl",
                     koap_data_dir: str = "data/koap_data"):
    print("="*70)
    print(f"ОЦЕНКА МОДЕЛИ {'С RAG' if use_rag else 'БЕЗ RAG'}")
    print(f"Модель: {model_path}")
    print("="*70)
    
    rag_system = None
    if use_rag:
        print("\n🔧 Инициализация RAG системы...")
        rag_system = MinimalRAG(use_cpu=True)
        
        if os.path.exists(rag_index_path):
            print(f"Загрузка сохраненного индекса: {rag_index_path}")
            rag_system.load_index(rag_index_path)
        else:
            print(f"Индекс {rag_index_path} не найден. Создание нового...")
            if load_koap_data(rag_system, koap_data_dir):
                rag_system.build_index()
                rag_system.save_index(rag_index_path)
            else:
                print("Не удалось загрузить данные КоАП. Переключение на режим без RAG.")
                use_rag = False
    
    print("\n📚 Загрузка тестовых данных...")
    test_data = load_test_data()
    if not test_data:
        print("Тестовые данные не найдены!")
        return
    
    print(f"Загружено {len(test_data)} тестовых примеров")
    
    predictions_with_rag = []
    predictions_without_rag = []
    references = []
    questions = []
    
    print("\n🤖 Генерация ответов...")
    
    if not use_rag or True:
        print("Загрузка модели для генерации без RAG...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = 0 if torch.cuda.is_available() else -1
        model = AutoModelForCausalLM.from_pretrained(model_path)
        generator_without_rag = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
    
    for item in tqdm(test_data, desc="Обработка вопросов"):
        question = item["instruction"]
        reference = item["output"]
        
        questions.append(question)
        references.append(reference)
        
        if use_rag and rag_system:
            try:
                answer_with_rag = rag_system.answer(question)
                predictions_with_rag.append(answer_with_rag)
            except Exception as e:
                print(f"Ошибка при генерации с RAG: {e}")
                predictions_with_rag.append("Ошибка генерации")
        else:
            predictions_with_rag.append("RAG не использовался")
        
        try:
            prompt = f"Вопрос: {question}\nОтвет:"
            
            result = generator_without_rag(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            full_text = result[0]['generated_text']
            if "Ответ:" in full_text:
                answer_without_rag = full_text.split("Ответ:")[-1].strip()
            else:
                answer_without_rag = full_text
            
            predictions_without_rag.append(answer_without_rag)
            
        except Exception as e:
            print(f"Ошибка при генерации без RAG: {e}")
            predictions_without_rag.append("Ошибка генерации")
    
    print("\n" + "="*70)
    print("ВЫЧИСЛЕНИЕ МЕТРИК")
    print("="*70)
    
    results = {
        "model_path": model_path,
        "use_rag": use_rag,
        "test_samples": len(test_data),
        "questions": questions,
        "references": references
    }
    
    if use_rag:
        print("\n📊 МЕТРИКИ С ИСПОЛЬЗОВАНИЕМ RAG:")
        print("-" * 40)
        
        rouge_scores_rag = calculate_rouge(predictions_with_rag, references)
        print("ROUGE метрики:")
        for key, value in rouge_scores_rag.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
        
        bleu_scores_rag = calculate_bleu(predictions_with_rag, references)
        print(f"\nBLEU метрика: {bleu_scores_rag['bleu']:.4f}")
        
        exact_match_rag = calculate_exact_match(predictions_with_rag, references)
        print(f"Exact Match: {exact_match_rag['exact_match']:.4f}")
        
        results["metrics_with_rag"] = {
            "rouge": rouge_scores_rag,
            "bleu": bleu_scores_rag,
            "exact_match": exact_match_rag,
            "predictions": predictions_with_rag
        }
    
    print("\n📊 МЕТРИКИ БЕЗ RAG:")
    print("-" * 40)
    
    rouge_scores_no_rag = calculate_rouge(predictions_without_rag, references)
    print("ROUGE метрики:")
    for key, value in rouge_scores_no_rag.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    bleu_scores_no_rag = calculate_bleu(predictions_without_rag, references)
    print(f"\nBLEU метрика: {bleu_scores_no_rag['bleu']:.4f}")
    
    exact_match_no_rag = calculate_exact_match(predictions_without_rag, references)
    print(f"Exact Match: {exact_match_no_rag['exact_match']:.4f}")
    
    results["metrics_without_rag"] = {
        "rouge": rouge_scores_no_rag,
        "bleu": bleu_scores_no_rag,
        "exact_match": exact_match_no_rag,
        "predictions": predictions_without_rag
    }
    
    print("\n" + "="*70)
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ")
    print("="*70)
    
    if use_rag:
        print("\n📈 УЛУЧШЕНИЕ/УХУДШЕНИЕ ПРИ ИСПОЛЬЗОВАНИИ RAG:")
        print("-" * 40)
        
        metrics_to_compare = ['rouge1', 'rouge2', 'rougeL', 'bleu', 'exact_match']
        for metric in metrics_to_compare:
            if metric == 'bleu':
                rag_val = bleu_scores_rag['bleu']
                no_rag_val = bleu_scores_no_rag['bleu']
            elif metric == 'exact_match':
                rag_val = exact_match_rag['exact_match']
                no_rag_val = exact_match_no_rag['exact_match']
            else:
                rag_val = rouge_scores_rag[metric]
                no_rag_val = rouge_scores_no_rag[metric]
            
            improvement = ((rag_val - no_rag_val) / no_rag_val * 100) if no_rag_val > 0 else 0
            trend = "🔼 УЛУЧШЕНИЕ" if improvement > 0 else "🔽 УХУДШЕНИЕ"
            
            print(f"{metric.upper():12} | Без RAG: {no_rag_val:.4f} | С RAG: {rag_val:.4f} | {trend}: {improvement:+.2f}%")
    
    print("\n📏 АНАЛИЗ ДЛИН ОТВЕТОВ:")
    print("-" * 40)
    
    if use_rag:
        avg_len_rag = sum(len(p.split()) for p in predictions_with_rag if isinstance(p, str)) / len(predictions_with_rag)
        print(f"Средняя длина ответа с RAG: {avg_len_rag:.1f} слов")
    
    avg_len_no_rag = sum(len(p.split()) for p in predictions_without_rag if isinstance(p, str)) / len(predictions_without_rag)
    print(f"Средняя длина ответа без RAG: {avg_len_no_rag:.1f} слов")
    
    avg_len_ref = sum(len(r.split()) for r in references) / len(references)
    print(f"Средняя длина эталонного ответа: {avg_len_ref:.1f} слов")
    
    print("\n" + "="*70)
    print("ПРИМЕРЫ ОТВЕТОВ (первые 3 вопроса)")
    print("="*70)
    
    for i in range(min(3, len(questions))):
        print(f"\n{i+1}. ВОПРОС: {questions[i]}")
        print(f"   ЭТАЛОН: {references[i]}")
        
        if use_rag:
            print(f"   С RAG: {predictions_with_rag[i][:150]}...")
        
        print(f"   БЕЗ RAG: {predictions_without_rag[i][:150]}...")
        print("-" * 50)
    
    output_file = f"evaluation_results_{'with_rag' if use_rag else 'without_rag'}_{len(test_data)}_samples.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Результаты сохранены в {output_file}")
    
    return results

def quick_test_with_rag():
    print("🚀 БЫСТРЫЙ ТЕСТ RAG СИСТЕМЫ")
    
    rag = MinimalRAG(use_cpu=True)
    
    if not rag.load_index("rag_index.pkl"):
        print("Индекс не найден. Запустите полную оценку сначала.")
        return
    
    test_questions = [
        "Какая статья КоАП РФ регулирует превышение скорости?",
        "Какой штраф за управление автомобилем без прав?",
        "Что такое административное правонарушение?"
    ]
    
    print("\n" + "="*60)
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. ВОПРОС: {question}")
        print("-" * 40)
        
        context_chunks = rag.search(question, top_k=2)
        if context_chunks:
            print("Найденные фрагменты:")
            for j, chunk in enumerate(context_chunks, 1):
                article = chunk['info'].get('article', 'Не указана')[:50]
                print(f"  {j}. {article}")
        
        answer = rag.answer(question)
        print(f"\nОТВЕТ: {answer}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка модели с RAG системой')
    parser.add_argument('--model_path', type=str, default="./rugpt3-koap-finetuned",
                       help='Путь к модели')
    parser.add_argument('--no_rag', action='store_true',
                       help='Оценивать только без RAG')
    parser.add_argument('--quick_test', action='store_true',
                       help='Быстрый тест RAG системы')
    parser.add_argument('--rag_index', type=str, default="rag_index.pkl",
                       help='Путь к сохраненному индексу RAG')
    parser.add_argument('--koap_data', type=str, default="data/koap_data",
                       help='Директория с данными КоАП')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test_with_rag()
    else:
        evaluate_with_rag(
            model_path=args.model_path,
            use_rag=not args.no_rag,
            rag_index_path=args.rag_index,
            koap_data_dir=args.koap_data
        )