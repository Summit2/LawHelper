import numpy as np
import pickle
import os
from typing import List, Dict
import re
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class MinimalRAG:
    def __init__(self, use_cpu: bool = False):
        print("Инициализация моделей...")
        
        self.embedder = SentenceTransformer("cointegrated/rubert-tiny2")
        model_name = 'rugpt3-koap-finetuned'
        
        print(f"Загрузка генеративной модели: {model_name}")
        
        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16
            print("Используется GPU")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("Используется CPU")
        
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
        print("Модели загружены успешно!")
    
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

def load_koap_data(rag_system: MinimalRAG, data_dir: str = "data"):
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

def example_usage():
    rag = MinimalRAG(use_cpu=True)
    
    data_dir = "koap_data"
    
    if os.path.exists("rag_index.pkl"):
        print("Загрузка сохраненного индекса...")
        rag.load_index("rag_index.pkl")
    else:
        print("Создание нового индекса...")
        if load_koap_data(rag, data_dir):
            rag.build_index()
            rag.save_index()
        else:
            print("Не удалось загрузить данные КоАП.")
            print("Создайте директорию 'koap_data' и поместите туда файлы КоАП в формате .txt")
            return
    
    test_questions = [
        "Какая статья регулирует превышение скорости?",
        "Что считается административным правонарушением?",
        "Какие наказания предусмотрены за нарушение ПДД?",
        "Какой срок давности по административным правонарушениям?",
        "Кто рассматривает дела об административных правонарушениях?"
    ]
    
    print("\n" + "="*60)
    print("Тестирование RAG системы для КоАП РФ")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Вопрос: {question}")
        print("-" * 40)
        
        answer = rag.answer(question)
        print(f"Ответ: {answer}")
        
        context_chunks = rag.search(question, top_k=2)
        if context_chunks:
            print(f"\nИспользованные фрагменты:")
            for j, chunk in enumerate(context_chunks, 1):
                print(f"  {j}. [Статья: {chunk['info'].get('article', 'Не указана')}]")
                print(f"     {chunk['text'][:150]}...")

def interactive_mode():
    print("Инициализация RAG системы...")
    rag = MinimalRAG(use_cpu=True)
    
    if not rag.load_index("rag_index.pkl"):
        print("Сначала создайте индекс с помощью example_usage()")
        return
    
    print("\n" + "="*60)
    print("RAG система для КоАП РФ готова к работе")
    print("Введите ваш вопрос (или 'выход' для завершения)")
    print("="*60)
    
    while True:
        question = input("\nВопрос: ").strip()
        
        if question.lower() in ['выход', 'exit', 'quit']:
            print("Завершение работы...")
            break
        
        if not question:
            continue
        
        print("Обработка запроса...")
        answer = rag.answer(question)
        print(f"\nОтвет: {answer}")
        
        context_chunks = rag.search(question, top_k=1)
        if context_chunks:
            print(f"\nИсточник: {context_chunks[0]['info'].get('article', 'Не указана')}")

if __name__ == "__main__":
    example_usage()