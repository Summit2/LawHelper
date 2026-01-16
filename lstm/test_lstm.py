import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
import os
import re
from collections import Counter
from typing import List, Dict
from tqdm import tqdm
import evaluate
from sentence_transformers import SentenceTransformer
import faiss

class MinimalRAG:
    def __init__(self, use_cpu: bool = False):
        print("Инициализация RAG моделей...")
        
        self.embedder = SentenceTransformer("cointegrated/rubert-tiny2")
        
        self.index = None
        self.chunks = []
        self.chunk_info = []
        print("RAG модель для эмбеддингов загружена успешно!")
    
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

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_count = {}
        
    def build_vocab(self, texts, min_freq=1):
        counter = Counter()
        for text in texts:
            words = text.split()
            counter.update(words)
        
        idx = len(self.word2idx)
        for word, count in counter.items():
            if count >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.word_count[word] = count
                idx += 1
                
        print(f"Размер словаря: {len(self.word2idx)} слов")
        return len(self.word2idx)
    
    def text_to_sequence(self, text):
        words = text.split()
        sequence = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in words]
        return sequence
    
    def sequence_to_text(self, sequence):
        words = [self.idx2word.get(idx, "<UNK>") for idx in sequence]
        return " ".join(words)

class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        
        output = self.fc(lstm_out)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device='cpu'):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def generate(self, start_text, vocab, max_len=50, temperature=1.0, device='cpu'):
        self.eval()
        
        words = start_text.split()
        sequence = [vocab.word2idx["<SOS>"]] + [vocab.word2idx.get(word, vocab.word2idx["<UNK>"]) for word in words]
        sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
        
        generated = sequence.copy()
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_len):
                output, hidden = self(sequence_tensor, hidden)
                
                last_logits = output[:, -1, :] / temperature
                
                probs = torch.softmax(last_logits, dim=-1)
                
                next_token = torch.multinomial(probs, 1).item()
                
                if next_token == vocab.word2idx["<EOS>"]:
                    break
                
                generated.append(next_token)
                
                sequence_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)
        
        generated_text = vocab.sequence_to_text(generated[1:])
        return generated_text

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

def load_lstm_model(model_path="lstm_koap_model_best.pt", vocab_path="lstm_koap_model_vocab.pkl"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = LSTMLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint.get('dropout', 0.3)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"LSTM модель загружена из {model_path}")
    print(f"Конфигурация: vocab_size={checkpoint['vocab_size']}, "
          f"hidden_dim={checkpoint['hidden_dim']}, "
          f"layers={checkpoint['num_layers']}")
    
    return model, vocab, device

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

def extract_answer_from_generated(text, question):
    if f"Вопрос: {question}" in text:
        text = text.replace(f"Вопрос: {question}", "")
    
    if "Ответ:" in text:
        answer = text.split("Ответ:")[-1].strip()
    else:
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            answer = sentences[-2].strip() if sentences[-1].strip() == "" else sentences[-1].strip()
        else:
            answer = text.strip()
    
    return answer

def generate_with_lstm(model, vocab, prompt, max_len=100, temperature=0.7, device='cpu'):
    generated_text = model.generate(prompt, vocab, max_len=max_len, temperature=temperature, device=device)
    return generated_text

def evaluate_lstm_with_rag(
    model_path: str = "lstm_koap_model_best.pt",
    vocab_path: str = "lstm_koap_model_vocab.pkl",
    use_rag: bool = True,
    rag_index_path: str = "rag_index.pkl",
    koap_data_dir: str = "data/koap_data",
    test_data_path: str = "data/test_data.json",
    max_gen_len: int = 100,
    temperature: float = 0.7
):
    print("="*70)
    print(f"ОЦЕНКА LSTM МОДЕЛИ {'С RAG' if use_rag else 'БЕЗ RAG'}")
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
    
    print("\n🤖 Загрузка LSTM модели...")
    model, vocab, device = load_lstm_model(model_path, vocab_path)
    
    print("\n📚 Загрузка тестовых данных...")
    test_data = load_test_data(test_data_path)
    if not test_data:
        print("Тестовые данные не найдены!")
        return
    
    print(f"Загружено {len(test_data)} тестовых примеров")
    
    predictions_with_rag = []
    predictions_without_rag = []
    references = []
    questions = []
    
    print("\n🧠 Генерация ответов...")
    
    for item in tqdm(test_data, desc="Обработка вопросов"):
        question = item["instruction"]
        reference = item["output"]
        
        questions.append(question)
        references.append(reference)
        
        base_prompt = f"Вопрос: {question} Ответ:"
        
        if use_rag and rag_system:
            try:
                context_chunks = rag_system.search(question, top_k=3)
                
                if context_chunks:
                    context = "\n".join([f"Фрагмент {i+1} (статья: {chunk['info'].get('article', 'Не указана')}):\n{chunk['text']}" 
                                       for i, chunk in enumerate(context_chunks)])
                    
                    rag_prompt = f"""Ты - юридический ассистент, специализирующийся на Кодексе об административных правонарушениях РФ (КоАП РФ).

Тебе предоставлены фрагменты из КоАП РФ:

{context}

На основании предоставленных фрагментов КоАП РФ ответь на вопрос: {question}

Требования к ответу:
1. Будь максимально точным и используй только информацию из предоставленных фрагментов
2. Укажи, если информация неполная
3. Если ответа нет в предоставленных фрагментах, так и скажи
4. Формулируй ответ официальным юридическим языком

Ответ:"""
                    
                    generated_with_rag = generate_with_lstm(
                        model, vocab, rag_prompt, 
                        max_len=max_gen_len, 
                        temperature=temperature, 
                        device=device
                    )
                    
                    answer_with_rag = extract_answer_from_generated(generated_with_rag, question)
                    predictions_with_rag.append(answer_with_rag)
                else:
                    predictions_with_rag.append("Не найдено релевантной информации в КоАП РФ.")
                    
            except Exception as e:
                print(f"Ошибка при генерации с RAG: {e}")
                predictions_with_rag.append("Ошибка генерации")
        else:
            predictions_with_rag.append("RAG не использовался")
        
        try:
            generated_without_rag = generate_with_lstm(
                model, vocab, base_prompt, 
                max_len=max_gen_len, 
                temperature=temperature, 
                device=device
            )
            
            answer_without_rag = extract_answer_from_generated(generated_without_rag, question)
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
        "max_gen_len": max_gen_len,
        "temperature": temperature,
        "questions": questions,
        "references": references
    }
    
    if use_rag:
        print("\n📊 МЕТРИКИ LSTM С ИСПОЛЬЗОВАНИЕМ RAG:")
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
    
    print("\n📊 МЕТРИКИ LSTM БЕЗ RAG:")
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
    print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ LSTM")
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
    
    print("\n📏 АНАЛИЗ ДЛИН ОТВЕТОВ LSTM:")
    print("-" * 40)
    
    if use_rag:
        avg_len_rag = sum(len(p.split()) for p in predictions_with_rag if isinstance(p, str)) / len(predictions_with_rag)
        print(f"Средняя длина ответа с RAG: {avg_len_rag:.1f} слов")
    
    avg_len_no_rag = sum(len(p.split()) for p in predictions_without_rag if isinstance(p, str)) / len(predictions_without_rag)
    print(f"Средняя длина ответа без RAG: {avg_len_no_rag:.1f} слов")
    
    avg_len_ref = sum(len(r.split()) for r in references) / len(references)
    print(f"Средняя длина эталонного ответа: {avg_len_ref:.1f} слов")
    
    print("\n🔑 ПРОВЕРКА КЛЮЧЕВЫХ СЛОВ:")
    print("-" * 40)
    
    keywords = ["статья", "КоАП", "штраф", "рублей", "административн"]
    
    print("Без RAG:")
    for keyword in keywords:
        count = sum(1 for p in predictions_without_rag if keyword.lower() in p.lower())
        percentage = (count / len(predictions_without_rag)) * 100
        print(f"  '{keyword}': {count}/{len(predictions_without_rag)} ({percentage:.1f}%)")
    
    if use_rag:
        print("\nС RAG:")
        for keyword in keywords:
            count = sum(1 for p in predictions_with_rag if keyword.lower() in p.lower())
            percentage = (count / len(predictions_with_rag)) * 100
            print(f"  '{keyword}': {count}/{len(predictions_with_rag)} ({percentage:.1f}%)")
    
    print("\n" + "="*70)
    print("ПРИМЕРЫ ОТВЕТОВ LSTM (первые 3 вопроса)")
    print("="*70)
    
    for i in range(min(3, len(questions))):
        print(f"\n{i+1}. ВОПРОС: {questions[i]}")
        print(f"   ЭТАЛОН: {references[i]}")
        
        if use_rag:
            print(f"   LSTM С RAG: {predictions_with_rag[i][:150]}...")
        
        print(f"   LSTM БЕЗ RAG: {predictions_without_rag[i][:150]}...")
        print("-" * 50)
    
    timestamp = os.path.splitext(os.path.basename(model_path))[0]
    output_file = f"lstm_evaluation_{'with_rag' if use_rag else 'without_rag'}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Результаты сохранены в {output_file}")
    
    return results

def quick_test_lstm():
    print("🚀 БЫСТРЫЙ ТЕСТ LSTM С RAG СИСТЕМОЙ")
    
    model, vocab, device = load_lstm_model()
    
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
            print("Найденные фрагменты в RAG:")
            for j, chunk in enumerate(context_chunks, 1):
                article = chunk['info'].get('article', 'Не указана')[:50]
                print(f"  {j}. {article}...")
        
        if context_chunks:
            context = "\n".join([f"Фрагмент {i+1}: {chunk['text'][:100]}..." 
                               for i, chunk in enumerate(context_chunks)])
            
            rag_prompt = f"""Контекст из КоАП РФ:
{context}

На основе контекста ответь на вопрос: {question}

Ответ:"""
            
            generated_with_rag = generate_with_lstm(model, vocab, rag_prompt, max_len=100, temperature=0.7, device=device)
            answer_with_rag = extract_answer_from_generated(generated_with_rag, question)
            print(f"\nLSTM С RAG: {answer_with_rag}")
        
        base_prompt = f"Вопрос: {question} Ответ:"
        generated_without_rag = generate_with_lstm(model, vocab, base_prompt, max_len=100, temperature=0.7, device=device)
        answer_without_rag = extract_answer_from_generated(generated_without_rag, question)
        print(f"\nLSTM БЕЗ RAG: {answer_without_rag}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка LSTM модели с RAG системой')
    parser.add_argument('--model_path', type=str, default="lstm_koap_model_best.pt",
                       help='Путь к модели LSTM')
    parser.add_argument('--vocab_path', type=str, default="lstm_koap_model_vocab.pkl",
                       help='Путь к словарю')
    parser.add_argument('--no_rag', action='store_true',
                       help='Оценивать только без RAG')
    parser.add_argument('--quick_test', action='store_true',
                       help='Быстрый тест LSTM с RAG системой')
    parser.add_argument('--rag_index', type=str, default="rag_index.pkl",
                       help='Путь к сохраненному индексу RAG')
    parser.add_argument('--koap_data', type=str, default="data/koap_data",
                       help='Директория с данными КоАП')
    parser.add_argument('--test_data', type=str, default="data/test_data.json",
                       help='Путь к тестовым данным')
    parser.add_argument('--max_len', type=int, default=100,
                       help='Максимальная длина генерируемого текста')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Температура для генерации')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test_lstm()
    else:
        evaluate_lstm_with_rag(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            use_rag=not args.no_rag,
            rag_index_path=args.rag_index,
            koap_data_dir=args.koap_data,
            test_data_path=args.test_data,
            max_gen_len=args.max_len,
            temperature=args.temperature
        )