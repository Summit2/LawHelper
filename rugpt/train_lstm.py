import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
from collections import Counter
import os
from tqdm import tqdm

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

class KoAPDataset(Dataset):
    def __init__(self, data, vocab, max_len=100):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        text = f"Вопрос: {item['instruction']} "
        if item.get('input'):
            text += f"Контекст: {item['input']} "
        text += f"Ответ: {item['output']}"
        
        sequence = self.vocab.text_to_sequence(text)
        
        sequence = [self.vocab.word2idx["<SOS>"]] + sequence + [self.vocab.word2idx["<EOS>"]]
        
        if len(sequence) > self.max_len:
            sequence = sequence[:self.max_len]
        else:
            sequence = sequence + [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(sequence))
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        
        input_seq = sequence_tensor[:-1]
        target_seq = sequence_tensor[1:]
        
        return input_seq, target_seq

def load_data():
    with open("data/train_data.json", 'r') as f:
        data = json.load(f)
    
    augmented_data = []
    for item in data:
        augmented_data.append(item)
        
        variants = [
            f"Какая статья регулирует {item['instruction'].lower().replace('какая статья коап рф регулирует ', '')}",
            f"Скажи, {item['instruction'].lower()}",
            f"Ответь на вопрос: {item['instruction']}",
        ]
        
        for variant in variants:
            augmented_data.append({
                "instruction": variant,
                "input": "",
                "output": item['output']
            })
    
    print(f"Всего примеров: {len(augmented_data)}")
    return augmented_data

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

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc="Обучение")
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        
        batch_size, seq_len = inputs.size()
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss * batch_size * seq_len
        total_tokens += batch_size * seq_len
        
        progress_bar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'ppl': f'{torch.exp(torch.tensor(batch_loss)).item():.2f}'
        })
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, _ = model(inputs)
            
            batch_size, seq_len = inputs.size()
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return avg_loss, perplexity

def main():
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.3
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    MAX_LEN = 100
    SAVE_PATH = "lstm_koap_model"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    print("Загрузка данных...")
    data = load_data()
    
    texts = []
    for item in data:
        text = f"Вопрос: {item['instruction']} "
        if item.get('input'):
            text += f"Контекст: {item['input']} "
        text += f"Ответ: {item['output']}"
        texts.append(text)
    
    print("Создание словаря...")
    vocab = Vocabulary()
    vocab_size = vocab.build_vocab(texts, min_freq=1)
    
    with open(f"{SAVE_PATH}_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    print("Создание датасета...")
    dataset = KoAPDataset(data, vocab, max_len=MAX_LEN)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")
    
    print("Создание модели...")
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Начало обучения...")
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        train_loss, train_ppl = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.2f}")
        
        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab_size': vocab_size,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT
            }, f"{SAVE_PATH}_best.pt")
            print(f"Модель сохранена (лучшая val loss: {val_loss:.4f})")
        
        if (epoch + 1) % 10 == 0:
            print("\nПример генерации:")
            
            test_prompts = [
                "Вопрос: Какая статья КоАП РФ регулирует превышение скорости? Ответ:",
                "Вопрос: Какой штраф за превышение скорости на 20-40 км/ч? Ответ:",
                "Вопрос: Что считается административным правонарушением? Ответ:"
            ]
            
            for prompt in test_prompts:
                generated = model.generate(prompt, vocab, max_len=50, temperature=0.7, device=device)
                print(f"Промпт: {prompt[:50]}...")
                print(f"Генерация: {generated[:100]}...")
                print("-" * 80)
    
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    
    print("\nФинальное тестирование модели:")
    
    checkpoint = torch.load(f"{SAVE_PATH}_best.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_cases = [
        {
            "prompt": "Вопрос: Какая статья КоАП РФ регулирует превышение скорости? Ответ:",
            "expected": "Превышение скорости регулируется статьей 12.9 КоАП РФ 'Превышение установленной скорости движения'."
        },
        {
            "prompt": "Вопрос: Какой штраф за превышение скорости на 20-40 км/ч? Ответ:",
            "expected": "За превышение скорости на величину более 20, но не более 40 км/ч предусмотрен административный штраф в размере 500 рублей (часть 1 статьи 12.9 КоАП РФ)."
        },
        {
            "prompt": "Вопрос: Что грозит за управление транспортным средством в состоянии опьянения? Ответ:",
            "expected": "Управление транспортным средством в состоянии опьянения влечет наложение административного штрафа в размере 30 000 рублей с лишением права управления транспортными средствами на срок от 1.5 до 2 лет (статья 12.8 КоАП РФ)."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nТест {i}:")
        print(f"Промпт: {test_case['prompt']}")
        
        for temp in [0.5, 0.7, 1.0]:
            generated = model.generate(test_case['prompt'], vocab, max_len=100, temperature=temp, device=device)
            print(f"Температура {temp}: {generated}")
        
        print(f"Ожидалось: {test_case['expected']}")
        print("-" * 80)
    
    print("\n🎮 Интерактивный режим (введите 'выход' для завершения):")
    
    while True:
        user_input = input("\nВведите промпт (или 'выход'): ").strip()
        
        if user_input.lower() == 'выход':
            break
        
        if user_input:
            if "Ответ:" not in user_input:
                user_input += " Ответ:"
            
            generated = model.generate(user_input, vocab, max_len=100, temperature=0.7, device=device)
            print(f"\nОтвет модели: {generated}")

if __name__ == "__main__":
    main()