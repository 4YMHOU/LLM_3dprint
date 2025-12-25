# chatting.py
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import PeftModel
import argparse
import torch

parser = argparse.ArgumentParser(description='Chatting with fine-tune causal LM with LoRA (PEFT)')
parser.add_argument('--model_name', type=str, required=True, help='Название модели(из HF)')
parser.add_argument('--model_dir', type=str, help='Директория с моделью')
parser.add_argument('--test_seed', type=int, required=False, help='Сид для генерации воспроизводимых ответов')

args = parser.parse_args()

if args.test_seed is not None:
    set_seed(args.test_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.test_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def chat(model, tokenizer):
    while True:
        prompt = input("Введите вопрос: ")
        if prompt.lower() == 'q':
            return
        
        # Форматируем промпт как в датасете для лучших результатов
        formatted_prompt = f"Вопрос: {prompt}\nОтвет:"
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Явно указываем pad_token_id и eos_token_id
        outputs = model.generate(
            **inputs, 
            max_new_tokens=300, 
            do_sample=True, 
            temperature=0.9, 
            top_p=0.9,
            top_k=50,  # Увеличиваем для лучшего качества
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем только ответ (после "Ответ:")
        if "Ответ:" in full_response:
            answer = full_response.split("Ответ:")[1].strip()
        else:
            answer = full_response
            
        print(f"Ответ: {answer}\n")

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Настраиваем pad_token если его нет
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Загружаем модель БЕЗ предупреждения о torch_dtype
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    dtype="auto",  # Используем dtype вместо torch_dtype
    device_map="auto",
)

# Загружаем LoRA адаптер если указан
if args.model_dir is not None:
    model = PeftModel.from_pretrained(model, args.model_dir)
    
chat(model, tokenizer)