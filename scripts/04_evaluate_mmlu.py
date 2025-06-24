import sys
import os
import argparse
import torch
import random
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import BitsAndBytesConfig

# Adiciona a pasta raiz ao path para encontrar os módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import load_model_and_tokenizer, generate_response, set_seed, create_mmlu_prompt, clean_mmlu_output

def main(args):
    set_seed(42)

    # --- 1. Carregar Modelo ---
    print(f"Carregando modelo base: {args.base_model_id}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_id, bnb_config)

    if args.adapter_path:
        print(f"Carregando adaptador LoRA de: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print("Avaliando o modelo base (sem adaptador).")
        model = base_model

    # --- 2. Preparar Suíte de Avaliação MMLU ---
    # Conforme requisito, 150 questões de 3 categorias. 
    categories = {
        "STEM": "college_computer_science",
        "Humanidades": "philosophy",
        "Ciências Sociais": "high_school_macroeconomics"
    }
    
    mmlu_suite = {}
    print("Preparando a suíte de avaliação MMLU...")
    for cat_name, hf_name in categories.items():
        # Pega 4 exemplos para o few-shot do split de 'dev'
        dev_split = load_dataset("cais/mmlu", hf_name, split="dev")
        few_shot_examples = dev_split.shuffle(seed=42).select(range(4))
        
        # Pega 50 exemplos para teste do split de 'test'
        test_split = load_dataset("cais/mmlu", hf_name, split="test")
        test_examples = test_split.shuffle(seed=42).select(range(5))
        
        mmlu_suite[cat_name] = {"test": test_examples, "shots": few_shot_examples}

    # --- 3. Executar Avaliação ---
    results = {"total": {"correct": 0, "count": 0}}
    for cat_name, data in mmlu_suite.items():
        print(f"\nAvaliando a categoria: {cat_name}...")
        results[cat_name] = {"correct": 0, "count": 0}
        
        for example in tqdm(data["test"], desc=f"Categoria {cat_name}"):
            prompt = create_mmlu_prompt(example, data["shots"])
            raw_output = generate_response(model, tokenizer, prompt, max_new_tokens=5)
            
            predicted_answer = clean_mmlu_output(raw_output)
            correct_answer = chr(65 + example['answer'])
            
            if predicted_answer == correct_answer:
                results[cat_name]["correct"] += 1
                results["total"]["correct"] += 1
            
            results[cat_name]["count"] += 1
            results["total"]["count"] += 1

    # --- 4. Calcular e Apresentar Acurácia ---
    # Conforme requisito 4.2 e 4.3
    print(f"\n===== Resultado Final da Avaliação MMLU =====")
    print(f"Modelo Avaliado: {args.adapter_path or 'Modelo Base'}")
    
    for cat_name, res in results.items():
        if cat_name != "total":
            accuracy = (res['correct'] / res['count']) * 100 if res['count'] > 0 else 0
            print(f"  - Acurácia em {cat_name}: {accuracy:.2f}% ({res['correct']}/{res['count']})")
    
    total_accuracy = (results['total']['correct'] / results['total']['count']) * 100 if results['total']['count'] > 0 else 0
    print(f"\n  Acurácia Agregada: {total_accuracy:.2f}% ({results['total']['correct']}/{results['total']['count']})")
    print(f"=============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de avaliação para MMLU.")
    parser.add_argument("--base_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="ID do modelo base.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Caminho para o adaptador LoRA. Se não for fornecido, avalia o modelo base.")
    
    args = parser.parse_args()
    main(args)