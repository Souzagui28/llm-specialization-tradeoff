import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def load_model_and_tokenizer(model_id: str, quantization_config: BitsAndBytesConfig):
    """
    Carrega o modelo e o tokenizer a partir de um ID do Hugging Face.
    """
    print(f"Carregando modelo base: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    print("Modelo base carregado.")

    print(f"Carregando tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer carregado e configurado.")
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction: str, device: str = "cuda", max_new_tokens=200):
    """
    Gera uma resposta para uma instrução usando o modelo e o tokenizer fornecidos.
    """
    messages = [
        {"role": "user", "content": instruction}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    #print(f'\nPergunta: {instruction}')
    #print("Gerando resposta...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    #print(f"Resposta do Modelo: {response}\n")
    return response

# Adicione esta função ao final de scripts/model_utils.py

def unload_model(model, tokenizer):
    """
    Libera a memória da VRAM ocupada pelo modelo e tokenizer.
    """
    print("Descarregando modelo e limpando cache da GPU...")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("Memória liberada.")


def clean_sql_output(raw_output: str) -> str:
    """
    Tenta extrair uma única consulta SQL da saída bruta de um LLM.
    """
    # Remove o prefixo "SQL:" se existir
    if raw_output.strip().upper().startswith("SQL:"):
        raw_output = raw_output.strip()[4:].strip()
    
    # Se o modelo usar blocos de código (```sql ... ```), extrai o conteúdo
    if "```" in raw_output:
        parts = raw_output.split("```")
        if len(parts) > 1:
            raw_output = parts[1]
            # Remove a palavra 'sql' se estiver no início da linha
            if raw_output.lower().startswith('sql'):
                raw_output = raw_output[3:]

    # Pega apenas a primeira linha que parece ser uma consulta e remove o ponto e vírgula final
    cleaned_sql = raw_output.split('\n')[0].strip()
    if cleaned_sql.endswith(';'):
        cleaned_sql = cleaned_sql[:-1]
        
    return cleaned_sql

import json
def save_results_to_jsonl(results_list, filepath):
    """
    Salva uma lista de dicionários em um arquivo .jsonl.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in results_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Resultados salvos em {filepath}")

import random
import numpy as np
def set_seed(seed_value=42):
    """Define as sementes para reprodutibilidade."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        # As duas linhas abaixo são para garantir um comportamento determinístico
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_text_row(example):
    """
    Formata um único exemplo do Spider para o formato de texto.
    """
    question = example['question']
    query = example['query']

    prompt = f"Traduza a pergunta a seguir para SQL.\n\n### Pergunta:\n{question}"

    # Formato de chat para o Mistral
    text = f"[INST] {prompt} [/INST]\n{query}"
    return {"text": text}


def create_mmlu_prompt(example, few_shot_examples):
    """Cria um prompt 4-shot para uma questão do MMLU."""
    prompt = "A seguir estão perguntas de múltipla escolha. Responda apenas com a letra da opção correta.\n\n"
    
    # Adiciona os exemplos de 4-shot
    for shot in few_shot_examples:
        prompt += f"Pergunta: {shot['question']}\n"
        prompt += f"A. {shot['choices'][0]}\n"
        prompt += f"B. {shot['choices'][1]}\n"
        prompt += f"C. {shot['choices'][2]}\n"
        prompt += f"D. {shot['choices'][3]}\n"
        prompt += f"Resposta: {chr(65 + shot['answer'])}\n\n" # Converte 0->A, 1->B, etc.

    # Adiciona a pergunta final
    prompt += f"Pergunta: {example['question']}\n"
    prompt += f"A. {example['choices'][0]}\n"
    prompt += f"B. {example['choices'][1]}\n"
    prompt += f"C. {example['choices'][2]}\n"
    prompt += f"D. {example['choices'][3]}\n"
    prompt += f"Resposta:"
    
    return f"[INST] {prompt} [/INST]"

def clean_mmlu_output(raw_output: str) -> str:
    """Extrai a primeira letra (A, B, C, ou D) da saída do modelo."""
    text = raw_output.strip().upper()
    for char in text:
        if char in ['A', 'B', 'C', 'D']:
            return char
    return "Z" # Retorna uma resposta inválida se nada for encontrado