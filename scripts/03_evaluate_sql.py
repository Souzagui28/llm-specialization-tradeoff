import sys
import os

# Isso permite que o script encontre a pasta 'custom_metrics'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import deepeval
from deepeval.test_case import LLMTestCase
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from transformers import BitsAndBytesConfig
from model_utils import load_model_and_tokenizer, generate_response, clean_sql_output, set_seed
from custom_metrics.execution_accuracy import ExecutionAccuracy

def main(args):
    """Função principal para avaliar um modelo na tarefa Text-to-SQL do Spider."""
    
    set_seed(42)
    
    # --- 1. Carregar Modelo e Tokenizer ---
    # Carrega o modelo base primeiro
    print(f"Carregando modelo base: {args.base_model_id}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model, tokenizer = load_model_and_tokenizer(args.base_model_id, bnb_config)

    # Se um caminho de adaptador for fornecido, carrega o modelo PEFT
    if args.adapter_path:
        print(f"Carregando adaptador LoRA de: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        model = model.merge_and_unload() # Opcional, mas pode acelerar a inferência
    else:
        print("Avaliando o modelo base (sem adaptador).")
        model = base_model

    # --- 2. Preparar Dados e Test Cases ---
    print("Carregando dataset de desenvolvimento do Spider...")
    # Conforme requisito 3.2, avaliar no Spider dev split
    spider_val = load_dataset("spider", split="validation")
    
    # Limita o número de exemplos para uma avaliação rápida
    if args.limit > 0:
        spider_val = spider_val.select(range(args.limit))

    test_cases = []
    print(f"Gerando predições para {len(spider_val)} exemplos...")
    for example in tqdm(spider_val, desc="Gerando SQL"):
        prompt = f"Traduza a pergunta a seguir para SQL.\n\n### Pergunta:\n{example['question']}"
        
        raw_sql = generate_response(model, tokenizer, f"[INST] {prompt} [/INST]")
        cleaned_sql = clean_sql_output(raw_sql)
        
        test_case = LLMTestCase(
            input=prompt,
            actual_output=cleaned_sql,
            expected_output=example['query'],
            context=[example['db_id']]
        )
        test_cases.append(test_case)

    # --- 3. Executar Avaliação com DeepEval ---
    print("\nIniciando avaliação com DeepEval e a métrica customizada...")
    
    results_by_db = {}
    for case in test_cases:
        db_id = case.context[0]
        if db_id not in results_by_db:
            results_by_db[db_id] = []
        results_by_db[db_id].append(case)

    overall_score = 0
    total_cases = len(test_cases)

    for db_id, cases in tqdm(results_by_db.items(), desc="Avaliando por DB"):
        db_path = os.path.join(args.db_root_path, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(db_path):
            print(f"AVISO: Banco de dados não encontrado em '{db_path}'. Pulando {len(cases)} casos.")
            total_cases -= len(cases)
            continue
            
        # Cria uma instância da métrica para o DB atual
        execution_metric = ExecutionAccuracy(db_path=db_path)
        
        # Avalia todos os casos para este DB
        for case in cases:
            execution_metric.measure(case)
            overall_score += execution_metric.score

    final_accuracy = (overall_score / total_cases) if total_cases > 0 else 0
    print(f"\n===== Resultado Final da Avaliação =====")
    print(f"Modelo Avaliado: {args.adapter_path or 'Modelo Base'}")
    print(f"Execution Accuracy: {final_accuracy:.2%}")
    print(f"Total de Acertos: {int(overall_score)} de {total_cases}")
    print(f"========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de avaliação para Text-to-SQL usando DeepEval.")
    parser.add_argument("--base_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="ID do modelo base no Hugging Face.")
    parser.add_argument("--adapter_path", type=str, default=None, help="Caminho para o adaptador LoRA fine-tunado. Se não for fornecido, avalia o modelo base.")
    parser.add_argument("--db_root_path", type=str, default="spider_db/database", help="Caminho para a pasta raiz contendo os bancos de dados SQLite do Spider.")
    parser.add_argument("--limit", type=int, default=100, help="Limitar a avaliação a N exemplos para testes rápidos. Use -1 para o dataset completo.")
    
    args = parser.parse_args()
    main(args)