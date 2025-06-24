import torch
from model_utils import load_model_and_tokenizer, generate_response, set_seed, save_results_to_jsonl, clean_sql_output
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import json

# Prompt Engineering (Requisito 1.1)
FEW_SHOT_PROMPT_TEMPLATE = """Sua única tarefa é traduzir a pergunta em linguagem natural para uma única consulta SQL. Não forneça explicações ou texto adicional.

Pergunta: How many heads of the departments are older than 56 ?
SQL: SELECT count(*) FROM head WHERE age > 56

Pergunta: List the name, born state and age of the heads of departments ordered by age.
SQL: SELECT name ,  born_state ,  age FROM head ORDER BY age

Pergunta: List the creation year, name and budget of all departments.
SQL: SELECT creation ,  name ,  budget_in_billions FROM department

Pergunta: {pergunta_atual}
SQL: """

def main():
    """
    Função principal para executar a avaliação do baseline (Fase 1).
    """
    set_seed(42)

    # --- Configurações ---
    MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Configuração de Quantização
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # --- Carregar Modelo e Tokenizer ---
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, bnb_config)

    spider_validation = load_dataset("spider", split="validation")

    resultados = [] # Continuamos usando a lista para calcular a acurácia no final
    filepath = "resultados_baseline.jsonl"
    
    # Abre o arquivo ANTES do loop
    with open(filepath, 'w', encoding='utf-8') as f:
        # Envelopa o iterável com tqdm
        for example in tqdm(spider_validation, desc="Avaliando Baseline no Spider"):
            pergunta_atual = example['question']
            prompt_final = FEW_SHOT_PROMPT_TEMPLATE.format(pergunta_atual=pergunta_atual)
            sql_bruta = generate_response(model, tokenizer, prompt_final)
            sql_gerada = clean_sql_output(sql_bruta)
            
            result_item = {
                'id': example['db_id'], 
                'pergunta': pergunta_atual, 
                'sql_esperado': example['query'], 
                'sql_gerado': sql_gerada
            }
            
            # Adiciona à lista na memória
            resultados.append(result_item)
            
            # Salva o item atual no arquivo IMEDIATAMENTE
            f.write(json.dumps(result_item, ensure_ascii=False) + '\n')

    print(f"Resultados completos salvos em {filepath}")

    # A lógica para calcular os sucessos continua a mesma
    sucessos = sum(1 for res in resultados if res['sql_gerado'].strip().lower() == res['sql_esperado'].strip().lower())
    total = len(resultados)
    # A métrica para a Fase 1 é a contagem bruta de sucesso/falha 
    print(f"\nContagem Bruta de Sucesso (Fase 1): {sucessos}/{total} ({sucessos/total:.2%})")


    """

    # --- Executar Testes de Inferência ---
    spider_validation = load_dataset("spider", split="validation")

    resultados = []
    for example in tqdm(spider_validation, desc="Avaliando Baseline no Spider"):
        pergunta_atual = example['question']
        prompt_final = FEW_SHOT_PROMPT_TEMPLATE.format(pergunta_atual=pergunta_atual)
        sql_bruta = generate_response(model, tokenizer, prompt_final)
        sql_gerada = clean_sql_output(sql_bruta)
        resultados.append({'id': example['db_id'], 'pergunta': pergunta_atual, 'sql_esperado': example['query'], 'sql_gerado': sql_gerada})
    save_results_to_jsonl(resultados, "resultados_baseline.jsonl")

    sucessos = sum(1 for res in resultados if res['sql_gerado'].strip().lower() == res['sql_esperado'].strip().lower())
    total = len(resultados)
    print(f"\nContagem Bruta de Sucesso (Fase 1): {sucessos}/{total} ({sucessos/total:.2%})")

    # --- Liberar Recursos ---
    # Descomente a linha abaixo quando quiser testar a liberação de memória
    # unload_model(model, tokenizer)
    """

if __name__ == "__main__":
    main()