import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from model_utils import set_seed

def create_text_row(example):
    """Formata um único exemplo do Spider para o formato de texto."""
    question = example['question']
    query = example['query']
    prompt = f"Traduza a pergunta a seguir para SQL.\n\n### Pergunta:\n{question}"
    text = f"[INST] {prompt} [/INST]\n{query}"
    return {"text": text}

def main(args):
    """Função principal que orquestra o processo de fine-tuning."""
    set_seed(42)

    # --- 1. Carregar Dados e Pré-processar ---
    print("Carregando e processando o dataset Spider (training split)...")
    train_dataset = load_dataset("spider", split="train")
    processed_train_dataset = train_dataset.map(
        create_text_row,
        remove_columns=train_dataset.column_names
    )
    print(f"Dataset processado. Exemplo de formatação:\n{processed_train_dataset[0]['text']}")

    # --- 2. Carregar Modelo e Tokenizer ---
    print(f"Carregando modelo base: {args.model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Configuração do LoRA ---
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 4. Configuração do Treinamento com SFTConfig ---
    output_dir_exp = os.path.join(args.output_dir, f"lr_{args.learning_rate}_epochs_{args.num_train_epochs}")
    training_arguments = SFTConfig(
        output_dir=output_dir_exp,
        max_steps=75,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        max_seq_length=1024,
        report_to="none",
        seed=42,
    )
    
    # --- 5. Inicializar e Executar o Treinador ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_train_dataset,
        peft_config=lora_config,
        args=training_arguments,
    )

    print(f"Iniciando o treinamento QLoRA com lr={args.learning_rate} e epochs={args.num_train_epochs}...")
    trainer.train()
    
    # --- 6. Salvar o Adaptador Final ---
    final_adapter_path = os.path.join(output_dir_exp, "final_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    print(f"Treinamento concluído! Adaptador LoRA salvo em: {final_adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script para Fine-tuning de LLMs com LoRA na tarefa Text-to-SQL (versão simplificada).")
    
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="ID do modelo no Hugging Face.")
    parser.add_argument("--output_dir", type=str, default="models/finetuning_results", help="Diretório para salvar os resultados.")
    
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Taxa de aprendizado.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Número de épocas de treinamento.")
    
    parser.add_argument("--lora_r", type=int, default=16, help="Rank (r) do LoRA.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha do LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout do LoRA.")

    args = parser.parse_args()
    main(args)