# Análise Quantitativa do Trade-off entre Especialização e Generalização em LLMs

## Visão Geral do Projeto

Este repositório contém o código e os resultados do quarto trabalho prático para as disciplinas de Tópicos Especiais em Bancos de Dados (ICC220) e Tópicos Especiais em Recuperação de Informação (PPGINF528).

O projeto consiste em uma avaliação empírica do processo de *fine-tuning* em Modelos de Linguagem de Grande Porte (LLMs). O objetivo central é especializar um LLM para a tarefa de Text-to-SQL usando o dataset Spider e, simultaneamente, medir o impacto dessa especialização no conhecimento geral do modelo.A análise quantifica o ganho de desempenho na tarefa-alvo e a perda de capacidade em tarefas de conhecimento geral (fenômeno conhecido como "esquecimento catastrófico"), utilizando o benchmark MMLU.

### Informações da Disciplina

  * **Universidade:** Universidade Federal do Amazonas (UFAM) - Instituto de Computação (IComp)
  * **Cursos:**
      * ICC220 - Tópicos Especiais em Bancos de Dados (Graduação)
      * PPGINF528 - Tópicos Especiais em Recuperação de Informação (Pós-Graduação)
  * **Semestre:** 2025/01
  * **Professores:** André Carvalho e Altigran da Silva 

-----

## Como Configurar e Executar o Projeto

Siga os passos abaixo para configurar o ambiente e reproduzir todos os resultados.

### 1\. Clonar o Repositório

```bash
git clone https://github.com/Souzagui28/llm-specialization-tradeoff.git
cd llm-specialization-tradeoff
```

### 2\. Criar o Ambiente Conda

```bash
# Crie um novo ambiente chamado 'icc220-tp4' com Python 3.10
conda create --name icc220-tp4 python=3.10 -y

# Ative o ambiente recém-criado
conda activate icc220-tp4
```

### 3\. Instalar as Dependências

```bash
pip install -r requirements.txt
```

### 4\. Baixar os Bancos de Dados do Spider

A avaliação da métrica `ExecutionAccuracy` requer os arquivos de banco de dados `.sqlite` do Spider, que não são baixados pela biblioteca `datasets`.

1.  Faça o download do arquivo `spider.zip` a partir do [site oficial do Spider](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view).
2.  Na raiz deste projeto, crie uma pasta chamada `spider_db`.
3.  Descompacte o arquivo baixado e mova a pasta `database` que está dentro dele para dentro da pasta `spider_db`. A estrutura final deve ser `spider_db/database/`.

### 5\. Autenticação no Hugging Face (Passo Crucial)

O modelo base `mistralai/Mistral-7B-Instruct-v0.3` é um "gated model" e requer autenticação.

  * **No Site:** Vá para a [página do modelo](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), faça login e aceite os termos de uso.
  * **No Terminal:** Com o ambiente `conda` ativado, execute `huggingface-cli login` e insira um token de acesso que pode ser gerado em [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 6\. Executando o Pipeline Experimental

Os scripts devem ser executados em ordem. Todos os resultados são salvos em arquivos para análise posterior.

#### Fase 1: Avaliação do Modelo Base (Text-to-SQL)

Este script avalia o modelo base (não treinado) no dataset Spider e salva os resultados em `resultados_baseline.jsonl`.

```bash
python scripts/01_run_baseline.py
```

#### Fase 2: Fine-Tuning LoRA

Este script treina dois modelos com diferentes hiperparâmetros e salva os adaptadores LoRA em `models/finetuning_results/`.

  * **Experimento 1:**
    ```bash
    python scripts/02_run_finetuning.py --learning_rate 2e-4
    ```
  * **Experimento 2:**
    ```bash
    python scripts/02_run_finetuning.py --learning_rate 1e-4
    ```

#### Fase 3: Avaliação de Desempenho na Tarefa-Alvo

Este script avalia a métrica `ExecutionAccuracy` para o modelo base e os dois modelos fine-tunados.

  * **Avaliar Modelo Base:**
    ```bash
    python scripts/03_evaluate_sql.py 
    ```
  * **Avaliar Experimento 1:**
    ```bash
    python scripts/03_evaluate_sql.py --adapter_path models/finetuning_results/lr_0.0002_max_steps_100/final_adapter 
    ```
  * **Avaliar Experimento 2:**
    ```bash
    python scripts/03_evaluate_sql.py --adapter_path models/finetuning_results/lr_0.0001_max_steps_100/final_adapter 
    ```

*(Opcional: use `--limit 50` para uma execução mais rápida de teste.)*

#### Fase 4: Avaliação de Regressão de Capacidade (MMLU)

Este script avalia a performance em conhecimento geral no MMLU.

  * **Avaliar Modelo Base:**
    ```bash
    python scripts/04_evaluate_mmlu.py
    ```
  * **Avaliar Experimento 1:**
    ```bash
    python scripts/04_evaluate_mmlu.py --adapter_path models/finetuning_results/lr_0.0002_max_steps_100/final_adapter
    ```
  * **Avaliar Experimento 2:**
    ```bash
    python scripts/04_evaluate_mmlu.py --adapter_path models/finetuning_results/lr_0.0001_max_steps_100/final_adapter
    ```

-----

## Estrutura do Repositório

```
.
├── README.md               # Documentação do projeto.
├── requirements.txt        # Lista de dependências Python.
├── custom_metrics/         # Contém a métrica customizada para DeepEval.
│   └── execution_accuracy.py
├── scripts/                # Contém os scripts executáveis do pipeline.
│   ├── 01_run_baseline.py    # Executa a avaliação do modelo base (Fase 1).
│   ├── 02_run_finetuning.py  # Executa o fine-tuning LoRA (Fase 2).
│   ├── 03_evaluate_sql.py    # Avalia a Execution Accuracy no Spider (Fase 3).
│   ├── 04_evaluate_mmlu.py   # Avalia a acurácia no MMLU (Fase 4).
│   └── model_utils.py        # Funções auxiliares para carregar modelos e gerar texto.
├── models/                 # Diretório para salvar os adaptadores LoRA.
└── spider_db/              # Contém os bancos de dados do dataset Spider.
    └── database/
```
