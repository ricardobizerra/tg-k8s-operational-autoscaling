Versões: [English](#en-individual-statistical-models-and-ensemble-for-proactive-auto-scaling-in-kubernetes-a-predictive-and-operational-comparison) | [Português](#pt-br-modelos-estatísticos-individuais-e-ensemble-para-autoescalonamento-proativo-em-ambiente-kubernetes-comparação-preditiva-e-operacional)

# [EN] Individual Statistical Models and Ensemble for Proactive Auto-Scaling in Kubernetes: A Predictive and Operational Comparison

The objective of this work is to compare statistical time-series models, applied individually and in dynamic selection Ensembles for proactive auto-scaling in a Kubernetes cluster, using trace replay to assess ML-driven Pod autoscaling.

This work is part of my bachelor's thesis in Computer Science at the Center for Informatics (CIn) of the Federal University of Pernambuco (UFPE).

## Reference papers

- FOG, Jonathan Wisborg et al. **Comparing Neural and Statistical Time-Series Models for Proactive Auto-Scaling in Kubernetes.** In: 2025 IEEE International Conference on Service-Oriented System Engineering (SOSE). IEEE, 2025\. p. 151-161.

- SAMIR, Mohamed; WASSIF, Khaled T.; MAKADY, Soha H. **Proactive auto-scaling approach of production applications using an ensemble model.** IEEE Access, \[S. l.\], v. 11, p. 25008-25019, 2023\.

## Directory structure

- [`datasets`](./datasets): Central point for the datasets used in the experiments.
  - [`cegedim`](./datasets/cegedim): Datasets from the SAMIR, 2023 paper.
- [`experiments`](./experiments): Scripts, configurations, and results related to the experiments.
  - [`data`](./experiments/data): Processed datasets used in the experiments.
  - [`k8s`](./experiments/k8s): Scripts and manifests for Kubernetes environment management, orchestration, load generation, and result analysis.
  - [`models`](./experiments/models): Implementation of time series forecasting models used for autoscaling (e.g., Prophet, Exponential Smoothing, FFT, Ensemble).
  - [`results`](./experiments/results): Output data and evaluation results from the experiments.
  - [`training`](./experiments/training): Data loading, processing pipelines, and model training/tuning routines.

## Running the Experiments

This section provides a comprehensive reference compiling all manual commands required to execute the experiments from start to finish.

### Experiment 1: Predictive Modeling

To generate the model tuning artifacts and dataset splits, run the following commands from the project root. This step is required before running Experiment 2.

```bash
# 1. All non-ensemble models
docker compose run --rm experiments python main.py --model Prophet --dataset all --stages all --n-trials 30 && \
docker compose run --rm experiments python main.py --model FFT --dataset all --stages all --n-trials 30 && \
docker compose run --rm experiments python main.py --model ExponentialSmoothing --dataset all --stages all --n-trials 30

# 2. Ensemble models

# App A (CPU)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_A_CPU_exp1.csv \
  --stages all \
  --ensemble-forward-window 24 \
  --n-trials 30

# App A (RAM)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_A_RAM_exp1.csv \
  --stages all \
  --ensemble-forward-window 36 \
  --n-trials 30

# App B (CPU)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_B_CPU_exp1.csv \
  --stages all \
  --ensemble-forward-window 24 \
  --n-trials 30

# App B (RAM)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_B_RAM_exp1.csv \
  --stages all \
  --ensemble-forward-window 36 \
  --n-trials 30

# App C (CPU)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_C_CPU_exp1.csv \
  --stages all \
  --ensemble-forward-window 24 \
  --n-trials 30

# App C (RAM)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_C_RAM_exp1.csv \
  --stages all \
  --ensemble-forward-window 36 \
  --n-trials 30
```

### Experiment 2: Full Execution Guide

#### 0. Prerequisites

Before running the experiments, ensure you have the following installed on your system:
- **Docker** and **Docker Compose**
- **Kubernetes Cluster** (e.g., Kind, Minikube, Docker Desktop K8s, K3s)
- **kubectl** configured to interact with your cluster
- **Python 3**

**Build the Docker Compose Environment:**
We use a granular, role-based Docker Compose architecture. Build the compose services together, including the Kubernetes images that now live in `docker-compose.yml`:
```bash
docker compose build target-app scaler exp2-trace-generator exp2-analyzer exp2-locust exp2-orchestrator
```

#### 1. Setup & Pre-flight

All commands below should be executed from the project root.

**1.1 Image Import to Kubernetes**
Depending on your Kubernetes environment, you may need to load the built images into your cluster.
- If using Docker Desktop K8s or Minikube (with Docker driver), the built images are already available locally.
- If using Kind, directly load the images from your local Docker daemon:
  ```bash
  kind load docker-image tcc/target-app:latest
  kind load docker-image tcc/scaler:latest
  ```

**1.2 Deploy Infrastructure**
```bash
# Create the infrastructure namespace first
kubectl create namespace tcc-infra

# Deploy Prometheus
kubectl apply -f experiments/k8s/manifests/prometheus/configmap.yaml
kubectl apply -f experiments/k8s/manifests/prometheus/deployment.yaml
```

#### 2. Calibration (Phase 6)

You must determine the peak resource parameters before running the trace generation and the orchestrator. With the automated calibration, this is written directly to `experiments/k8s/calibration.json`. The orchestrator will refuse to start if any calibration values are missing.

**2.1 Start Calibration Pod**
```bash
kubectl run calibration-pod --image=tcc/target-app:latest \
  --image-pull-policy=Never --port=8000 --env="CPU_WORK_ITERATIONS=50000" -n tcc-infra

# In a separate terminal, start port-forwarding for the app and for Prometheus:
kubectl port-forward pod/calibration-pod 8001:8000 -n tcc-infra &
kubectl port-forward svc/prometheus 30090:9090 -n tcc-infra &
```

**2.2 Run Auto-Calibration**
Use the `calibrate.py` script to automatically ramp Locust, query Prometheus, and find peak resources.
```bash
# Run for App A (Repeat for App B and C with --app B and --app C)
python3 experiments/k8s/load/calibrate.py \
  --namespace tcc-infra \
  --app A \
  --port 8001 \
  --prometheus-url http://localhost:30090 \
  --max-users 60 \
  --step-size 5 \
  --step-duration 30 \
  --output experiments/k8s/calibration.json
```

#### 3. Generate Load Traces

Once `calibration.json` is populated, use the automated values to generate the correct load traces for Locust and the RAM injector. The `warmup-steps` are exactly calculated to cover 7 days based on the sampling frequency of each dataset (1 hour for CPU = 168 steps, 40 minutes for RAM = 252 steps).

```bash
# --- Application A ---
docker compose run --rm exp2-trace-generator bash -c "\
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_A_CPU_exp2_replay.csv --metric cpu --peak-cpu-cores \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['cpu']['peak_cpu_cores'])\") --peak-rps \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['cpu']['peak_rps'])\") --warmup-steps 168 && \
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_A_RAM_exp2_replay.csv --metric ram --min-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['ram']['min_memory_mb'])\") --max-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['ram']['max_memory_mb'])\") --warmup-steps 252 --memory-step-mb 1"

# --- Application B ---
docker compose run --rm exp2-trace-generator bash -c "\
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_B_CPU_exp2_replay.csv --metric cpu --peak-cpu-cores \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['cpu']['peak_cpu_cores'])\") --peak-rps \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['cpu']['peak_rps'])\") --warmup-steps 168 && \
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_B_RAM_exp2_replay.csv --metric ram --min-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['ram']['min_memory_mb'])\") --max-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['ram']['max_memory_mb'])\") --warmup-steps 252 --memory-step-mb 1"

# --- Application C ---
docker compose run --rm exp2-trace-generator bash -c "\
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_C_CPU_exp2_replay.csv --metric cpu --peak-cpu-cores \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['cpu']['peak_cpu_cores'])\") --peak-rps \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['cpu']['peak_rps'])\") --warmup-steps 168 && \
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_C_RAM_exp2_replay.csv --metric ram --min-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['ram']['min_memory_mb'])\") --max-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['ram']['max_memory_mb'])\") --warmup-steps 252 --memory-step-mb 1"
```

#### 4. Execution Session (Phase 8)

Once calibration is complete and traces are generated, run the experiment batches.
Ensure the results persistence directory exists on your host so that results are correctly saved in case of failures.

```bash
# Create the results persistence directory on the host
sudo mkdir -p /var/tcc-results && sudo chmod 777 /var/tcc-results
```

> The orchestrator automatically creates a timestamped subdirectory inside your `--output-dir` for each run (e.g., `results/exp2/2026-06-07_12-00-00/`). All state and metrics files are saved there. 

> **Resuming a Batch:** If you hit `Ctrl+C` to abort a batch (which triggers a clean teardown of current namespaces) or if it crashes, you can resume it. To do so, point the `--output-dir` exactly to the timestamped directory of the interrupted run and append the `--resume` flag. The orchestrator will pick up the `pending` queue right where it left off.

> **Batch Size Recommendation:** We recommend using `--batch-size 1` due to calibration constraints, especially if your local environment does not have enough resources to match the same capacities of the original applications.

**4.1 Execution Batch**
Run the experiments using the combined scaling mode.

> Even though only CPU datasets (`a-cpu b-cpu c-cpu`) are explicitly provided in the command below, the `combined` scaling mode guarantees that the models for both metrics (CPU and RAM) will be instantiated and evaluated.

```bash
docker compose run --rm exp2-orchestrator python k8s/orchestrator.py \
  --exp1-results-dir results/ \
  --trace-dir k8s/load/traces/ \
  --output-dir results/exp2/ \
  --models Prophet ExponentialSmoothing FFT Ensemble \
  --datasets a-cpu b-cpu c-cpu \
  --scaling-modes combined \
  --cpu-trace-step-seconds 15 \
  --ram-trace-step-seconds 15 \
  --batch-size 1
```

#### 5. Analysis (Phase 7)

Once all batches are complete, generate the CSVs and charts.

```bash
docker compose run --rm exp2-analyzer python k8s/analyze_results.py \
  --exp2-results-dir results/exp2/ \
  --exp1-results-dir results/ \
  --output-dir results/exp2/analysis/
```

# [PT-BR] Modelos Estatísticos Individuais e Ensemble para Autoescalonamento Proativo em Ambiente Kubernetes: Comparação Preditiva e Operacional

O objetivo deste trabalho é comparar modelos estatísticos de Séries Temporais, aplicados individualmente e em Ensembles de seleção dinâmica para autoescalonamento proativo em um cluster Kubernetes, usando trace replay para avaliar o dimensionamento de Pods impulsionado por Aprendizado de Máquina.

Este trabalho é parte do meu Trabalho de Graduação (TG/TCC) em Ciência da Computação pelo Centro de Informática (CIn) da Universidade Federal de Pernambuco (UFPE).

## Artigos de referência

- FOG, Jonathan Wisborg et al. **Comparing Neural and Statistical Time-Series Models for Proactive Auto-Scaling in Kubernetes.** In: 2025 IEEE International Conference on Service-Oriented System Engineering (SOSE). IEEE, 2025\. p. 151-161.

- SAMIR, Mohamed; WASSIF, Khaled T.; MAKADY, Soha H. **Proactive auto-scaling approach of production applications using an ensemble model.** IEEE Access, [S. l.], v. 11, p. 25008-25019, 2023.

## Estrutura de pastas

- [`datasets`](./datasets): Pasta para centralizar os datasets utilizados nos experimentos.
  - [`cegedim`](./datasets/cegedim): Datasets do artigo de SAMIR, 2023.
- [`experiments`](./experiments): Scripts, configurações e resultados relacionados aos experimentos.
  - [`data`](./experiments/data): Conjuntos de dados processados usados nos experimentos.
  - [`k8s`](./experiments/k8s): Scripts e manifestos para gerenciamento do ambiente Kubernetes, orquestração, geração de carga e análise de resultados.
  - [`models`](./experiments/models): Implementação de modelos de previsão de séries temporais usados para autoscaling (por exemplo, Prophet, Exponential Smoothing, FFT, Ensemble).
  - [`results`](./experiments/results): Dados de saída e resultados de avaliação dos experimentos.
  - [`training`](./experiments/training): Pipelines de carregamento e processamento de dados, e rotinas de treinamento/ajuste de modelos.

## Executando os Experimentos

Para gerar os artefatos de ajuste de modelo e as divisões de dataset, execute os comandos a seguir a partir da raiz do projeto. Esta etapa é necessária antes de executar o Experimento 2.

```bash
# 1. Todos os modelos non-ensemble
docker compose run --rm experiments python main.py --model Prophet --dataset all --stages all --n-trials 30 && \
docker compose run --rm experiments python main.py --model FFT --dataset all --stages all --n-trials 30 && \
docker compose run --rm experiments python main.py --model ExponentialSmoothing --dataset all --stages all --n-trials 30

# 2. Modelos Ensemble

# App A (CPU)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_A_CPU_exp1.csv \
  --stages all \
  --ensemble-forward-window 24 \
  --n-trials 30

# App A (RAM)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_A_RAM_exp1.csv \
  --stages all \
  --ensemble-forward-window 36 \
  --n-trials 30

# App B (CPU)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_B_CPU_exp1.csv \
  --stages all \
  --ensemble-forward-window 24 \
  --n-trials 30

# App B (RAM)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_B_RAM_exp1.csv \
  --stages all \
  --ensemble-forward-window 36 \
  --n-trials 30

# App C (CPU)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_C_CPU_exp1.csv \
  --stages all \
  --ensemble-forward-window 24 \
  --n-trials 30

# App C (RAM)
docker compose run --rm experiments python main.py \
  --model Ensemble \
  --ensemble-models Prophet FFT ExponentialSmoothing \
  --dataset data/cegedim/cegedim_application_C_RAM_exp1.csv \
  --stages all \
  --ensemble-forward-window 36 \
  --n-trials 30
```

### Experimento 2: Guia Completo de Execução

#### 0. Pré-requisitos

Antes de executar os experimentos, garanta que você possui o seguinte instalado no seu sistema:
- **Docker** e **Docker Compose**
- **Cluster Kubernetes** (ex: Kind, Minikube, Docker Desktop K8s, K3s)
- **kubectl** configurado para interagir com o seu cluster
- **Python 3**

**Construir o Ambiente Docker Compose:**
Utilizamos uma arquitetura Docker Compose granular e baseada em funções. Construa os serviços compose em conjunto, incluindo as imagens Kubernetes que agora ficam em `docker-compose.yml`:
```bash
docker compose build target-app scaler exp2-trace-generator exp2-analyzer exp2-locust exp2-orchestrator
```

#### 1. Configuração e Preparação

Todos os comandos abaixo devem ser executados a partir da raiz do projeto.

**1.1 Importação de Imagens para o Kubernetes**
Dependendo do seu ambiente Kubernetes, pode ser necessário carregar as imagens construídas no seu cluster.
- Se usar o Docker Desktop K8s ou Minikube (com o driver Docker), as imagens construídas já estão disponíveis localmente.
- Se usar o Kind, carregue diretamente as imagens do seu daemon local do Docker:
  ```bash
  kind load docker-image tcc/target-app:latest
  kind load docker-image tcc/scaler:latest
  ```

**1.2 Implantar a Infraestrutura**
```bash
# Criar o namespace de infraestrutura primeiro
kubectl create namespace tcc-infra

# Implantar o Prometheus
kubectl apply -f experiments/k8s/manifests/prometheus/configmap.yaml
kubectl apply -f experiments/k8s/manifests/prometheus/deployment.yaml
```

#### 2. Calibração (Fase 6)

Você deve determinar os parâmetros de recursos de pico antes de rodar a geração de traços e o orquestrador. Com a calibração automatizada, isso é escrito diretamente no `experiments/k8s/calibration.json`. O orquestrador irá se recusar a iniciar se algum valor de calibração estiver faltando.

**2.1 Iniciar Pod de Calibração**
```bash
kubectl run calibration-pod --image=tcc/target-app:latest \
  --image-pull-policy=Never --port=8000 --env="CPU_WORK_ITERATIONS=50000" -n tcc-infra

# Em um terminal separado, inicie o port-forwarding para a aplicação e para o Prometheus:
kubectl port-forward pod/calibration-pod 8001:8000 -n tcc-infra &
kubectl port-forward svc/prometheus 30090:9090 -n tcc-infra &
```

**2.2 Executar a Auto-Calibração**
Utilize o script `calibrate.py` para automaticamente elevar os usuários do Locust, consultar o Prometheus, e encontrar os picos de recursos.
```bash
# Executar para a App A (Repita para a App B e C com --app B e --app C)
python3 experiments/k8s/load/calibrate.py \
  --namespace tcc-infra \
  --app A \
  --port 8001 \
  --prometheus-url http://localhost:30090 \
  --max-users 60 \
  --step-size 5 \
  --step-duration 30 \
  --output experiments/k8s/calibration.json
```

#### 3. Gerar Traços de Carga

Assim que o `calibration.json` for populado, utilize os valores automatizados para gerar os traços de carga corretos para o Locust e o injetor de RAM. Os passos de aquecimento (`warmup-steps`) são calculados com exatidão para cobrir 7 dias baseado na frequência de amostragem de cada dataset (1 hora para CPU = 168 passos, 40 minutos para RAM = 252 passos).

```bash
# --- Aplicação A ---
docker compose run --rm exp2-trace-generator bash -c "\
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_A_CPU_exp2_replay.csv --metric cpu --peak-cpu-cores \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['cpu']['peak_cpu_cores'])\") --peak-rps \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['cpu']['peak_rps'])\") --warmup-steps 168 && \
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_A_RAM_exp2_replay.csv --metric ram --min-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['ram']['min_memory_mb'])\") --max-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_a']['ram']['max_memory_mb'])\") --warmup-steps 252 --memory-step-mb 1"

# --- Aplicação B ---
docker compose run --rm exp2-trace-generator bash -c "\
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_B_CPU_exp2_replay.csv --metric cpu --peak-cpu-cores \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['cpu']['peak_cpu_cores'])\") --peak-rps \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['cpu']['peak_rps'])\") --warmup-steps 168 && \
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_B_RAM_exp2_replay.csv --metric ram --min-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['ram']['min_memory_mb'])\") --max-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_b']['ram']['max_memory_mb'])\") --warmup-steps 252 --memory-step-mb 1"

# --- Aplicação C ---
docker compose run --rm exp2-trace-generator bash -c "\
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_C_CPU_exp2_replay.csv --metric cpu --peak-cpu-cores \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['cpu']['peak_cpu_cores'])\") --peak-rps \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['cpu']['peak_rps'])\") --warmup-steps 168 && \
  python k8s/load/trace_replay.py --csv data/cegedim/cegedim_application_C_RAM_exp2_replay.csv --metric ram --min-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['ram']['min_memory_mb'])\") --max-memory-mb \$(python -c \"import json; print(json.load(open('k8s/calibration.json'))['app_c']['ram']['max_memory_mb'])\") --warmup-steps 252 --memory-step-mb 1"
```

#### 4. Sessão de Execução (Fase 8)

Assim que a calibração for completada e os traços forem gerados, execute os batches de experimento.
Garanta que o diretório de persistência de resultados existe na sua máquina host de forma que os resultados sejam corretamente salvos em caso de falhas.

```bash
# Criar o diretório de persistência de resultados na máquina host
sudo mkdir -p /var/tcc-results && sudo chmod 777 /var/tcc-results
```

> O orquestrador automaticamente cria um subdiretório contendo o horário (timestamp) dentro do seu `--output-dir` para cada execução (ex: `results/exp2/2026-06-07_12-00-00/`). Todos os arquivos de métricas e estado são salvos lá.
> 
> **Retomando um Batch:** Se você apertar `Ctrl+C` para abortar um batch (o que desencadeia uma limpeza completa dos namespaces atuais) ou se ocorrer algum erro fatal, você pode retomá-lo. Para isso, aponte o `--output-dir` exatamente para o diretório de timestamp da execução interrompida e adicione a flag `--resume`. O orquestrador irá continuar a fila `pending` de onde parou.

> **Recomendação de Batch Size:** Nós recomendamos usar `--batch-size 1` devido a restrições de calibração, especialmente se o seu ambiente local não possui recursos suficientes para igualar as mesmas capacidades das aplicações originais.

**4.1 Batch de Execução**
Execute os experimentos usando o modo de escalonamento combinado.

> Apesar de apenas datasets de CPU (`a-cpu b-cpu c-cpu`) serem fornecidos explicitamente no comando abaixo, o modo de escalonamento `combined` garante que os modelos para ambas as métricas (CPU e RAM) serão instanciados e avaliados.

```bash
docker compose run --rm exp2-orchestrator python k8s/orchestrator.py \
  --exp1-results-dir results/ \
  --trace-dir k8s/load/traces/ \
  --output-dir results/exp2/ \
  --models Prophet ExponentialSmoothing FFT Ensemble \
  --datasets a-cpu b-cpu c-cpu \
  --scaling-modes combined \
  --cpu-trace-step-seconds 15 \
  --ram-trace-step-seconds 15 \
  --batch-size 1
```

#### 5. Análise (Fase 7)

Assim que todos os batches forem finalizados, gere os arquivos CSV e os gráficos.

```bash
docker compose run --rm exp2-analyzer python k8s/analyze_results.py \
  --exp2-results-dir results/exp2/ \
  --exp1-results-dir results/ \
  --output-dir results/exp2/analysis/
```

#### 6. Atualizando Traços e Calibração

Se você modificar as divisões (splits) do dataset ou as distribuições matemáticas, você **deve** obrigatoriamente regerar manualmente os traços e recalibrar o sistema antes de executar os batches do orquestrador.

Execute estes passos em ordem a partir da raiz do projeto:

1. **Regerar as divisões em CSV**:
   ```bash
   python3 experiments/split_datasets.py
   ```
2. **Retreinar os modelos de baseline**:
   ```bash
   python3 experiments/training/pipeline.py
   ```
3. **Executar a Fase de Calibração** (Execute a Seção 2 acima para achar os parâmetros de pico de CPU e RAM).
4. **Regerar os Traços JSON** (Execute os comandos do gerador de traço na Seção 3 acima).

Assim que estes passos forem completados, você pode prosseguir seguramente para a Sessão de Execução (Seção 4).
