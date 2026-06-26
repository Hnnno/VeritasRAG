# Pipeline de Ajuste Fino (Fine-Tuning) con QLoRA — Guía Completa

## Descripción General

Este pipeline implementa el ajuste fino eficiente de modelos de lenguaje de código abierto
(Llama 3, Qwen 2.5, Mistral) utilizando **QLoRA** — una técnica que combina cuantización
en 4 bits con adaptadores LoRA para entrenar en hardware de consumo con mínima pérdida de calidad.

**Caso de uso de ejemplo:** Especializar un LLM para generar llamadas estructuradas a
una API REST de e-commerce a partir de instrucciones en lenguaje natural.

---

## Arquitectura del Pipeline

```
[Datos] → [finetune_qlora.py] → [Adaptadores LoRA] → [Modelo Fusionado]
                                                              ↓
                                               [quantize_to_gguf.py]
                                                              ↓
                                              [Modelo GGUF Q4_K_M ~4.4GB]
                                                              ↓
                                              [local_inference.py]
                                              (Ollama o llama.cpp)
```

---

## Requisitos del Sistema

### Hardware mínimo para entrenamiento (modelo 7B)
- **GPU:** NVIDIA con ≥12 GB VRAM (RTX 3060 Ti, 3080, A100, etc.)
- **RAM:** 24 GB recomendado (16 GB mínimo)
- **Almacenamiento:** 50 GB libres (modelo base + checkpoints + GGUF)

### Hardware mínimo para inferencia (modelo 7B Q4_K_M)
- **RAM:** 6-8 GB disponibles
- **CPU:** Cualquier procesador moderno x86_64 o Apple Silicon
- **GPU:** Opcional (mejora la velocidad de generación ~5-10x)

### Software
```bash
# Python 3.10+
python --version

# CUDA 11.8+ (para GPU NVIDIA)
nvcc --version

# Dependencias Python
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install trl>=0.8.6
pip install datasets>=2.18.0
pip install bitsandbytes>=0.43.0
pip install accelerate>=0.29.0
pip install flash-attn>=2.5.8  # Opcional: Flash Attention 2 para mayor velocidad
```

---

## Estructura de Archivos

```
.
├── finetune_qlora.py      # Script principal de entrenamiento QLoRA
├── quantize_to_gguf.py    # Conversión del modelo a formato GGUF
├── local_inference.py     # Inferencia local (Ollama / llama.cpp)
├── README.md              # Esta guía
└── output/
    ├── qlora-adapter/     # Adaptadores LoRA entrenados (<100 MB)
    ├── merged-model/      # Modelo fusionado en formato HuggingFace (~14 GB)
    └── gguf/
        ├── api-specialist-f16.gguf        # GGUF float16 (~14 GB, intermedio)
        ├── api-specialist-q4_k_m.gguf     # GGUF 4-bit (~4.4 GB) ← USAR ESTE
        ├── api-specialist-q8_0.gguf       # GGUF 8-bit (~7.7 GB)
        └── Modelfile                      # Configuración para Ollama
```

---

## Paso 1: Entrenamiento (finetune_qlora.py)

### Conceptos clave implementados

| Técnica | Descripción | Beneficio |
|---------|-------------|-----------|
| **QLoRA** | Cuantización 4-bit + LoRA | Entrena 7B en 12GB VRAM |
| **NF4** | Normal Float 4-bit | Mayor precisión que FP4 estándar |
| **Double Quant** | Cuantización anidada | Ahorra ~0.5 bits adicionales/parámetro |
| **Flash Attention 2** | Atención optimizada | ~30% más rápido en GPUs Ampere+ |
| **Gradient Checkpointing** | Recalcula activaciones | Reduce VRAM ~40-60% |
| **Paged AdamW** | Optimizador paginado | Evita OOM en GPUs limitadas |

### Ejecución básica (datos sintéticos de demostración)
```bash
python finetune_qlora.py
```

### Ejecución con modelo personalizado y dataset real
```bash
python finetune_qlora.py \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output_dir "./mis-adaptadores" \
    --merged_dir "./modelo-fusionado" \
    --hf_dataset "iamtarun/python_code_instructions_18k_alpaca" \
    --use_real_data
```

### Parámetros LoRA recomendados por tamaño de dataset

| Dataset | r (rank) | lora_alpha | dropout | epochs |
|---------|----------|------------|---------|--------|
| <500 ejemplos | 8 | 16 | 0.05 | 5 |
| 500-5K ejemplos | 16 | 32 | 0.05 | 3 |
| 5K-50K ejemplos | 32 | 64 | 0.1 | 2 |
| >50K ejemplos | 64 | 128 | 0.1 | 1 |

### Monitoreo del entrenamiento
```bash
# Opcional: conectar a Weights & Biases para visualización de métricas
pip install wandb
wandb login
# Cambiar en finetune_qlora.py: report_to="wandb"
```

---

## Paso 2: Cuantización (quantize_to_gguf.py)

### Comparativa de formatos GGUF

| Formato | Bits/param | Tamaño (7B) | RAM mínima | Pérdida calidad |
|---------|-----------|-------------|------------|-----------------|
| F16 | 16 | ~14 GB | 16 GB | Ninguna |
| Q8_0 | 8.5 | ~7.7 GB | 10 GB | Mínima (<1%) |
| **Q4_K_M** | **4.8** | **~4.4 GB** | **6 GB** | **Baja (~3%)** |
| Q4_0 | 4.5 | ~4.1 GB | 6 GB | Moderada (~5%) |
| Q3_K_M | 3.9 | ~3.5 GB | 5 GB | Alta (~8%) |

**Recomendación:** Q4_K_M para uso general, Q8_0 si tienes RAM suficiente.

### Ejecución
```bash
python quantize_to_gguf.py
```

---

## Paso 3: Inferencia Local (local_inference.py)

### Opción A: Ollama (recomendado para la mayoría de usuarios)

```bash
# 1. Instalar Ollama
# Linux/macOS: curl -fsSL https://ollama.ai/install.sh | sh
# Windows: descargar desde https://ollama.ai/download

# 2. Importar el modelo ajustado
ollama create api-specialist -f ./output/gguf/Modelfile

# 3. Probar el modelo
ollama run api-specialist "Consulta el stock del SKU-1234 en Madrid"

# 4. Usar la API REST (en otra terminal)
curl http://localhost:11434/api/chat -d '{
  "model": "api-specialist",
  "messages": [{"role": "user", "content": "Crea un pedido para cliente 99134"}],
  "stream": false
}'

# 5. Ejecutar el script de demo
python local_inference.py --backend ollama --modo demo

# 6. Modo interactivo
python local_inference.py --backend ollama --modo interactivo
```

### Opción B: llama-cpp-python (sin servidor externo)

```bash
# Instalar con soporte GPU NVIDIA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall

# Instalar para CPU puro
pip install llama-cpp-python

# Ejecutar demo
python local_inference.py \
    --backend llama_cpp \
    --modelo_gguf ./output/gguf/api-specialist-q4_k_m.gguf \
    --gpu_layers -1 \
    --modo demo
```

---

## Preparar tus Propios Datos de Entrenamiento

### Formato requerido (ChatML)

El script `finetune_qlora.py` espera datos en formato de conversación:

```python
{
    "messages": [
        {"role": "system", "content": "Tu instrucción de sistema aquí"},
        {"role": "user", "content": "Pregunta o instrucción del usuario"},
        {"role": "assistant", "content": "Respuesta esperada del modelo"}
    ]
}
```

### Generación con Claude API (para datasets sintéticos a escala)

```python
import anthropic

client = anthropic.Anthropic()

def generar_ejemplo(categoria: str, accion: str) -> dict:
    prompt = f"""Genera un ejemplo de entrenamiento para un asistente de API REST.
    Categoría: {categoria}
    Acción: {accion}
    
    Responde SOLO con JSON en este formato:
    {{"instruccion": "...", "llamada_api": {{...}}}}"""
    
    mensaje = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(mensaje.content[0].text)
```

### Datasets públicos recomendados en Hugging Face

```bash
# Instrucciones de código Python (18K ejemplos)
iamtarun/python_code_instructions_18k_alpaca

# Seguimiento de instrucciones general (52K ejemplos, base Alpaca)  
tatsu-lab/alpaca

# Llamadas a funciones/APIs (alta calidad)
glaiveai/glaive-function-calling-v2
```

---

## Solución de Problemas Comunes

### Error: CUDA out of memory
```python
# En finetune_qlora.py, reducir:
per_device_train_batch_size = 1    # Reducir de 2 a 1
gradient_accumulation_steps = 16   # Aumentar para mantener batch efectivo
max_seq_length = 1024              # Reducir longitud de secuencia
```

### El modelo genera JSON inválido después del ajuste fino
- Aumentar los ejemplos de entrenamiento (mínimo 200 ejemplos de calidad)
- Reducir la temperatura de inferencia a 0.05-0.1
- Verificar que el system prompt de entrenamiento coincide con el de inferencia

### Ollama no reconoce el modelo
```bash
# Verificar que el path en el Modelfile es absoluto y correcto
cat ./output/gguf/Modelfile
# Reimportar el modelo
ollama rm api-specialist
ollama create api-specialist -f ./output/gguf/Modelfile
```

### Baja velocidad de inferencia en CPU
```bash
# Usar más hilos en llama.cpp (no más que núcleos físicos)
python local_inference.py --backend llama_cpp --gpu_layers 0
# Editar N_THREADS = número_de_nucleos_fisicos en local_inference.py
```

---

## Referencias y Recursos

- **QLoRA original:** Dettmers et al. (2023) — https://arxiv.org/abs/2305.14314
- **LoRA:** Hu et al. (2021) — https://arxiv.org/abs/2106.09685
- **PEFT Library:** https://github.com/huggingface/peft
- **TRL Library:** https://github.com/huggingface/trl
- **llama.cpp:** https://github.com/ggerganov/llama.cpp
- **Ollama:** https://ollama.ai
- **Qwen 2.5 Models:** https://huggingface.co/Qwen
- **Llama 3 Models:** https://huggingface.co/meta-llama
