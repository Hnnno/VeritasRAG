"""
==============================================================================
Pipeline de Ajuste Fino (Fine-Tuning) con QLoRA para LLMs de Código Abierto
Modelos compatibles: Llama 3, Qwen 2.5, Mistral, Phi-3, etc.
Objetivo: Especialización en formatos de datos industriales y llamadas a APIs
==============================================================================
Autor: Pipeline de producción - Uso en entornos con GPU limitada
Requisitos: pip install transformers peft trl datasets bitsandbytes accelerate
"""

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging as hf_logging,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN GLOBAL DEL PIPELINE
# ==============================================================================

# --- Configuración del modelo base ---
# Cambia MODEL_ID por el modelo que desees ajustar.
# Opciones recomendadas (en orden de menor a mayor tamaño):
#   - "Qwen/Qwen2.5-1.5B-Instruct"   → muy ligero, ideal para pruebas rápidas
#   - "Qwen/Qwen2.5-7B-Instruct"     → buen balance calidad/recursos
#   - "meta-llama/Meta-Llama-3-8B-Instruct" → requiere aceptar licencia en HF
#   - "mistralai/Mistral-7B-Instruct-v0.3"  → alternativa sólida sin licencia
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# --- Directorio de salida ---
# Aquí se guardarán los adaptadores LoRA entrenados (NO el modelo completo).
# El tamaño es mucho menor al del modelo base (típicamente < 100MB).
OUTPUT_DIR = "./output/qlora-adapter"

# --- Nombre del modelo final fusionado ---
# Después del entrenamiento, fusionamos los pesos LoRA con el modelo base.
MERGED_MODEL_DIR = "./output/merged-model"

# --- Nombre del experimento en W&B (opcional) ---
# Si tienes Weights & Biases configurado, registra las métricas automáticamente.
WANDB_PROJECT = "domain-finetuning-pipeline"


# ==============================================================================
# SECCIÓN 2: GENERACIÓN DE DATOS SINTÉTICOS DE ENTRENAMIENTO
# ==============================================================================
# Este conjunto de datos simula un caso de uso real: un sistema de IA que debe
# aprender a generar llamadas estructuradas a una API REST de comercio electrónico.
# En producción, sustituye esto por tus datos reales o un dataset de Hugging Face.

def generate_synthetic_api_dataset() -> List[Dict]:
    """
    Genera ejemplos sintéticos de instrucción → respuesta en formato de chat.
    
    El formato utilizado sigue el estándar de ChatML, que la mayoría de los
    modelos modernos (Qwen, Llama 3, Mistral) entienden nativamente.
    
    Cada ejemplo contiene:
    - system: contexto del rol del asistente
    - user: instrucción del usuario (descripción en lenguaje natural)
    - assistant: respuesta esperada (llamada a API estructurada en JSON)
    """
    
    # Instrucción del sistema que define el dominio y las reglas de negocio.
    # Esta instrucción se repetirá en todos los ejemplos para establecer el contexto.
    SYSTEM_PROMPT = """Eres un asistente especializado en integración de APIs de e-commerce.
Tu tarea es convertir instrucciones en lenguaje natural en llamadas API estructuradas.
Siempre responde con un objeto JSON válido que incluya: endpoint, method, headers y body.
Nunca añadas explicaciones adicionales fuera del JSON."""

    # Lista de ejemplos instrucción-respuesta que cubren diferentes endpoints de la API.
    # Cada ejemplo enseña al modelo un patrón diferente de uso de la API.
    ejemplos = [
        # --- Ejemplo 1: Consulta de inventario ---
        {
            "instruction": "Quiero saber cuántas unidades del producto SKU-2847 tenemos disponibles en el almacén de Madrid.",
            "response": json.dumps({
                "endpoint": "/api/v2/inventory/check",
                "method": "GET",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Content-Type": "application/json",
                    "X-Warehouse-Region": "ES-MAD"
                },
                "body": {
                    "sku": "SKU-2847",
                    "warehouse_id": "WH_MADRID_01",
                    "include_reserved": True
                }
            }, ensure_ascii=False, indent=2)
        },
        # --- Ejemplo 2: Creación de pedido ---
        {
            "instruction": "Crea un pedido urgente para el cliente con ID 99134. Necesita 3 unidades de SKU-1122 y 1 unidad de SKU-5566. Envío exprés a la dirección registrada.",
            "response": json.dumps({
                "endpoint": "/api/v2/orders/create",
                "method": "POST",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Content-Type": "application/json",
                    "X-Priority": "urgent"
                },
                "body": {
                    "customer_id": "99134",
                    "shipping_method": "express",
                    "use_registered_address": True,
                    "items": [
                        {"sku": "SKU-1122", "quantity": 3},
                        {"sku": "SKU-5566", "quantity": 1}
                    ],
                    "metadata": {"source": "ai_assistant", "priority": "high"}
                }
            }, ensure_ascii=False, indent=2)
        },
        # --- Ejemplo 3: Actualización de precio ---
        {
            "instruction": "Actualiza el precio del producto SKU-3344 a 29.99 EUR. Aplica un descuento del 15% para clientes premium con nivel Gold o superior.",
            "response": json.dumps({
                "endpoint": "/api/v2/products/pricing",
                "method": "PATCH",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Content-Type": "application/json"
                },
                "body": {
                    "sku": "SKU-3344",
                    "base_price": 29.99,
                    "currency": "EUR",
                    "discounts": [
                        {
                            "type": "customer_tier",
                            "tiers": ["gold", "platinum", "diamond"],
                            "discount_percent": 15,
                            "valid_from": None,
                            "valid_until": None
                        }
                    ]
                }
            }, ensure_ascii=False, indent=2)
        },
        # --- Ejemplo 4: Consulta de estado de envío ---
        {
            "instruction": "El cliente pregunta por el estado de su envío con número de seguimiento ES-TRK-20241201-8847.",
            "response": json.dumps({
                "endpoint": "/api/v2/shipments/track",
                "method": "GET",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Accept": "application/json"
                },
                "body": {
                    "tracking_number": "ES-TRK-20241201-8847",
                    "include_history": True,
                    "language": "es"
                }
            }, ensure_ascii=False, indent=2)
        },
        # --- Ejemplo 5: Procesamiento de devolución ---
        {
            "instruction": "Procesa la devolución del pedido ORD-77823. El cliente indica que los productos llegaron dañados. Reembolso completo a la tarjeta original.",
            "response": json.dumps({
                "endpoint": "/api/v2/returns/process",
                "method": "POST",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Content-Type": "application/json",
                    "X-Return-Policy": "standard"
                },
                "body": {
                    "order_id": "ORD-77823",
                    "reason": "damaged_goods",
                    "refund_method": "original_payment",
                    "refund_full_amount": True,
                    "return_shipping_label": True,
                    "notify_customer": True,
                    "notes": "Cliente reporta daños en los productos recibidos"
                }
            }, ensure_ascii=False, indent=2)
        },
        # --- Ejemplo 6: Búsqueda de productos con filtros ---
        {
            "instruction": "Busca todos los productos de la categoría 'electrónica' con precio entre 50 y 200 EUR, stock disponible mayor a 10 unidades, y ordenados por popularidad descendente.",
            "response": json.dumps({
                "endpoint": "/api/v2/products/search",
                "method": "GET",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Accept": "application/json"
                },
                "body": {
                    "category": "electronica",
                    "filters": {
                        "price_range": {"min": 50.00, "max": 200.00, "currency": "EUR"},
                        "stock": {"operator": "gt", "value": 10},
                        "status": "active"
                    },
                    "sort": {"field": "popularity", "order": "desc"},
                    "pagination": {"page": 1, "limit": 20}
                }
            }, ensure_ascii=False, indent=2)
        },
        # --- Ejemplo 7: Actualización masiva de inventario ---
        {
            "instruction": "Ajusta el inventario del almacén de Barcelona: suma 500 unidades de SKU-9901, resta 200 de SKU-9902 por merma, y bloquea 150 unidades de SKU-9903 para reserva.",
            "response": json.dumps({
                "endpoint": "/api/v2/inventory/bulk-update",
                "method": "POST",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Content-Type": "application/json",
                    "X-Warehouse-Region": "ES-BCN"
                },
                "body": {
                    "warehouse_id": "WH_BARCELONA_01",
                    "operations": [
                        {"sku": "SKU-9901", "action": "add", "quantity": 500, "reason": "restock"},
                        {"sku": "SKU-9902", "action": "subtract", "quantity": 200, "reason": "shrinkage"},
                        {"sku": "SKU-9903", "action": "reserve", "quantity": 150, "reason": "manual_hold"}
                    ],
                    "audit_user": "{{USER_ID}}",
                    "notify_low_stock": True
                }
            }, ensure_ascii=False, indent=2)
        },
        # --- Ejemplo 8: Configuración de campaña de descuentos ---
        {
            "instruction": "Crea una campaña de descuento del 20% para todos los productos de la categoría 'ropa de invierno', válida del 15 al 31 de enero, solo para nuevos clientes registrados en los últimos 6 meses.",
            "response": json.dumps({
                "endpoint": "/api/v2/campaigns/create",
                "method": "POST",
                "headers": {
                    "Authorization": "Bearer {{API_TOKEN}}",
                    "Content-Type": "application/json"
                },
                "body": {
                    "name": "Descuento Ropa Invierno - Nuevos Clientes",
                    "type": "category_discount",
                    "target_categories": ["ropa-invierno"],
                    "discount": {"type": "percentage", "value": 20},
                    "validity": {
                        "start_date": "2025-01-15T00:00:00Z",
                        "end_date": "2025-01-31T23:59:59Z"
                    },
                    "eligibility": {
                        "customer_type": "new",
                        "registration_window_days": 180
                    },
                    "combinable_with_other_offers": False,
                    "status": "draft"
                }
            }, ensure_ascii=False, indent=2)
        },
    ]
    
    # Construye el dataset en formato de conversación (ChatML).
    # El modelo aprende a completar el turno del "assistant" dado el contexto.
    dataset_rows = []
    for ejemplo in ejemplos:
        dataset_rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ejemplo["instruction"]},
                {"role": "assistant", "content": ejemplo["response"]}
            ]
        })
    
    return dataset_rows


# ==============================================================================
# SECCIÓN 3: CONFIGURACIÓN DE CUANTIZACIÓN EN 4-BIT (QLoRA)
# ==============================================================================

def get_bnb_config() -> BitsAndBytesConfig:
    """
    Configura la cuantización NF4 (Normal Float 4-bit) de BitsAndBytes.
    
    QLoRA = Quantized Low-Rank Adaptation:
    - El modelo base se carga en precisión 4-bit (ahorra ~75% de VRAM)
    - Los adaptadores LoRA se entrenan en bfloat16 (precisión completa)
    - Permite entrenar modelos de 7B en GPUs con ≥12GB de VRAM (ej: RTX 3060)
    
    Parámetros clave:
    - load_in_4bit: activa la cuantización de 4 bits
    - bnb_4bit_quant_type="nf4": formato NF4 es más preciso que el FP4 estándar
    - bnb_4bit_use_double_quant: cuantiza también las constantes de cuantización
      (doble cuantización → ahorra ~0.5 bits adicionales por parámetro)
    - bnb_4bit_compute_dtype: dtype para los cálculos durante el forward pass
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NF4 > FP4 en distribuciones normales
        bnb_4bit_use_double_quant=True,       # Cuantización anidada para mayor ahorro
        bnb_4bit_compute_dtype=torch.bfloat16 # bfloat16 es más estable que float16
    )


# ==============================================================================
# SECCIÓN 4: CONFIGURACIÓN DE LOS ADAPTADORES LORA
# ==============================================================================

def get_lora_config() -> LoraConfig:
    """
    Configura los adaptadores LoRA (Low-Rank Adaptation).
    
    Principio de LoRA:
    En lugar de modificar toda la matriz de pesos W (de dimensión d×k),
    aprendemos dos matrices pequeñas: A (d×r) y B (r×k), donde r << d.
    El cambio de peso se representa como: ΔW = B × A
    Esto reduce los parámetros entrenables de d×k a r×(d+k).
    
    Parámetros clave:
    - r (rank): dimensión del rango de los adaptadores
        → Valores pequeños (4-8): menos parámetros, más rápido, puede underfit
        → Valores grandes (32-64): más capacidad, más lento, riesgo de overfit
        → Recomendado: 16 para datasets medianos (1K-50K ejemplos)
    
    - lora_alpha: factor de escala para los pesos LoRA
        → La escala efectiva es: alpha/r
        → Convención común: alpha = 2*r (escala ≈ 2.0)
        → Un alpha más alto = mayor tasa de aprendizaje efectiva para LoRA
    
    - lora_dropout: regularización para prevenir sobreajuste
        → 0.05-0.1 para datasets pequeños
        → 0.0 para datasets grandes (>100K ejemplos)
    
    - target_modules: capas del transformer donde se insertan los adaptadores
        → "all-linear": aplica LoRA a todas las capas lineales (más potente)
        → Lista específica: ["q_proj", "v_proj"] es el mínimo recomendado
        → Para Qwen/Llama: incluir k_proj, o_proj mejora resultados
    
    - bias="none": no entrenar sesgos (mantener los del modelo base)
    
    - task_type=CAUSAL_LM: indica que es un modelo de lenguaje autoregresivo
    """
    return LoraConfig(
        r=16,                          # Rango de los adaptadores LoRA
        lora_alpha=32,                 # Factor de escala (alpha/r = 2.0)
        lora_dropout=0.05,             # Dropout para regularización
        target_modules="all-linear",   # Aplica LoRA a todas las capas lineales
        bias="none",                   # No entrenar sesgos
        task_type=TaskType.CAUSAL_LM,  # Modelado de lenguaje causal (autoregresivo)
        inference_mode=False,          # Modo entrenamiento (no inferencia)
    )


# ==============================================================================
# SECCIÓN 5: CONFIGURACIÓN DE LOS HIPERPARÁMETROS DE ENTRENAMIENTO
# ==============================================================================

def get_training_config(output_dir: str) -> SFTConfig:
    """
    Define los hiperparámetros del proceso de entrenamiento supervisado (SFT).
    
    SFT = Supervised Fine-Tuning: aprendizaje supervisado donde el modelo
    aprende a predecir el token siguiente dado el contexto anterior.
    
    Parámetros críticos:
    
    TASA DE APRENDIZAJE (learning_rate):
    - Rango típico para fine-tuning: 1e-5 a 3e-4
    - Con QLoRA: 2e-4 es un buen punto de partida
    - Muy alta → pérdida inestable, el modelo "olvida" conocimiento base
    - Muy baja → convergencia lenta o quedarse atascado en mínimo local
    
    BATCH SIZE Y GRADIENT ACCUMULATION:
    - per_device_train_batch_size: muestras procesadas simultáneamente por GPU
        → Limitado por VRAM: con 12GB y modelo 7B en 4-bit, usa 2-4
    - gradient_accumulation_steps: acumula gradientes de N pasos antes de actualizar
        → Batch size efectivo = per_device_batch * accumulation_steps
        → Con batch=2 y accumulation=8 → batch efectivo = 16 (mejor estabilidad)
    
    ÉPOCAS (num_train_epochs):
    - Para datasets pequeños (<1K ejemplos): 3-5 épocas
    - Para datasets medianos (1K-50K): 1-3 épocas  
    - Monitorear la pérdida de validación para detectar sobreajuste
    
    PLANIFICADOR DE LR (lr_scheduler_type):
    - "cosine": decaimiento suave, generalmente el mejor para LLMs
    - "linear": decaimiento lineal, más simple
    - warmup_ratio: % de pasos para calentar la tasa de aprendizaje gradualmente
        → Evita gradientes explosivos al inicio del entrenamiento
    
    PRECISIÓN MIXTA (bf16/fp16):
    - bf16=True: usar bfloat16 en GPUs Ampere+ (A100, RTX 3000/4000 series)
    - fp16=True: alternativa para GPUs más antiguas (V100, RTX 2000 series)
    - Reduce uso de VRAM ~50% vs float32, con mínima pérdida de precisión
    
    GRADIENT CHECKPOINTING:
    - Técnica para reducir VRAM a costa de mayor tiempo de cómputo
    - En lugar de guardar todas las activaciones, las recalcula en backward pass
    - Reduce VRAM ~40-60%, aumenta tiempo de entrenamiento ~20-30%
    
    OPTIMIZADOR (optim):
    - "paged_adamw_32bit": versión paginada de AdamW, libera memoria CPU/GPU
        → Esencial para QLoRA en GPUs con VRAM limitada
    - Los estados del optimizador se "paginan" a CPU cuando no se usan
    """
    return SFTConfig(
        # --- Directorios y registro ---
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        report_to="none",              # Cambiar a "wandb" si tienes W&B configurado
        
        # --- Hiperparámetros de entrenamiento ---
        num_train_epochs=3,            # Número de épocas sobre el dataset completo
        learning_rate=2e-4,            # Tasa de aprendizaje inicial
        lr_scheduler_type="cosine",    # Decaimiento coseno de la tasa de aprendizaje
        warmup_ratio=0.03,             # 3% de pasos iniciales para calentamiento gradual
        
        # --- Control del tamaño de batch y memoria ---
        per_device_train_batch_size=2,  # Muestras por GPU por paso (ajustar según VRAM)
        per_device_eval_batch_size=2,   # Muestras por GPU en evaluación
        gradient_accumulation_steps=8,  # Acumulación: batch efectivo = 2*8 = 16
        gradient_checkpointing=True,    # Activa checkpointing para ahorrar VRAM
        
        # --- Precisión numérica ---
        bf16=True,                     # bfloat16 en GPUs compatibles (Ampere+)
        fp16=False,                    # NO usar simultáneamente con bf16
        
        # --- Optimizador ---
        optim="paged_adamw_32bit",     # AdamW paginado, esencial para QLoRA
        weight_decay=0.001,            # Regularización L2 para prevenir sobreajuste
        max_grad_norm=0.3,             # Clipping de gradientes para estabilidad
        
        # --- Configuración de secuencia ---
        max_seq_length=2048,           # Longitud máxima de contexto (tokens)
        packing=False,                 # False: un ejemplo por secuencia (más claro)
                                       # True: empaqueta múltiples ejemplos (más rápido)
        
        # --- Frecuencia de guardado y evaluación ---
        save_steps=50,                 # Guarda checkpoint cada N pasos
        eval_steps=50,                 # Evalúa en validation set cada N pasos
        logging_steps=10,              # Registra métricas cada N pasos
        evaluation_strategy="steps",   # Estrategia de evaluación por pasos
        save_total_limit=3,            # Mantiene solo los 3 mejores checkpoints
        load_best_model_at_end=True,   # Carga el mejor modelo al finalizar
        
        # --- Dataset ---
        dataset_text_field="text",     # Campo del dataset que contiene el texto formateado
    )


# ==============================================================================
# SECCIÓN 6: FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ==============================================================================

def train(
    model_id: str = MODEL_ID,
    output_dir: str = OUTPUT_DIR,
    merged_dir: str = MERGED_MODEL_DIR,
    use_synthetic_data: bool = True,
    hf_dataset_name: Optional[str] = None,
):
    """
    Función principal que orquesta todo el pipeline de ajuste fino.
    
    Flujo de ejecución:
    1. Carga y prepara el dataset
    2. Inicializa el tokenizador
    3. Carga el modelo base en 4-bit (QLoRA)
    4. Configura y aplica los adaptadores LoRA
    5. Entrena con SFTTrainer
    6. Guarda los adaptadores LoRA entrenados
    7. Fusiona LoRA con el modelo base y guarda el modelo completo
    
    Args:
        model_id: Identificador del modelo en Hugging Face Hub
        output_dir: Directorio para guardar los adaptadores LoRA
        merged_dir: Directorio para guardar el modelo fusionado
        use_synthetic_data: Si True, usa datos sintéticos generados
        hf_dataset_name: Nombre del dataset en HF Hub (si use_synthetic_data=False)
    """
    
    print("=" * 70)
    print("INICIANDO PIPELINE DE AJUSTE FINO CON QLORA")
    print(f"Modelo base: {model_id}")
    print(f"Directorio de salida: {output_dir}")
    print("=" * 70)
    
    # --- PASO 1: PREPARACIÓN DEL DATASET ---
    # Carga o genera los datos de entrenamiento y los formatea en ChatML
    print("\n[1/7] Preparando dataset de entrenamiento...")
    
    if use_synthetic_data:
        # Genera datos sintéticos de ejemplo (para demostración y pruebas)
        raw_data = generate_synthetic_api_dataset()
        # Divide en entrenamiento (80%) y validación (20%)
        split_idx = int(len(raw_data) * 0.8)
        train_data = raw_data[:split_idx]
        eval_data = raw_data[split_idx:]
        
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        print(f"   Dataset sintético: {len(train_data)} entrenamiento, {len(eval_data)} validación")
    else:
        # Carga un dataset real desde Hugging Face Hub
        # Ejemplo: hf_dataset_name = "iamtarun/python_code_instructions_18k_alpaca"
        dataset = load_dataset(hf_dataset_name, split="train")
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"   Dataset HF '{hf_dataset_name}': {len(train_dataset)} train, {len(eval_dataset)} eval")
    
    # --- PASO 2: INICIALIZACIÓN DEL TOKENIZADOR ---
    # El tokenizador convierte texto en tokens (enteros) que el modelo puede procesar
    print(f"\n[2/7] Cargando tokenizador desde '{model_id}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,   # Necesario para modelos con código personalizado (Qwen)
        padding_side="right",     # Padding a la derecha para entrenamiento causal
    )
    
    # El token de padding es necesario para procesar batches de diferente longitud.
    # Usamos el EOS token como pad token si no existe uno específico.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Función para formatear los mensajes al formato de chat del modelo
    # apply_chat_template convierte la lista de mensajes al formato correcto:
    # Por ejemplo, para Qwen/Llama 3: <|im_start|>system\n...<|im_end|>\n
    def format_messages_to_text(examples):
        """
        Aplica la plantilla de chat del modelo a cada ejemplo del dataset.
        Esto produce el texto formateado que el modelo aprenderá a generar.
        """
        texts = []
        for messages in examples["messages"]:
            # tokenize=False devuelve texto, no tokens
            # add_generation_prompt=False: no añadir marcador de generación al final
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    
    # Aplica el formateo al dataset completo
    train_dataset = train_dataset.map(format_messages_to_text, batched=True)
    eval_dataset = eval_dataset.map(format_messages_to_text, batched=True)
    
    print(f"   Ejemplo de texto formateado:\n{train_dataset[0]['text'][:300]}...")
    
    # --- PASO 3: CARGA DEL MODELO BASE EN 4-BIT ---
    # El modelo se carga directamente cuantizado a 4-bit para ahorrar VRAM
    print(f"\n[3/7] Cargando modelo base en 4-bit desde '{model_id}'...")
    print("   (Esto puede tardar varios minutos la primera vez...)")
    
    bnb_config = get_bnb_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,   # Configuración de cuantización 4-bit
        device_map="auto",                 # Distribuye automáticamente entre GPU/CPU
        trust_remote_code=True,            # Para modelos con código personalizado
        torch_dtype=torch.bfloat16,        # Tipo de dato para capas no cuantizadas
        attn_implementation="flash_attention_2",  # Flash Attention 2 para mayor velocidad
                                                   # Retirar si no está instalado
    )
    
    # Prepara el modelo para entrenamiento con k-bit (habilita gradient checkpointing,
    # convierte las capas de normalización a float32 para estabilidad)
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    
    print(f"   Modelo cargado. Parámetros totales: {model.num_parameters():,}")
    
    # --- PASO 4: APLICACIÓN DE ADAPTADORES LORA ---
    # Envuelve el modelo cuantizado con los adaptadores LoRA entrenables
    print("\n[4/7] Aplicando adaptadores LoRA al modelo...")
    
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Muestra el resumen de parámetros entrenables
    # Típicamente: ~0.5-2% del total de parámetros del modelo
    model.print_trainable_parameters()
    
    # --- PASO 5: CONFIGURACIÓN DEL ENTRENADOR ---
    # SFTTrainer (Supervised Fine-Tuning Trainer) de la librería TRL
    # Simplifica el bucle de entrenamiento con soporte nativo para PEFT y formato de chat
    print("\n[5/7] Configurando SFTTrainer...")
    
    training_config = get_training_config(output_dir)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_config,
        peft_config=lora_config,   # Pasa la config LoRA directamente (evita doble aplicación)
    )
    
    # --- PASO 6: EJECUCIÓN DEL ENTRENAMIENTO ---
    print("\n[6/7] Iniciando entrenamiento...")
    print("   Monitorea las métricas en la consola o en W&B si está configurado.")
    
    # Detecta si existe un checkpoint previo para reanudar el entrenamiento
    last_checkpoint = None
    if os.path.isdir(output_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(output_dir)
    ):
        checkpoints = [
            d for d in os.listdir(output_dir) if d.startswith("checkpoint-")
        ]
        last_checkpoint = os.path.join(output_dir, sorted(checkpoints)[-1])
        print(f"   Reanudando desde checkpoint: {last_checkpoint}")
    
    # Ejecuta el entrenamiento (puede tardar desde minutos hasta horas según el hardware)
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    
    # Registra las métricas finales del entrenamiento
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print(f"\n   Entrenamiento completado:")
    print(f"   - Pérdida final: {metrics.get('train_loss', 'N/A'):.4f}")
    print(f"   - Tiempo total: {metrics.get('train_runtime', 0)/60:.1f} minutos")
    
    # --- PASO 7: GUARDADO DEL MODELO ---
    # Primero guardamos solo los adaptadores LoRA (ligeros, rápidos de cargar)
    print(f"\n[7/7] Guardando adaptadores LoRA en '{output_dir}'...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("   Adaptadores LoRA guardados exitosamente.")
    print(f"   Tamaño aproximado: <100MB (vs {'>10GB' if '7B' in model_id else '>3GB'} del modelo completo)")
    
    # --- PASO OPCIONAL: FUSIÓN DE LORA CON EL MODELO BASE ---
    # Para producción y para convertir a GGUF, necesitamos el modelo completo fusionado.
    # Este paso requiere cargar el modelo base en float16 (sin cuantización).
    print(f"\n[BONUS] Fusionando adaptadores LoRA con el modelo base...")
    print("   Cargando modelo base en float16 para fusión (requiere más RAM)...")
    
    # Descarga los adaptadores del modelo cuantizado de la GPU
    del model
    torch.cuda.empty_cache()
    
    # Carga el modelo base sin cuantización para fusión
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,   # float16 para reducir RAM (vs float32)
        device_map="cpu",            # CPU para evitar problemas de VRAM durante fusión
        trust_remote_code=True,
    )
    
    # Carga los adaptadores LoRA entrenados
    peft_model = PeftModel.from_pretrained(base_model, output_dir)
    
    # Fusiona los pesos LoRA en el modelo base (unload_and_merge)
    # Resultado: modelo estándar sin capas PEFT, listo para exportar a GGUF
    print("   Fusionando pesos (esto puede tardar unos minutos)...")
    merged_model = peft_model.merge_and_unload()
    
    # Guarda el modelo fusionado completo
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    
    print(f"   ✓ Modelo fusionado guardado en '{merged_dir}'")
    print(f"   Listo para convertir a GGUF con el script de cuantización.")
    
    print("\n" + "=" * 70)
    print("PIPELINE DE AJUSTE FINO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\nPróximos pasos:")
    print(f"  1. Ejecuta 'python quantize_to_gguf.py' para convertir a GGUF/INT4")
    print(f"  2. Ejecuta 'python local_inference.py' para probar la inferencia local")
    print("=" * 70)


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    # Configuración de argumentos de línea de comandos para flexibilidad
    parser = argparse.ArgumentParser(
        description="Pipeline de ajuste fino QLoRA para LLMs de código abierto"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=MODEL_ID,
        help="ID del modelo en Hugging Face Hub (ej: 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directorio de salida para los adaptadores LoRA"
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        default=MERGED_MODEL_DIR,
        help="Directorio para el modelo base fusionado con LoRA"
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="Nombre del dataset en HF Hub (si no se usan datos sintéticos)"
    )
    parser.add_argument(
        "--use_real_data",
        action="store_true",
        help="Usar dataset real de HF Hub en lugar de datos sintéticos"
    )
    
    args = parser.parse_args()
    
    # Verifica disponibilidad de GPU antes de comenzar
    if not torch.cuda.is_available():
        print("⚠️  ADVERTENCIA: No se detectó GPU CUDA.")
        print("   El entrenamiento en CPU es posible pero muy lento.")
        print("   Recomendado: GPU con ≥12GB VRAM (ej: RTX 3060 Ti, A100)")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU detectada: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
    
    # Ejecuta el pipeline de entrenamiento
    train(
        model_id=args.model_id,
        output_dir=args.output_dir,
        merged_dir=args.merged_dir,
        use_synthetic_data=not args.use_real_data,
        hf_dataset_name=args.hf_dataset if args.use_real_data else None,
    )
