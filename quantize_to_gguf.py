"""
==============================================================================
Script de Cuantización: Conversión del Modelo Ajustado a Formato GGUF/INT4
==============================================================================
Propósito: Convertir el modelo fusionado (de finetune_qlora.py) a formato GGUF
para su despliegue local eficiente con llama.cpp u Ollama.

Flujo del script:
  1. Verifica que el modelo fusionado existe
  2. Clona/actualiza el repositorio llama.cpp (con el script de conversión)
  3. Convierte el modelo de formato HuggingFace a GGUF (float16)
  4. Cuantiza el GGUF a INT4/INT8 según la calidad deseada
  5. Crea un Modelfile para importar en Ollama (opcional)

Formatos de cuantización disponibles (de menor a mayor calidad/tamaño):
  - Q4_K_M : 4-bit cuantización, balance óptimo calidad/tamaño ← RECOMENDADO
  - Q5_K_M : 5-bit, mayor calidad, ~25% más grande que Q4
  - Q8_0   : 8-bit, casi sin pérdida de calidad, 2x más grande que Q4
  - F16    : float16 sin cuantización, máxima calidad (muy grande)

Requisitos:
  pip install huggingface_hub
  apt-get install git cmake build-essential  (para compilar llama.cpp)
==============================================================================
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ==============================================================================

# Directorio donde está el modelo fusionado (salida de finetune_qlora.py)
# Este modelo debe estar en formato HuggingFace estándar (safetensors)
MERGED_MODEL_DIR = "./output/merged-model"

# Directorio donde se guardarán los archivos GGUF cuantizados
GGUF_OUTPUT_DIR = "./output/gguf"

# Nombre del repositorio llama.cpp (se clonará si no existe)
LLAMA_CPP_DIR = "./llama.cpp"

# Nombre base para los archivos GGUF de salida
MODEL_NAME = "api-specialist"

# Tipos de cuantización a generar
# Puedes seleccionar uno o varios; más tipos = más tiempo y espacio en disco
QUANTIZATION_TYPES = [
    "Q4_K_M",   # 4-bit, calidad media-alta → RECOMENDADO para la mayoría de casos
    "Q8_0",     # 8-bit, casi sin pérdida de calidad → Para hardware con más RAM
]


# ==============================================================================
# SECCIÓN 2: FUNCIONES AUXILIARES
# ==============================================================================

def run_command(cmd: list, description: str, cwd: str = None) -> bool:
    """
    Ejecuta un comando del sistema y muestra el progreso.
    
    Args:
        cmd: Lista con el comando y sus argumentos
        description: Descripción legible del paso que se está ejecutando
        cwd: Directorio de trabajo para el comando (None = directorio actual)
    
    Returns:
        True si el comando se ejecutó exitosamente, False en caso de error
    """
    print(f"\n>>> {description}")
    print(f"    Comando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,  # Muestra la salida en tiempo real
            text=True
        )
        print(f"    ✓ Completado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    ✗ Error al ejecutar el comando: {e}")
        return False
    except FileNotFoundError as e:
        print(f"    ✗ Comando no encontrado: {e}")
        print(f"    Asegúrate de tener instaladas las dependencias necesarias.")
        return False


def verificar_dependencias() -> bool:
    """
    Verifica que las herramientas necesarias estén instaladas en el sistema.
    
    Herramientas requeridas:
    - git: para clonar el repositorio llama.cpp
    - python3: para ejecutar los scripts de conversión de llama.cpp
    - cmake y make: para compilar llama.cpp (opcional si solo usas Ollama)
    
    Returns:
        True si todas las dependencias están disponibles
    """
    print("Verificando dependencias del sistema...")
    dependencias = ["git", "python3", "cmake"]
    faltantes = []
    
    for dep in dependencias:
        if shutil.which(dep) is None:
            faltantes.append(dep)
            print(f"  ✗ {dep}: NO encontrado")
        else:
            print(f"  ✓ {dep}: OK")
    
    if faltantes:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(faltantes)}")
        print("   Instálalas con:")
        print("   Ubuntu/Debian: sudo apt-get install git cmake build-essential python3-pip")
        print("   macOS: brew install git cmake python3")
        return False
    
    return True


# ==============================================================================
# SECCIÓN 3: CONFIGURACIÓN DE LLAMA.CPP
# ==============================================================================

def setup_llama_cpp() -> bool:
    """
    Clona y compila el repositorio llama.cpp si no existe.
    
    llama.cpp es la implementación en C++ que permite ejecutar modelos GGUF
    de forma eficiente en CPUs y GPUs con soporte de cuantización nativa.
    
    Opciones de compilación:
    - CPU puro: cmake .. (más compatible, más lento)
    - Con CUDA: cmake .. -DLLAMA_CUDA=on (más rápido en GPU NVIDIA)
    - Con Metal: cmake .. -DLLAMA_METAL=on (para GPU en macOS)
    
    Para el pipeline de cuantización, solo necesitamos los scripts Python,
    no es estrictamente necesario compilar los binarios si usamos Ollama.
    """
    llama_path = Path(LLAMA_CPP_DIR)
    
    if llama_path.exists():
        # Actualiza el repositorio existente
        print(f"\nActualizando llama.cpp en '{LLAMA_CPP_DIR}'...")
        success = run_command(
            ["git", "pull", "origin", "master"],
            "Actualizando llama.cpp",
            cwd=str(llama_path)
        )
    else:
        # Clona el repositorio desde GitHub
        print(f"\nClonando llama.cpp en '{LLAMA_CPP_DIR}'...")
        success = run_command(
            ["git", "clone", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR],
            "Clonando repositorio llama.cpp"
        )
    
    if not success:
        return False
    
    # Instala las dependencias Python de llama.cpp para la conversión
    requirements_file = llama_path / "requirements.txt"
    if requirements_file.exists():
        run_command(
            ["pip", "install", "-r", str(requirements_file), "-q"],
            "Instalando dependencias Python de llama.cpp"
        )
    
    return True


# ==============================================================================
# SECCIÓN 4: CONVERSIÓN A FORMATO GGUF
# ==============================================================================

def convertir_a_gguf_f16(modelo_dir: str, output_dir: str, nombre_modelo: str) -> str:
    """
    Convierte el modelo de formato HuggingFace a GGUF en float16.
    
    Este es el PRIMER paso de cuantización:
    HuggingFace (safetensors) → GGUF F16 (paso intermedio)
    
    El GGUF F16 sirve como base para generar versiones cuantizadas (Q4, Q8, etc.)
    Es posible usar el GGUF F16 directamente, pero su tamaño es grande (~14GB para 7B).
    
    El script convert_hf_to_gguf.py de llama.cpp:
    - Lee los pesos del modelo en formato safetensors/pytorch
    - Reorganiza la arquitectura de pesos al formato GGUF
    - Escribe los metadatos del modelo (vocab, configuración, arquitectura)
    - Produce un único archivo .gguf con todo el modelo
    
    Args:
        modelo_dir: Directorio del modelo fusionado en formato HuggingFace
        output_dir: Directorio donde guardar el GGUF
        nombre_modelo: Nombre base para el archivo de salida
    
    Returns:
        Ruta al archivo GGUF F16 generado
    """
    os.makedirs(output_dir, exist_ok=True)
    gguf_f16_path = os.path.join(output_dir, f"{nombre_modelo}-f16.gguf")
    
    # Busca el script de conversión en el repositorio llama.cpp
    # El script puede estar en la raíz o en el directorio convert/
    conversion_script = None
    posibles_rutas = [
        os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py"),
        os.path.join(LLAMA_CPP_DIR, "convert.py"),
    ]
    
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            conversion_script = ruta
            break
    
    if not conversion_script:
        print("⚠️  Script de conversión no encontrado en llama.cpp")
        print("   Intentando con la ruta estándar más reciente...")
        conversion_script = posibles_rutas[0]
    
    # Ejecuta la conversión HuggingFace → GGUF F16
    success = run_command(
        [
            "python3", conversion_script,
            modelo_dir,                    # Directorio del modelo fuente
            "--outtype", "f16",            # Tipo de salida: float16
            "--outfile", gguf_f16_path,    # Ruta del archivo GGUF de salida
        ],
        f"Convirtiendo modelo HuggingFace a GGUF F16"
    )
    
    if not success:
        print("\n⚠️  La conversión falló. Verifica que el modelo fusionado está completo.")
        print(f"   Directorio del modelo: {modelo_dir}")
        print(f"   Contenido esperado: config.json, tokenizer.json, *.safetensors")
        sys.exit(1)
    
    print(f"   ✓ GGUF F16 generado: {gguf_f16_path}")
    return gguf_f16_path


def cuantizar_gguf(gguf_f16_path: str, output_dir: str, nombre_modelo: str,
                   tipos_cuantizacion: list) -> list:
    """
    Cuantiza el modelo GGUF F16 a formatos de menor precisión (Q4, Q8, etc.).
    
    SEGUNDO paso: GGUF F16 → GGUF Q4_K_M / Q8_0 / etc.
    
    La cuantización reduce los pesos de float16 (16 bits) a enteros de menor
    precisión, con una pérdida mínima de calidad pero con grandes ganancias:
    
    Comparativa de formatos para un modelo de 7B parámetros:
    ┌──────────┬──────────┬─────────────┬─────────────────┐
    │ Formato  │ Bits/param│ Tamaño aprox│ RAM mínima      │
    ├──────────┼──────────┼─────────────┼─────────────────┤
    │ F16      │ 16       │ ~14 GB      │ ~16 GB          │
    │ Q8_0     │ 8.5      │ ~7.7 GB     │ ~10 GB          │
    │ Q4_K_M   │ 4.8      │ ~4.4 GB     │ ~6 GB           │
    │ Q4_0     │ 4.5      │ ~4.1 GB     │ ~6 GB           │
    └──────────┴──────────┴─────────────┴─────────────────┘
    
    La variante "_K_M" (K-Quants, Medium) usa un esquema de cuantización
    más sofisticado que el Q4_0 básico, con mejor calidad para igual tamaño.
    
    Args:
        gguf_f16_path: Ruta al GGUF F16 (archivo intermedio)
        output_dir: Directorio de salida para los GGUF cuantizados
        nombre_modelo: Nombre base para los archivos de salida
        tipos_cuantizacion: Lista de tipos de cuantización a aplicar
    
    Returns:
        Lista de rutas a los archivos GGUF cuantizados generados
    """
    # Busca el binario de cuantización de llama.cpp
    # El binario se llama 'quantize' o 'llama-quantize' según la versión
    quantize_bin = None
    posibles_bins = [
        os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize"),
        os.path.join(LLAMA_CPP_DIR, "build", "bin", "quantize"),
        os.path.join(LLAMA_CPP_DIR, "quantize"),
    ]
    
    for bin_path in posibles_bins:
        if os.path.exists(bin_path):
            quantize_bin = bin_path
            break
    
    if not quantize_bin:
        print("\n⚠️  Binario 'quantize' no encontrado.")
        print("   Compilando llama.cpp para obtener el binario de cuantización...")
        
        # Compila llama.cpp si no está compilado
        build_dir = os.path.join(LLAMA_CPP_DIR, "build")
        os.makedirs(build_dir, exist_ok=True)
        
        run_command(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"],
                   "Configurando compilación de llama.cpp", cwd=build_dir)
        run_command(["make", "-j4", "quantize"],
                   "Compilando binario de cuantización", cwd=build_dir)
        
        # Verifica si la compilación fue exitosa
        for bin_path in posibles_bins:
            if os.path.exists(bin_path):
                quantize_bin = bin_path
                break
        
        if not quantize_bin:
            print("   ✗ No se pudo compilar llama.cpp.")
            print("   Alternativa: Usa 'ollama create' directamente (ver Modelfile).")
            return []
    
    archivos_generados = []
    
    for tipo_q in tipos_cuantizacion:
        # Define la ruta del archivo GGUF cuantizado de salida
        gguf_output = os.path.join(output_dir, f"{nombre_modelo}-{tipo_q.lower()}.gguf")
        
        print(f"\nCuantizando a {tipo_q}...")
        print(f"  Entrada:  {gguf_f16_path}")
        print(f"  Salida:   {gguf_output}")
        
        success = run_command(
            [quantize_bin, gguf_f16_path, gguf_output, tipo_q],
            f"Cuantización {tipo_q}"
        )
        
        if success and os.path.exists(gguf_output):
            tamaño_mb = os.path.getsize(gguf_output) / (1024 * 1024)
            print(f"  ✓ {tipo_q}: {tamaño_mb:.1f} MB")
            archivos_generados.append(gguf_output)
        else:
            print(f"  ✗ Falló la cuantización {tipo_q}")
    
    return archivos_generados


# ==============================================================================
# SECCIÓN 5: CREACIÓN DEL MODELFILE PARA OLLAMA
# ==============================================================================

def crear_modelfile_ollama(gguf_path: str, output_dir: str, nombre_modelo: str) -> str:
    """
    Genera el Modelfile necesario para importar el modelo en Ollama.
    
    Ollama es una herramienta que simplifica la descarga y ejecución de LLMs
    locales. El Modelfile define:
    - FROM: ruta al archivo GGUF del modelo
    - PARAMETER: configuración de inferencia (temperatura, contexto, etc.)
    - SYSTEM: instrucción del sistema por defecto para el modelo
    - TEMPLATE: plantilla de formato de mensajes (si difiere del estándar)
    
    Comandos de uso posterior:
      ollama create api-specialist -f ./Modelfile
      ollama run api-specialist "Crea un pedido para el cliente 12345"
    
    Args:
        gguf_path: Ruta absoluta al archivo GGUF a importar en Ollama
        output_dir: Directorio donde guardar el Modelfile
        nombre_modelo: Nombre que tendrá el modelo en Ollama
    
    Returns:
        Ruta al Modelfile generado
    """
    # Usa ruta absoluta en el Modelfile para evitar problemas de paths relativos
    gguf_abs_path = os.path.abspath(gguf_path)
    
    # Contenido del Modelfile
    # El formato sigue la especificación oficial de Ollama
    modelfile_content = f"""# Modelfile para Ollama - Modelo especializado en APIs de e-commerce
# Generado automáticamente por el pipeline de cuantización
#
# INSTRUCCIONES DE USO:
#   1. Importar el modelo:
#      ollama create {nombre_modelo} -f ./Modelfile
#
#   2. Ejecutar el modelo:
#      ollama run {nombre_modelo}
#
#   3. Consultar via API REST:
#      curl http://localhost:11434/api/generate -d '{{
#        "model": "{nombre_modelo}",
#        "prompt": "Crea un pedido urgente para el cliente 99134"
#      }}'

# Ruta al archivo GGUF del modelo cuantizado
FROM {gguf_abs_path}

# --- Parámetros de generación ---
# temperature: controla la creatividad/aleatoriedad de las respuestas
#   0.0 = determinista (siempre la misma respuesta)
#   0.7 = balance entre coherencia y variedad
#   1.0+ = respuestas más creativas/aleatorias
#   Para APIs/JSON: usar 0.1-0.3 (queremos respuestas precisas y consistentes)
PARAMETER temperature 0.1

# top_p: muestreo por núcleo - considera tokens hasta cubrir este % de probabilidad
#   0.9 = considera tokens que cubren el 90% de probabilidad acumulada
#   Valores más bajos = respuestas más conservadoras
PARAMETER top_p 0.9

# top_k: limita la selección a los K tokens más probables en cada paso
#   40 = valor estándar, buen balance entre calidad y diversidad
PARAMETER top_k 40

# num_ctx: tamaño de la ventana de contexto en tokens
#   2048 = suficiente para la mayoría de llamadas API + historial
#   4096/8192 = para conversaciones largas o documentos extensos
PARAMETER num_ctx 2048

# repeat_penalty: penaliza la repetición de tokens recientes
#   1.0 = sin penalización
#   1.1 = penalización suave (evita frases repetitivas)
PARAMETER repeat_penalty 1.1

# num_predict: máximo de tokens a generar por respuesta
#   512 = suficiente para llamadas API estructuradas en JSON
#   -1 = sin límite (no recomendado para producción)
PARAMETER num_predict 512

# --- Instrucción del sistema ---
# Define el rol y comportamiento por defecto del modelo
# Este SYSTEM prompt se usa cuando no se especifica uno en la petición
SYSTEM \"\"\"Eres un asistente especializado en integración de APIs de e-commerce.
Tu tarea es convertir instrucciones en lenguaje natural en llamadas API estructuradas.
Siempre responde con un objeto JSON válido que incluya: endpoint, method, headers y body.
Nunca añadas explicaciones adicionales fuera del JSON.\"\"\"
"""
    
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    print(f"\n✓ Modelfile creado en: {modelfile_path}")
    return modelfile_path


# ==============================================================================
# SECCIÓN 6: FUNCIÓN PRINCIPAL DE CUANTIZACIÓN
# ==============================================================================

def main():
    """
    Orquesta el pipeline completo de cuantización:
    1. Verifica dependencias y la existencia del modelo fusionado
    2. Configura llama.cpp
    3. Convierte el modelo a GGUF F16
    4. Cuantiza a los formatos INT4/INT8 especificados
    5. Genera el Modelfile para Ollama
    6. Muestra las instrucciones de despliegue final
    """
    
    print("=" * 70)
    print("PIPELINE DE CUANTIZACIÓN: HuggingFace → GGUF (Q4/Q8)")
    print("=" * 70)
    
    # --- Verificación previa: el modelo fusionado debe existir ---
    if not os.path.exists(MERGED_MODEL_DIR):
        print(f"\n✗ ERROR: No se encontró el modelo fusionado en '{MERGED_MODEL_DIR}'")
        print("   Primero ejecuta 'python finetune_qlora.py' para generar el modelo.")
        sys.exit(1)
    
    # Verifica que el directorio contiene los archivos esperados
    archivos_modelo = list(Path(MERGED_MODEL_DIR).glob("*.safetensors")) + \
                      list(Path(MERGED_MODEL_DIR).glob("*.bin"))
    
    if not archivos_modelo:
        print(f"\n⚠️  ADVERTENCIA: No se encontraron pesos del modelo en '{MERGED_MODEL_DIR}'")
        print("   Archivos esperados: *.safetensors o *.bin")
    else:
        print(f"\n✓ Modelo encontrado: {len(archivos_modelo)} archivo(s) de pesos")
        for archivo in archivos_modelo:
            tamaño = os.path.getsize(archivo) / (1024**3)
            print(f"   - {archivo.name}: {tamaño:.2f} GB")
    
    # --- Paso 1: Verificar dependencias del sistema ---
    if not verificar_dependencias():
        print("\n⚠️  Continuando de todas formas (algunas funciones pueden fallar)...")
    
    # --- Paso 2: Configurar llama.cpp ---
    if not setup_llama_cpp():
        print("\n✗ No se pudo configurar llama.cpp. Abortando.")
        sys.exit(1)
    
    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    
    # --- Paso 3: Convertir a GGUF F16 ---
    print(f"\n{'─'*70}")
    print("PASO 1/3: Conversión a GGUF F16 (formato intermedio)")
    print(f"{'─'*70}")
    
    gguf_f16_path = convertir_a_gguf_f16(
        MERGED_MODEL_DIR,
        GGUF_OUTPUT_DIR,
        MODEL_NAME
    )
    
    # --- Paso 4: Cuantizar a formatos comprimidos ---
    print(f"\n{'─'*70}")
    print(f"PASO 2/3: Cuantización a {QUANTIZATION_TYPES}")
    print(f"{'─'*70}")
    
    archivos_cuantizados = cuantizar_gguf(
        gguf_f16_path,
        GGUF_OUTPUT_DIR,
        MODEL_NAME,
        QUANTIZATION_TYPES
    )
    
    # --- Paso 5: Crear Modelfile para Ollama ---
    print(f"\n{'─'*70}")
    print("PASO 3/3: Generando Modelfile para Ollama")
    print(f"{'─'*70}")
    
    # Usa el primer GGUF cuantizado generado (preferiblemente Q4_K_M)
    gguf_para_ollama = archivos_cuantizados[0] if archivos_cuantizados else gguf_f16_path
    
    modelfile_path = crear_modelfile_ollama(
        gguf_para_ollama,
        GGUF_OUTPUT_DIR,
        MODEL_NAME
    )
    
    # --- Resumen final con instrucciones de despliegue ---
    print("\n" + "=" * 70)
    print("CUANTIZACIÓN COMPLETADA - INSTRUCCIONES DE DESPLIEGUE")
    print("=" * 70)
    
    print("\n📦 Archivos generados:")
    for archivo in [gguf_f16_path] + archivos_cuantizados + [modelfile_path]:
        if os.path.exists(archivo):
            tamaño = os.path.getsize(archivo) / (1024**2)
            print(f"   ✓ {archivo} ({tamaño:.0f} MB)")
    
    print("\n🚀 OPCIÓN A: Despliegue con Ollama (MÁS FÁCIL)")
    print("   # Instalar Ollama: https://ollama.ai/download")
    print(f"   ollama create {MODEL_NAME} -f {modelfile_path}")
    print(f"   ollama run {MODEL_NAME}")
    print(f"   # Servidor API automático en: http://localhost:11434")
    
    print("\n🔧 OPCIÓN B: Despliegue directo con llama.cpp")
    if archivos_cuantizados:
        gguf = archivos_cuantizados[0]
        print(f"   # Servidor HTTP local:")
        print(f"   {LLAMA_CPP_DIR}/build/bin/llama-server -m {gguf} --port 8080 -c 2048")
        print(f"   # Inferencia en consola:")
        print(f"   {LLAMA_CPP_DIR}/build/bin/llama-cli -m {gguf} -p 'Tu instrucción aquí'")
    
    print("\n📊 Para usar con el script de inferencia Python:")
    print("   python local_inference.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
