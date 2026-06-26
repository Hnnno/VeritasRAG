"""
==============================================================================
Script de Inferencia Local: Uso del Modelo Ajustado en Entornos de Bajo Recurso
==============================================================================
Soporte para dos backends de inferencia:
  A) Ollama  → más fácil, gestión automática de GPU/CPU, API REST integrada
  B) llama.cpp (via Python bindings) → más control, sin servidor externo

Requisitos mínimos de hardware para modelo 7B cuantizado a Q4_K_M:
  - RAM: 6-8 GB disponibles
  - CPU: Cualquier procesador moderno x86_64 o ARM (ej: Apple M1/M2)
  - GPU: Opcional pero mejora significativamente la velocidad
    → NVIDIA: ≥4GB VRAM (ej: GTX 1660, RTX 3060)
    → AMD: ≥6GB VRAM con ROCm
    → Apple Silicon: comparte RAM entre CPU/GPU

Velocidades de inferencia aproximadas (tokens/segundo):
  ┌─────────────────────┬──────────────┐
  │ Hardware            │ Tokens/seg   │
  ├─────────────────────┼──────────────┤
  │ CPU solo (8 cores)  │ 3-8 t/s      │
  │ GTX 1660 (6GB)     │ 15-30 t/s    │
  │ RTX 3060 (12GB)    │ 30-60 t/s    │
  │ Apple M2            │ 20-40 t/s    │
  └─────────────────────┴──────────────┘

Instalación de dependencias:
  # Para Ollama (backend A):
  # 1. Descargar desde https://ollama.ai/download e instalar
  # 2. ollama create api-specialist -f ./output/gguf/Modelfile
  # 3. ollama serve (inicia el servidor en background)

  # Para llama-cpp-python (backend B):
  # CPU puro:
  pip install llama-cpp-python
  # Con soporte CUDA (GPU NVIDIA):
  CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall
  # Con soporte Metal (GPU Apple):
  CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
==============================================================================
"""

import json
import time
import sys
from typing import Optional, Generator


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN DE LA INFERENCIA LOCAL
# ==============================================================================

# --- Selección del backend de inferencia ---
# "ollama"    → usa el servidor Ollama local (debe estar ejecutándose)
# "llama_cpp" → usa directamente la librería llama-cpp-python (sin servidor)
BACKEND = "ollama"

# --- Configuración del backend Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"   # URL del servidor Ollama
OLLAMA_MODEL_NAME = "api-specialist"         # Nombre del modelo importado en Ollama

# --- Configuración del backend llama.cpp directo ---
# Ruta al archivo GGUF cuantizado (salida del script de cuantización)
GGUF_MODEL_PATH = "./output/gguf/api-specialist-q4_k_m.gguf"

# --- Parámetros de generación comunes para ambos backends ---
# Estos parámetros controlan el comportamiento de la generación de texto

# Número de capas del modelo a cargar en GPU (para llama.cpp)
#   -1 = todas las capas en GPU (máxima velocidad, requiere más VRAM)
#    0 = solo CPU (más lento pero funciona sin GPU)
#   N  = N capas en GPU (balance entre velocidad y VRAM disponible)
N_GPU_LAYERS = -1  # Cambiar a 0 si no tienes GPU compatible

# Número de hilos CPU para la inferencia (llama.cpp)
#   Recomendado: número de núcleos físicos (no lógicos/hyperthreading)
N_THREADS = 8

# Tamaño de la ventana de contexto en tokens
#   Mayor contexto = más RAM, pero permite conversaciones más largas
CONTEXT_SIZE = 2048

# Instrucción del sistema que define el dominio del asistente
SYSTEM_PROMPT = """Eres un asistente especializado en integración de APIs de e-commerce.
Tu tarea es convertir instrucciones en lenguaje natural en llamadas API estructuradas.
Siempre responde con un objeto JSON válido que incluya: endpoint, method, headers y body.
Nunca añadas explicaciones adicionales fuera del JSON."""


# ==============================================================================
# SECCIÓN 2: CLIENTE PARA BACKEND OLLAMA
# ==============================================================================

class OllamaClient:
    """
    Cliente para interactuar con el servidor Ollama local.
    
    Ollama expone una API REST compatible con OpenAI en localhost:11434.
    Ventajas:
    - Gestión automática de GPU/CPU
    - Soporte para múltiples modelos cargados simultáneamente
    - API REST fácil de integrar con aplicaciones web
    - Historial de conversación integrado
    
    Endpoints principales:
    - POST /api/generate  → generación de texto (completion)
    - POST /api/chat      → chat con historial de mensajes
    - GET  /api/tags      → lista de modelos disponibles
    - POST /api/pull      → descargar modelos del registro de Ollama
    """
    
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL_NAME):
        """
        Inicializa el cliente Ollama.
        
        Args:
            base_url: URL base del servidor Ollama (por defecto localhost:11434)
            model: Nombre del modelo a usar (debe estar importado en Ollama)
        """
        # Importación tardía: solo se importa 'requests' si usamos Ollama
        try:
            import requests
            self.requests = requests
        except ImportError:
            print("✗ Librería 'requests' no encontrada.")
            print("  Instálala con: pip install requests")
            sys.exit(1)
        
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session = self.requests.Session()
        
        # Verifica la conexión con el servidor
        self._verificar_servidor()
    
    def _verificar_servidor(self):
        """
        Verifica que el servidor Ollama está ejecutándose y el modelo está disponible.
        Imprime información útil de diagnóstico si hay problemas.
        """
        try:
            # Consulta los modelos disponibles
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            modelos = response.json().get("models", [])
            nombres_modelos = [m["name"] for m in modelos]
            
            print(f"✓ Servidor Ollama activo en {self.base_url}")
            print(f"  Modelos disponibles: {', '.join(nombres_modelos) or 'ninguno'}")
            
            # Advierte si el modelo solicitado no está disponible
            if not any(self.model in nombre for nombre in nombres_modelos):
                print(f"\n⚠️  El modelo '{self.model}' no está disponible en Ollama.")
                print(f"   Para importarlo, ejecuta:")
                print(f"   ollama create {self.model} -f ./output/gguf/Modelfile")
                
        except self.requests.exceptions.ConnectionError:
            print(f"✗ No se puede conectar al servidor Ollama en {self.base_url}")
            print("  Soluciones:")
            print("  1. Instalar Ollama: https://ollama.ai/download")
            print("  2. Iniciar el servidor: ollama serve")
            print("  3. Verificar que el puerto 11434 no está bloqueado")
            sys.exit(1)
    
    def generar(
        self,
        instruccion: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        """
        Genera una respuesta del modelo dado un prompt de instrucción.
        
        Usa el endpoint /api/chat con formato de mensajes para mantener
        el system prompt separado del mensaje del usuario.
        
        Args:
            instruccion: Instrucción del usuario en lenguaje natural
            temperature: Temperatura de muestreo (0.0-1.0)
            max_tokens: Número máximo de tokens a generar
            stream: Si True, transmite la respuesta token por token
        
        Returns:
            Respuesta completa del modelo como string
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruccion}
            ],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
            }
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,  # 2 minutos de timeout para respuestas largas
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                # Modo streaming: imprime tokens conforme se generan
                texto_completo = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            texto = chunk["message"].get("content", "")
                            print(texto, end="", flush=True)
                            texto_completo += texto
                        if chunk.get("done", False):
                            print()  # Nueva línea al terminar
                return texto_completo
            else:
                # Modo estándar: espera la respuesta completa
                data = response.json()
                return data["message"]["content"]
                
        except self.requests.exceptions.Timeout:
            print("✗ Timeout: La generación tardó más de 2 minutos.")
            print("  Considera reducir max_tokens o usar un modelo más pequeño.")
            return ""
        except self.requests.exceptions.RequestException as e:
            print(f"✗ Error en la petición a Ollama: {e}")
            return ""


# ==============================================================================
# SECCIÓN 3: CLIENTE PARA BACKEND LLAMA.CPP (SIN SERVIDOR)
# ==============================================================================

class LlamaCppClient:
    """
    Cliente para inferencia directa con llama-cpp-python (sin servidor).
    
    llama-cpp-python es un binding Python de llama.cpp que permite:
    - Cargar y ejecutar modelos GGUF directamente en Python
    - Sin necesidad de servidor externo
    - Control preciso sobre el hardware (GPU layers, threads, etc.)
    - Integración fácil en aplicaciones Python existentes
    
    Ventajas vs Ollama:
    - No requiere proceso de servidor separado
    - Más control sobre parámetros de bajo nivel
    - Mejor para integración en aplicaciones Python
    
    Desventajas vs Ollama:
    - Más difícil de instalar (compilación C++)
    - No tiene API REST lista para usar
    - Requiere cargar el modelo en cada proceso
    """
    
    def __init__(
        self,
        model_path: str = GGUF_MODEL_PATH,
        n_gpu_layers: int = N_GPU_LAYERS,
        n_threads: int = N_THREADS,
        n_ctx: int = CONTEXT_SIZE,
    ):
        """
        Inicializa el cliente llama.cpp cargando el modelo en memoria.
        
        Args:
            model_path: Ruta al archivo GGUF del modelo cuantizado
            n_gpu_layers: Capas a cargar en GPU (-1=todas, 0=solo CPU)
            n_threads: Hilos CPU para la inferencia
            n_ctx: Tamaño de la ventana de contexto en tokens
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            print("✗ Librería 'llama-cpp-python' no encontrada.")
            print("  Instalación para CPU:  pip install llama-cpp-python")
            print("  Con GPU NVIDIA:        CMAKE_ARGS='-DLLAMA_CUDA=on' pip install llama-cpp-python")
            print("  Con GPU Apple:         CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python")
            sys.exit(1)
        
        import os
        if not os.path.exists(model_path):
            print(f"✗ Modelo GGUF no encontrado: {model_path}")
            print("  Ejecuta primero 'python quantize_to_gguf.py' para generar el GGUF.")
            sys.exit(1)
        
        print(f"Cargando modelo GGUF desde: {model_path}")
        print(f"  Capas en GPU: {n_gpu_layers} (-1=todas)")
        print(f"  Hilos CPU: {n_threads}")
        print(f"  Contexto: {n_ctx} tokens")
        print("  (La primera carga puede tardar 10-30 segundos...)")
        
        # Carga el modelo GGUF en memoria
        # n_gpu_layers controla cuántas capas del transformer se cargan en VRAM
        # Más capas en GPU = más rápido pero requiere más VRAM
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,  # Capas en GPU
            n_threads=n_threads,         # Hilos de CPU para capas en CPU
            n_ctx=n_ctx,                 # Tamaño del contexto (ventana de atención)
            n_batch=512,                 # Tokens procesados en paralelo en el prompt
            verbose=False,               # True para logs detallados de llama.cpp
            chat_format="chatml",        # Formato de chat: ChatML (para Qwen/Llama 3)
                                         # Alternativas: "llama-2", "mistral-instruct"
        )
        
        print("✓ Modelo cargado exitosamente")
    
    def generar(
        self,
        instruccion: str,
        temperature: float = 0.1,
        max_tokens: int = 512,
        stream: bool = False,
    ) -> str:
        """
        Genera una respuesta usando la interfaz de chat de llama.cpp.
        
        Args:
            instruccion: Instrucción del usuario
            temperature: Temperatura de muestreo (0.0 determinista, 1.0 creativo)
            max_tokens: Máximo de tokens a generar en la respuesta
            stream: Transmitir tokens conforme se generan (para UX más fluida)
        
        Returns:
            Texto de respuesta generado por el modelo
        """
        # Formatea los mensajes en el formato de chat estándar
        mensajes = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruccion}
        ]
        
        # Genera la respuesta usando la interfaz de chat
        # create_chat_completion aplica automáticamente la plantilla ChatML
        respuesta = self.llm.create_chat_completion(
            messages=mensajes,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,           # Top-p sampling (nucleus sampling)
            top_k=40,             # Top-k sampling
            repeat_penalty=1.1,   # Penalización por repetición de tokens
            stream=stream,
        )
        
        if stream:
            # Modo streaming: itera sobre chunks y los imprime en tiempo real
            texto_completo = ""
            for chunk in respuesta:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    texto = delta["content"]
                    print(texto, end="", flush=True)
                    texto_completo += texto
            print()  # Nueva línea al terminar el streaming
            return texto_completo
        else:
            # Modo estándar: devuelve el texto de la primera opción
            return respuesta["choices"][0]["message"]["content"]


# ==============================================================================
# SECCIÓN 4: UTILIDADES DE EVALUACIÓN Y FORMATEO
# ==============================================================================

def formatear_respuesta_json(respuesta: str) -> dict:
    """
    Intenta parsear la respuesta del modelo como JSON válido.
    
    El modelo está entrenado para responder solo con JSON, pero puede
    incluir texto adicional o tener formato incorrecto en algunos casos.
    Esta función extrae el JSON de la respuesta de forma robusta.
    
    Args:
        respuesta: Texto crudo de la respuesta del modelo
    
    Returns:
        Diccionario Python con el JSON parseado, o dict con error si falla
    """
    # Estrategia 1: Parsear directamente como JSON
    try:
        return json.loads(respuesta.strip())
    except json.JSONDecodeError:
        pass
    
    # Estrategia 2: Buscar el bloque JSON entre llaves
    # El modelo podría incluir texto antes/después del JSON
    inicio = respuesta.find("{")
    fin = respuesta.rfind("}") + 1
    
    if inicio != -1 and fin > inicio:
        try:
            return json.loads(respuesta[inicio:fin])
        except json.JSONDecodeError:
            pass
    
    # Estrategia 3: Buscar bloque de código JSON (```json ... ```)
    if "```json" in respuesta:
        try:
            inicio_bloque = respuesta.find("```json") + 7
            fin_bloque = respuesta.find("```", inicio_bloque)
            if fin_bloque > inicio_bloque:
                return json.loads(respuesta[inicio_bloque:fin_bloque].strip())
        except json.JSONDecodeError:
            pass
    
    # Si ninguna estrategia funciona, retorna la respuesta cruda
    return {"error": "JSON inválido", "respuesta_cruda": respuesta}


def evaluar_calidad_respuesta(respuesta_json: dict) -> dict:
    """
    Evalúa si la respuesta del modelo cumple con el formato esperado de la API.
    
    Verifica que el JSON de respuesta contiene los campos obligatorios
    definidos por el contrato de la API: endpoint, method, headers y body.
    
    Args:
        respuesta_json: Respuesta del modelo como diccionario Python
    
    Returns:
        Diccionario con el resultado de la evaluación y los campos faltantes
    """
    campos_obligatorios = ["endpoint", "method", "headers", "body"]
    campos_faltantes = []
    
    # Verifica la presencia de cada campo obligatorio
    for campo in campos_obligatorios:
        if campo not in respuesta_json:
            campos_faltantes.append(campo)
    
    # Valida el método HTTP (debe ser un verbo estándar)
    metodos_validos = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    metodo = respuesta_json.get("method", "").upper()
    metodo_valido = metodo in metodos_validos
    
    # Verifica que el endpoint comienza con /api/
    endpoint = respuesta_json.get("endpoint", "")
    endpoint_valido = endpoint.startswith("/api/")
    
    es_valido = (
        len(campos_faltantes) == 0 and
        metodo_valido and
        endpoint_valido
    )
    
    return {
        "valido": es_valido,
        "campos_faltantes": campos_faltantes,
        "metodo_valido": metodo_valido,
        "endpoint_valido": endpoint_valido,
        "puntaje": (
            (len(campos_obligatorios) - len(campos_faltantes)) / len(campos_obligatorios)
            * (1 if metodo_valido else 0.8)
            * (1 if endpoint_valido else 0.8)
        )
    }


# ==============================================================================
# SECCIÓN 5: FUNCIÓN PRINCIPAL DE DEMOSTRACIÓN
# ==============================================================================

def demo_inferencia():
    """
    Ejecuta una demostración de inferencia local con el modelo ajustado.
    
    Prueba el modelo con varios casos de uso que representan diferentes
    tipos de operaciones de la API de e-commerce.
    
    La demostración mide:
    - Tiempo de respuesta por consulta
    - Validez del JSON generado
    - Cumplimiento del contrato de la API (campos obligatorios)
    """
    
    print("=" * 70)
    print("DEMO DE INFERENCIA LOCAL - MODELO AJUSTADO EN DOMINIO")
    print(f"Backend: {BACKEND.upper()}")
    print("=" * 70)
    
    # Inicializa el cliente según el backend seleccionado
    print("\nInicializando cliente de inferencia...")
    
    if BACKEND == "ollama":
        cliente = OllamaClient(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL_NAME
        )
    elif BACKEND == "llama_cpp":
        cliente = LlamaCppClient(
            model_path=GGUF_MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_threads=N_THREADS,
            n_ctx=CONTEXT_SIZE
        )
    else:
        print(f"✗ Backend desconocido: '{BACKEND}'")
        print("  Opciones válidas: 'ollama' o 'llama_cpp'")
        sys.exit(1)
    
    # Casos de prueba que cubren diferentes operaciones de la API
    # Algunos son similares a los datos de entrenamiento (para verificar aprendizaje)
    # Otros son nuevos (para evaluar generalización)
    casos_de_prueba = [
        {
            "descripcion": "Consulta de inventario (similar al entrenamiento)",
            "instruccion": "¿Cuántas unidades del producto SKU-7744 tenemos en el almacén de Sevilla?",
            "tipo": "conocido"
        },
        {
            "descripcion": "Creación de pedido estándar",
            "instruccion": "Genera un pedido para el cliente 45678 con 2 unidades de SKU-0011 y 5 de SKU-0022. Envío estándar a la dirección registrada.",
            "tipo": "conocido"
        },
        {
            "descripcion": "Actualización de stock de emergencia (nuevo caso)",
            "instruccion": "Por una incidencia en el proveedor, necesito bloquear temporalmente todo el stock del producto SKU-5500 en todos los almacenes. Marca los 340 artículos como en revisión de calidad.",
            "tipo": "nuevo"
        },
        {
            "descripcion": "Campaña flash de 24 horas (nuevo caso)",
            "instruccion": "Crea una promoción flash de 35% de descuento para los 10 productos más vendidos de la semana. Duración: exactamente 24 horas desde ahora. Solo para clientes con más de 2 compras anteriores.",
            "tipo": "nuevo"
        },
        {
            "descripcion": "Consulta multi-almacén (caso complejo nuevo)",
            "instruccion": "Necesito un informe del stock total de SKU-1234, SKU-5678 y SKU-9012 sumando los 5 almacenes de la península ibérica. Incluye el stock reservado y el umbral de reposición.",
            "tipo": "complejo"
        },
    ]
    
    # Almacena los resultados para el resumen final
    resultados = []
    
    print(f"\n{'─' * 70}")
    print("EJECUTANDO CASOS DE PRUEBA")
    print(f"{'─' * 70}")
    
    for i, caso in enumerate(casos_de_prueba, 1):
        print(f"\n[{i}/{len(casos_de_prueba)}] {caso['descripcion']} [{caso['tipo'].upper()}]")
        print(f"  Instrucción: {caso['instruccion'][:80]}{'...' if len(caso['instruccion']) > 80 else ''}")
        print(f"  Respuesta:")
        
        # Mide el tiempo de generación
        inicio = time.time()
        
        respuesta_cruda = cliente.generar(
            instruccion=caso["instruccion"],
            temperature=0.1,    # Temperatura baja para respuestas consistentes en JSON
            max_tokens=512,
            stream=False
        )
        
        tiempo_generacion = time.time() - inicio
        
        # Procesa y evalúa la respuesta
        respuesta_json = formatear_respuesta_json(respuesta_cruda)
        evaluacion = evaluar_calidad_respuesta(respuesta_json)
        
        # Muestra la respuesta formateada
        if "error" not in respuesta_json:
            print(json.dumps(respuesta_json, ensure_ascii=False, indent=4))
        else:
            print(f"  [RESPUESTA CRUDA]: {respuesta_cruda}")
        
        # Muestra métricas de calidad
        estado = "✓ VÁLIDO" if evaluacion["valido"] else "⚠ PARCIAL"
        print(f"\n  {estado} | Tiempo: {tiempo_generacion:.2f}s | Puntaje: {evaluacion['puntaje']:.0%}")
        
        if evaluacion["campos_faltantes"]:
            print(f"  Campos faltantes: {evaluacion['campos_faltantes']}")
        
        resultados.append({
            "caso": caso["descripcion"],
            "tipo": caso["tipo"],
            "valido": evaluacion["valido"],
            "puntaje": evaluacion["puntaje"],
            "tiempo_seg": tiempo_generacion
        })
    
    # --- Resumen estadístico de la evaluación ---
    print(f"\n{'═' * 70}")
    print("RESUMEN DE LA EVALUACIÓN")
    print(f"{'═' * 70}")
    
    casos_validos = sum(1 for r in resultados if r["valido"])
    puntaje_promedio = sum(r["puntaje"] for r in resultados) / len(resultados)
    tiempo_promedio = sum(r["tiempo_seg"] for r in resultados) / len(resultados)
    
    print(f"\n  Casos válidos:     {casos_validos}/{len(resultados)} ({casos_validos/len(resultados):.0%})")
    print(f"  Puntaje promedio:  {puntaje_promedio:.0%}")
    print(f"  Tiempo promedio:   {tiempo_promedio:.2f}s por consulta")
    
    print(f"\n  Detalle por caso:")
    for r in resultados:
        estado = "✓" if r["valido"] else "⚠"
        print(f"    {estado} [{r['tipo']:8}] {r['caso'][:45]:45} {r['puntaje']:.0%} en {r['tiempo_seg']:.1f}s")
    
    print(f"\n{'═' * 70}")
    
    # Recomendaciones basadas en los resultados
    if puntaje_promedio >= 0.9:
        print("🎉 EXCELENTE: El modelo maneja correctamente el dominio de la API.")
        print("   Considera desplegarlo en producción.")
    elif puntaje_promedio >= 0.7:
        print("✅ BUENO: El modelo funciona bien. Algunas mejoras posibles:")
        print("   - Añadir más ejemplos de entrenamiento para casos complejos")
        print("   - Ajustar el system prompt para casos específicos que fallen")
    else:
        print("⚠️  MEJORABLE: El modelo necesita más ajuste.")
        print("   Recomendaciones:")
        print("   - Aumentar el dataset de entrenamiento (mínimo 200-500 ejemplos)")
        print("   - Aumentar num_train_epochs en finetune_qlora.py")
        print("   - Revisar si el formato del sistema prompt coincide con el modelo base")
    
    print("=" * 70)


# ==============================================================================
# SECCIÓN 6: MODO INTERACTIVO (CHAT EN CONSOLA)
# ==============================================================================

def modo_interactivo():
    """
    Inicia un chat interactivo en la consola para probar el modelo manualmente.
    
    Permite introducir instrucciones en lenguaje natural y ver las llamadas
    API generadas en tiempo real. Útil para demos y pruebas rápidas.
    
    Comandos especiales:
      'salir' o 'exit' → termina el chat
      'limpiar'        → limpia la pantalla
      'ayuda'          → muestra ejemplos de instrucciones
    """
    print("=" * 70)
    print("CHAT INTERACTIVO - Modelo especializado en APIs de e-commerce")
    print(f"Backend: {BACKEND.upper()} | Escribe 'salir' para terminar")
    print("=" * 70)
    
    print("\nEjemplos de instrucciones:")
    print("  • 'Consulta el stock del producto SKU-1234 en Madrid'")
    print("  • 'Crea un pedido urgente para el cliente 9999 con 5 unidades de SKU-0001'")
    print("  • 'Aplica un 10% de descuento a todos los productos de la categoría libros'")
    print()
    
    # Inicializa el cliente (mismo que en la demo)
    if BACKEND == "ollama":
        cliente = OllamaClient()
    else:
        cliente = LlamaCppClient()
    
    while True:
        try:
            instruccion = input("\n📝 Tu instrucción: ").strip()
            
            if not instruccion:
                continue
            
            if instruccion.lower() in ("salir", "exit", "quit"):
                print("¡Hasta luego!")
                break
            
            if instruccion.lower() == "ayuda":
                print("\nEjemplos de instrucciones válidas:")
                print("  1. Cancela el pedido ORD-55123 y reembolsa al método de pago original")
                print("  2. ¿Cuál es el estado del envío ES-TRK-2024-9999?")
                print("  3. Busca productos de electrónica entre 100 y 500 EUR con más de 50 reseñas")
                continue
            
            print("\n🤖 Llamada API generada:")
            
            inicio = time.time()
            respuesta = cliente.generar(instruccion, stream=True)
            tiempo = time.time() - inicio
            
            # Intenta formatear como JSON para mejor legibilidad
            respuesta_json = formatear_respuesta_json(respuesta)
            if "error" not in respuesta_json and not respuesta.strip().startswith("{"):
                # Solo re-imprime si el streaming no mostró JSON formateado
                print(json.dumps(respuesta_json, ensure_ascii=False, indent=2))
            
            print(f"\n⏱️  Generado en {tiempo:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\nInterrumpido por el usuario. ¡Hasta luego!")
            break


# ==============================================================================
# PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inferencia local con modelo ajustado - API e-commerce specialist"
    )
    parser.add_argument(
        "--modo",
        choices=["demo", "interactivo"],
        default="demo",
        help="Modo de ejecución: 'demo' (casos de prueba automáticos) o 'interactivo' (chat)"
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "llama_cpp"],
        default=BACKEND,
        help="Backend de inferencia: 'ollama' (servidor) o 'llama_cpp' (directo)"
    )
    parser.add_argument(
        "--modelo_gguf",
        type=str,
        default=GGUF_MODEL_PATH,
        help="Ruta al archivo GGUF (solo para backend llama_cpp)"
    )
    parser.add_argument(
        "--ollama_modelo",
        type=str,
        default=OLLAMA_MODEL_NAME,
        help="Nombre del modelo en Ollama (solo para backend ollama)"
    )
    parser.add_argument(
        "--gpu_layers",
        type=int,
        default=N_GPU_LAYERS,
        help="Capas en GPU para llama_cpp: -1=todas, 0=solo CPU"
    )
    
    args = parser.parse_args()
    
    # Aplica los argumentos a las variables globales
    BACKEND = args.backend
    GGUF_MODEL_PATH = args.modelo_gguf
    OLLAMA_MODEL_NAME = args.ollama_modelo
    N_GPU_LAYERS = args.gpu_layers
    
    print(f"\nConfiguración de inferencia:")
    print(f"  Backend:  {BACKEND}")
    if BACKEND == "ollama":
        print(f"  Modelo:   {OLLAMA_MODEL_NAME}")
        print(f"  URL:      {OLLAMA_BASE_URL}")
    else:
        print(f"  GGUF:     {GGUF_MODEL_PATH}")
        print(f"  GPU capas: {N_GPU_LAYERS} (-1=todas)")
        print(f"  Hilos CPU: {N_THREADS}")
    
    # Ejecuta el modo seleccionado
    if args.modo == "demo":
        demo_inferencia()
    else:
        modo_interactivo()
