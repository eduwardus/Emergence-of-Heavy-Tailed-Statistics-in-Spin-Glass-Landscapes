# -*- coding: utf-8 -*-
"""M19_step1_generate_instances.py - Generación de instancias de spin glass con control de curtosis"""

from google.colab import drive
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import threading
import random
from IPython.display import display, Javascript

# =========================================================
# 0. ANTI-DESCONEXIÓN Y MONTAR DRIVE
# =========================================================

def start_keep_alive():
    """Mantiene la sesión activa automáticamente"""
    js_code = """
    function ClickConnect(){
        console.log("🔄 " + new Date().toLocaleTimeString() + " - Manteniendo sesión activa...");
        document.querySelector("colab-connect-button")?.click();
        document.querySelector("#top-toolbar > colab-connect-button")?.click();
    }
    setInterval(ClickConnect, 60000);
    """
    display(Javascript(js_code))
    print("✅ Anti-desconexión activado - la sesión no se dormirá")

    def keep_alive():
        while True:
            time.sleep(60)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Sesión activa...", end="\r")
    
    thread = threading.Thread(target=keep_alive, daemon=True)
    thread.start()

start_keep_alive()

# Montar Drive
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount('/content/drive')

BASE_DRIVE = "/content/drive/MyDrive/M19_V3"
INSTANCES_DIR = os.path.join(BASE_DRIVE, "data/instances")
os.makedirs(INSTANCES_DIR, exist_ok=True)

print("="*70)
print("M19 - STEP 1: GENERACIÓN DE INSTANCIAS DE SPIN GLASS")
print("Control explícito de curtosis (κ_J)")
print("="*70)
print(f"Directorio de salida: {INSTANCES_DIR}")
print("="*70)

# =========================================================
# 1. FUNCIONES DE GENERACIÓN DE COUPLINGS
# =========================================================

def generate_gaussian_couplings(N, mu=0, sigma=1, seed=None):
    """J_ij ~ N(mu, sigma)"""
    if seed is not None:
        np.random.seed(seed)
    J = np.random.normal(mu, sigma, (N, N))
    J = np.triu(J, 1) + np.triu(J, 1).T
    return J

def generate_laplace_couplings(N, mu=0, b=1, seed=None):
    """J_ij ~ Laplace(mu, b)"""
    if seed is not None:
        np.random.seed(seed)
    J = np.random.laplace(mu, b, (N, N))
    J = np.triu(J, 1) + np.triu(J, 1).T
    return J

def generate_student_t_couplings(N, df, seed=None):
    """J_ij ~ Student-t con df grados de libertad"""
    if seed is not None:
        np.random.seed(seed)
    J = np.random.standard_t(df, (N, N))
    J = np.triu(J, 1) + np.triu(J, 1).T
    return J

def generate_gaussian_mixture_couplings(N, p=0.9, sigma1=1, sigma2=5, seed=None):
    """Mezcla de dos Gaussianas: con prob p se usa sigma1, con 1-p sigma2"""
    if seed is not None:
        np.random.seed(seed)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if np.random.random() < p:
                J[i,j] = np.random.normal(0, sigma1)
            else:
                J[i,j] = np.random.normal(0, sigma2)
            J[j,i] = J[i,j]
    return J

# =========================================================
# 2. NORMALIZACIÓN Y CÁLCULO DE CURTOSIS
# =========================================================

def normalize_couplings(J):
    """J = (J - mean) / std, garantiza media 0 y varianza 1"""
    mean_J = np.mean(J)
    std_J = np.std(J)
    if std_J < 1e-10:
        return J, mean_J, std_J
    J_norm = (J - mean_J) / std_J
    return J_norm, mean_J, std_J

def compute_kurtosis(J):
    """Excess kurtosis: κ = E[(J-μ)^4]/σ^4 - 3, usando triángulo superior"""
    N = J.shape[0]
    upper_tri = J[np.triu_indices(N, k=1)]
    mu = np.mean(upper_tri)
    sigma = np.std(upper_tri)
    if sigma < 1e-10:
        return 0.0
    kurt = np.mean(((upper_tri - mu) / sigma) ** 4) - 3
    return kurt

# =========================================================
# 3. GENERACIÓN DE UNA INSTANCIA COMPLETA
# =========================================================

def generate_instance(N, dist_type, params, seed):
    """Genera una instancia y devuelve J, kappa_J, metadata"""
    np.random.seed(seed)
    
    if dist_type == "gaussian":
        mu = params.get('mu', 0)
        sigma = params.get('sigma', 1)
        J_raw = generate_gaussian_couplings(N, mu, sigma, seed)
    elif dist_type == "laplace":
        mu = params.get('mu', 0)
        b = params.get('b', 1)
        J_raw = generate_laplace_couplings(N, mu, b, seed)
    elif dist_type == "student_t":
        df = params.get('df', 3)
        J_raw = generate_student_t_couplings(N, df, seed)
    elif dist_type == "gaussian_mixture":
        p = params.get('p', 0.9)
        sigma1 = params.get('sigma1', 1)
        sigma2 = params.get('sigma2', 5)
        J_raw = generate_gaussian_mixture_couplings(N, p, sigma1, sigma2, seed)
    else:
        raise ValueError(f"Distribución no soportada: {dist_type}")
    
    J, mean_J, std_J = normalize_couplings(J_raw)
    kappa_J = compute_kurtosis(J)
    
    metadata = {
        'N': N,
        'dist_type': dist_type,
        'params': params,
        'seed': seed,
        'kappa_J': kappa_J,
        'normalization': {'mean': float(mean_J), 'std': float(std_J)},
        'generation_timestamp': datetime.now().isoformat()
    }
    
    return J, kappa_J, metadata

# =========================================================
# 4. CONFIGURACIÓN DEL EXPERIMENTO
# =========================================================

N = 50
NUM_INSTANCES_PER_DIST = 20   # Por ahora generamos 20 por distribución (luego se pueden añadir más)
TOTAL_INSTANCES = 100         # 5 distribuciones × 20

# Distribuciones a generar (baseline para M19_V3)
DISTRIBUTIONS = [
    ("gaussian", {"mu": 0, "sigma": 1}),
    ("laplace", {"mu": 0, "b": 1}),
    ("student_t", {"df": 3}),
    ("student_t", {"df": 5}),
    ("gaussian_mixture", {"p": 0.9, "sigma1": 1, "sigma2": 5})
]

# Verificar instancias existentes para evitar duplicados
existing_files = set([f.name for f in Path(INSTANCES_DIR).glob("*.npz")])
print(f"\n📊 Instancias existentes en directorio: {len(existing_files)}")

# =========================================================
# 5. GENERACIÓN DE INSTANCIAS (SOLO LAS QUE FALTAN)
# =========================================================

print("\n🚀 GENERANDO INSTANCIAS")
print("-"*50)

new_instances = []
start_time = time.time()

for dist_type, params in DISTRIBUTIONS:
    print(f"\n📦 Distribución: {dist_type} {params}")
    count_generated = 0
    # Intentamos generar hasta NUM_INSTANCES_PER_DIST instancias nuevas
    while count_generated < NUM_INSTANCES_PER_DIST:
        # Semilla única basada en timestamp y contador
        seed = int((time.time() * 1e6) % 1e9) + count_generated * 10000 + random.randint(0, 10000)
        
        try:
            J, kappa, meta = generate_instance(N, dist_type, params, seed)
            rounded_kappa = round(kappa, 2)
            base_name = f"instance_{dist_type}_k{rounded_kappa}_seed{seed}.npz"
            # Asegurar nombre único
            counter = 1
            name = base_name
            while name in existing_files or name in [inst[0] for inst in new_instances]:
                name = f"instance_{dist_type}_k{rounded_kappa}_seed{seed}_v{counter}.npz"
                counter += 1
            
            # Guardar
            filepath = os.path.join(INSTANCES_DIR, name)
            np.savez_compressed(
                filepath,
                J=J,
                kappa_J=kappa,
                dist_type=dist_type,
                params=params,
                N=N,
                seed=seed,
                generation_timestamp=meta['generation_timestamp']
            )
            new_instances.append((name, kappa, dist_type))
            count_generated += 1
            print(f"   ✓ [{count_generated}/{NUM_INSTANCES_PER_DIST}] {name} (κ={kappa:.3f})")
            
        except Exception as e:
            print(f"   ❌ Error generando instancia: {e}")
            continue

# =========================================================
# 6. RESUMEN FINAL
# =========================================================

total_time = time.time() - start_time

print("\n" + "="*70)
print("📊 RESUMEN FINAL")
print("="*70)

print(f"✅ Generadas {len(new_instances)} nuevas instancias")
print(f"   Tiempo total: {total_time:.2f} segundos")
print(f"   Velocidad: {len(new_instances)/total_time:.2f} instancias/segundo")

# Estadísticas de κ
kappa_values = [k for _, k, _ in new_instances]
if kappa_values:
    print(f"\n📈 Estadísticas de κ generadas:")
    print(f"   Min: {np.min(kappa_values):.3f}")
    print(f"   Max: {np.max(kappa_values):.3f}")
    print(f"   Mean: {np.mean(kappa_values):.3f}")
    print(f"   Std: {np.std(kappa_values):.3f}")

total_instances = len(list(Path(INSTANCES_DIR).glob("*.npz")))
print(f"\n📁 Total instancias en directorio: {total_instances}")

# Guardar resumen en JSON
summary = {
    "experiment": "M19_step1_generate_instances",
    "timestamp": datetime.now().isoformat(),
    "N": N,
    "num_instances_per_dist": NUM_INSTANCES_PER_DIST,
    "total_new_instances": len(new_instances),
    "total_instances_now": total_instances,
    "generation_time_seconds": total_time,
    "kappa_stats": {
        "min": float(np.min(kappa_values)) if kappa_values else None,
        "max": float(np.max(kappa_values)) if kappa_values else None,
        "mean": float(np.mean(kappa_values)) if kappa_values else None,
        "std": float(np.std(kappa_values)) if kappa_values else None
    },
    "new_instances": [
        {"name": name, "kappa": k, "dist_type": d}
        for name, k, d in new_instances
    ]
}

summary_path = os.path.join(INSTANCES_DIR, "generation_summary_step1.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n📄 Resumen guardado: {summary_path}")
print("="*70)
print("✅ GENERACIÓN COMPLETADA")
print("="*70)
