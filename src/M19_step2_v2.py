# -*- coding: utf-8 -*-
"""M19_step2_v2.py - VERSIÓN CORREGIDA CON d_G CONTINUO (PCA + entropía de varianzas)"""

from google.colab import drive
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import time
import threading
from functools import wraps
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes

# =========================================================
# 0. MONTAR DRIVE Y PREPARAR DIRECTORIOS
# =========================================================

if not os.path.exists("/content/drive/MyDrive"):
    drive.mount('/content/drive')

BASE_DRIVE = "/content/drive/MyDrive/M19_V2"
dirs = [
    BASE_DRIVE,
    f"{BASE_DRIVE}/data",
    f"{BASE_DRIVE}/data/instances",
    f"{BASE_DRIVE}/data/binned",
    f"{BASE_DRIVE}/data/landscape"
]

print("="*70)
print("M19 - STEP 2 VERSIÓN CORREGIDA")
print("d_G CONTINUO mediante PCA + entropía de varianzas")
print("="*70)

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"✓ {d}")

# =========================================================
# 1. COPIAR INSTANCIAS LOCALES A DRIVE
# =========================================================

LOCAL_INSTANCES = "M19/data/instances"
DRIVE_INSTANCES = f"{BASE_DRIVE}/data/instances"

if os.path.exists(LOCAL_INSTANCES):
    local_files = list(Path(LOCAL_INSTANCES).glob("*.npz"))
    if local_files:
        print(f"\n📦 Copiando {len(local_files)} instancias locales a Drive...")
        for f in local_files:
            shutil.copy2(f, DRIVE_INSTANCES)
        print("   ✅ Copia completada")

# =========================================================
# 2. VERIFICAR INSTANCIAS
# =========================================================

instances_in_drive = list(Path(DRIVE_INSTANCES).glob("*.npz"))
print(f"\n📊 Instancias en Drive: {len(instances_in_drive)}")

if len(instances_in_drive) == 0:
    print("\n❌ ERROR: No hay instancias en Drive")
    print("   Asegúrate de que M19_generate_instances.py se ejecutó primero")
    exit(1)

# =========================================================
# 3. REORGANIZAR POR BINS
# =========================================================

BINNED_DIR = f"{BASE_DRIVE}/data/binned"
KAPPA_BINS = [(-1, 0), (0, 1), (1, 3), (3, 6), (6, 10), (10, 20), (20, 50)]

for low, high in KAPPA_BINS:
    os.makedirs(os.path.join(BINNED_DIR, f"kappa_{low}_{high}"), exist_ok=True)

def get_kappa_bin(kappa):
    for low, high in KAPPA_BINS:
        if low <= kappa < high:
            return f"kappa_{low}_{high}"
    return None

print("\n📂 Reorganizando por bins...")
for npz_path in instances_in_drive:
    try:
        data = np.load(npz_path)
        kappa = float(data['kappa_J'])
        bin_name = get_kappa_bin(kappa)
        if bin_name:
            dest = os.path.join(BINNED_DIR, bin_name, npz_path.name)
            if not os.path.exists(dest):
                shutil.copy2(npz_path, dest)
    except Exception as e:
        print(f"   ⚠️ Error: {e}")

# =========================================================
# 4. FUNCIÓN PROCRUSTES MANUAL
# =========================================================

def procrustes_alignment(X, Y):
    """
    Alinea Y a X usando transformación de Procrustes ortogonal.
    X: matriz base (n_samples, n_features)
    Y: matriz a alinear (n_samples, n_features)
    Returns: Y alineada
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Calcular transformación ortogonal
    R, _ = orthogonal_procrustes(Y.T, X.T)

    # Aplicar transformación
    Y_aligned = (R @ Y.T).T

    return Y_aligned

# =========================================================
# 5. FUNCIONES DE ENERGÍA Y DINÁMICA
# =========================================================

def energy(s, J):
    return -np.sum(np.triu(J * np.outer(s, s), 1))

def delta_energy(i, s, J):
    return 2 * s[i] * np.dot(J[i], s)

def greedy_descent(s, J):
    N = len(s)
    s_current = s.copy()
    while True:
        flipped = False
        for i in np.random.permutation(N):
            dE = delta_energy(i, s_current, J)
            if dE < 0:
                s_current[i] *= -1
                flipped = True
        if not flipped:
            break
    return s_current

# =========================================================
# 6. NUEVA FUNCIÓN d_G CONTINUO (PCA + entropía de varianzas)
# =========================================================

def compute_dG_continuous(solutions):
    """
    Calcula dimensión efectiva continua usando PCA y entropía de varianzas.
    Esto reemplaza la métrica discreta anterior.

    Returns:
        d_eff: float, dimensión continua (1.0 → ~10)
    """
    if len(solutions) < 2:
        return 1.0

    try:
        # 1. Alinear configuraciones con Procrustes
        base = solutions[0]
        aligned = []

        for sol in solutions:
            # Alinear sol a base
            sol_aligned = procrustes_alignment(base, sol)
            aligned.append(sol_aligned)

        # 2. Flatten embeddings
        features = [X.flatten() for X in aligned]
        X_mat = np.vstack(features)

        # 3. PCA
        pca = PCA()
        pca.fit(X_mat)

        ev = pca.explained_variance_
        ev = ev / (np.sum(ev) + 1e-10)  # Normalizar

        # 4. Dimensión efectiva (entropía de varianzas)
        # d_eff = exp(-sum(p_i * log(p_i)))
        d_eff = np.exp(-np.sum(ev * np.log(ev + 1e-12)))

        # Acotar a rango razonable
        d_eff = min(max(d_eff, 1.0), 50.0)

        return d_eff

    except Exception as e:
        # Fallback: usar número de mínimos como aproximación
        return min(len(solutions), 50.0)

# =========================================================
# 7. EXPLORADOR CON TIMEOUT Y GUARDADO PARCIAL
# =========================================================

NUM_STARTS = 50
TIMEOUT_SECONDS = 600  # 10 minutos por instancia
LANDSCAPE_DIR = f"{BASE_DRIVE}/data/landscape"
CHECKPOINT_FILE = os.path.join(LANDSCAPE_DIR, "checkpoint.json")

class TimeoutExplorer:
    def __init__(self, J, kappa_J, N, num_starts):
        self.J = J
        self.kappa_J = kappa_J
        self.N = N
        self.num_starts = num_starts
        self.unique_minima = {}
        self.completed_starts = 0
        self.is_running = True

    def run(self):
        for start_idx in range(self.num_starts):
            if not self.is_running:
                break
            s0 = np.random.choice([-1, 1], size=self.N)
            s_min = greedy_descent(s0, self.J)
            config_key = tuple(s_min)
            if config_key not in self.unique_minima:
                self.unique_minima[config_key] = s_min.copy()
            self.completed_starts += 1
        return self.get_results()

    def stop(self):
        self.is_running = False

    def get_results(self):
        """Calcula d_G continuo usando PCA sobre las configuraciones encontradas"""
        solutions = list(self.unique_minima.values())

        # Calcular d_G continuo
        dG_continuous = compute_dG_continuous(solutions)

        # Métricas adicionales
        num_unique_minima = len(solutions)

        # Calcular energías para gap
        energies = []
        for s in solutions:
            energies.append(energy(s, self.J))
        energies = sorted(energies)

        if len(energies) > 1:
            energy_gap = energies[1] - energies[0]
        else:
            energy_gap = 0.0

        if energies:
            degeneracy_ratio = sum(abs(E - energies[0]) < 1e-6 for E in energies) / len(energies)
        else:
            degeneracy_ratio = 0.0

        return {
            'dG_continuous': dG_continuous,
            'num_unique_minima': num_unique_minima,
            'energy_gap': energy_gap,
            'degeneracy_ratio': degeneracy_ratio,
            'kappa_J': self.kappa_J,
            'N': self.N,
            'completed_starts': self.completed_starts,
            'total_starts_requested': self.num_starts
        }

def run_with_timeout(explorer, timeout_seconds):
    def target():
        explorer.run()

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        explorer.stop()
        thread.join(2)
        results = explorer.get_results()
        results['_partial'] = True
        results['_timeout_seconds'] = timeout_seconds
        return results

    return explorer.get_results()

# =========================================================
# 8. PROCESAR TODAS LAS INSTANCIAS
# =========================================================

print("\n" + "="*70)
print("🚀 PROCESANDO INSTANCIAS (con d_G continuo)")
print(f"⏱️  Timeout por instancia: {TIMEOUT_SECONDS//60} minutos")
print(f"🎲 Starts por instancia: {NUM_STARTS}")
print("="*70)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_files', []))
    return set()

def save_checkpoint(processed_files):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'processed_files': list(processed_files),
            'last_update': datetime.now().isoformat()
        }, f, indent=2)

# Recopilar instancias
all_instances = []
for bin_name in os.listdir(BINNED_DIR):
    bin_path = os.path.join(BINNED_DIR, bin_name)
    if os.path.isdir(bin_path) and bin_name.startswith("kappa_"):
        for npz_file in Path(bin_path).glob("*.npz"):
            all_instances.append((bin_name, npz_file))

total = len(all_instances)
processed_files = load_checkpoint()
remaining = [inst for inst in all_instances if inst[1].name not in processed_files]

print(f"Total instancias: {total}")
print(f"Ya procesadas: {len(processed_files)}")
print(f"Pendientes: {len(remaining)}")

for idx, (bin_name, npz_path) in enumerate(remaining):
    current_num = len(processed_files) + 1
    print(f"\n[{current_num}/{total}] {bin_name}/{npz_path.name}")

    try:
        data = np.load(npz_path)
        J = data['J']
        kappa_J = float(data['kappa_J'])
        N = int(data['N'])

        print(f"   κ = {kappa_J:.3f} | N = {N}")
        print(f"   Iniciando exploración (timeout {TIMEOUT_SECONDS//60} min)...")

        explorer = TimeoutExplorer(J, kappa_J, N, NUM_STARTS)
        start_time = time.time()
        results = run_with_timeout(explorer, TIMEOUT_SECONDS)
        elapsed = time.time() - start_time

        # Mostrar resultados
        dG_cont = results['dG_continuous']
        num_min = results['num_unique_minima']
        is_partial = results.get('_partial', False)

        if is_partial:
            print(f"   ⏰ TIMEOUT después de {elapsed:.1f}s")
            print(f"      → Resultados PARCIALES ({results['completed_starts']}/{NUM_STARTS} starts)")
            results['status'] = 'partial_timeout'
        else:
            print(f"   ✅ Completado en {elapsed:.1f}s")
            results['status'] = 'complete'

        print(f"      → dG_continuous = {dG_cont:.3f}")
        print(f"      → Mínimos únicos = {num_min}")

        # Guardar resultado
        output_filename = f"landscape_{npz_path.name}"
        output_path = os.path.join(LANDSCAPE_DIR, output_filename)

        results['processed_at'] = datetime.now().isoformat()
        results['elapsed_seconds'] = elapsed

        np.savez_compressed(output_path, **results)

        # Actualizar checkpoint
        processed_files.add(npz_path.name)
        save_checkpoint(processed_files)

    except Exception as e:
        print(f"   ❌ Error: {e}")

# =========================================================
# 9. RESUMEN FINAL
# =========================================================

print("\n" + "="*70)
print("📊 RESUMEN FINAL")
print("="*70)

landscape_files = list(Path(LANDSCAPE_DIR).glob("landscape_*.npz"))
print(f"Total landscape files generados: {len(landscape_files)}")

# Estadísticas rápidas de dG_continuous
dG_values = []
for f in landscape_files:
    try:
        data = np.load(f)
        if 'dG_continuous' in data:
            dG_values.append(float(data['dG_continuous']))
    except:
        pass

if dG_values:
    print(f"\n📈 dG_continuous statistics:")
    print(f"   Min: {np.min(dG_values):.3f}")
    print(f"   Max: {np.max(dG_values):.3f}")
    print(f"   Mean: {np.mean(dG_values):.3f}")
    print(f"   Std: {np.std(dG_values):.3f}")

print(f"\n📁 Resultados en: {LANDSCAPE_DIR}")
print("="*70)
print("✅ M19_step2 V2 COMPLETADO")
print("   d_G ahora es CONTINUO (PCA + entropía de varianzas)")
print("👉 Ejecuta M19_step3 con estos nuevos datos")
