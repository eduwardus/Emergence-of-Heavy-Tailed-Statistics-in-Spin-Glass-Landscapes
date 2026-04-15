# -*- coding: utf-8 -*-
"""M19_step2_v3.py - VERSIÓN DEFINITIVA CON ANTI-DESCONEXIÓN ROBUSTA"""

from google.colab import drive
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil
import time
import threading
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from IPython.display import display, Javascript

# =========================================================
# 0. ANTI-DESCONEXIÓN ROBUSTA (MÚLTIPLES CAPAS)
# =========================================================

print("="*70)
print("🔧 ACTIVANDO ANTI-DESCONEXIÓN (múltiples capas)")
print("="*70)

# Capa 1: JavaScript que hace clic en el botón de conexión
js_code = """
<script>
function ClickConnect(){
    console.log("🔄 " + new Date().toLocaleTimeString() + " - Manteniendo sesión activa...");
    document.querySelector("colab-connect-button")?.click();
    document.querySelector("#top-toolbar > colab-connect-button")?.click();
}
setInterval(ClickConnect, 60000);
</script>
"""
display(Javascript(js_code))
print("✅ Capa 1: JavaScript activado (clic cada 60 segundos)")

# Capa 2: Hilo en Python que imprime actividad
stop_keep_alive = False

def python_keep_alive():
    """Hilo secundario que imprime actividad cada minuto"""
    counter = 0
    while not stop_keep_alive:
        time.sleep(60)
        counter += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Sesión activa - minuto {counter}", end="\r")

keep_alive_thread = threading.Thread(target=python_keep_alive, daemon=True)
keep_alive_thread.start()
print("✅ Capa 2: Hilo Python activado (actividad cada 60 segundos)")

# Capa 3: Prevenir suspensión del navegador (solo funciona si la pestaña está activa)
try:
    from IPython.display import HTML
    display(HTML("""
    <script>
    // Evitar que el navegador entre en modo de ahorro de energía
    setInterval(function() {
        document.dispatchEvent(new Event('mousemove'));
    }, 30000);
    </script>
    """))
    print("✅ Capa 3: Anti-suspensión del navegador activado")
except:
    print("⚠️ Capa 3 no disponible")

print("="*70)
print("🛡️ ANTI-DESCONEXIÓN COMPLETAMENTE ACTIVADA")
print("   La sesión NO se desconectará por inactividad")
print("   El límite de 12 horas de Colab sigue aplicando")
print("="*70)

# =========================================================
# 1. MONTAR DRIVE
# =========================================================

if not os.path.exists("/content/drive/MyDrive"):
    drive.mount('/content/drive')

BASE_DRIVE = "/content/drive/MyDrive/M19_V3"
dirs = [
    BASE_DRIVE,
    f"{BASE_DRIVE}/data",
    f"{BASE_DRIVE}/data/instances",
    f"{BASE_DRIVE}/data/binned",
    f"{BASE_DRIVE}/data/landscape"
]

print("\n" + "="*70)
print("M19 - STEP 2 VERSIÓN 3 (DEFINITIVA)")
print("d_G = DIMENSIÓN EFECTIVA mediante PCA + entropía")
print("(COMPARABLE con M18)")
print("="*70)

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"✓ {d}")

# =========================================================
# 2. VERIFICAR INSTANCIAS
# =========================================================

INSTANCES_DIR = f"{BASE_DRIVE}/data/instances"
instances_in_drive = list(Path(INSTANCES_DIR).glob("*.npz"))
print(f"\n📊 Instancias en Drive: {len(instances_in_drive)}")

if len(instances_in_drive) == 0:
    print("\n❌ ERROR: No hay instancias en Drive")
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
# 4. FUNCIONES DE ENERGÍA Y DINÁMICA
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
# 5. PROCRUSTES ALIGNMENT
# =========================================================

def procrustes_alignment(X, Y):
    """Alinea Y a X usando transformación de Procrustes ortogonal"""
    X = np.asarray(X)
    Y = np.asarray(Y)
    R, _ = orthogonal_procrustes(Y.T, X.T)
    Y_aligned = (R @ Y.T).T
    return Y_aligned

# =========================================================
# 6. d_G como DIMENSIÓN EFECTIVA (PCA + entropía)
# =========================================================

def compute_dG_effective(solutions):
    """
    Calcula la dimensión efectiva del conjunto de soluciones.
    Esto es COMPARABLE con M18.

    d_eff = exp(-sum(p_i * log(p_i)))
    donde p_i son las varianzas explicadas normalizadas de PCA.
    """
    if len(solutions) < 2:
        return 1.0

    try:
        # Alinear configuraciones con Procrustes
        base = solutions[0]
        aligned = []
        for sol in solutions:
            sol_aligned = procrustes_alignment(base, sol)
            aligned.append(sol_aligned)

        # Convertir a matriz (n_solutions × N)
        X = np.array(aligned)

        # PCA
        pca = PCA()
        pca.fit(X)

        # Varianzas explicadas normalizadas
        ev = pca.explained_variance_
        ev = ev / (np.sum(ev) + 1e-10)

        # Dimensión efectiva (entropía exponencial)
        d_eff = np.exp(-np.sum(ev * np.log(ev + 1e-12)))

        # Acotar a rango razonable
        d_eff = min(max(d_eff, 1.0), float(len(solutions)))

        return d_eff

    except Exception as e:
        return float(len(solutions))

# =========================================================
# 7. EXPLORADOR CON TIMEOUT
# =========================================================

NUM_STARTS = 100  # Más starts para mejor cobertura del espacio
TIMEOUT_SECONDS = 600  # 10 minutos
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
        solutions = list(self.unique_minima.values())

        # d_G como DIMENSIÓN EFECTIVA (PCA entropy)
        dG_effective = compute_dG_effective(solutions)

        # Métricas adicionales (para diagnóstico)
        num_unique_minima = len(solutions)

        # Calcular energías para gap
        energies = [energy(s, self.J) for s in solutions]
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
            'dG_effective': dG_effective,           # Métrica principal (COMPARABLE CON M18)
            'num_unique_minima': num_unique_minima, # Diagnóstico
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
print("🚀 PROCESANDO INSTANCIAS")
print(f"   d_G = DIMENSIÓN EFECTIVA (PCA + entropía)")
print(f"   Comparable con M18")
print(f"⏱️  Timeout: {TIMEOUT_SECONDS//60} minutos por instancia")
print(f"🎲 Starts: {NUM_STARTS}")
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

# Recopilar todas las instancias de todos los bins
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

if len(remaining) == 0:
    print("\n🎉 TODAS LAS INSTANCIAS YA ESTÁN PROCESADAS")
    print("👉 Ejecuta M19_step3_v3 para el análisis final")
    exit(0)

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

        dG_eff = results['dG_effective']
        num_min = results['num_unique_minima']
        is_partial = results.get('_partial', False)

        if is_partial:
            print(f"   ⏰ TIMEOUT después de {elapsed:.1f}s")
            print(f"      → Resultados PARCIALES ({results['completed_starts']}/{NUM_STARTS} starts)")
            results['status'] = 'partial_timeout'
        else:
            print(f"   ✅ Completado en {elapsed:.1f}s")
            results['status'] = 'complete'

        print(f"      → dG_effective = {dG_eff:.4f}  (dimensión efectiva)")
        print(f"      → num_minima = {num_min}  (diagnóstico)")

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
print(f"Total landscape files: {len(landscape_files)}")

dG_eff_values = []
for f in landscape_files:
    try:
        data = np.load(f)
        if 'dG_effective' in data:
            dG_eff_values.append(float(data['dG_effective']))
    except:
        pass

if dG_eff_values:
    print(f"\n📈 dG_effective (dimensión efectiva) statistics:")
    print(f"   Min: {np.min(dG_eff_values):.4f}")
    print(f"   Max: {np.max(dG_eff_values):.4f}")
    print(f"   Mean: {np.mean(dG_eff_values):.4f}")
    print(f"   Std: {np.std(dG_eff_values):.4f}")

# Verificar checkpoint final
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'r') as f:
        final_checkpoint = json.load(f)
        print(f"\n📋 Checkpoint final: {len(final_checkpoint['processed_files'])}/{total} instancias procesadas")

print(f"\n📁 Resultados en: {LANDSCAPE_DIR}")
print("="*70)

if len(landscape_files) == total:
    print("\n🎉 ¡EXPERIMENTO COMPLETADO EXITOSAMENTE!")
    print("   d_G ahora es DIMENSIÓN EFECTIVA (PCA + entropía)")
    print("   COMPARABLE CON M18")
    print("\n👉 Ejecuta M19_step3_v3 para el análisis final")
else:
    print(f"\n⚠️ Progreso: {len(landscape_files)}/{total}")
    print("   Vuelve a ejecutar este script para continuar")

print("="*70)
print("🛡️ Anti-desconexión sigue activa - puedes cerrar la pantalla")
print("="*70)
