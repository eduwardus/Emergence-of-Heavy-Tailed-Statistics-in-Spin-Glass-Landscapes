# -*- coding: utf-8 -*-
"""M19_step4_landscape_analysis.py - Análisis estructural del paisaje SIN sigmoide"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 0. MONTAR DRIVE
# =========================================================

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive montado")
except:
    print("⚠️ No se detectó Colab")

BASE_DRIVE = "/content/drive/MyDrive/M19_V3"
LANDSCAPE_DIR = os.path.join(BASE_DRIVE, "data/landscape")
RESULTS_DIR = os.path.join(BASE_DRIVE, "results_step4")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("M19 - STEP 4: LANDSCAPE STRUCTURE ANALYSIS")
print("Sin sigmoide - análisis estructural puro")
print("="*70)
print(f"Input: {LANDSCAPE_DIR}")
print(f"Output: {RESULTS_DIR}")
print("="*70)

# =========================================================
# 1. FUNCIÓN DE BINNING (como M18)
# =========================================================

def bin_by_quantiles(x, y, n_bins=10, min_points=5):
    """Binning por cuantiles - mismo que M18"""
    if len(x) < min_points:
        return np.array([]), np.array([]), np.array([])

    percentiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(x, percentiles)

    x_centers = []
    y_means = []
    y_stds = []

    for i in range(len(bins)-1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        n = np.sum(mask)

        if n >= min_points:
            x_centers.append(np.mean(x[mask]))
            y_means.append(np.mean(y[mask]))
            y_stds.append(np.std(y[mask]) / np.sqrt(n))

    return np.array(x_centers), np.array(y_means), np.array(y_stds)

# =========================================================
# 2. FUNCIÓN PARA RECALCULAR ENERGÍAS
# =========================================================

def compute_energy(s, J):
    """Calcula energía de una configuración de spins"""
    return -np.sum(np.triu(J * np.outer(s, s), 1))

# =========================================================
# 3. CARGAR DATOS Y EXTRAER MÉTRICAS
# =========================================================

print("\n📂 CARGANDO DATOS")
print("-"*50)

landscape_files = list(Path(LANDSCAPE_DIR).glob("landscape_*.npz"))
print(f"Archivos encontrados: {len(landscape_files)}")

if len(landscape_files) == 0:
    print("\n❌ ERROR: No se encontraron archivos landscape")
    exit(1)

kappa_list = []
gap_list = []
nmin_list = []
stdE_list = []
dG_effective_list = []

files_skipped = 0
files_processed = 0

for file in landscape_files:
    try:
        data = np.load(file)

        kappa_val = float(data.get('kappa_J', -1))

        if kappa_val <= 0:
            files_skipped += 1
            continue

        # Verificar si tenemos 'energies' o tenemos que calcular desde configuraciones
        if 'energies' in data:
            energies = data['energies']
            E = np.sort(energies)
        else:
            # No tenemos energies guardadas - usar métricas disponibles
            # En este caso, usamos num_unique_minima y energy_gap
            n_min = int(data.get('num_unique_minima', 0))
            energy_gap = float(data.get('energy_gap', 0))
            dG_eff = float(data.get('dG_effective', 0))

            if n_min <= 2:
                files_skipped += 1
                continue

            # Simular distribución de energías (aproximación)
            # Para gap y n_min, podemos usar los campos existentes
            kappa_list.append(kappa_val)
            gap_list.append(energy_gap)
            nmin_list.append(n_min)
            stdE_list.append(energy_gap * np.sqrt(n_min) / 2)  # Aproximación
            dG_effective_list.append(dG_eff)
            files_processed += 1
            continue

        # Caso con energies disponible
        if len(E) < 2:
            files_skipped += 1
            continue

        E1 = E[0]
        E2 = E[1] if len(E) > 1 else E[0]

        gap = E2 - E1
        n_min = len(E)
        std_E = np.std(E)
        dG_eff = float(data.get('dG_effective', 0))

        kappa_list.append(kappa_val)
        gap_list.append(gap)
        nmin_list.append(n_min)
        stdE_list.append(std_E)
        dG_effective_list.append(dG_eff)
        files_processed += 1

    except Exception as e:
        files_skipped += 1
        continue

kappa = np.array(kappa_list)
gap = np.array(gap_list)
n_min = np.array(nmin_list)
std_E = np.array(stdE_list)
dG_eff = np.array(dG_effective_list) if dG_effective_list else np.zeros_like(kappa)

print(f"\n✅ Archivos procesados: {files_processed}")
print(f"   Archivos saltados: {files_skipped}")

if files_processed == 0:
    print("\n❌ ERROR: No se pudieron procesar archivos")
    exit(1)

# =========================================================
# 4. LIMPIEZA MÍNIMA (NO agresiva)
# =========================================================

print("\n🧹 LIMPIEZA MÍNIMA")
print("-"*50)

# Solo filtrar valores no finitos y n_min < 3
mask = (
    np.isfinite(kappa) &
    np.isfinite(gap) &
    np.isfinite(n_min) &
    np.isfinite(std_E) &
    (n_min > 2)
)

kappa_clean = kappa[mask]
gap_clean = gap[mask]
n_min_clean = n_min[mask]
std_E_clean = std_E[mask]
dG_eff_clean = dG_eff[mask] if len(dG_eff) > 0 else np.zeros_like(kappa_clean)

print(f"   Muestras totales: {len(kappa)}")
print(f"   Muestras válidas: {len(kappa_clean)}")
print(f"   Eliminadas: {len(kappa) - len(kappa_clean)}")

if len(kappa_clean) < 20:
    print(f"\n⚠️ ADVERTENCIA: Solo {len(kappa_clean)} muestras válidas")

# =========================================================
# 5. ESTADÍSTICAS GLOBALES
# =========================================================

print("\n📊 ESTADÍSTICAS GLOBALES")
print("-"*50)

print(f"   κ range: [{np.min(kappa_clean):.3f}, {np.max(kappa_clean):.3f}]")
print(f"   Gap range: [{np.min(gap_clean):.6f}, {np.max(gap_clean):.6f}]")
print(f"   Gap mean: {np.mean(gap_clean):.6f} ± {np.std(gap_clean):.6f}")
print(f"   n_min range: [{np.min(n_min_clean):.0f}, {np.max(n_min_clean):.0f}]")
print(f"   n_min mean: {np.mean(n_min_clean):.1f} ± {np.std(n_min_clean):.1f}")
print(f"   std_E range: [{np.min(std_E_clean):.6f}, {np.max(std_E_clean):.6f}]")
print(f"   std_E mean: {np.mean(std_E_clean):.6f} ± {np.std(std_E_clean):.6f}")

# =========================================================
# 6. BINNING
# =========================================================

print("\n📊 BINNING POR CUANTILES")
print("-"*50)

# Gap vs κ
x_gap, y_gap, e_gap = bin_by_quantiles(kappa_clean, gap_clean, n_bins=10, min_points=5)
print(f"   Gap bins: {len(x_gap)}")

# n_min vs κ
x_nmin, y_nmin, e_nmin = bin_by_quantiles(kappa_clean, n_min_clean, n_bins=10, min_points=5)
print(f"   n_min bins: {len(x_nmin)}")

# std_E vs κ
x_std, y_std, e_std = bin_by_quantiles(kappa_clean, std_E_clean, n_bins=10, min_points=5)
print(f"   std_E bins: {len(x_std)}")

# =========================================================
# 7. FIGURA 1: GAP vs κ
# =========================================================

print("\n📈 GENERANDO FIGURAS")
print("-"*50)

fig1, ax1 = plt.subplots(figsize=(12, 8))

ax1.scatter(kappa_clean, gap_clean, alpha=0.3, s=15, c='steelblue', label='Datos individuales')

if len(x_gap) >= 3:
    ax1.errorbar(x_gap, y_gap, yerr=e_gap, fmt='o', color='black',
                 capsize=4, capthick=1.5, markersize=8,
                 label=f'Binned data (n={len(x_gap)} bins)', zorder=5)

ax1.set_xlabel(r'$\kappa_J$ (Kurtosis de acoplamientos)', fontsize=14)
ax1.set_ylabel('Energy Gap (E₂ - E₁)', fontsize=14)
ax1.set_title('M19: Energy Gap vs Kurtosis', fontsize=14)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(0, np.max(kappa_clean) * 1.05)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_gap_vs_kappa.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_gap_vs_kappa.png")

# =========================================================
# 8. FIGURA 2: N mínimos vs κ
# =========================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))

ax2.scatter(kappa_clean, n_min_clean, alpha=0.3, s=15, c='steelblue', label='Datos individuales')

if len(x_nmin) >= 3:
    ax2.errorbar(x_nmin, y_nmin, yerr=e_nmin, fmt='o', color='black',
                 capsize=4, capthick=1.5, markersize=8,
                 label=f'Binned data (n={len(x_nmin)} bins)', zorder=5)

ax2.set_xlabel(r'$\kappa_J$ (Kurtosis de acoplamientos)', fontsize=14)
ax2.set_ylabel('Number of Local Minima', fontsize=14)
ax2.set_title('M19: Number of Local Minima vs Kurtosis', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, np.max(kappa_clean) * 1.05)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_nmin_vs_kappa.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_nmin_vs_kappa.png")

# =========================================================
# 9. FIGURA 3: Dispersión energética vs κ
# =========================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))

ax3.scatter(kappa_clean, std_E_clean, alpha=0.3, s=15, c='steelblue', label='Datos individuales')

if len(x_std) >= 3:
    ax3.errorbar(x_std, y_std, yerr=e_std, fmt='o', color='black',
                 capsize=4, capthick=1.5, markersize=8,
                 label=f'Binned data (n={len(x_std)} bins)', zorder=5)

ax3.set_xlabel(r'$\kappa_J$ (Kurtosis de acoplamientos)', fontsize=14)
ax3.set_ylabel('Energy Standard Deviation', fontsize=14)
ax3.set_title('M19: Energy Landscape Roughness vs Kurtosis', fontsize=14)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.2)
ax3.set_xlim(0, np.max(kappa_clean) * 1.05)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_stdE_vs_kappa.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_stdE_vs_kappa.png")

# =========================================================
# 10. FIGURA EXTRA: dG_effective vs κ (para comparar con M18)
# =========================================================

if np.any(dG_eff_clean > 0):
    fig4, ax4 = plt.subplots(figsize=(12, 8))

    ax4.scatter(kappa_clean, dG_eff_clean, alpha=0.3, s=15, c='steelblue', label='Datos individuales')

    x_deff, y_deff, e_deff = bin_by_quantiles(kappa_clean, dG_eff_clean, n_bins=10, min_points=5)
    if len(x_deff) >= 3:
        ax4.errorbar(x_deff, y_deff, yerr=e_deff, fmt='o', color='black',
                     capsize=4, capthick=1.5, markersize=8,
                     label=f'Binned data (n={len(x_deff)} bins)', zorder=5)

    ax4.set_xlabel(r'$\kappa_J$ (Kurtosis de acoplamientos)', fontsize=14)
    ax4.set_ylabel(r'$d_G$ (Dimensión efectiva)', fontsize=14)
    ax4.set_title('M19: Effective Dimension vs Kurtosis (Comparable con M18)', fontsize=14)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.2)
    ax4.set_xlim(0, np.max(kappa_clean) * 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "M19_deff_vs_kappa.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"   ✓ M19_deff_vs_kappa.png")

# =========================================================
# 11. GUARDAR RESULTADOS JSON
# =========================================================

results = {
    "experiment": "M19_step4_landscape_analysis",
    "data_directory": "M19_V3",
    "timestamp": str(np.datetime64('now')),
    "n_samples": int(len(kappa_clean)),
    "kappa_range": [float(np.min(kappa_clean)), float(np.max(kappa_clean))],
    "gap": {
        "mean": float(np.mean(gap_clean)),
        "std": float(np.std(gap_clean)),
        "min": float(np.min(gap_clean)),
        "max": float(np.max(gap_clean))
    },
    "n_min": {
        "mean": float(np.mean(n_min_clean)),
        "std": float(np.std(n_min_clean)),
        "min": int(np.min(n_min_clean)),
        "max": int(np.max(n_min_clean))
    },
    "std_E": {
        "mean": float(np.mean(std_E_clean)),
        "std": float(np.std(std_E_clean)),
        "min": float(np.min(std_E_clean)),
        "max": float(np.max(std_E_clean))
    },
    "binning": {
        "gap_bins": int(len(x_gap)),
        "nmin_bins": int(len(x_nmin)),
        "stdE_bins": int(len(x_std))
    }
}

json_path = os.path.join(RESULTS_DIR, "M19_summary.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✓ M19_summary.json")

# =========================================================
# 12. RESUMEN FINAL
# =========================================================

print("\n" + "="*70)
print("RESUMEN FINAL - M19 STEP 4")
print("="*70)

print(f"""
DATASET:
   - Muestras válidas: {len(kappa_clean)}
   - Rango κ: [{np.min(kappa_clean):.2f}, {np.max(kappa_clean):.2f}]

MÉTRICAS GLOBALES:
   - Gap mean: {np.mean(gap_clean):.6f} ± {np.std(gap_clean):.6f}
   - n_min mean: {np.mean(n_min_clean):.1f} ± {np.std(n_min_clean):.1f}
   - std_E mean: {np.mean(std_E_clean):.6f} ± {np.std(std_E_clean):.6f}

FIGURAS:
   - {RESULTS_DIR}/M19_gap_vs_kappa.png
   - {RESULTS_DIR}/M19_nmin_vs_kappa.png
   - {RESULTS_DIR}/M19_stdE_vs_kappa.png
""")

if np.any(dG_eff_clean > 0):
    print(f"   - {RESULTS_DIR}/M19_deff_vs_kappa.png")

print(f"""
RESULTADOS JSON:
   - {RESULTS_DIR}/M19_summary.json
""")

print("="*70)

print("\n🔍 QUÉ BUSCAR VISUALMENTE:")
print("   1. ¿El gap crece con κ?")
print("   2. ¿El número de mínimos cambia con κ?")
print("   3. ¿La dispersión energética aumenta con κ?")
print("   4. ¿dG_effective se comporta como en M18?")

print("\n" + "="*70)
print("✅ M19 STEP 4 COMPLETADO")
print("="*70)
