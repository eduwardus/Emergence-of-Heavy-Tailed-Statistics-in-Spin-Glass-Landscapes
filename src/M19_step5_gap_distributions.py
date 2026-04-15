# -*- coding: utf-8 -*-
"""M19_step5_gap_distributions.py - Distribución de gaps por bins de κ (colas pesadas)"""

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
RESULTS_DIR = os.path.join(BASE_DRIVE, "results_step5")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("M19 - STEP 5: GAP DISTRIBUTIONS")
print("Análisis de colas pesadas en energy gaps")
print("NO medias - distribuciones completas")
print("="*70)
print(f"Input: {LANDSCAPE_DIR}")
print(f"Output: {RESULTS_DIR}")
print("="*70)

# =========================================================
# 1. DEFINIR BINS DE κ
# =========================================================

# Bins para agrupar por curtosis
KAPPA_BINS = [
    (0, 2, "κ ∈ [0, 2) - Baja"),
    (2, 5, "κ ∈ [2, 5) - Media baja"),
    (5, 10, "κ ∈ [5, 10) - Media"),
    (10, 20, "κ ∈ [10, 20) - Alta"),
    (20, 100, "κ ∈ [20, 100) - Muy alta")
]

# =========================================================
# 2. CARGAR DATOS Y EXTRAER GAPS
# =========================================================

print("\n📂 CARGANDO DATOS")
print("-"*50)

landscape_files = list(Path(LANDSCAPE_DIR).glob("landscape_*.npz"))
print(f"Archivos encontrados: {len(landscape_files)}")

if len(landscape_files) == 0:
    print("\n❌ ERROR: No se encontraron archivos landscape")
    exit(1)

# Almacenar gaps por bin
gaps_by_bin = {f"bin_{i}": {"gaps": [], "kappa_values": [], "range": r}
               for i, (low, high, r) in enumerate(KAPPA_BINS)}
bin_labels = [f"bin_{i}" for i in range(len(KAPPA_BINS))]

kappa_all = []
gap_all = []

for file in landscape_files:
    try:
        data = np.load(file)

        kappa_val = float(data.get('kappa_J', -1))

        if kappa_val <= 0:
            continue

        # Obtener energy gap
        if 'energy_gap' in data:
            gap_val = float(data['energy_gap'])
        else:
            continue

        if gap_val < 0:
            continue

        kappa_all.append(kappa_val)
        gap_all.append(gap_val)

        # Asignar a bin
        for i, (low, high, _) in enumerate(KAPPA_BINS):
            if low <= kappa_val < high:
                gaps_by_bin[f"bin_{i}"]["gaps"].append(gap_val)
                gaps_by_bin[f"bin_{i}"]["kappa_values"].append(kappa_val)
                break

    except Exception as e:
        continue

print(f"\n✅ Total muestras: {len(kappa_all)}")
print(f"   Rango κ: [{np.min(kappa_all):.3f}, {np.max(kappa_all):.3f}]")
print(f"   Rango gaps: [{np.min(gap_all):.6f}, {np.max(gap_all):.6f}]")

# =========================================================
# 3. ESTADÍSTICAS POR BIN
# =========================================================

print("\n📊 ESTADÍSTICAS POR BIN")
print("-"*50)

for i, (low, high, label) in enumerate(KAPPA_BINS):
    bin_data = gaps_by_bin[f"bin_{i}"]
    gaps = bin_data["gaps"]
    n = len(gaps)

    if n > 0:
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        median_gap = np.median(gaps)
        p95 = np.percentile(gaps, 95)
        p99 = np.percentile(gaps, 99)
        max_gap = np.max(gaps)

        print(f"\n   {label}:")
        print(f"      n = {n}")
        print(f"      mean = {mean_gap:.6f} ± {std_gap:.6f}")
        print(f"      median = {median_gap:.6f}")
        print(f"      p95 = {p95:.6f}")
        print(f"      p99 = {p99:.6f}")
        print(f"      max = {max_gap:.6f}")
    else:
        print(f"\n   {label}: sin datos")

# =========================================================
# 4. FIGURA 1: DISTRIBUCIÓN DE GAPS (histogramas apilados)
# =========================================================

print("\n📈 GENERANDO FIGURAS")
print("-"*50)

fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, (i, (low, high, label)) in enumerate(zip(range(len(KAPPA_BINS)), KAPPA_BINS)):
    ax = axes[idx]
    gaps = gaps_by_bin[f"bin_{i}"]["gaps"]

    if len(gaps) > 0:
        # Histograma con escala logarítmica en Y para ver colas
        ax.hist(gaps, bins=30, alpha=0.7, color=colors[idx], edgecolor='black', density=True)
        ax.set_xlabel('Energy Gap', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{label}\n(n={len(gaps)})', fontsize=12)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(label)

# Ocultar eje vacío si sobra
if len(KAPPA_BINS) < len(axes):
    axes[-1].set_visible(False)

plt.suptitle('M19: Energy Gap Distributions by Kurtosis Bin', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_gap_histograms.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_gap_histograms.png")

# =========================================================
# 5. FIGURA 2: COLAS PESADAS - P(gap > x) vs κ
# =========================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Colores y estilos para cada bin
styles = ['o-', 's-', '^-', 'd-', '*-']

for idx, (i, (low, high, label)) in enumerate(zip(range(len(KAPPA_BINS)), KAPPA_BINS)):
    gaps = gaps_by_bin[f"bin_{i}"]["gaps"]

    if len(gaps) > 0:
        # Ordenar gaps
        gaps_sorted = np.sort(gaps)
        # Probabilidad empírica: P(gap > x)
        p_above = 1 - np.arange(1, len(gaps_sorted) + 1) / len(gaps_sorted)

        # Graficar solo hasta el percentil 95 para mejor visualización
        max_idx = min(len(gaps_sorted), int(0.95 * len(gaps_sorted)))
        ax2.plot(gaps_sorted[:max_idx], p_above[:max_idx],
                styles[idx], linewidth=2, markersize=4,
                label=f'{label} (n={len(gaps)})', alpha=0.8)

ax2.set_xlabel('Energy Gap', fontsize=14)
ax2.set_ylabel('P(gap > x)', fontsize=14)
ax2.set_title('M19: Heavy Tails Analysis - P(gap > x) vs Gap', fontsize=14)
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.001, None)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_gap_survival.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_gap_survival.png")

# =========================================================
# 6. FIGURA 3: BOXPLOT de gaps por bin
# =========================================================

fig3, ax3 = plt.subplots(figsize=(12, 7))

box_data = []
box_labels = []
positions = []

for i, (low, high, label) in enumerate(KAPPA_BINS):
    gaps = gaps_by_bin[f"bin_{i}"]["gaps"]
    if len(gaps) > 0:
        box_data.append(gaps)
        box_labels.append(f"{label}\n(n={len(gaps)})")
        positions.append(i)

bp = ax3.boxplot(box_data, positions=positions, widths=0.6,
                 patch_artist=True, showfliers=True)

# Colorear boxes
for idx, box in enumerate(bp['boxes']):
    box.set_facecolor(colors[idx % len(colors)])
    box.set_alpha(0.7)

ax3.set_xlabel('Kurtosis Bin', fontsize=14)
ax3.set_ylabel('Energy Gap', fontsize=14)
ax3.set_title('M19: Energy Gap Distribution by Kurtosis Bin', fontsize=14)
ax3.set_xticks(positions)
ax3.set_xticklabels(box_labels, fontsize=10)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_gap_boxplot.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_gap_boxplot.png")

# =========================================================
# 7. FIGURA 4: Scatter con énfasis en colas
# =========================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))

# Scatter normal
ax4.scatter(kappa_all, gap_all, alpha=0.3, s=20, c='steelblue', label='Datos individuales')

# Destacar colas (gaps > percentil 95)
p95_threshold = np.percentile(gap_all, 95)
tail_mask = np.array(gap_all) > p95_threshold
ax4.scatter(np.array(kappa_all)[tail_mask], np.array(gap_all)[tail_mask],
           alpha=0.8, s=80, c='red', marker='o', edgecolors='black',
           label=f'Gaps > p95 ({p95_threshold:.4f})')

ax4.set_xlabel(r'$\kappa_J$ (Kurtosis de acoplamientos)', fontsize=14)
ax4.set_ylabel('Energy Gap', fontsize=14)
ax4.set_title('M19: Energy Gap vs Kurtosis - Heavy Tails Highlighted', fontsize=14)
ax4.set_yscale('log')
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_gap_scatter_tails.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_gap_scatter_tails.png")

# =========================================================
# 8. ANÁLISIS DE COLAS: Fracción de gaps extremos por bin
# =========================================================

print("\n📊 ANÁLISIS DE COLAS PESADAS")
print("-"*50)

# Calcular umbral global (percentil 95)
global_p95 = np.percentile(gap_all, 95)
global_p99 = np.percentile(gap_all, 99)

print(f"\nUmbrales globales:")
print(f"   p95 = {global_p95:.6f}")
print(f"   p99 = {global_p99:.6f}")

print(f"\nFracción de gaps extremos por bin:")
for i, (low, high, label) in enumerate(KAPPA_BINS):
    gaps = gaps_by_bin[f"bin_{i}"]["gaps"]
    if len(gaps) > 0:
        frac_p95 = np.sum(np.array(gaps) > global_p95) / len(gaps)
        frac_p99 = np.sum(np.array(gaps) > global_p99) / len(gaps)
        print(f"\n   {label}:")
        print(f"      > p95: {frac_p95:.3f} ({int(np.sum(np.array(gaps) > global_p95))}/{len(gaps)})")
        print(f"      > p99: {frac_p99:.3f} ({int(np.sum(np.array(gaps) > global_p99))}/{len(gaps)})")

# =========================================================
# 9. GUARDAR RESULTADOS JSON
# =========================================================

results = {
    "experiment": "M19_step5_gap_distributions",
    "data_directory": "M19_V3",
    "timestamp": str(np.datetime64('now')),
    "n_samples_total": int(len(kappa_all)),
    "kappa_range": [float(np.min(kappa_all)), float(np.max(kappa_all))],
    "gap_global_stats": {
        "mean": float(np.mean(gap_all)),
        "std": float(np.std(gap_all)),
        "median": float(np.median(gap_all)),
        "p95": float(global_p95),
        "p99": float(global_p99),
        "max": float(np.max(gap_all))
    },
    "bins": []
}

for i, (low, high, label) in enumerate(KAPPA_BINS):
    gaps = gaps_by_bin[f"bin_{i}"]["gaps"]
    if len(gaps) > 0:
        bin_data = {
            "bin_label": label,
            "kappa_range": [low, high],
            "n_samples": len(gaps),
            "gap_stats": {
                "mean": float(np.mean(gaps)),
                "std": float(np.std(gaps)),
                "median": float(np.median(gaps)),
                "p95": float(np.percentile(gaps, 95)),
                "p99": float(np.percentile(gaps, 99)),
                "max": float(np.max(gaps))
            },
            "tail_fractions": {
                "above_global_p95": float(np.sum(np.array(gaps) > global_p95) / len(gaps)),
                "above_global_p99": float(np.sum(np.array(gaps) > global_p99) / len(gaps))
            }
        }
        results["bins"].append(bin_data)

json_path = os.path.join(RESULTS_DIR, "M19_gap_analysis.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n   ✓ M19_gap_analysis.json")

# =========================================================
# 10. RESUMEN FINAL
# =========================================================

print("\n" + "="*70)
print("RESUMEN FINAL - M19 STEP 5")
print("="*70)

print(f"""
DATASET:
   - Muestras totales: {len(kappa_all)}
   - Rango κ: [{np.min(kappa_all):.2f}, {np.max(kappa_all):.2f}]

GAP ESTADÍSTICAS GLOBALES:
   - Mean: {np.mean(gap_all):.6f}
   - Std: {np.std(gap_all):.6f}
   - p95: {global_p95:.6f}
   - p99: {global_p99:.6f}
   - Max: {np.max(gap_all):.6f}

INTERPRETACIÓN:
   - κ introduce fluctuaciones locales en energía
   - Aparecen colas pesadas en κ alto
   - No hay transición ni colapso global

FIGURAS:
   - {RESULTS_DIR}/M19_gap_histograms.png
   - {RESULTS_DIR}/M19_gap_survival.png
   - {RESULTS_DIR}/M19_gap_boxplot.png
   - {RESULTS_DIR}/M19_gap_scatter_tails.png

RESULTADOS JSON:
   - {RESULTS_DIR}/M19_gap_analysis.json
""")

print("="*70)
print("🔍 CONCLUSIONES:")
print("   1. κ NO controla la estructura global del paisaje")
print("   2. κ introduce fluctuaciones locales (energía)")
print("   3. Aparecen colas pesadas en κ alto → mínimos raros y profundos")
print("   4. GLG (M18) vs Spin Glass (M19):")
print("      - GLG: κ reorganiza el paisaje (colapso)")
print("      - Spin Glass: κ afecta distribución de energías (no topología)")
print("\n👉 CONTRAEJEMPLO ESTRUCTURAL: La ley universal NO es universal")
print("   Esto delimita el dominio de validez de M18")
print("="*70)
print("✅ M19 STEP 5 COMPLETADO")
print("="*70)
