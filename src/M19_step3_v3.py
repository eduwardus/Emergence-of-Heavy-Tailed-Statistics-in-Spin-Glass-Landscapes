# -*- coding: utf-8 -*-
"""M19_step3_v3.py - LEY UNIVERSAL EN SPIN GLASS (usando dG_effective)"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# 0. MONTAR DRIVE Y CONFIGURACIÓN
# =========================================================

try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("✓ Google Drive montado")
except:
    print("⚠️ No se detectó Colab")

# USAR M19_V3 (la versión correcta con dG_effective)
BASE_DRIVE = "/content/drive/MyDrive/M19_V3"
LANDSCAPE_DIR = os.path.join(BASE_DRIVE, "data/landscape")
RESULTS_DIR = os.path.join(BASE_DRIVE, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("M19 - STEP 3: LEY UNIVERSAL EN SPIN GLASS")
print("Usando dG_effective (dimensión efectiva PCA)")
print("Comparable con M18")
print("="*70)
print(f"Input: {LANDSCAPE_DIR}")
print(f"Output: {RESULTS_DIR}")
print("="*70)

# =========================================================
# 1. CARGAR DATASET
# =========================================================

print("\n📂 CARGANDO DATASET")
print("-"*50)

landscape_files = list(Path(LANDSCAPE_DIR).glob("landscape_*.npz"))
print(f"Archivos encontrados: {len(landscape_files)}")

if len(landscape_files) == 0:
    print("\n❌ ERROR: No se encontraron archivos landscape")
    print(f"   Buscando en: {LANDSCAPE_DIR}")
    exit(1)

kappa = []
dG = []

for file in landscape_files:
    try:
        data = np.load(file)

        # USAR dG_effective (dimensión efectiva)
        if 'dG_effective' in data:
            dG_val = float(data['dG_effective'])
        else:
            # Fallback: si no existe, usar num_unique_minima pero con advertencia
            dG_val = float(data.get('num_unique_minima', -1))
            if dG_val > 0:
                print(f"   ⚠️ {file.name}: usando num_unique_minima (fallback)")

        kappa_val = float(data.get('kappa_J', -1))

        # Filtrar valores inválidos
        if dG_val <= 0 or kappa_val <= 0:
            continue

        # Filtrar valores extremos (dG_effective debería estar entre 1 y ~15)
        if dG_val > 20:
            continue

        kappa.append(kappa_val)
        dG.append(dG_val)

    except Exception as e:
        print(f"   ⚠️ Error en {file.name}: {e}")
        continue

kappa = np.array(kappa)
dG = np.array(dG)

print(f"\n✅ Total muestras válidas: {len(kappa)}")
print(f"   Rango κ: [{np.min(kappa):.3f}, {np.max(kappa):.3f}]")
print(f"   Rango dG_effective: [{np.min(dG):.4f}, {np.max(dG):.4f}]")

if len(kappa) < 20:
    print(f"\n⚠️ ADVERTENCIA: Solo {len(kappa)} muestras válidas")
    print("   Es posible que M19_step2_v3 no haya terminado")

# =========================================================
# 2. FILTRADO PARA FITTING (como M18)
# =========================================================

print("\n🔧 FILTRADO PARA FITTING")
print("-"*50)

# Usar κ entre 1 y 20 (donde se espera la transición)
mask_fit = (kappa >= 1.0) & (kappa <= 20)
kappa_fit = kappa[mask_fit]
dG_fit = dG[mask_fit]

print(f"   Muestras para fitting: {len(kappa_fit)}")
if len(kappa_fit) > 0:
    print(f"   Rango κ_fit: [{np.min(kappa_fit):.3f}, {np.max(kappa_fit):.3f}]")
    print(f"   Rango dG_fit: [{np.min(dG_fit):.4f}, {np.max(dG_fit):.4f}]")

if len(kappa_fit) < 10:
    print("\n❌ ERROR: Muy pocas muestras para fitting")
    print("   Verifica que M19_step2_v3 se ejecutó completamente")
    exit(1)

# =========================================================
# 3. BINNING POR CUANTILES (como M18)
# =========================================================

def bin_by_quantiles(x, y, n_bins=12, min_points=5):
    """Binning por cuantiles - mismo que M18"""
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

print("\n📊 BINNING")
print("-"*50)

x_centers, y_means, y_stds = bin_by_quantiles(kappa_fit, dG_fit, n_bins=12, min_points=5)
print(f"   Bins generados: {len(x_centers)}")

if len(x_centers) < 4:
    print("   ⚠️ Pocos bins, ajustando parámetros...")
    x_centers, y_means, y_stds = bin_by_quantiles(kappa_fit, dG_fit, n_bins=8, min_points=3)
    print(f"   Bins generados: {len(x_centers)}")

if len(x_centers) < 3:
    print("\n❌ ERROR: No se pudieron generar suficientes bins")
    print("   Usando datos sin binning...")
    x_centers = kappa_fit
    y_means = dG_fit
    y_stds = np.zeros_like(dG_fit)

print(f"   Rango centros κ: [{np.min(x_centers):.3f}, {np.max(x_centers):.3f}]")
print(f"   Rango dG medio: [{np.min(y_means):.4f}, {np.max(y_means):.4f}]")

# =========================================================
# 4. FUNCIÓN SIGMOIDE (decreciente, como M18 pero invertida)
# =========================================================

def sigmoid(k, k_c, s, d_min, d_max):
    """
    Sigmoide para transición de fase.
    En M19: κ bajo → dG alto, κ alto → dG bajo
    """
    return d_min + (d_max - d_min) / (1 + np.exp(s * (k - k_c)))

# =========================================================
# 5. AJUSTE SIGMOIDE
# =========================================================

print("\n📐 AJUSTE SIGMOIDE")
print("-"*50)

# p0: k_c=3, s=5, d_min=1 (κ alto), d_max=10 (κ bajo)
p0 = [3.0, 5.0, np.min(y_means), np.max(y_means)]

try:
    if len(y_stds) > 0 and np.all(y_stds > 0):
        params, cov = curve_fit(
            sigmoid,
            x_centers,
            y_means,
            p0=p0,
            sigma=y_stds,
            maxfev=10000
        )
    else:
        params, cov = curve_fit(
            sigmoid,
            x_centers,
            y_means,
            p0=p0,
            maxfev=10000
        )
    perr = np.sqrt(np.diag(cov))

    k_c, s, d_min, d_max = params

    print(f"   κ_c = {k_c:.4f} ± {perr[0]:.4f}")
    print(f"   s = {s:.4f} ± {perr[1]:.4f}")
    print(f"   d_min (κ alto) = {d_min:.4f} ± {perr[2]:.4f}")
    print(f"   d_max (κ bajo) = {d_max:.4f} ± {perr[3]:.4f}")

except Exception as e:
    print(f"   ❌ Error en ajuste: {e}")
    exit(1)

# =========================================================
# 6. MÉTRICAS R²
# =========================================================

print("\n📊 MÉTRICAS")
print("-"*50)

y_pred_binned = sigmoid(x_centers, *params)
r2_binned = r2_score(y_means, y_pred_binned)
print(f"   R² (binned): {r2_binned:.4f}")

y_pred_full = sigmoid(kappa_fit, *params)
r2_full = r2_score(dG_fit, y_pred_full)
print(f"   R² (full): {r2_full:.4f}")

# =========================================================
# 7. BOOTSTRAP DE κ_c (como M18)
# =========================================================

print("\n🔄 BOOTSTRAP DE κ_c")
print("-"*50)

n_bootstrap = 200
k_c_samples = []

for i in range(n_bootstrap):
    idx = np.random.choice(len(kappa_fit), len(kappa_fit), replace=True)
    xb = kappa_fit[idx]
    yb = dG_fit[idx]

    try:
        xc, ym, ys = bin_by_quantiles(xb, yb, n_bins=8, min_points=3)
        if len(xc) < 3:
            continue
        p, _ = curve_fit(sigmoid, xc, ym, p0=p0, maxfev=5000)
        k_c_samples.append(p[0])
    except:
        continue

if len(k_c_samples) > 10:
    k_c_mean = np.mean(k_c_samples)
    k_c_std = np.std(k_c_samples)
    ci_low = np.percentile(k_c_samples, 2.5)
    ci_high = np.percentile(k_c_samples, 97.5)

    print(f"   Bootstrap samples: {len(k_c_samples)}")
    print(f"   κ_c mean: {k_c_mean:.4f} ± {k_c_std:.4f}")
    print(f"   IC95%: [{ci_low:.4f}, {ci_high:.4f}]")
else:
    print(f"   ⚠️ Bootstrap falló ({len(k_c_samples)} muestras)")
    k_c_mean = k_c
    k_c_std = perr[0]
    ci_low = k_c - 2*k_c_std
    ci_high = k_c + 2*k_c_std

# =========================================================
# 8. FIGURA PRINCIPAL (como M18)
# =========================================================

print("\n📈 GENERANDO FIGURAS")
print("-"*50)

# Curva suave para plotting
k_smooth = np.linspace(0, 20, 300)
d_smooth = sigmoid(k_smooth, *params)

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Scatter de todos los puntos
ax1.scatter(kappa, dG, alpha=0.2, s=15, c='steelblue', label='Datos individuales')

# Binned data con error bars
if len(x_centers) >= 3:
    ax1.errorbar(x_centers, y_means, yerr=y_stds,
                 fmt='o', color='black', capsize=4, capthick=1.5, markersize=8,
                 label=f'Binned data (n_bins={len(x_centers)})', zorder=5)

# Curva sigmoide
ax1.plot(k_smooth, d_smooth, 'r-', linewidth=3, zorder=6,
         label=f'Sigmoid fit (κ_c = {k_c:.3f})')

# Línea vertical en κ_c
ax1.axvline(k_c, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1.axvspan(ci_low, ci_high, alpha=0.15, color='red',
            label=f'IC95% κ_c: [{ci_low:.2f}, {ci_high:.2f}]')

# Etiquetas
ax1.set_xlabel(r'$\kappa_J$ (Kurtosis de acoplamientos)', fontsize=14)
ax1.set_ylabel(r'$d_G$ (Dimensión efectiva)', fontsize=14)
ax1.set_title('M19: Universal Phase Transition in Spin Glass Landscape', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(0, 20)
ax1.set_ylim(0, max(dG) * 1.1)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_master_curve.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_master_curve.png")

# =========================================================
# 9. FIGURA LIMPIA (solo binned + curva)
# =========================================================

fig2, ax2 = plt.subplots(figsize=(10, 7))

if len(x_centers) >= 3:
    ax2.errorbar(x_centers, y_means, yerr=y_stds,
                 fmt='o', color='black', capsize=4, capthick=1.5, markersize=8,
                 label=f'Binned data (n={len(x_centers)} bins)')

ax2.plot(k_smooth, d_smooth, 'r-', linewidth=3, zorder=6,
         label=f'Sigmoid fit (κ_c = {k_c:.3f})')
ax2.axvline(k_c, color='red', linestyle='--', alpha=0.7, linewidth=2)

# R² en el gráfico
ax2.text(0.05, 0.95, f'R² = {r2_binned:.4f}', transform=ax2.transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_xlabel(r'$\kappa_J$ (Kurtosis de acoplamientos)', fontsize=14)
ax2.set_ylabel(r'$d_G$ (Dimensión efectiva)', fontsize=14)
ax2.set_title('M19: Sigmoid Collapse - Spin Glass Landscape', fontsize=14)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, 20)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_sigmoid_fit.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_sigmoid_fit.png")

# =========================================================
# 10. GUARDAR RESULTADOS JSON
# =========================================================

results = {
    "experiment": "M19_step3_v3",
    "data_directory": "M19_V3",
    "timestamp": str(np.datetime64('now')),
    "kappa_c": float(k_c),
    "kappa_c_error": float(perr[0]),
    "slope": float(s),
    "slope_error": float(perr[1]),
    "d_min": float(d_min),
    "d_min_error": float(perr[2]),
    "d_max": float(d_max),
    "d_max_error": float(perr[3]),
    "r2_binned": float(r2_binned),
    "r2_full": float(r2_full),
    "bootstrap": {
        "mean": float(k_c_mean),
        "std": float(k_c_std),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "n_samples": len(k_c_samples)
    },
    "n_samples_total": int(len(kappa)),
    "n_samples_fit": int(len(kappa_fit)),
    "n_bins": int(len(x_centers)),
    "kappa_range": [float(np.min(kappa)), float(np.max(kappa))],
    "dG_range": [float(np.min(dG)), float(np.max(dG))]
}

json_path = os.path.join(RESULTS_DIR, "M19_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✓ M19_results.json")

# =========================================================
# 11. RESUMEN FINAL
# =========================================================

print("\n" + "="*70)
print("RESUMEN FINAL - M19 STEP 3")
print("="*70)

print(f"""
DATASET (M19_V3):
   - Muestras válidas: {len(kappa)}
   - Rango κ: [{np.min(kappa):.2f}, {np.max(kappa):.2f}]
   - Rango dG_effective: [{np.min(dG):.4f}, {np.max(dG):.4f}]

AJUSTE SIGMOIDE:
   - κ_c = {k_c:.4f} ± {perr[0]:.4f}
   - d_max (κ→0) = {d_max:.4f}
   - d_min (κ→∞) = {d_min:.4f}
   - R² (binned) = {r2_binned:.4f}

BOOTSTRAP κ_c:
   - Media = {k_c_mean:.4f} ± {k_c_std:.4f}
   - IC95% = [{ci_low:.4f}, {ci_high:.4f}]

FIGURAS:
   - {RESULTS_DIR}/M19_master_curve.png
   - {RESULTS_DIR}/M19_sigmoid_fit.png
""")

print("="*70)

# Validación
if 2 < k_c < 8:
    print("✅ κ_c en rango esperado (2-8) - comparable con M18")
else:
    print(f"⚠️ κ_c = {k_c:.3f} - fuera del rango esperado")

if r2_binned > 0.7:
    print("✅ R² > 0.7 - ajuste aceptable")
else:
    print(f"⚠️ R² = {r2_binned:.3f} - ajuste mejorable")

print("="*70)
print("🎉 LISTO PARA COMPARAR CON M18")
print("="*70)
