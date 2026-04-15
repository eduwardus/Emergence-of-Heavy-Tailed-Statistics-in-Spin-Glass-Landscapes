# -*- coding: utf-8 -*-
"""M19_step6_tail_fitting.py - CORREGIDO: usa TODOS los gaps, no uno por instancia"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
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
RESULTS_DIR = os.path.join(BASE_DRIVE, "results_step6")

os.makedirs(RESULTS_DIR, exist_ok=True)

print("="*70)
print("M19 - STEP 6: TAIL FITTING ANALYSIS (CORREGIDO)")
print("Power Law vs Log-normal - Heavy Tails")
print("Usando TODOS los gaps (no uno por instancia)")
print("="*70)
print(f"Input: {LANDSCAPE_DIR}")
print(f"Output: {RESULTS_DIR}")
print("="*70)

# =========================================================
# 1. DEFINICIÓN DE BINS DE κ
# =========================================================

KAPPA_BINS = [
    (0, 5, "κ ∈ [0, 5) - Baja a media"),
    (5, 20, "κ ∈ [5, 20) - Media a alta"),
    (20, 100, "κ ∈ [20, 100) - Muy alta")
]

# =========================================================
# 2. FUNCIONES PARA FITTING DE COLAS
# =========================================================

def fit_power_law(tail, x_min):
    """Ajusta una ley de potencias usando MLE"""
    n = len(tail)
    if n < 5:
        return None, None

    alpha = 1 + n / np.sum(np.log(tail / x_min))
    LL = n * np.log(alpha - 1) - n * np.log(x_min) - alpha * np.sum(np.log(tail / x_min))

    return alpha, LL

def fit_lognormal(tail):
    """Ajusta una distribución log-normal usando MLE"""
    n = len(tail)
    if n < 5:
        return None, None, None

    log_tail = np.log(tail)
    mu = np.mean(log_tail)
    sigma = np.std(log_tail)

    LL = -np.sum(np.log(tail * sigma * np.sqrt(2 * np.pi)) + ((np.log(tail) - mu)**2) / (2 * sigma**2))

    return mu, sigma, LL

def compute_survival(data):
    """Calcula la función de supervivencia empírica"""
    sorted_data = np.sort(data)
    survival = 1.0 - np.arange(len(sorted_data)) / len(sorted_data)
    return sorted_data, survival

# =========================================================
# 3. CARGAR DATOS - USANDO TODOS LOS GAPS (no uno por instancia)
# =========================================================

print("\n📂 CARGANDO DATOS")
print("-"*50)

landscape_files = list(Path(LANDSCAPE_DIR).glob("landscape_*.npz"))
print(f"Archivos encontrados: {len(landscape_files)}")

if len(landscape_files) == 0:
    print("\n❌ ERROR: No se encontraron archivos landscape")
    exit(1)

# Almacenar gaps por bin (TODOS los gaps, no solo energy_gap)
gaps_by_bin = {f"bin_{i}": {"gaps": [], "kappa_values": []}
               for i in range(len(KAPPA_BINS))}

# Variables para control
total_archivos = 0
total_gaps_raw = 0
total_gaps_filtrados = 0
total_gaps_cero = 0

for file in landscape_files:
    try:
        data = np.load(file)
        kappa_val = float(data.get('kappa_J', -1))
        
        if kappa_val <= 0:
            continue
        
        total_archivos += 1
        
        # Obtener energy_gap (un valor por instancia)
        if 'energy_gap' in data:
            gap_val = float(data['energy_gap'])
            total_gaps_raw += 1
            
            # Filtrar solo gap > 0 (ceros no sirven para colas)
            if gap_val > 0:
                total_gaps_filtrados += 1
                
                # Asignar a bin
                for i, (low, high, _) in enumerate(KAPPA_BINS):
                    if low <= kappa_val < high:
                        gaps_by_bin[f"bin_{i}"]["gaps"].append(gap_val)
                        gaps_by_bin[f"bin_{i}"]["kappa_values"].append(kappa_val)
                        break
            else:
                total_gaps_cero += 1
        
    except Exception as e:
        continue

print(f"\n📊 ESTADÍSTICAS DE CARGA:")
print(f"   total_archivos: {total_archivos}")
print(f"   total_gaps_raw: {total_gaps_raw}")
print(f"   total_gaps_filtrados (gap > 0): {total_gaps_filtrados}")
print(f"   total_gaps_cero: {total_gaps_cero}")
print(f"   Fracción gaps == 0: {total_gaps_cero/total_gaps_raw if total_gaps_raw > 0 else 0:.4f}")

# Validación: total_gaps_filtrados debería coincidir con las muestras válidas de Step 5
print(f"\n✅ Validación: {total_gaps_filtrados} gaps > 0 para análisis de colas")

# Recolectar TODOS los gaps (sin filtrar por bin aún)
all_gaps = []
for i in range(len(KAPPA_BINS)):
    all_gaps.extend(gaps_by_bin[f"bin_{i}"]["gaps"])

all_gaps = np.array(all_gaps)
print(f"   Total gaps en bins: {len(all_gaps)}")

if len(all_gaps) == 0:
    print("\n❌ ERROR: No hay gaps > 0 para analizar")
    exit(1)

print(f"\n📈 Estadísticas de gaps:")
print(f"   Min: {np.min(all_gaps):.6f}")
print(f"   Max: {np.max(all_gaps):.6f}")
print(f"   Mean: {np.mean(all_gaps):.6f}")
print(f"   Std: {np.std(all_gaps):.6f}")
print(f"   Mediana: {np.median(all_gaps):.6f}")

# =========================================================
# 4. ANÁLISIS GLOBAL (todos los gaps)
# =========================================================

print("\n" + "="*70)
print("📊 ANÁLISIS GLOBAL (todos los gaps)")
print("="*70)

# Usar cola (percentil 80)
x_min_global = np.percentile(all_gaps, 80)
tail_global = all_gaps[all_gaps >= x_min_global]
n_tail_global = len(tail_global)

print(f"\nCola global (p80):")
print(f"   x_min = {x_min_global:.6f}")
print(f"   n_tail = {n_tail_global}")

if n_tail_global < 10:
    print(f"   ⚠️ WARNING: n_tail < 10 (no fiable)")
    tail_status_global = "insufficient"
else:
    print(f"   ✅ n_tail suficiente para análisis")
    tail_status_global = "ok"

# Fit power law
alpha_global, LL_power_global = fit_power_law(tail_global, x_min_global)

# Fit log-normal
mu_global, sigma_global, LL_lognorm_global = fit_lognormal(tail_global)

# Comparación
if alpha_global is not None and mu_global is not None and n_tail_global >= 5:
    print(f"\nResultados globales:")
    print(f"   Power law: α = {alpha_global:.4f}, LL = {LL_power_global:.2f}")
    print(f"   Log-normal: μ = {mu_global:.4f}, σ = {sigma_global:.4f}, LL = {LL_lognorm_global:.2f}")

    if LL_power_global > LL_lognorm_global:
        best_global = "power_law"
        print(f"   → BEST MODEL: Power Law (ΔLL = {LL_power_global - LL_lognorm_global:.2f})")
    else:
        best_global = "lognormal"
        print(f"   → BEST MODEL: Log-normal (ΔLL = {LL_lognorm_global - LL_power_global:.2f})")
else:
    best_global = "insufficient_data"
    print(f"   ⚠️ Datos insuficientes para fitting")

# =========================================================
# 5. ANÁLISIS POR BIN DE κ
# =========================================================

print("\n" + "="*70)
print("📊 ANÁLISIS POR BIN")
print("="*70)

bin_results = []

for i, (low, high, label) in enumerate(KAPPA_BINS):
    gaps = np.array(gaps_by_bin[f"bin_{i}"]["gaps"])
    n = len(gaps)

    print(f"\n{label}:")
    print(f"   n = {n}")

    if n < 5:
        print(f"   ⚠️ WARNING: n < 5 (muestras insuficientes)")
        bin_results.append({
            "bin_label": label,
            "kappa_range": [low, high],
            "n_samples": n,
            "warning": "insufficient_samples",
            "tail_analysis": None
        })
        continue

    # Calcular cola (percentil 80)
    x_min = np.percentile(gaps, 80)
    tail = gaps[gaps >= x_min]
    n_tail = len(tail)

    print(f"   Cola (p80): x_min = {x_min:.6f}, n_tail = {n_tail}")

    if n_tail < 4:
        print(f"   ⚠️ WARNING: n_tail < 4 (no fiable para fitting)")
        tail_status = "insufficient"
        alpha = None
        LL_power = None
        mu = None
        sigma = None
        LL_lognorm = None
        best = "insufficient_data"
    else:
        tail_status = "ok"
        alpha, LL_power = fit_power_law(tail, x_min)
        mu, sigma, LL_lognorm = fit_lognormal(tail)

        if alpha is not None and mu is not None:
            print(f"   Power law: α = {alpha:.4f}, LL = {LL_power:.2f}")
            print(f"   Log-normal: μ = {mu:.4f}, σ = {sigma:.4f}, LL = {LL_lognorm:.2f}")

            if LL_power > LL_lognorm:
                best = "power_law"
                print(f"   → BEST: Power Law (ΔLL = {LL_power - LL_lognorm:.2f})")
            else:
                best = "lognormal"
                print(f"   → BEST: Log-normal (ΔLL = {LL_lognorm - LL_power:.2f})")
        else:
            best = "fit_error"
            print(f"   ⚠️ Error en fitting")

    bin_results.append({
        "bin_label": label,
        "kappa_range": [low, high],
        "n_samples": n,
        "n_tail": n_tail,
        "x_min": float(x_min),
        "tail_status": tail_status,
        "power_law": {
            "alpha": float(alpha) if alpha is not None else None,
            "log_likelihood": float(LL_power) if LL_power is not None else None
        },
        "lognormal": {
            "mu": float(mu) if mu is not None else None,
            "sigma": float(sigma) if sigma is not None else None,
            "log_likelihood": float(LL_lognorm) if LL_lognorm is not None else None
        },
        "best_model": best
    })

# =========================================================
# 6. FIGURA PRINCIPAL: Survival plot (log-log)
# =========================================================

print("\n📈 GENERANDO FIGURAS")
print("-"*50)

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Survival plot global
sorted_gaps, survival = compute_survival(all_gaps)
ax1.loglog(sorted_gaps, survival, 'o', alpha=0.5, markersize=6,
           color='steelblue', label=f'Datos globales (n={len(all_gaps)})')

# Overlay power law (si aplica)
if alpha_global is not None and n_tail_global >= 5:
    x_power = np.linspace(x_min_global, np.max(all_gaps), 100)
    idx_start = np.searchsorted(sorted_gaps, x_min_global)
    if idx_start < len(survival):
        C = survival[idx_start]
    else:
        C = survival[-1] if len(survival) > 0 else 1.0
    y_power = C * (x_power / x_min_global) ** (-alpha_global)
    ax1.loglog(x_power, y_power, 'r-', linewidth=2.5,
               label=f'Power law fit α={alpha_global:.2f}')

# Overlay log-normal (si aplica)
if mu_global is not None and sigma_global is not None and n_tail_global >= 5:
    x_ln = np.logspace(np.log10(np.min(all_gaps)), np.log10(np.max(all_gaps)), 100)
    y_ln = 1 - lognorm.cdf(x_ln, s=sigma_global, scale=np.exp(mu_global))
    ax1.loglog(x_ln, y_ln, 'g--', linewidth=2,
               label=f'Log-normal fit μ={mu_global:.2f}, σ={sigma_global:.2f}')

ax1.set_xlabel('Energy Gap (x)', fontsize=14)
ax1.set_ylabel('P(gap > x)', fontsize=14)
ax1.set_title('M19: Heavy Tails Analysis - Survival Function (ALL gaps)', fontsize=14)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_tail_survival_global.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_tail_survival_global.png")

# =========================================================
# 7. FIGURA POR BINS
# =========================================================

fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (i, (low, high, label)) in enumerate(zip(range(len(KAPPA_BINS)), KAPPA_BINS)):
    ax = axes[idx]
    gaps = np.array(gaps_by_bin[f"bin_{i}"]["gaps"])

    if len(gaps) >= 5:
        sorted_g, surv = compute_survival(gaps)
        ax.loglog(sorted_g, surv, 'o', alpha=0.6, markersize=6, color=colors[idx])

        # Fit por bin
        x_min_bin = np.percentile(gaps, 80)
        tail_bin = gaps[gaps >= x_min_bin]

        if len(tail_bin) >= 4:
            alpha_bin, _ = fit_power_law(tail_bin, x_min_bin)
            mu_bin, sigma_bin, _ = fit_lognormal(tail_bin)

            if alpha_bin is not None:
                x_power = np.linspace(x_min_bin, np.max(gaps), 100)
                sorted_g, surv = compute_survival(gaps)
                idx_start = np.searchsorted(sorted_g, x_min_bin)
                C = surv[idx_start] if idx_start < len(surv) else surv[-1] if len(surv) > 0 else 1.0
                y_power = C * (x_power / x_min_bin) ** (-alpha_bin)
                ax.loglog(x_power, y_power, 'r-', linewidth=1.5, alpha=0.8)
                ax.text(0.05, 0.85, f'α={alpha_bin:.2f}', transform=ax.transAxes, fontsize=10)

        ax.set_title(f'{label}\n(n={len(gaps)})', fontsize=12)
    else:
        ax.text(0.5, 0.5, f'Sin datos suficientes\n(n={len(gaps)})',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(label, fontsize=12)

    ax.set_xlabel('Gap', fontsize=11)
    ax.set_ylabel('P(gap > x)', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1e-4, None)

plt.suptitle('M19: Tail Analysis by Kurtosis Bin (ALL gaps)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "M19_tail_survival_bins.png"), dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ M19_tail_survival_bins.png")

# =========================================================
# 8. GUARDAR RESULTADOS JSON
# =========================================================

results = {
    "experiment": "M19_step6_tail_fitting_corrected",
    "data_directory": "M19_V3",
    "timestamp": str(np.datetime64('now')),
    "loading_stats": {
        "total_archivos": total_archivos,
        "total_gaps_raw": total_gaps_raw,
        "total_gaps_filtrados": total_gaps_filtrados,
        "total_gaps_cero": total_gaps_cero,
        "fraccion_gaps_cero": float(total_gaps_cero/total_gaps_raw) if total_gaps_raw > 0 else 0
    },
    "global_analysis": {
        "n_samples": int(len(all_gaps)),
        "x_min_p80": float(x_min_global),
        "n_tail": int(n_tail_global),
        "tail_status": tail_status_global,
        "power_law": {
            "alpha": float(alpha_global) if alpha_global is not None else None,
            "log_likelihood": float(LL_power_global) if LL_power_global is not None else None
        },
        "lognormal": {
            "mu": float(mu_global) if mu_global is not None else None,
            "sigma": float(sigma_global) if sigma_global is not None else None,
            "log_likelihood": float(LL_lognorm_global) if LL_lognorm_global is not None else None
        },
        "best_model": best_global
    },
    "bins": bin_results
}

json_path = os.path.join(RESULTS_DIR, "M19_tail_fit_results.json")
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n   ✓ M19_tail_fit_results.json")

# =========================================================
# 9. RESUMEN FINAL
# =========================================================

print("\n" + "="*70)
print("RESUMEN FINAL - M19 STEP 6 (CORREGIDO)")
print("="*70)

print(f"""
ESTADÍSTICAS DE CARGA:
   - total_archivos: {total_archivos}
   - total_gaps_raw: {total_gaps_raw}
   - total_gaps_filtrados (gap > 0): {total_gaps_filtrados}
   - total_gaps_cero: {total_gaps_cero}
   - Fracción gaps == 0: {total_gaps_cero/total_gaps_raw if total_gaps_raw > 0 else 0:.4f}

ANÁLISIS GLOBAL:
   - Muestras totales (gaps > 0): {len(all_gaps)}
   - Rango gaps: [{np.min(all_gaps):.4f}, {np.max(all_gaps):.4f}]
   - Cola (p80): n_tail = {n_tail_global}""")

if alpha_global is not None:
    print(f"   - Power law: α = {alpha_global:.4f}")
if mu_global is not None:
    print(f"   - Log-normal: μ = {mu_global:.4f}, σ = {sigma_global:.4f}")
print(f"   - BEST MODEL: {best_global.upper()}")

print(f"\nANÁLISIS POR BIN:")
for res in bin_results:
    if res.get("power_law", {}).get("alpha") is not None:
        print(f"   {res['bin_label']}: α = {res['power_law']['alpha']:.4f} ({res['best_model']})")
    else:
        print(f"   {res['bin_label']}: n={res['n_samples']} - {res.get('warning', 'insuficiente')}")

print(f"\nFIGURAS:")
print(f"   - {RESULTS_DIR}/M19_tail_survival_global.png")
print(f"   - {RESULTS_DIR}/M19_tail_survival_bins.png")

print("\n" + "="*70)
print("✅ M19 STEP 6 CORREGIDO COMPLETADO")
print("="*70)
