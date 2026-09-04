"""
run_final_exp_v1.py
===================

Versión limpia, documentada y reproducible del pipeline final empleado para
el modelo exponencial del TFM.

El script conserva las decisiones científicas y computacionales utilizadas
para producir las siete cadenas finales:

- b_exp_H0rd
- s_exp
- c_exp
- bs_exp_H0rd_sep
- cb_exp
- cs_exp
- cbs_exp

Procedencia de las implementaciones
-----------------------------------
1. BAO, SN, CB, CS y CBS reproducen la implementación usada en
   ``run_final_exp.py``.
2. La cadena CMB-only ``c_exp`` utiliza la implementación definitiva que
   consiguió converger:
   - cuadratura Gauss-Legendre de orden 256;
   - corte Omega_DE(z*) <= 1e-2;
   - covarianza aprendida en la ejecución interrumpida;
   - proposal_scale = 0.5;
   - learn_proposal = False;
   - Rminus1_stop = 0.005;
   - Rminus1_cl_stop = 0.2.

Los scripts históricos originales deben conservarse sin modificar. Este archivo
es la versión v1 preparada para lectura, reproducción y publicación.

Datos y rutas
-------------
Por defecto se buscan los datos en la carpeta del proyecto, pero la ruta puede
cambiarse mediante ``--data-dir``. Las variables observacionales se inicializan
como ``None`` hasta que se ejecuta ``load_late_time_data()``. Esto permite
importar el archivo sin imponer una organización concreta de carpetas.

Ejemplos
--------
Mostrar el procedimiento y las rutas esperadas, sin ejecutar cadenas:

    python run_final_exp_v1.py

Comprobar la configuración de una run:

    python run_final_exp_v1.py --check --run c_exp

Ejecutar una run en otra carpeta de salida:

    python run_final_exp_v1.py \
        --execute \
        --run c_exp \
        --data-dir /ruta/a/los/datos \
        --chains-final-dir /ruta/a/chains_final \
        --output-dir /ruta/a/reproduccion

Dependencias principales
------------------------
- Python 3.9+
- NumPy
- pandas
- SciPy
- Cobaya
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import argparse
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.integrate import IntegrationWarning, quad, solve_ivp
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expi, logsumexp


# =============================================================================
# Rutas y carga de datos
# =============================================================================

MODULE_DIR = Path(__file__).resolve().parent

# Permite situar el archivo tanto en la raíz del proyecto como en TFM/models/.
if (MODULE_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt").exists():
    PROJECT_DIR = MODULE_DIR
else:
    PROJECT_DIR = MODULE_DIR.parent

DATA_DIR = PROJECT_DIR

BAO_MEAN_FILE = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
BAO_COV_FILE = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"
SN_DATA_FILE = DATA_DIR / "PantheonSH0ES_unique_data.txt"
SN_COV_FILE = DATA_DIR / "PantheonSH0ES_unique_cov"

bao_data: pd.DataFrame | None = None
bao_obs: np.ndarray | None = None
bao_cov: np.ndarray | None = None
bao_cov_chol = None

sn_data: pd.DataFrame | None = None
sn_cov: np.ndarray | None = None
sn_z: np.ndarray | None = None
sn_mu_obs: np.ndarray | None = None
sn_cov_chol = None


def load_late_time_data(data_dir: str | Path | None = None) -> None:
    """Carga los datos BAO y Pantheon+SH0ES usados en las cadenas finales."""
    global DATA_DIR
    global BAO_MEAN_FILE, BAO_COV_FILE, SN_DATA_FILE, SN_COV_FILE
    global bao_data, bao_obs, bao_cov, bao_cov_chol
    global sn_data, sn_cov, sn_z, sn_mu_obs, sn_cov_chol

    if data_dir is not None:
        DATA_DIR = Path(data_dir).expanduser().resolve()
        BAO_MEAN_FILE = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
        BAO_COV_FILE = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"
        SN_DATA_FILE = DATA_DIR / "PantheonSH0ES_unique_data.txt"
        SN_COV_FILE = DATA_DIR / "PantheonSH0ES_unique_cov"

    missing = [
        path
        for path in (BAO_MEAN_FILE, BAO_COV_FILE, SN_DATA_FILE, SN_COV_FILE)
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "No se encontraron todos los datos necesarios:\n" + missing_text
        )

    bao_data = pd.read_csv(
        BAO_MEAN_FILE,
        comment="#",
        sep=r"\s+",
        header=None,
        names=["z", "value", "quantity"],
    )
    bao_obs = bao_data["value"].to_numpy(dtype=float)
    bao_cov = np.loadtxt(BAO_COV_FILE)
    bao_cov_chol = cho_factor(bao_cov, lower=True, check_finite=False)

    sn_data = pd.read_csv(SN_DATA_FILE, sep=r"\s+")
    sn_cov = np.loadtxt(SN_COV_FILE)
    sn_z = sn_data["zHD"].to_numpy(dtype=float)
    sn_mu_obs = sn_data["MU_SH0ES"].to_numpy(dtype=float)
    sn_cov_chol = cho_factor(sn_cov, lower=True, check_finite=False)


def _require_bao_data() -> None:
    if bao_data is None or bao_obs is None or bao_cov_chol is None:
        raise RuntimeError(
            "Los datos BAO no están cargados. Ejecuta load_late_time_data()."
        )


def _require_sn_data() -> None:
    if sn_z is None or sn_mu_obs is None or sn_cov_chol is None:
        raise RuntimeError(
            "Los datos SN no están cargados. Ejecuta load_late_time_data()."
        )


# Carga automática cuando los cuatro archivos están disponibles.
if all(path.exists() for path in (BAO_MEAN_FILE, BAO_COV_FILE, SN_DATA_FILE, SN_COV_FILE)):
    load_late_time_data()


# =============================================================================
# Datos comprimidos de CMB y constantes cosmológicas
# =============================================================================

cmb_obs = np.array([1.7504, 301.77, 0.022371])

cmb_cov = 1.0e-8 * np.array(
    [
        [1559.83, -1325.41, -36.45],
        [-1325.41, 714691.80, 269.77],
        [-36.45, 269.77, 2.10],
    ]
)

cmb_cov_chol = cho_factor(cmb_cov, lower=True, check_finite=False)

c_light = 299792.458  # km/s
T_CMB = 2.726  # K
N_eff = 3.046
z_star_cmb = 1090.0
H0_rad_fid = 70.0

OMEGA_DE_STAR_MAX_EXP = 1.0e-2
CMB_GL_ORDER_EXP = 256
CMB_Z_MAX = 1.0e7

LOG_FLOAT_MAX = np.log(np.finfo(float).max)


# =============================================================================
# Priors utilizados en las cadenas finales
# =============================================================================

EXP_OMEGA_M_PRIOR = {"min": 0.05, "max": 0.7}
EXP_H0_PRIOR = {"min": 50.0, "max": 100.0}
EXP_H0RD_PRIOR = {"min": 8000.0, "max": 12000.0}
EXP_OMEGA_B_PRIOR = {"min": 0.020, "max": 0.025}
EXP_W0_PRIOR = {"min": -3.0, "max": 0.5}
EXP_WE_PRIOR = {"min": -5.0, "max": 12.0}

FINAL_PRIORS = {
    "Omega_m": EXP_OMEGA_M_PRIOR,
    "H0": EXP_H0_PRIOR,
    "H0rd": EXP_H0RD_PRIOR,
    "omega_b": EXP_OMEGA_B_PRIOR,
    "w0": EXP_W0_PRIOR,
    "we": EXP_WE_PRIOR,
}


# =============================================================================
# Funciones cosmológicas del modelo exponencial
# =============================================================================


def h_from_H0(H0: float) -> float:
    """Devuelve h = H0 / 100 para H0 en km/s/Mpc."""
    return H0 / 100.0


def omega_r_from_h(h: float) -> float:
    """Densidad de radiación actual usada en las cadenas finales."""
    return 4.18e-5 / h**2


def exp_from_log_safe(log_x):
    """Exponencial estable que devuelve infinito ante overflow."""
    log_x = np.asarray(log_x, dtype=float)
    result = np.empty_like(log_x)
    overflow = log_x > LOG_FLOAT_MAX
    result[overflow] = np.inf
    result[~overflow] = np.exp(log_x[~overflow])

    if result.ndim == 0:
        return result.item()
    return result


def log_f_de_exp(z, w0: float, we: float):
    """Logaritmo de la evolución relativa de la densidad de energía oscura."""
    z = np.asarray(z, dtype=float)
    y = z / (1.0 + z)

    if abs(we) < 1.0e-8:
        return 3.0 * (1.0 + w0) * np.log1p(z)

    integral = np.log1p(z) + w0 * np.exp(we) * (
        expi(-we) - expi(-we * (1.0 - y))
    )
    return 3.0 * integral


def f_de_exp(z, w0: float, we: float):
    """Evolución relativa de la densidad de energía oscura."""
    return exp_from_log_safe(log_f_de_exp(z, w0, we))


def E_exp(z, Omega_m: float, H0: float, w0: float, we: float):
    """Función de expansión adimensional E(z) = H(z)/H0."""
    scalar_input = np.ndim(z) == 0
    z = np.asarray(z, dtype=float)

    h = h_from_H0(H0)
    Omega_r = omega_r_from_h(h)
    Omega_de = 1.0 - Omega_m - Omega_r

    if Omega_m <= 0.0 or Omega_r <= 0.0 or Omega_de <= 0.0:
        if scalar_input:
            return np.nan
        return np.full_like(z, np.nan, dtype=float)

    log1pz = np.log1p(z)
    log_radiation = np.log(Omega_r) + 4.0 * log1pz
    log_matter = np.log(Omega_m) + 3.0 * log1pz
    log_dark_energy = np.log(Omega_de) + log_f_de_exp(z, w0, we)

    log_E_squared = np.logaddexp(
        np.logaddexp(log_radiation, log_matter),
        log_dark_energy,
    )
    log_E = 0.5 * log_E_squared

    E = np.exp(np.minimum(log_E, LOG_FLOAT_MAX))
    E = np.where(log_E > LOG_FLOAT_MAX, np.inf, E)

    if scalar_input:
        return float(E)
    return E


def omega_de_exp(z, Omega_m: float, H0: float, w0: float, we: float):
    """Fracción de energía oscura calculada de forma estable."""
    h = h_from_H0(H0)
    Omega_r = omega_r_from_h(h)
    Omega_de = 1.0 - Omega_m - Omega_r

    if Omega_m <= 0.0 or Omega_r <= 0.0 or Omega_de <= 0.0:
        return np.nan

    log_1pz = np.log1p(z)
    log_radiation = np.log(Omega_r) + 4.0 * log_1pz
    log_matter = np.log(Omega_m) + 3.0 * log_1pz
    log_dark_energy = np.log(Omega_de) + log_f_de_exp(z, w0, we)
    log_total = logsumexp(
        [log_radiation, log_matter, log_dark_energy],
        axis=0,
    )

    return np.exp(log_dark_energy - log_total)


# =============================================================================
# Distancias cosmológicas
# =============================================================================


def chi_of_z(z: float, E_func: Callable, *E_params) -> float:
    """Distancia comóvil adimensional chi(z)."""
    integrand = lambda zp: 1.0 / E_func(zp, *E_params)
    return quad(integrand, 0.0, z)[0]


def DM_of_z(z: float, H0: float, E_func: Callable, *E_params) -> float:
    """Distancia comóvil transversal D_M(z) en Mpc."""
    return (c_light / H0) * chi_of_z(z, E_func, *E_params)


def DH_of_z(z: float, H0: float, E_func: Callable, *E_params) -> float:
    """Distancia de Hubble D_H(z) en Mpc."""
    return c_light / (H0 * E_func(z, *E_params))


def DV_of_z(z: float, H0: float, E_func: Callable, *E_params) -> float:
    """Distancia BAO isotrópica D_V(z) en Mpc."""
    DM = DM_of_z(z, H0, E_func, *E_params)
    DH = DH_of_z(z, H0, E_func, *E_params)
    return (z * DM**2 * DH) ** (1.0 / 3.0)


def DL_of_z(z: float, H0: float, E_func: Callable, *E_params) -> float:
    """Distancia de luminosidad D_L(z) en Mpc."""
    return (1.0 + z) * DM_of_z(z, H0, E_func, *E_params)


# =============================================================================
# Física temprana y horizonte sonoro
# =============================================================================


def omega_m_physical(Omega_m: float, H0: float) -> float:
    """Densidad física de materia omega_m = Omega_m h^2."""
    return Omega_m * h_from_H0(H0) ** 2


def rd_approx(omega_m: float, omega_b: float, N_eff: float = N_eff) -> float:
    """Aproximación DESI para el horizonte sonoro en el drag epoch."""
    return (
        147.05
        * (omega_m / 0.1432) ** (-0.23)
        * (N_eff / 3.04) ** (-0.1)
        * (omega_b / 0.02236) ** (-0.13)
    )


def baryon_loading(omega_b: float) -> float:
    """Factor de carga bariónica usado en la velocidad del sonido."""
    return 31500.0 * omega_b * (T_CMB / 2.7) ** (-4.0)


def sound_speed_over_c(z, omega_b: float):
    """Velocidad del sonido del plasma fotón-barión en unidades de c."""
    Rb = baryon_loading(omega_b)
    return 1.0 / np.sqrt(3.0 * (1.0 + Rb / (1.0 + z)))


def rs_sound_horizon(
    z: float,
    H0: float,
    omega_b: float,
    E_func: Callable,
    *E_params,
    z_max: float = CMB_Z_MAX,
) -> float:
    """Horizonte sonoro comóvil r_s(z) en Mpc mediante quad."""
    integrand = lambda zp: sound_speed_over_c(zp, omega_b) / E_func(
        zp, *E_params
    )
    integral = quad(integrand, z, z_max)[0]
    return (c_light / H0) * integral


# =============================================================================
# Observables BAO
# =============================================================================


def rd_from_cosmology(
    Omega_m: float,
    H0: float,
    omega_b: float,
    N_eff: float = N_eff,
) -> float:
    """Horizonte sonoro r_d calculado desde parámetros físicos."""
    return rd_approx(
        omega_m=omega_m_physical(Omega_m, H0),
        omega_b=omega_b,
        N_eff=N_eff,
    )


def DM_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params):
    return DM_of_z(z, H0, E_func, *E_params) / rd_from_cosmology(
        Omega_m, H0, omega_b
    )


def DH_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params):
    return DH_of_z(z, H0, E_func, *E_params) / rd_from_cosmology(
        Omega_m, H0, omega_b
    )


def DV_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params):
    return DV_of_z(z, H0, E_func, *E_params) / rd_from_cosmology(
        Omega_m, H0, omega_b
    )


def bao_theory_vector(Omega_m, H0, omega_b, E_func, *E_params):
    """Vector BAO de la rama física, en el orden del archivo DESI."""
    _require_bao_data()
    model = []

    for _, row in bao_data.iterrows():
        z = row["z"]
        quantity = row["quantity"]

        if quantity == "DM_over_rs":
            value = DM_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params)
        elif quantity == "DH_over_rs":
            value = DH_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params)
        elif quantity == "DV_over_rs":
            value = DV_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params)
        else:
            raise ValueError(f"Observable BAO no reconocido: {quantity}")

        model.append(value)

    return np.asarray(model, dtype=float)


def bao_theory_vector_H0rd(H0rd, E_func, *E_params):
    """Vector BAO de la rama efectiva H0*r_d."""
    _require_bao_data()
    model = []

    for _, row in bao_data.iterrows():
        z = row["z"]
        quantity = row["quantity"]
        chi = chi_of_z(z, E_func, *E_params)
        Ez = E_func(z, *E_params)

        if quantity == "DM_over_rs":
            value = (c_light / H0rd) * chi
        elif quantity == "DH_over_rs":
            value = (c_light / H0rd) / Ez
        elif quantity == "DV_over_rs":
            value = (c_light / H0rd) * (z * chi**2 / Ez) ** (1.0 / 3.0)
        else:
            raise ValueError(f"Observable BAO no reconocido: {quantity}")

        model.append(value)

    return np.asarray(model, dtype=float)


# =============================================================================
# Observables de supernovas
# =============================================================================


def sn_theory_vector_fast(
    H0,
    E_func,
    *E_params,
    rtol=1.0e-10,
    atol=1.0e-12,
):
    """
    Vector de módulos de distancia usado en las cadenas exponenciales.

    Resuelve una sola vez dchi/dz = 1/E(z) mediante DOP853 y evalúa la
    solución en todos los redshifts de Pantheon+SH0ES.
    """
    _require_sn_data()
    z_values = np.asarray(sn_z, dtype=float)

    if np.any(~np.isfinite(z_values)):
        raise ValueError("sn_z contiene valores no finitos")
    if np.any(z_values <= 0.0):
        raise ValueError("Todos los redshifts de SNIa deben ser positivos")

    unique_z, inverse_indices = np.unique(z_values, return_inverse=True)

    def dchi_dz(z, chi):
        Ez = float(np.asarray(E_func(z, *E_params)))
        if not np.isfinite(Ez) or Ez <= 0.0:
            raise ValueError(f"E(z) no válida en z={z}: {Ez}")
        return np.array([1.0 / Ez])

    solution = solve_ivp(
        fun=dchi_dz,
        t_span=(0.0, float(unique_z[-1])),
        y0=np.array([0.0]),
        t_eval=unique_z,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(
            "Falló la integración rápida de SNIa: " + solution.message
        )

    chi_values = solution.y[0][inverse_indices]
    luminosity_distance = (
        c_light / H0 * (1.0 + z_values) * chi_values
    )

    if np.any(~np.isfinite(luminosity_distance)):
        raise ValueError("Se obtuvieron distancias de luminosidad no finitas")
    if np.any(luminosity_distance <= 0.0):
        raise ValueError("Se obtuvieron distancias de luminosidad no positivas")

    return 5.0 * np.log10(luminosity_distance) + 25.0


# =============================================================================
# CMB de referencia: utilizada en CB, CS y CBS
# =============================================================================


def cmb_distance_prior_theory_vector(
    Omega_m,
    H0,
    omega_b,
    E_func,
    *E_params,
):
    """Vector (R, ell_A, omega_b) mediante integración adaptativa original."""
    DM_star = DM_of_z(z_star_cmb, H0, E_func, *E_params)
    rs_star = rs_sound_horizon(
        z_star_cmb,
        H0,
        omega_b,
        E_func,
        *E_params,
    )

    R = np.sqrt(Omega_m) * H0 * DM_star / c_light
    ell_A = np.pi * DM_star / rs_star
    return np.array([R, ell_A, omega_b])


# =============================================================================
# CMB optimizada: utilizada exclusivamente en la cadena final c_exp
# =============================================================================


def _build_cmb_gl_cache(order: int = CMB_GL_ORDER_EXP) -> dict[str, np.ndarray]:
    """Construye los nodos y pesos finales de Gauss-Legendre."""
    nodes, weights = leggauss(order)
    x_star = np.log1p(z_star_cmb)
    x_max = np.log1p(CMB_Z_MAX)

    x_chi = 0.5 * x_star * (nodes + 1.0)
    w_chi = 0.5 * x_star * weights

    x_rs = 0.5 * (x_max - x_star) * (nodes + 1.0) + x_star
    w_rs = 0.5 * (x_max - x_star) * weights

    return {
        "x_chi": x_chi,
        "w_chi": w_chi,
        "x_rs": x_rs,
        "w_rs": w_rs,
    }


CMB_GL_CACHE_EXP = _build_cmb_gl_cache()


def cmb_background_gl_exp(omega_b, E_func, *E_params):
    """Integrales CMB con Gauss-Legendre de orden 256 y x = ln(1+z)."""
    cache = CMB_GL_CACHE_EXP

    x_chi = cache["x_chi"]
    z_chi = np.expm1(x_chi)
    E_chi = np.asarray(E_func(z_chi, *E_params), dtype=float)

    if np.any(~np.isfinite(E_chi)) or np.any(E_chi <= 0.0):
        raise ValueError("E(z) no válida en la integral de distancia CMB.")

    chi_star = np.sum(cache["w_chi"] * np.exp(x_chi) / E_chi)

    x_rs = cache["x_rs"]
    z_rs = np.expm1(x_rs)
    E_rs = np.asarray(E_func(z_rs, *E_params), dtype=float)

    if np.any(~np.isfinite(E_rs)) or np.any(E_rs <= 0.0):
        raise ValueError(
            "E(z) no válida en la integral del horizonte sonoro."
        )

    rs_dimensionless = np.sum(
        cache["w_rs"]
        * np.exp(x_rs)
        * sound_speed_over_c(z_rs, omega_b)
        / E_rs
    )

    return float(chi_star), float(rs_dimensionless)


def cmb_distance_prior_theory_vector_gl_exp(
    Omega_m,
    H0,
    omega_b,
    E_func,
    *E_params,
):
    """Vector CMB optimizado utilizado en c_exp."""
    chi_star, rs_dimensionless = cmb_background_gl_exp(
        omega_b,
        E_func,
        *E_params,
    )

    DM_star = c_light / H0 * chi_star
    rs_star = c_light / H0 * rs_dimensionless
    R = np.sqrt(Omega_m) * chi_star
    ell_A = np.pi * DM_star / rs_star

    return np.array([R, ell_A, omega_b])


def cmb_theory_exp_opt(Omega_m, H0, omega_b, w0, we):
    """Predicción CMB GL256 para el modelo exponencial."""
    return cmb_distance_prior_theory_vector_gl_exp(
        Omega_m,
        H0,
        omega_b,
        E_exp,
        Omega_m,
        H0,
        w0,
        we,
    )


# =============================================================================
# Chi cuadrado
# =============================================================================


def chi2_gaussian(data, theory, cov_chol):
    """Chi cuadrado gaussiano con la Cholesky usada en las cadenas finales."""
    delta = data - theory
    solved = cho_solve(cov_chol, delta, check_finite=False)
    return delta @ solved


def chi2_bao(Omega_m, H0, omega_b, E_func, *E_params):
    _require_bao_data()
    theory = bao_theory_vector(Omega_m, H0, omega_b, E_func, *E_params)
    return chi2_gaussian(bao_obs, theory, bao_cov_chol)


def chi2_bao_H0rd(H0rd, E_func, *E_params):
    _require_bao_data()
    theory = bao_theory_vector_H0rd(H0rd, E_func, *E_params)
    return chi2_gaussian(bao_obs, theory, bao_cov_chol)


def chi2_sn(H0, E_func, *E_params):
    _require_sn_data()
    theory = sn_theory_vector_fast(H0, E_func, *E_params)
    return chi2_gaussian(sn_mu_obs, theory, sn_cov_chol)


def chi2_cmb(Omega_m, H0, omega_b, E_func, *E_params):
    theory = cmb_distance_prior_theory_vector(
        Omega_m,
        H0,
        omega_b,
        E_func,
        *E_params,
    )
    return chi2_gaussian(cmb_obs, theory, cmb_cov_chol)


# =============================================================================
# Likelihoods BAO y SN utilizadas en las cadenas finales
# =============================================================================


def loglike_bao_exp(Omega_m, H0, omega_b, w0, we):
    return -0.5 * chi2_bao(
        Omega_m,
        H0,
        omega_b,
        E_exp,
        Omega_m,
        H0,
        w0,
        we,
    )


def loglike_bao_exp_H0rd(Omega_m, H0rd, w0, we):
    return -0.5 * chi2_bao_H0rd(
        H0rd,
        E_exp,
        Omega_m,
        H0_rad_fid,
        w0,
        we,
    )


def loglike_sn_exp(Omega_m, H0, w0, we):
    return -0.5 * chi2_sn(
        H0,
        E_exp,
        Omega_m,
        H0,
        w0,
        we,
    )


# =============================================================================
# CMB de referencia: utilizada en las combinaciones CB, CS y CBS
# =============================================================================


def loglike_cmb_exp(Omega_m, H0, omega_b, w0, we):
    """Likelihood CMB original sin aplicar el corte."""
    return -0.5 * chi2_cmb(
        Omega_m,
        H0,
        omega_b,
        E_exp,
        Omega_m,
        H0,
        w0,
        we,
    )


def loglike_cmb_exp_with_cut(
    Omega_m,
    H0,
    omega_b,
    w0,
    we,
    omega_de_star_max,
):
    """Likelihood CMB original con el corte de energía oscura temprana."""
    omega_de_star = omega_de_exp(
        z_star_cmb,
        Omega_m,
        H0,
        w0,
        we,
    )

    if not np.isfinite(omega_de_star) or omega_de_star > omega_de_star_max:
        return -np.inf

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            warnings.filterwarnings("error", category=IntegrationWarning)
            loglike = loglike_cmb_exp(Omega_m, H0, omega_b, w0, we)
    except (
        RuntimeWarning,
        IntegrationWarning,
        FloatingPointError,
        OverflowError,
        ValueError,
    ):
        return -np.inf

    if not np.isfinite(loglike):
        return -np.inf
    return loglike


def loglike_cmb_exp_reference_with_cut(Omega_m, H0, omega_b, w0, we):
    """CMB original usada en cb_exp, cs_exp y cbs_exp."""
    return loglike_cmb_exp_with_cut(
        Omega_m,
        H0,
        omega_b,
        w0,
        we,
        OMEGA_DE_STAR_MAX_EXP,
    )


# Nombre histórico empleado por las configuraciones originales combinadas.
loglike_cmb_exp_loose = loglike_cmb_exp_reference_with_cut


# =============================================================================
# CMB final optimizada: utilizada en c_exp
# =============================================================================


def loglike_cmb_exp_with_cut_opt(Omega_m, H0, omega_b, w0, we):
    """
    Likelihood CMB definitiva de c_exp.

    Conserva el corte Omega_DE(z*) <= 1e-2 y usa la cuadratura fija
    Gauss-Legendre de orden 256 validada en opt_result.ipynb.
    """
    omega_de_star = omega_de_exp(
        z_star_cmb,
        Omega_m,
        H0,
        w0,
        we,
    )

    if (
        not np.isfinite(omega_de_star)
        or omega_de_star > OMEGA_DE_STAR_MAX_EXP
    ):
        return -np.inf

    theory = cmb_theory_exp_opt(Omega_m, H0, omega_b, w0, we)
    return -0.5 * chi2_gaussian(cmb_obs, theory, cmb_cov_chol)


# Alias semántico para el único resultado CMB-only final.
loglike_cmb_exp_final = loglike_cmb_exp_with_cut_opt


# =============================================================================
# Covarianzas de propuesta y configuración reproducible
# =============================================================================

FINAL_RMINUS1_STOP = 0.005

# Estas funciones no modifican los scripts históricos. Preparan únicamente
# las rutas y configuraciones necesarias para reproducir las cadenas.


def require_covmat(path: str | Path) -> Path:
    """
    Comprueba que existe una covarianza de propuesta.

    Las covmats forman parte del procedimiento seguido: no se sustituyen
    silenciosamente por propuestas diagonales porque eso cambiaría el proceso
    de muestreo.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            "No existe la covmat de propuesta requerida:\n"
            f"  {path}\n"
            "Indica la carpeta correcta mediante --chains-final-dir."
        )

    return path


def prepare_covmat_with_renamed_parameter(
    source_path: str | Path,
    destination_path: str | Path,
    old_name: str,
    new_name: str,
) -> Path:
    """
    Copia una covmat y sustituye solo el nombre indicado en su cabecera.

    Se reproduce así la preparación usada para BS exponencial, donde se tomó
    la covmat final de CPLn2 y se cambió ``w2`` por ``we`` sin modificar la
    matriz numérica.
    """
    source_path = require_covmat(source_path)
    destination_path = Path(destination_path).expanduser().resolve()

    lines = source_path.read_text(encoding="utf-8").splitlines()

    if not lines:
        raise ValueError(f"La covmat está vacía: {source_path}")

    header_tokens = [
        token
        for token in lines[0].split()
        if token != "#"
    ]
    prefix = "# " if lines[0].lstrip().startswith("#") else ""

    if old_name in header_tokens:
        header_tokens = [
            new_name if token == old_name else token
            for token in header_tokens
        ]
    elif new_name not in header_tokens:
        raise ValueError(
            f"La cabecera de {source_path} no contiene "
            f"ni {old_name!r} ni {new_name!r}: {lines[0]}"
        )

    lines[0] = prefix + " ".join(header_tokens)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return destination_path


def build_covmat_files(
    chains_final_dir: str | Path,
    *,
    validate: bool = True,
) -> dict:
    """
    Construye el mapa de covmats usado por las siete cadenas exponenciales.

    ``chains_final_dir`` es la carpeta raíz que contiene:
    - final/
    - prior_tests/
    - sampler_tests/
    """
    chains_final_dir = Path(chains_final_dir).expanduser().resolve()

    final_dir = chains_final_dir / "final"
    prior_tests_dir = chains_final_dir / "prior_tests"
    sampler_tests_dir = chains_final_dir / "sampler_tests"

    covmat_paths = {
        "BAO": prior_tests_dir / "b_exp_H0rd_priorwide.covmat",
        "SN": prior_tests_dir / "s_exp_priorcheck_fast.covmat",
        # La cadena CMB-only final usa la covmat aprendida durante
        # la ejecución interrumpida y archivada.
        "CMB": sampler_tests_dir / "c_exp_interrupted.covmat",
        "BAO+SN": (
            prior_tests_dir
            / "bs_CPLn2_H0rd_sep_we_for_exp.covmat"
        ),
        "CMB+BAO": prior_tests_dir / "cb_exp_omegaDE1e-2.covmat",
        "CMB+SN": prior_tests_dir / "cbs_exp_priorcheck.covmat",
        "CMB+BAO+SN": prior_tests_dir / "cbs_exp_priorcheck.covmat",
    }

    if not validate:
        return {
            key: path.expanduser().resolve()
            for key, path in covmat_paths.items()
        }

    covmat_paths["BAO+SN"] = prepare_covmat_with_renamed_parameter(
        final_dir / "bs_CPLn2_H0rd_sep.covmat",
        covmat_paths["BAO+SN"],
        old_name="w2",
        new_name="we",
    )

    return {
        key: require_covmat(path)
        for key, path in covmat_paths.items()
    }


def build_run_infos(
    chains_final_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    validate_covmats: bool = True,
    create_output_dir: bool = True,
) -> dict[str, dict]:
    """
    Construye las siete configuraciones de Cobaya.

    Las seis configuraciones distintas de ``c_exp`` conservan la estructura
    original. ``c_exp`` incorpora directamente la likelihood GL256 y la
    configuración final del sampler que produjo la cadena convergida.
    """
    chains_final_dir = Path(chains_final_dir).expanduser().resolve()

    if output_dir is None:
        output_dir = chains_final_dir / "final"
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    if create_output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    covmats = build_covmat_files(
        chains_final_dir,
        validate=validate_covmats,
    )

    info_b_exp_H0rd = {
        "likelihood": {
            "bao_exp_H0rd": loglike_bao_exp_H0rd,
        },
        "params": {
            "Omega_m": {
                "prior": EXP_OMEGA_M_PRIOR,
                "ref": 0.30,
                "proposal": 0.01,
                "latex": r"\Omega_m",
            },
            "H0rd": {
                "prior": EXP_H0RD_PRIOR,
                "ref": 10000.0,
                "proposal": 50.0,
                "latex": r"H_0 r_d",
            },
            "w0": {
                "prior": EXP_W0_PRIOR,
                "ref": -0.9,
                "proposal": 0.08,
                "latex": r"w_0",
            },
            "we": {
                "prior": EXP_WE_PRIOR,
                "ref": 2.0,
                "proposal": 0.4,
                "latex": r"w_{\rm e}",
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["BAO"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000,
            }
        },
        "output": str(output_dir / "b_exp_H0rd"),
    }

    info_s_exp = {
        "likelihood": {
            "sn_exp": loglike_sn_exp,
        },
        "params": {
            "Omega_m": {
                "prior": EXP_OMEGA_M_PRIOR,
                "ref": 0.30,
                "proposal": 0.01,
                "latex": r"\Omega_m",
            },
            "H0": {
                "prior": EXP_H0_PRIOR,
                "ref": 73.0,
                "proposal": 0.2,
                "latex": r"H_0",
            },
            "w0": {
                "prior": EXP_W0_PRIOR,
                "ref": -0.9,
                "proposal": 0.08,
                "latex": r"w_0",
            },
            "we": {
                "prior": EXP_WE_PRIOR,
                "ref": 2.0,
                "proposal": 0.4,
                "latex": r"w_{\rm e}",
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000,
            }
        },
        "output": str(output_dir / "s_exp"),
    }

    # Configuración definitiva de c_exp.
    #
    # La etiqueta interna "cmb_exp_loose" se conserva para mantener la
    # continuidad con los metadatos de la run histórica. La función externa
    # real es la versión optimizada GL256 con el corte early-DE incorporado.
    info_c_exp = {
        "likelihood": {
            "cmb_exp_loose": {
                "external": loglike_cmb_exp_with_cut_opt,
                "input_params": [
                    "Omega_m",
                    "H0",
                    "omega_b",
                    "w0",
                    "we",
                ],
            },
        },
        "params": {
            "Omega_m": {
                "prior": EXP_OMEGA_M_PRIOR,
                "ref": 0.30,
                "proposal": 0.01,
                "latex": r"\Omega_m",
            },
            "H0": {
                "prior": EXP_H0_PRIOR,
                "ref": 70.0,
                "proposal": 0.5,
                "latex": r"H_0",
            },
            "omega_b": {
                "prior": EXP_OMEGA_B_PRIOR,
                "ref": 0.0224,
                "proposal": 0.0001,
                "latex": r"\omega_b",
            },
            "w0": {
                "prior": EXP_W0_PRIOR,
                "ref": -0.9,
                "proposal": 0.08,
                "latex": r"w_0",
            },
            "we": {
                "prior": EXP_WE_PRIOR,
                "ref": 0.0,
                "proposal": 0.4,
                "latex": r"w_{\rm e}",
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB"]),
                "proposal_scale": 0.5,
                "learn_proposal": False,
                "Rminus1_stop": 0.005,
                "Rminus1_cl_stop": 0.2,
                "max_tries": 10000,
                "burn_in": 0,
            }
        },
        "output": str(output_dir / "c_exp"),
    }

    info_bs_exp_H0rd = {
        "likelihood": {
            "bao_exp_H0rd": loglike_bao_exp_H0rd,
            "sn_exp": loglike_sn_exp,
        },
        "params": {
            "Omega_m": {
                "prior": EXP_OMEGA_M_PRIOR,
                "ref": 0.30,
                "proposal": 0.01,
                "latex": r"\Omega_m",
            },
            "H0": {
                "prior": EXP_H0_PRIOR,
                "ref": 73.0,
                "proposal": 0.2,
                "latex": r"H_0",
            },
            "H0rd": {
                "prior": EXP_H0RD_PRIOR,
                "ref": 10000.0,
                "proposal": 50.0,
                "latex": r"H_0 r_d",
            },
            "w0": {
                "prior": EXP_W0_PRIOR,
                "ref": -0.9,
                "proposal": 0.08,
                "latex": r"w_0",
            },
            "we": {
                "prior": EXP_WE_PRIOR,
                "ref": 2.0,
                "proposal": 0.4,
                "latex": r"w_{\rm e}",
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["BAO+SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000,
            }
        },
        "output": str(output_dir / "bs_exp_H0rd_sep"),
    }

    info_cb_exp = {
        "likelihood": {
            "cmb_exp_loose": loglike_cmb_exp_reference_with_cut,
            "bao_exp": loglike_bao_exp,
        },
        "params": {
            "Omega_m": {
                "prior": EXP_OMEGA_M_PRIOR,
                "ref": 0.32,
                "proposal": 0.01,
                "latex": r"\Omega_m",
            },
            "H0": {
                "prior": EXP_H0_PRIOR,
                "ref": 65.0,
                "proposal": 0.8,
                "latex": r"H_0",
            },
            "omega_b": {
                "prior": EXP_OMEGA_B_PRIOR,
                "ref": 0.0224,
                "proposal": 0.0001,
                "latex": r"\omega_b",
            },
            "w0": {
                "prior": EXP_W0_PRIOR,
                "ref": -0.5,
                "proposal": 0.15,
                "latex": r"w_0",
            },
            "we": {
                "prior": EXP_WE_PRIOR,
                "ref": 2.0,
                "proposal": 0.5,
                "latex": r"w_{\rm e}",
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB+BAO"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000,
            }
        },
        "output": str(output_dir / "cb_exp"),
    }

    info_cs_exp = {
        "likelihood": {
            "cmb_exp_loose": loglike_cmb_exp_reference_with_cut,
            "sn_exp": loglike_sn_exp,
        },
        "params": {
            "Omega_m": {
                "prior": EXP_OMEGA_M_PRIOR,
                "ref": 0.273,
                "proposal": 0.005,
                "latex": r"\Omega_m",
            },
            "H0": {
                "prior": EXP_H0_PRIOR,
                "ref": 72.2,
                "proposal": 0.3,
                "latex": r"H_0",
            },
            "omega_b": {
                "prior": EXP_OMEGA_B_PRIOR,
                "ref": 0.02248,
                "proposal": 0.0001,
                "latex": r"\omega_b",
            },
            "w0": {
                "prior": EXP_W0_PRIOR,
                "ref": -0.6,
                "proposal": 0.05,
                "latex": r"w_0",
            },
            "we": {
                "prior": EXP_WE_PRIOR,
                "ref": 2.5,
                "proposal": 0.25,
                "latex": r"w_{\rm e}",
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB+SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000,
            }
        },
        "output": str(output_dir / "cs_exp"),
    }

    info_cbs_exp = {
        "likelihood": {
            "cmb_exp_loose": loglike_cmb_exp_reference_with_cut,
            "bao_exp": loglike_bao_exp,
            "sn_exp": loglike_sn_exp,
        },
        "params": {
            "Omega_m": {
                "prior": EXP_OMEGA_M_PRIOR,
                "ref": 0.273,
                "proposal": 0.005,
                "latex": r"\Omega_m",
            },
            "H0": {
                "prior": EXP_H0_PRIOR,
                "ref": 72.2,
                "proposal": 0.3,
                "latex": r"H_0",
            },
            "omega_b": {
                "prior": EXP_OMEGA_B_PRIOR,
                "ref": 0.02248,
                "proposal": 0.0001,
                "latex": r"\omega_b",
            },
            "w0": {
                "prior": EXP_W0_PRIOR,
                "ref": -0.6,
                "proposal": 0.05,
                "latex": r"w_0",
            },
            "we": {
                "prior": EXP_WE_PRIOR,
                "ref": 2.5,
                "proposal": 0.25,
                "latex": r"w_{\rm e}",
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB+BAO+SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000,
            }
        },
        "output": str(output_dir / "cbs_exp"),
    }

    return {
        "b_exp_H0rd": info_b_exp_H0rd,
        "s_exp": info_s_exp,
        "c_exp": info_c_exp,
        "bs_exp_H0rd_sep": info_bs_exp_H0rd,
        "cb_exp": info_cb_exp,
        "cs_exp": info_cs_exp,
        "cbs_exp": info_cbs_exp,
    }


# =============================================================================
# Registro del procedimiento utilizado
# =============================================================================

FINAL_RUN_ORDER = (
    "cbs_exp",
    "cs_exp",
    "bs_exp_H0rd_sep",
    "cb_exp",
    "s_exp",
    "c_exp",
    "b_exp_H0rd",
)

FINAL_CHAIN_IMPLEMENTATIONS = {
    "b_exp_H0rd": {
        "datasets": ("BAO",),
        "implementation": "BAO H0rd original",
        "sampler": "MCMC adaptativo de Cobaya",
    },
    "s_exp": {
        "datasets": ("SN",),
        "implementation": "SN solve_ivp conjunta + chi2 gaussiano original",
        "sampler": "MCMC adaptativo de Cobaya",
    },
    "bs_exp_H0rd_sep": {
        "datasets": ("BAO", "SN"),
        "implementation": "BAO H0rd original + SN solve_ivp conjunta",
        "sampler": "MCMC adaptativo de Cobaya",
    },
    "cb_exp": {
        "datasets": ("CMB", "BAO"),
        "implementation": "CMB quad original con corte + BAO físico original",
        "sampler": "MCMC adaptativo de Cobaya",
    },
    "cs_exp": {
        "datasets": ("CMB", "SN"),
        "implementation": "CMB quad original con corte + SN solve_ivp conjunta",
        "sampler": "MCMC adaptativo de Cobaya",
    },
    "cbs_exp": {
        "datasets": ("CMB", "BAO", "SN"),
        "implementation": (
            "CMB quad original con corte + BAO físico original "
            "+ SN solve_ivp conjunta"
        ),
        "sampler": "MCMC adaptativo de Cobaya",
    },
    "c_exp": {
        "datasets": ("CMB",),
        "implementation": "CMB Gauss-Legendre 256 con corte early-DE",
        "sampler": (
            "covmat aprendida, proposal_scale=0.5, "
            "learn_proposal=False"
        ),
        "convergence": {
            "accepted_samples": 3567400,
            "acceptance_rate": 0.026,
            "Rminus1_means": 0.0046881337948928,
            "Rminus1_bounds": 0.039093,
        },
    },
}


def print_procedure_summary(
    run_infos: dict[str, dict],
    selected_runs: list[str],
    data_dir: str | Path,
) -> None:
    """Muestra el procedimiento y la configuración sin ejecutar cadenas."""
    print("=" * 88)
    print("MODELO EXPONENCIAL — PIPELINE FINAL V1")
    print("=" * 88)
    print(f"Directorio de datos: {Path(data_dir).expanduser().resolve()}")
    print("\nRuns seleccionadas:")

    for run_name in selected_runs:
        implementation = FINAL_CHAIN_IMPLEMENTATIONS[run_name]
        info = run_infos[run_name]
        sampler = info["sampler"]["mcmc"]

        print(f"\n- {run_name}")
        print(f"  Datos: {', '.join(implementation['datasets'])}")
        print(f"  Implementación: {implementation['implementation']}")
        print(f"  Sampler: {implementation['sampler']}")
        print(f"  Covmat: {sampler['covmat']}")
        print(f"  Output: {info['output']}")

        if run_name == "c_exp":
            print(
                "  Criterios: "
                f"Rminus1_stop={sampler['Rminus1_stop']}, "
                f"Rminus1_cl_stop={sampler['Rminus1_cl_stop']}"
            )


def format_elapsed_time(seconds: float) -> str:
    """Formatea un tiempo en segundos como texto legible."""
    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.2f} s"

    minutes, seconds = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)} min {seconds:.1f} s"

    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} h {int(minutes)} min {seconds:.1f} s"


def run_with_timer(
    info: dict,
    run_name: str,
    *,
    test: bool = False,
) -> tuple:
    """
    Ejecuta o comprueba una configuración de Cobaya.

    La importación se realiza aquí para que el archivo pueda leerse e
    importarse en entornos sin Cobaya, siempre que no se solicite una run.
    """
    try:
        from cobaya.run import run
    except ImportError as error:
        raise ImportError(
            "No se encuentra Cobaya. Instálalo antes de ejecutar o "
            "comprobar las cadenas."
        ) from error

    run_info = info

    if test:
        from copy import deepcopy

        run_info = deepcopy(info)
        run_info.pop("output", None)

    start = time.time()

    result = run(
        run_info,
        test=test,
        no_mpi=True,
        force=False,
    )

    elapsed = time.time() - start
    action = "CHECK" if test else "RUN"
    print(
        f"{action} {run_name} completado en "
        f"{format_elapsed_time(elapsed)}"
    )

    return result


def resolve_selected_runs(
    requested_runs: list[str] | None,
) -> list[str]:
    """Resuelve ``all`` o una lista explícita conservando el orden final."""
    if not requested_runs or "all" in requested_runs:
        return list(FINAL_RUN_ORDER)

    requested_set = set(requested_runs)

    return [
        run_name
        for run_name in FINAL_RUN_ORDER
        if run_name in requested_set
    ]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce o inspecciona las cadenas finales "
            "del modelo exponencial."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=(
            "Carpeta con los cuatro archivos BAO/SN. "
            "Por defecto: carpeta del proyecto."
        ),
    )

    parser.add_argument(
        "--chains-final-dir",
        type=Path,
        default=PROJECT_DIR / "chains_final",
        help=(
            "Carpeta raíz con final/, prior_tests/ y sampler_tests/."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Carpeta de salida. Por defecto se usa "
            "<chains-final-dir>/final."
        ),
    )

    parser.add_argument(
        "--run",
        action="append",
        choices=("all",) + FINAL_RUN_ORDER,
        help=(
            "Run que se quiere describir, comprobar o ejecutar. "
            "Puede repetirse. Sin --run se muestran todas."
        ),
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--check",
        action="store_true",
        help="Valida la configuración con Cobaya sin iniciar el muestreo.",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
        help="Ejecuta las runs seleccionadas.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    selected_runs = resolve_selected_runs(args.run)

    if args.execute and not args.run:
        raise SystemExit(
            "Para evitar ejecutar las siete cadenas por accidente, "
            "usa al menos un argumento --run o --run all."
        )

    active_mode = args.check or args.execute

    # En modo descriptivo no se exige que los datos o las covmats estén
    # presentes: se muestran las rutas esperadas. Para comprobar o ejecutar
    # una run sí se valida todo y los None se sustituyen por datos reales.
    if active_mode:
        load_late_time_data(args.data_dir)

    run_infos = build_run_infos(
        chains_final_dir=args.chains_final_dir,
        output_dir=args.output_dir,
        validate_covmats=active_mode,
        create_output_dir=active_mode,
    )

    print_procedure_summary(
        run_infos,
        selected_runs,
        data_dir=args.data_dir,
    )

    if not active_mode:
        print(
            "\nModo descriptivo: no se ha ejecutado ninguna cadena.\n"
            "Usa --check para validar o --execute para iniciar el muestreo."
        )
        return

    status = []

    for run_name in selected_runs:
        print("\n" + "=" * 88)
        print(
            "COMPROBANDO" if args.check else "EJECUTANDO",
            run_name,
        )
        print("=" * 88)

        try:
            run_with_timer(
                run_infos[run_name],
                run_name,
                test=args.check,
            )

            status.append({
                "run": run_name,
                "status": "ok",
            })

        except Exception as error:
            traceback.print_exc()

            status.append({
                "run": run_name,
                "status": "failed",
                "error": repr(error),
            })

            # En check se revisan todas las configuraciones.
            # En ejecución también se conserva el comportamiento histórico:
            # una run fallida no impide intentar la siguiente.

    print("\n" + "=" * 88)
    print("RESUMEN")
    print("=" * 88)

    for item in status:
        print(item)


if __name__ == "__main__":
    main()
