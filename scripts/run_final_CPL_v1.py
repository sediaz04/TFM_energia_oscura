"""
run_final_CPL_v1.py
===================

Versión limpia, documentada y reproducible del pipeline final empleado para
el modelo CPL.

El script reproduce las siete combinaciones utilizadas en el TFM:

- b_CPL_H0rd
- s_CPL
- c_CPL
- bs_CPL_H0rd_sep
- cb_CPL_sep
- cs_CPL_sep
- cbs_CPL_sep

Esta versión v1 conserva las implementaciones que generaron las cadenas
finales. En particular:

- BAO usa la implementación original basada en ``quad``;
- BAO-only y BAO+SN usan la parametrización efectiva ``H0rd``;
- SN conserva la implementación original, con una integral independiente
  por supernova;
- CMB usa la implementación original basada en ``quad``;
- el chi cuadrado gaussiano conserva ``cho_solve``;
- no se introducen las optimizaciones posteriores desarrolladas en
  ``opt_result.ipynb``.

Los scripts históricos originales deben conservarse sin modificar. Este
archivo es la versión v1 preparada para lectura, reproducción y publicación.

Datos y rutas
-------------
Los datos BAO y Pantheon+SH0ES se inicializan como ``None`` hasta ejecutar
``load_late_time_data()``. Esto permite importar el archivo sin imponer una
organización concreta de carpetas. Las rutas pueden suministrarse desde la
línea de comandos.

Las covarianzas de propuesta se leen de ``chains_cmb``, tal como se hizo en
el script que generó las cadenas finales.

Ejemplos
--------
Mostrar el procedimiento sin ejecutar cadenas:

    python run_final_CPL_v1.py

Comprobar una configuración con Cobaya:

    python run_final_CPL_v1.py --check --run c_CPL

Ejecutar una cadena en una carpeta alternativa:

    python run_final_CPL_v1.py \
        --execute \
        --run c_CPL \
        --data-dir /ruta/a/los/datos \
        --chains-cmb-dir /ruta/a/chains_cmb \
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
from typing import Dict, List, Optional, Tuple, Union
import argparse
import time
import traceback

import numpy as np
import pandas as pd

from scipy.integrate import quad
from scipy.linalg import cho_factor, cho_solve


# =============================================================================
# Rutas y carga configurable de datos
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR

BAO_MEAN_FILE = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
BAO_COV_FILE = DATA_DIR / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"
SN_DATA_FILE = DATA_DIR / "PantheonSH0ES_unique_data.txt"
SN_COV_FILE = DATA_DIR / "PantheonSH0ES_unique_cov"

# Estas variables permanecen como None hasta que los datos se cargan.
# Las likelihoods comprueban su estado antes de utilizarlas.
bao_data: Optional[pd.DataFrame] = None
bao_obs: Optional[np.ndarray] = None
bao_cov: Optional[np.ndarray] = None
bao_cov_chol = None

sn_data: Optional[pd.DataFrame] = None
sn_cov: Optional[np.ndarray] = None
sn_z: Optional[np.ndarray] = None
sn_mu_obs: Optional[np.ndarray] = None
sn_cov_chol = None


def load_late_time_data(
    data_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Carga DESI BAO y Pantheon+SH0ES.

    Parameters
    ----------
    data_dir
        Carpeta que contiene los cuatro archivos observacionales. Cuando no
        se indica, se usa la carpeta del propio script.
    """
    global DATA_DIR
    global BAO_MEAN_FILE, BAO_COV_FILE, SN_DATA_FILE, SN_COV_FILE
    global bao_data, bao_obs, bao_cov, bao_cov_chol
    global sn_data, sn_cov, sn_z, sn_mu_obs, sn_cov_chol

    if data_dir is not None:
        DATA_DIR = Path(data_dir).expanduser().resolve()
        BAO_MEAN_FILE = (
            DATA_DIR
            / "desi_2024_gaussian_bao_ALL_GCcomb_mean.txt"
        )
        BAO_COV_FILE = (
            DATA_DIR
            / "desi_2024_gaussian_bao_ALL_GCcomb_cov.txt"
        )
        SN_DATA_FILE = DATA_DIR / "PantheonSH0ES_unique_data.txt"
        SN_COV_FILE = DATA_DIR / "PantheonSH0ES_unique_cov"

    required_files = (
        BAO_MEAN_FILE,
        BAO_COV_FILE,
        SN_DATA_FILE,
        SN_COV_FILE,
    )

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        missing_text = "\n".join(
            f"- {path}"
            for path in missing
        )
        raise FileNotFoundError(
            "No se encontraron todos los datos necesarios:\n"
            + missing_text
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
    bao_cov_chol = cho_factor(
        bao_cov,
        lower=True,
        check_finite=False,
    )

    sn_data = pd.read_csv(
        SN_DATA_FILE,
        sep=r"\s+",
    )
    sn_cov = np.loadtxt(SN_COV_FILE)
    sn_z = sn_data["zHD"].to_numpy(dtype=float)
    sn_mu_obs = sn_data["MU_SH0ES"].to_numpy(dtype=float)
    sn_cov_chol = cho_factor(
        sn_cov,
        lower=True,
        check_finite=False,
    )


def _require_bao_data() -> None:
    """Comprueba que los datos BAO están inicializados."""
    if (
        bao_data is None
        or bao_obs is None
        or bao_cov_chol is None
    ):
        raise RuntimeError(
            "Los datos BAO no están cargados. "
            "Ejecuta load_late_time_data()."
        )


def _require_sn_data() -> None:
    """Comprueba que los datos SN están inicializados."""
    if (
        sn_z is None
        or sn_mu_obs is None
        or sn_cov_chol is None
    ):
        raise RuntimeError(
            "Los datos SN no están cargados. "
            "Ejecuta load_late_time_data()."
        )


# Carga automática cuando la estructura local coincide con la usada
# durante el TFM. En otras organizaciones se usa --data-dir.
if all(
    path.exists()
    for path in (
        BAO_MEAN_FILE,
        BAO_COV_FILE,
        SN_DATA_FILE,
        SN_COV_FILE,
    )
):
    load_late_time_data()


# =============================================================================
# Datos comprimidos de CMB
# =============================================================================

cmb_obs = np.array([
    1.7504,
    301.77,
    0.022371,
])

cmb_cov = 1.0e-8 * np.array([
    [1559.83, -1325.41, -36.45],
    [-1325.41, 714691.80, 269.77],
    [-36.45, 269.77, 2.10],
])

cmb_cov_chol = cho_factor(
    cmb_cov,
    lower=True,
    check_finite=False,
)


# ============================================================
# Constantes cosmológicas y funciones auxiliares
# ============================================================

c_light = 299792.458  # km/s

T_CMB = 2.726  # K
N_eff = 3.046
z_star_cmb = 1090.0
H0_rad_fid = 70.0


def h_from_H0(H0):
    """
    H0 en km/s/Mpc.
    Devuelve h = H0 / 100.
    """
    return H0 / 100.0


def omega_r_from_h(h):
    """
    Densidad de radiación actual.

    Siguiendo la aproximación usada en el paper moderno:
        Omega_r = 4.18e-5 h^{-2}

    Incluye fotones + neutrinos relativistas estándar.
    """
    return 4.18e-5 / h**2

# ============================================================
# Funciones cosmológicas: CPL
# ============================================================

def f_de_CPL(z, w0, wa):
    """
    Evolución relativa de la densidad de energía oscura para CPL:

        rho_DE(z) / rho_DE(0)

    con w(a) = w0 + wa (1 - a).
    """
    zp1 = 1.0 + np.asarray(z)

    return zp1**(3.0 * (1.0 + w0 + wa)) * np.exp(
        -3.0 * wa * z / zp1
    )


def E_CPL(z, Omega_m, H0, w0, wa):
    """
    Función de expansión adimensional E(z) = H(z) / H0
    para el modelo CPL plano.

    Incluye materia, radiación y energía oscura dinámica.
    La densidad actual de energía oscura se fija por planitud.
    """
    h = h_from_H0(H0)
    Omega_r = omega_r_from_h(h)
    Omega_de = 1.0 - Omega_m - Omega_r

    zp1 = 1.0 + np.asarray(z)

    E2 = (
        Omega_m * zp1**3
        + Omega_r * zp1**4
        + Omega_de * f_de_CPL(z, w0, wa)
    )

    return np.sqrt(E2)


# ============================================================
# Distancias cosmológicas
# ============================================================

def chi_of_z(z, E_func, *E_params):
    """
    Distancia comóvil adimensional:

        chi(z) = integral_0^z dz' / E(z')
    """
    integrand = lambda zp: 1.0 / E_func(zp, *E_params)

    return quad(integrand, 0.0, z)[0]


def DM_of_z(z, H0, E_func, *E_params):
    """
    Distancia comóvil transversal D_M(z) en Mpc.

    En cosmología plana coincide con la distancia comóvil radial.
    """
    return (c_light / H0) * chi_of_z(z, E_func, *E_params)


def DH_of_z(z, H0, E_func, *E_params):
    """
    Distancia de Hubble D_H(z) en Mpc:

        D_H(z) = c / H(z)
    """
    return c_light / (H0 * E_func(z, *E_params))


def DV_of_z(z, H0, E_func, *E_params):
    """
    Distancia BAO isotrópica D_V(z) en Mpc.

    Combina información transversal y radial.
    """
    DM = DM_of_z(z, H0, E_func, *E_params)
    DH = DH_of_z(z, H0, E_func, *E_params)

    return (z * DM**2 * DH) ** (1.0 / 3.0)


def DL_of_z(z, H0, E_func, *E_params):
    """
    Distancia de luminosidad D_L(z) en Mpc.

    En cosmología plana:

        D_L(z) = (1 + z) D_M(z)
    """
    return (1.0 + z) * DM_of_z(z, H0, E_func, *E_params)
    
# ============================================================
# Física temprana y horizonte sonoro
# ============================================================

def omega_m_physical(Omega_m, H0):
    """
    Densidad física de materia:

        omega_m = Omega_m h^2
    """
    h = h_from_H0(H0)

    return Omega_m * h**2


def rd_approx(omega_m, omega_b, N_eff=3.046):
    """
    Aproximación para el horizonte sonoro en el drag epoch r_d.

    Fórmula usada en DESI para física temprana estándar:

        r_d = 147.05
              * (omega_m / 0.1432)^(-0.23)
              * (N_eff / 3.04)^(-0.1)
              * (omega_b / 0.02236)^(-0.13)  Mpc

    donde:

        omega_m = Omega_m h^2
        omega_b = Omega_b h^2
    """
    return (
        147.05
        * (omega_m / 0.1432)**(-0.23)
        * (N_eff / 3.04)**(-0.1)
        * (omega_b / 0.02236)**(-0.13)
    )


def baryon_loading(omega_b):
    """
    Factor bar_R_b usado en la velocidad del sonido del plasma fotón-barión:

        bar_R_b = 31500 * omega_b * (T_CMB / 2.7)^(-4)
    """
    return 31500.0 * omega_b * (T_CMB / 2.7)**(-4.0)


def sound_speed_over_c(z, omega_b):
    """
    Velocidad del sonido en unidades de c:

        c_s(z)/c = 1 / sqrt(3 * (1 + bar_R_b / (1+z)))
    """
    Rb = baryon_loading(omega_b)

    return 1.0 / np.sqrt(3.0 * (1.0 + Rb / (1.0 + z)))


def rs_sound_horizon(z, H0, omega_b, E_func, *E_params, z_max=1.0e7):
    """
    Horizonte sonoro comóvil r_s(z) en Mpc:

        r_s(z) = integral_z^infty c_s(z') / H(z') dz'

    En la práctica se integra hasta z_max.
    """
    integrand = lambda zp: sound_speed_over_c(zp, omega_b) / E_func(
        zp,
        *E_params
    )

    integral = quad(
        integrand,
        z,
        z_max
    )[0]

    return (c_light / H0) * integral


# ============================================================
# Observables BAO: rama física
# ============================================================

def rd_from_cosmology(Omega_m, H0, omega_b, N_eff=N_eff):
    """
    Horizonte sonoro en el drag epoch r_d en Mpc.

    Se calcula a partir de los parámetros físicos mediante
    la aproximación rápida adoptada para r_d.
    """
    omega_m = omega_m_physical(Omega_m, H0)

    return rd_approx(
        omega_m=omega_m,
        omega_b=omega_b,
        N_eff=N_eff
    )


def DM_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params):
    """
    Observable BAO transversal:

        D_M(z) / r_d
    """
    rd = rd_from_cosmology(Omega_m, H0, omega_b)
    DM = DM_of_z(z, H0, E_func, *E_params)

    return DM / rd


def DH_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params):
    """
    Observable BAO radial:

        D_H(z) / r_d
    """
    rd = rd_from_cosmology(Omega_m, H0, omega_b)
    DH = DH_of_z(z, H0, E_func, *E_params)

    return DH / rd


def DV_over_rd(z, Omega_m, H0, omega_b, E_func, *E_params):
    """
    Observable BAO isotrópico:

        D_V(z) / r_d
    """
    rd = rd_from_cosmology(Omega_m, H0, omega_b)
    DV = DV_of_z(z, H0, E_func, *E_params)

    return DV / rd

def bao_theory_vector(Omega_m, H0, omega_b, E_func, *E_params):
    """
    Vector teórico BAO en el mismo orden que el vector observado.

    Usa la rama física:

        H0, omega_b  →  r_d(Omega_m, H0, omega_b)

    y calcula los observables:

        D_M / r_d
        D_H / r_d
        D_V / r_d
    """
    _require_bao_data()

    model = []

    for _, row in bao_data.iterrows():
        z = row["z"]
        quantity = row["quantity"]

        if quantity == "DM_over_rs":
            value = DM_over_rd(
                z,
                Omega_m,
                H0,
                omega_b,
                E_func,
                *E_params
            )

        elif quantity == "DH_over_rs":
            value = DH_over_rd(
                z,
                Omega_m,
                H0,
                omega_b,
                E_func,
                *E_params
            )

        elif quantity == "DV_over_rs":
            value = DV_over_rd(
                z,
                Omega_m,
                H0,
                omega_b,
                E_func,
                *E_params
            )

        else:
            raise ValueError(f"Observable BAO no reconocido: {quantity}")

        model.append(value)

    return np.array(model)  

# ============================================================
# Observables BAO: rama efectiva H0rd
# ============================================================

def bao_theory_vector_H0rd(H0rd, E_func, *E_params):
    """
    Vector teórico BAO usando la parametrización efectiva H0*r_d.

    En esta rama no se separan H0 y r_d. Los observables se escriben como:

        D_M / r_d = (c / H0 r_d) * chi(z)

        D_H / r_d = (c / H0 r_d) / E(z)

        D_V / r_d = (c / H0 r_d) * [z chi(z)^2 / E(z)]^(1/3)
    """
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
            value = (c_light / H0rd) * (z * chi**2 / Ez)**(1.0 / 3.0)

        else:
            raise ValueError(f"Observable BAO no reconocido: {quantity}")

        model.append(value)

    return np.array(model)    

# ============================================================
# Observables teóricos: supernovas
# ============================================================

def mu_theory_of_z(z, H0, E_func, *E_params):
    """
    Módulo de distancia teórico para supernovas.

    Usa la distancia de luminosidad D_L(z) en Mpc:

        mu(z) = 5 log10(D_L / Mpc) + 25
    """
    DL = DL_of_z(z, H0, E_func, *E_params)

    return 5.0 * np.log10(DL) + 25.0


def sn_theory_vector(H0, E_func, *E_params):
    """
    Vector teórico de módulos de distancia para el conjunto SN.

    El orden del vector coincide con el orden de sn_z y sn_mu_obs.
    """
    _require_sn_data()

    return np.array([
        mu_theory_of_z(z, H0, E_func, *E_params)
        for z in sn_z
    ])

# ============================================================
# Observables teóricos: CMB distance priors
# ============================================================

def cmb_distance_prior_theory_vector(Omega_m, H0, omega_b, E_func, *E_params):
    """
    Vector teórico para los distance priors de CMB.

    Calcula:

        R      = sqrt(Omega_m) H0 D_M(z*) / c
        ell_A  = pi D_M(z*) / r_s(z*)
        omega_b

    en el mismo orden que cmb_obs.
    """
    z_star = z_star_cmb

    DM_star = DM_of_z(z_star, H0, E_func, *E_params)
    rs_star = rs_sound_horizon(
        z_star,
        H0,
        omega_b,
        E_func,
        *E_params
    )

    R = np.sqrt(Omega_m) * H0 * DM_star / c_light
    ell_A = np.pi * DM_star / rs_star

    return np.array([
        R,
        ell_A,
        omega_b
    ])

# ============================================================
# Chi cuadrado
# ============================================================

def chi2_gaussian(data, theory, cov_chol):
    """
    Chi cuadrado gaussiano con covarianza completa.

    Usa la factorización de Cholesky de la matriz de covarianza
    para resolver de forma estable:

        chi2 = delta^T C^{-1} delta

    donde delta = data - theory.
    """
    delta = data - theory

    solved = cho_solve(
        cov_chol,
        delta,
        check_finite=False
    )

    return delta @ solved


def chi2_bao(Omega_m, H0, omega_b, E_func, *E_params):
    """
    Chi cuadrado de BAO usando la rama física.

    En esta rama r_d se calcula a partir de Omega_m, H0 y omega_b.
    Se usa para combinaciones que incluyen CMB.
    """
    theory = bao_theory_vector(
        Omega_m,
        H0,
        omega_b,
        E_func,
        *E_params
    )

    return chi2_gaussian(
        bao_obs,
        theory,
        bao_cov_chol
    )


def chi2_bao_H0rd(H0rd, E_func, *E_params):
    """
    Chi cuadrado de BAO usando la parametrización efectiva H0*r_d.

    Se usa para BAO-only y BAO+SN, donde BAO no separa H0 y r_d.
    """
    theory = bao_theory_vector_H0rd(
        H0rd,
        E_func,
        *E_params
    )

    return chi2_gaussian(
        bao_obs,
        theory,
        bao_cov_chol
    )


def chi2_sn(H0, E_func, *E_params):
    """
    Chi cuadrado de supernovas.

    Compara el vector observado MU_SH0ES con el vector teórico
    de módulos de distancia.
    """
    theory = sn_theory_vector(
        H0,
        E_func,
        *E_params
    )

    return chi2_gaussian(
        sn_mu_obs,
        theory,
        sn_cov_chol
    )


def chi2_cmb(Omega_m, H0, omega_b, E_func, *E_params):
    """
    Chi cuadrado de CMB distance priors.

    Compara el vector teórico (R, ell_A, omega_b)
    con los priors observacionales comprimidos.
    """
    theory = cmb_distance_prior_theory_vector(
        Omega_m,
        H0,
        omega_b,
        E_func,
        *E_params
    )

    return chi2_gaussian(
        cmb_obs,
        theory,
        cmb_cov_chol
    )

# ============================================================
# Log-likelihoods: CPL
# ============================================================

def loglike_bao_CPL(Omega_m, H0, omega_b, w0, wa):
    """
    Log-likelihood BAO para CPL usando la rama física.

    Se usa en combinaciones que incluyen CMB.
    """
    return -0.5 * chi2_bao(
        Omega_m,
        H0,
        omega_b,
        E_CPL,
        Omega_m,
        H0,
        w0,
        wa
    )


def loglike_bao_CPL_H0rd(Omega_m, H0rd, w0, wa):
    """
    Log-likelihood BAO para CPL usando la rama efectiva H0*r_d.

    Se usa para BAO-only y BAO+SN.
    """
    return -0.5 * chi2_bao_H0rd(
        H0rd,
        E_CPL,
        Omega_m,
        H0_rad_fid,
        w0,
        wa
    )


def loglike_sn_CPL(Omega_m, H0, w0, wa):
    """
    Log-likelihood de supernovas para CPL.
    """
    return -0.5 * chi2_sn(
        H0,
        E_CPL,
        Omega_m,
        H0,
        w0,
        wa
    )


def loglike_cmb_CPL(Omega_m, H0, omega_b, w0, wa):
    """
    Log-likelihood de CMB distance priors para CPL.
    """
    return -0.5 * chi2_cmb(
        Omega_m,
        H0,
        omega_b,
        E_CPL,
        Omega_m,
        H0,
        w0,
        wa
    )


# =============================================================================
# Priors y criterio de convergencia utilizados
# =============================================================================

FINAL_RMINUS1_STOP = 0.005

CPL_OMEGA_M_PRIOR = {"min": 0.1, "max": 0.7}
CPL_H0_PRIOR = {"min": 50.0, "max": 100.0}
CPL_H0RD_PRIOR = {"min": 8000.0, "max": 12000.0}
CPL_OMEGA_B_PRIOR = {"min": 0.020, "max": 0.025}
CPL_W0_PRIOR = {"min": -2.0, "max": 2.0}
CPL_WA_PRIOR = {"min": -10.0, "max": 5.0}

FINAL_PRIORS = {
    "Omega_m": CPL_OMEGA_M_PRIOR,
    "H0": CPL_H0_PRIOR,
    "H0rd": CPL_H0RD_PRIOR,
    "omega_b": CPL_OMEGA_B_PRIOR,
    "w0": CPL_W0_PRIOR,
    "wa": CPL_WA_PRIOR,
}


# =============================================================================
# Covarianzas de propuesta
# =============================================================================

def require_covmat(path: Union[str, Path]) -> Path:
    """Comprueba que existe una covarianza de propuesta."""
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            "No existe la covmat de propuesta requerida:\n"
            f"  {path}"
        )

    return path


def build_covmat_files(
    chains_cmb_dir: Union[str, Path],
    *,
    validate: bool = True,
) -> Dict[str, Path]:
    """
    Construye las rutas de las siete covmats de CPL.

    En modo descriptivo solo devuelve las rutas esperadas. En modo de
    comprobación o ejecución verifica que existan.
    """
    chains_cmb_dir = Path(
        chains_cmb_dir
    ).expanduser().resolve()

    paths = {
        "BAO": chains_cmb_dir / "b_CPL_H0rd_wide.covmat",
        "SN": chains_cmb_dir / "s_CPL.covmat",
        "CMB": chains_cmb_dir / "c_CPL.covmat",
        "BAO+SN": chains_cmb_dir / "bs_CPL_H0rd_sep.covmat",
        "CMB+BAO": chains_cmb_dir / "cb_CPL_sep.covmat",
        "CMB+SN": chains_cmb_dir / "cs_CPL_sep.covmat",
        "CMB+BAO+SN": chains_cmb_dir / "cbs_CPL_sep.covmat",
    }

    if not validate:
        return {
            key: path.expanduser().resolve()
            for key, path in paths.items()
        }

    return {
        key: require_covmat(path)
        for key, path in paths.items()
    }



# =============================================================================
# Configuraciones Cobaya
# =============================================================================

def build_run_infos(
    chains_cmb_dir: Union[str, Path],
    chains_final_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    *,
    validate_covmats: bool = True,
    create_output_dir: bool = True,
) -> Dict[str, dict]:
    """
    Construye las siete configuraciones finales de CPL.

    No introduce las optimizaciones posteriores de BAO, SN, CMB o chi2.
    """
    chains_final_dir = Path(
        chains_final_dir
    ).expanduser().resolve()

    if output_dir is None:
        output_dir = chains_final_dir / "final"
    else:
        output_dir = Path(
            output_dir
        ).expanduser().resolve()

    if create_output_dir:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    covmats = build_covmat_files(
        chains_cmb_dir=chains_cmb_dir,
        validate=validate_covmats,
    )

    info_b_CPL_H0rd = {
        "likelihood": {
            "bao_CPL_H0rd": loglike_bao_CPL_H0rd
        },
        "params": {
            "Omega_m": {
                "prior": CPL_OMEGA_M_PRIOR,
                "ref": 0.35,
                "proposal": 0.03,
                "latex": r"\Omega_m"
            },
            "H0rd": {
                "prior": CPL_H0RD_PRIOR,
                "ref": 9500.0,
                "proposal": 200.0,
                "latex": r"H_0 r_d"
            },
            "w0": {
                "prior": CPL_W0_PRIOR,
                "ref": -0.5,
                "proposal": 0.2,
                "latex": r"w_0"
            },
            "wa": {
                "prior": CPL_WA_PRIOR,
                "ref": -2.0,
                "proposal": 0.8,
                "latex": r"w_a"
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["BAO"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000
            }
        },
        "output": str(output_dir / "b_CPL_H0rd")
    }


    info_s_CPL = {
        "likelihood": {
            "sn_CPL": loglike_sn_CPL
        },
        "params": {
            "Omega_m": {
                "prior": CPL_OMEGA_M_PRIOR,
                "ref": 0.35,
                "proposal": 0.03,
                "latex": r"\Omega_m"
            },
            "H0": {
                "prior": CPL_H0_PRIOR,
                "ref": 73.0,
                "proposal": 0.5,
                "latex": r"H_0"
            },
            "w0": {
                "prior": CPL_W0_PRIOR,
                "ref": -0.9,
                "proposal": 0.15,
                "latex": r"w_0"
            },
            "wa": {
                "prior": CPL_WA_PRIOR,
                "ref": -1.0,
                "proposal": 0.8,
                "latex": r"w_a"
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000
            }
        },
        "output": str(output_dir / "s_CPL")
    }


    info_c_CPL = {
        "likelihood": {
            "cmb_CPL": loglike_cmb_CPL
        },
        "params": {
            "Omega_m": {
                "prior": CPL_OMEGA_M_PRIOR,
                "ref": 0.30,
                "proposal": 0.04,
                "latex": r"\Omega_m"
            },
            "H0": {
                "prior": CPL_H0_PRIOR,
                "ref": 70.0,
                "proposal": 2.0,
                "latex": r"H_0"
            },
            "omega_b": {
                "prior": CPL_OMEGA_B_PRIOR,
                "ref": 0.0224,
                "proposal": 0.0002,
                "latex": r"\omega_b"
            },
            "w0": {
                "prior": CPL_W0_PRIOR,
                "ref": -0.7,
                "proposal": 0.4,
                "latex": r"w_0"
            },
            "wa": {
                "prior": CPL_WA_PRIOR,
                "ref": -1.5,
                "proposal": 1.0,
                "latex": r"w_a"
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000
            }
        },
        "output": str(output_dir / "c_CPL")
    }

    # ============================================================
    # Configuraciones Cobaya: CPL combinado
    # ============================================================

    info_bs_CPL_H0rd = {
        "likelihood": {
            "bao_CPL_H0rd": loglike_bao_CPL_H0rd,
            "sn_CPL": loglike_sn_CPL,
        },
        "params": {
            "Omega_m": {
                "prior": CPL_OMEGA_M_PRIOR,
                "ref": 0.31,
                "proposal": 0.03,
                "latex": r"\Omega_m"
            },
            "H0": {
                "prior": CPL_H0_PRIOR,
                "ref": 72.5,
                "proposal": 0.5,
                "latex": r"H_0"
            },
            "H0rd": {
                "prior": CPL_H0RD_PRIOR,
                "ref": 10000.0,
                "proposal": 120.0,
                "latex": r"H_0 r_d"
            },
            "w0": {
                "prior": CPL_W0_PRIOR,
                "ref": -0.8,
                "proposal": 0.15,
                "latex": r"w_0"
            },
            "wa": {
                "prior": CPL_WA_PRIOR,
                "ref": -0.7,
                "proposal": 0.8,
                "latex": r"w_a"
            },
        
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["BAO+SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000
            }
        },
        "output": str(output_dir / "bs_CPL_H0rd_sep")
    }


    info_cb_CPL = {
        "likelihood": {
            "cmb_CPL": loglike_cmb_CPL,
            "bao_CPL": loglike_bao_CPL,
        },
        "params": {
            "Omega_m": {
                "prior": CPL_OMEGA_M_PRIOR,
                "ref": 0.35,
                "proposal": 0.03,
                "latex": r"\Omega_m"
            },
            "H0": {
                "prior": CPL_H0_PRIOR,
                "ref": 64.0,
                "proposal": 2.0,
                "latex": r"H_0"
            },
            "omega_b": {
                "prior": CPL_OMEGA_B_PRIOR,
                "ref": 0.0224,
                "proposal": 0.0002,
                "latex": r"\omega_b"
            },
            "w0": {
                "prior": CPL_W0_PRIOR,
                "ref": -0.3,
                "proposal": 0.3,
                "latex": r"w_0"
            },
            "wa": {
                "prior": CPL_WA_PRIOR,
                "ref": -2.0,
                "proposal": 0.8,
                "latex": r"w_a"
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB+BAO"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000
            }
        },
        "output": str(output_dir / "cb_CPL_sep")
    }


    info_cs_CPL = {
        "likelihood": {
            "cmb_CPL": loglike_cmb_CPL,
            "sn_CPL": loglike_sn_CPL,
        },
        "params": {
            "Omega_m": {
                "prior": CPL_OMEGA_M_PRIOR,
                "ref": 0.27,
                "proposal": 0.01,
                "latex": r"\Omega_m"
            },
            "H0": {
                "prior": CPL_H0_PRIOR,
                "ref": 72.5,
                "proposal": 0.5,
                "latex": r"H_0"
            },
            "omega_b": {
                "prior": CPL_OMEGA_B_PRIOR,
                "ref": 0.0225,
                "proposal": 0.0002,
                "latex": r"\omega_b"
            },
            "w0": {
                "prior": CPL_W0_PRIOR,
                "ref": -0.55,
                "proposal": 0.15,
                "latex": r"w_0"
            },
            "wa": {
                "prior": CPL_WA_PRIOR,
                "ref": -2.5,
                "proposal": 0.5,
                "latex": r"w_a"
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB+SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000
            }
        },
        "output": str(output_dir / "cs_CPL_sep")
    }


    info_cbs_CPL = {
        "likelihood": {
            "cmb_CPL": loglike_cmb_CPL,
            "bao_CPL": loglike_bao_CPL,
            "sn_CPL": loglike_sn_CPL,
        },
        "params": {
            "Omega_m": {
                "prior": CPL_OMEGA_M_PRIOR,
                "ref": 0.27,
                "proposal": 0.01,
                "latex": r"\Omega_m"
            },
            "H0": {
                "prior": CPL_H0_PRIOR,
                "ref": 72.5,
                "proposal": 0.5,
                "latex": r"H_0"
            },
            "omega_b": {
                "prior": CPL_OMEGA_B_PRIOR,
                "ref": 0.0225,
                "proposal": 0.0002,
                "latex": r"\omega_b"
            },
            "w0": {
                "prior": CPL_W0_PRIOR,
                "ref": -0.65,
                "proposal": 0.15,
                "latex": r"w_0"
            },
            "wa": {
                "prior": CPL_WA_PRIOR,
                "ref": -2.0,
                "proposal": 0.5,
                "latex": r"w_a"
            },
        },
        "sampler": {
            "mcmc": {
                "covmat": str(covmats["CMB+BAO+SN"]),
                "Rminus1_stop": FINAL_RMINUS1_STOP,
                "max_tries": 10000
            }
        },
        "output": str(output_dir / "cbs_CPL_sep")
    }

    return {
        "b_CPL_H0rd": info_b_CPL_H0rd,
        "s_CPL": info_s_CPL,
        "c_CPL": info_c_CPL,
        "bs_CPL_H0rd_sep": info_bs_CPL_H0rd,
        "cb_CPL_sep": info_cb_CPL,
        "cs_CPL_sep": info_cs_CPL,
        "cbs_CPL_sep": info_cbs_CPL,
    }


# =============================================================================
# Registro del procedimiento utilizado
# =============================================================================

FINAL_RUN_ORDER: Tuple[str, ...] = (
    "cbs_CPL_sep",
    "cs_CPL_sep",
    "bs_CPL_H0rd_sep",
    "cb_CPL_sep",
    "s_CPL",
    "c_CPL",
    "b_CPL_H0rd",
)

FINAL_CHAIN_IMPLEMENTATIONS = {
    "b_CPL_H0rd": {
        "datasets": ("BAO",),
        "implementation": "BAO H0rd original con quad",
    },
    "s_CPL": {
        "datasets": ("SN",),
        "implementation": (
            "SN original: una integral quad por supernova"
        ),
    },
    "c_CPL": {
        "datasets": ("CMB",),
        "implementation": "CMB original con quad",
    },
    "bs_CPL_H0rd_sep": {
        "datasets": ("BAO", "SN"),
        "implementation": (
            "BAO H0rd original + SN original por supernova"
        ),
    },
    "cb_CPL_sep": {
        "datasets": ("CMB", "BAO"),
        "implementation": (
            "CMB original con quad + BAO físico original"
        ),
    },
    "cs_CPL_sep": {
        "datasets": ("CMB", "SN"),
        "implementation": (
            "CMB original con quad + SN original por supernova"
        ),
    },
    "cbs_CPL_sep": {
        "datasets": ("CMB", "BAO", "SN"),
        "implementation": (
            "CMB original con quad + BAO físico original "
            "+ SN original por supernova"
        ),
    },
}


def print_procedure_summary(
    run_infos: Dict[str, dict],
    selected_runs: List[str],
    *,
    data_dir: Union[str, Path],
    chains_cmb_dir: Union[str, Path],
) -> None:
    """Muestra la configuración sin ejecutar cadenas."""
    print("=" * 88)
    print("MODELO CPL — PIPELINE FINAL V1")
    print("=" * 88)
    print(
        "Directorio de datos:",
        Path(data_dir).expanduser().resolve(),
    )
    print(
        "Covmats exploratorias:",
        Path(chains_cmb_dir).expanduser().resolve(),
    )
    print("\nRuns seleccionadas:")

    for run_name in selected_runs:
        procedure = FINAL_CHAIN_IMPLEMENTATIONS[run_name]
        info = run_infos[run_name]
        sampler = info["sampler"]["mcmc"]

        print(f"\n- {run_name}")
        print(
            "  Datos:",
            ", ".join(procedure["datasets"]),
        )
        print(
            "  Implementación:",
            procedure["implementation"],
        )
        print("  Sampler: MCMC adaptativo de Cobaya")
        print("  Covmat:", sampler["covmat"])
        print(
            "  Rminus1_stop:",
            sampler["Rminus1_stop"],
        )
        print("  Output:", info["output"])


def format_elapsed_time(seconds: float) -> str:
    """Formatea segundos como texto legible."""
    seconds = float(seconds)

    if seconds < 60.0:
        return f"{seconds:.2f} s"

    minutes, seconds = divmod(
        seconds,
        60.0,
    )

    if minutes < 60.0:
        return (
            f"{int(minutes)} min "
            f"{seconds:.1f} s"
        )

    hours, minutes = divmod(
        minutes,
        60.0,
    )

    return (
        f"{int(hours)} h "
        f"{int(minutes)} min "
        f"{seconds:.1f} s"
    )


def run_with_timer(
    info: dict,
    run_name: str,
    *,
    test: bool = False,
):
    """
    Ejecuta o comprueba una configuración de Cobaya.

    Cobaya se importa solo cuando se solicita ``--check`` o ``--execute``.
    """
    try:
        from cobaya.run import run
    except ImportError as error:
        raise ImportError(
            "No se encuentra Cobaya. Instálalo antes de "
            "comprobar o ejecutar las cadenas."
        ) from error

    run_info = info

    if test:
        from copy import deepcopy

        run_info = deepcopy(info)
        run_info.pop(
            "output",
            None,
        )

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
    requested_runs: Optional[List[str]],
) -> List[str]:
    """Resuelve ``all`` o una lista explícita."""
    if (
        not requested_runs
        or "all" in requested_runs
    ):
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
            "del modelo CPL."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_DIR,
        help=(
            "Carpeta con los cuatro archivos BAO/SN. "
            "Por defecto: carpeta del proyecto."
        ),
    )

    parser.add_argument(
        "--chains-cmb-dir",
        type=Path,
        default=PROJECT_DIR / "chains_cmb",
        help=(
            "Carpeta con las covmats exploratorias originales."
        ),
    )

    parser.add_argument(
        "--chains-final-dir",
        type=Path,
        default=PROJECT_DIR / "chains_final",
        help=(
            "Carpeta raíz de las cadenas finales."
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
        help=(
            "Valida la configuración con Cobaya sin iniciar "
            "el muestreo."
        ),
    )

    mode.add_argument(
        "--execute",
        action="store_true",
        help="Ejecuta las runs seleccionadas.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    selected_runs = resolve_selected_runs(
        args.run,
    )

    if args.execute and not args.run:
        raise SystemExit(
            "Para evitar ejecutar las siete cadenas por accidente, "
            "usa al menos un argumento --run o --run all."
        )

    active_mode = (
        args.check
        or args.execute
    )

    if active_mode:
        load_late_time_data(
            args.data_dir,
        )

    run_infos = build_run_infos(
        chains_cmb_dir=args.chains_cmb_dir,
        chains_final_dir=args.chains_final_dir,
        output_dir=args.output_dir,
        validate_covmats=active_mode,
        create_output_dir=active_mode,
    )

    print_procedure_summary(
        run_infos,
        selected_runs,
        data_dir=args.data_dir,
        chains_cmb_dir=args.chains_cmb_dir,
    )

    if not active_mode:
        print(
            "\nModo descriptivo: no se ha ejecutado ninguna cadena.\n"
            "Usa --check para validar o --execute para iniciar "
            "el muestreo."
        )
        return

    status = []

    for run_name in selected_runs:
        print("\n" + "=" * 88)
        print(
            "COMPROBANDO"
            if args.check
            else "EJECUTANDO",
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

    print("\n" + "=" * 88)
    print("RESUMEN")
    print("=" * 88)

    for item in status:
        print(item)


if __name__ == "__main__":
    main()
