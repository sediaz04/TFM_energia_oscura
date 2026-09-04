# Modelos de energía oscura con DESI, Pantheon+SH0ES y CMB

Repositorio asociado al Trabajo Fin de Máster dedicado al estudio de distintos modelos fenomenológicos de energía oscura mediante datos de BAO de DESI DR1, supernovas de tipo Ia de Pantheon+SH0ES y priors de distancia del CMB.

El repositorio contiene los scripts utilizados para el muestreo de los modelos finales, los productos extraídos de las cadenas, las figuras principales y el cuaderno completo de análisis de resultados.

## Modelos analizados

Se consideran cuatro modelos cosmológicos:

- ΛCDM plano.
- CPL.
- CPL₂, una extensión fenomenológica cuadrática de CPL.
- Modelo exponencial, EXP.

Las parametrizaciones, priors y detalles metodológicos se describen en la memoria del TFM y en los propios scripts.

## Datos

El análisis utiliza:

- DESI DR1 BAO.
- Pantheon+SH0ES.
- Priors de distancia del CMB de Bansal & Huterer (2025).

Los datos observacionales originales no se redistribuyen en este repositorio. Deben obtenerse de sus fuentes oficiales, indicadas a continuación:

- [DESI DR1 BAO](https://data.desi.lbl.gov/doc/releases/dr1/vac/bao-cosmo-params/): documentación oficial de los resultados cosmológicos BAO de DR1.
- [Pantheon+SH0ES](https://github.com/PantheonPlusSH0ES/DataRelease): repositorio oficial DataRelease.
- [Bansal & Huterer (2025)](https://arxiv.org/abs/2502.07185), *Expansion-history preferences of DESI and external data*.

En el caso de Pantheon+SH0ES se utilizó una versión preprocesada de los datos, siguiendo el procedimiento descrito en la memoria del TFM.

La carpeta [`data/`](data/) contiene información adicional sobre los datos esperados por los scripts.
En el caso de Pantheon+SH0ES se utilizó una versión preprocesada de los datos, siguiendo el procedimiento descrito en la memoria del TFM.

La carpeta [`data/`](data/) contiene información adicional sobre los datos esperados por los scripts.

## Estructura del repositorio

```text
TFM_energia_oscura/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── resultados_analisis_TFM.ipynb
│   └── resultados_analisis_TFM.pdf
├── scripts/
│   ├── run_final_LCDM_v1.py
│   ├── run_final_CPL_v1.py
│   ├── run_final_CPLn2_v1.py
│   └── run_final_exp_v1.py
└── results_final/
    ├── extraction/
    └── figures/
```

### `notebooks/`

Contiene el cuaderno final de resultados ya ejecutado:

- `resultados_analisis_TFM.ipynb`: cuaderno completo de análisis.
- `resultados_analisis_TFM.pdf`: versión estática del cuaderno para facilitar su consulta sin necesidad de ejecutar código.

### `scripts/`

Contiene los scripts finales utilizados para los cuatro modelos cosmológicos.

Cada script permite ejecutar las distintas combinaciones de datos consideradas en el trabajo.

### `results_final/extraction/`

Contiene productos derivados de las cadenas MCMC utilizados por el cuaderno de análisis:

- `chain_inventory.csv`
- `chain_diagnostics.csv`
- `posterior_summary.csv`
- `best_sampled_points.csv`
- `validation_warnings.csv`
- `chain_metadata.json`

### `results_final/figures/`

Contiene una selección de las figuras finales más relevantes del análisis en formatos PNG y PDF.

## Entorno de Python

El cuaderno final fue ejecutado con:

- Python 3.12.12
- NumPy 2.3.5
- pandas 2.3.3
- SciPy 1.16.3
- Matplotlib 3.10.8
- GetDist 1.7.6
- Cobaya 3.6.2

Las dependencias de Python se recogen en [`requirements.txt`](requirements.txt).

Para instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Reproducción del análisis

El cuaderno incluido en `notebooks/` se distribuye ya ejecutado, de forma que los resultados, tablas y figuras pueden consultarse directamente sin necesidad de disponer de las cadenas MCMC originales.

Las cadenas completas no se incluyen en este repositorio debido a su tamaño.

Los productos resumidos necesarios para el análisis se encuentran en `results_final/extraction/`.

La reejecución completa de los muestreos MCMC requiere además obtener localmente los datos observacionales correspondientes y proporcionar a los scripts las rutas necesarias.

## Consulta rápida

Para una revisión rápida de los resultados sin necesidad de ejecutar el cuaderno ni instalar dependencias, puede consultarse directamente:

[`notebooks/resultados_analisis_TFM.pdf`](notebooks/resultados_analisis_TFM.pdf)

El archivo `.ipynb` constituye la versión principal del análisis de resultados.
