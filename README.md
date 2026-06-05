# Galaxy Morphology Classification & Formation Simulation

This repository serves as a comprehensive research and computational hub, bridging the gap between theoretical astrophysics and data-driven modeling. It demonstrates the application of numerical simulations and supervised machine learning to decode the structure and evolution of galaxies.

---

## 🏛️ Project Overview

The project is divided into three distinct modules, each exploring different facets of galactic dynamics and analysis:

* [cite_start]**Module 1 (Theoretical Research):** An in-depth exploration of the physical processes governing galaxy formation, covering the Dark Matter framework, Baryonic physics, stellar evolution (Pop III to Pop II transitions), feedback mechanisms (Supernovae/AGN), and hierarchical assembly[cite: 847, 849, 850, 851, 852, 853, 854].
* **Module 2 (ML Classification Pipeline):** An end-to-end machine learning project designed to automate the classification of galaxy morphologies. [cite_start]We utilize physical structural indicators to label datasets and train high-performance models to predict morphological types[cite: 887, 889].
* [cite_start]**Module 3 (N-Body Simulation):** A custom 2D particle-based simulation built to visualize galaxy formation from first principles using Newtonian gravity, showcasing the direct impact of angular momentum on galaxy shape[cite: 458, 468, 923, 934].

---

## 🛠️ Technical Implementation



### Machine Learning Pipeline

Our ML workflow focuses on structured tabular data, ensuring robust performance through meticulous preprocessing and state-of-the-art algorithms:

* [cite_start]**Feature Engineering:** Extracted critical structural parameters including Sérsic indices ($n_i, n_z, n_y$), axis ratios ($b/a$), half-light radii, and stellar population color indices[cite: 273, 276, 280, 284].
* [cite_start]**Preprocessing:** Implemented leak-free pipelines using **Scikit-learn** for median imputation and feature standardization, ensuring model training remains isolated from validation/test data[cite: 300, 301, 304].
* [cite_start]**Models:** * **Random Forest:** Utilized for its ability to handle non-linear patterns and provide feature importance rankings[cite: 318, 320].
    * [cite_start]**XGBoost:** Employed for superior performance on tabular data, using gradient boosting to refine boundaries between elliptical and spiral classifications[cite: 354, 355].
* [cite_start]**Performance:** Achieved >99% classification accuracy, validated through confusion matrices and F1-score analysis[cite: 341, 399].

### N-Body Simulation Framework

The simulation uses a custom Python-based approach to model gravitational interactions:

* [cite_start]**Numerical Engine:** Uses Euler integration to evolve particle trajectories over time, with softening parameters ($\epsilon = 0.1$) to prevent numerical divergence during close encounters[cite: 468, 473, 474].
* [cite_start]**Physical Conservation:** Tracks total Energy and Angular Momentum throughout the simulation to validate the physical consistency of the model[cite: 488, 491, 548].
* **Experimental Scenarios:**
    * [cite_start]**High Angular Momentum:** Leads to stable, rotating spiral disk formation[cite: 479, 508].
    * [cite_start]**Low Angular Momentum:** Results in rapid, centrally concentrated elliptical-like structures[cite: 483, 512].
    * [cite_start]**Asymmetric/Turbulent:** Models realistic irregular structures[cite: 486, 515].

---

## 📊 Tech Stack

* [cite_start]**Languages:** Python 3.x [cite: 554]
* [cite_start]**ML Frameworks:** Scikit-learn, XGBoost [cite: 314, 350]
* [cite_start]**Data & Numerics:** NumPy, Pandas [cite: 555]
* [cite_start]**Visualization:** Matplotlib, FuncAnimation [cite: 555]

---

## 📂 Repository Structure

```text
├── module_1_theory/        # Theoretical reports and research notes
├── module_2.ipynb          # ML pipeline: preprocessing, training, evaluation
├── module_3.py             # Galaxy formation simulation script
└── README.md
