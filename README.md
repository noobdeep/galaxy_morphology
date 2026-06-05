# Galaxy Morphology Classification & Formation Simulation

This project explores the interplay between cosmological theory and data-driven computational modeling. It consists of three modules: theoretical research on galaxy formation, a machine learning pipeline for morphological classification, and an N-body gravitational simulation.

## Project Structure
- **Module 1 (Theoretical):** A comprehensive report on galaxy formation, including dark matter framework, baryonic physics, and hierarchical assembly.
- **Module 2 (ML Classification):** A supervised learning pipeline using Random Forest and XGBoost to classify galaxies into **elliptical** and **non-elliptical** categories based on photometric and structural features.
- **Module 3 (Simulation):** A 2D N-body gravitational simulation implemented in Python to model galaxy formation from first principles.

## Key Features
- **Machine Learning Pipeline:** - Engineered robust structural features ($n_i$, $b/a$, peak surface brightness, colors).
  - Implemented leakage-free preprocessing (median imputation, standardization).
  - Achieved near-perfect classification performance using Random Forest and XGBoost.
- **N-Body Simulation:**
  - Implemented gravitational interactions with softening parameters to avoid singularities.
  - Demonstrated conservation of angular momentum and energy across three scenarios: High, Low, and Asymmetric angular momentum.
  - Visualized structure formation (spirals vs. ellipticals) through particle animations.

## Technologies
- **Languages:** Python
- **ML Frameworks:** Scikit-learn, XGBoost
- **Data/Numerics:** NumPy, Pandas, Matplotlib
- **Simulation:** N-body gravitational dynamics (Euler integration)
