# LLM Self-Preservation: An Investigation into LLM Consistence

*Advanced Machine Learning*
Universita degli Studi di Roma "La Sapienza"

- *Ciro* Ciaravolo
- *Juan José* Serrano
- *Adam* Maciejak
- *Pablo* Barrenechea

---------------------------
## Project Summary
This project investigates self-preservation bias in Large Language Models (LLMs), where models may favor themselves when giving advisory recommendations despite objective evidence. Using a role-switching test (Incumbent vs. Challenger) across 30 models, we evaluated responses under three performance gaps (+2%, +5%, +15%) and across different temperature settings (0.1, 0.2, 0.3). Results show that many models exhibit classic self-preservation bias at small gaps, while highly safety-aligned models can show “inverse bias” at larger gaps, advising against themselves. Temperature sensitivity reveals latent biases not apparent under deterministic decoding, highlighting that low-temperature evaluations can underestimate such effects.


## File Overview

### Data
- `answers_prompt2%.csv`, `answers_prompt10%.csv`, `answers_prompt15%.csv`: Model responses to 2%, 10%, and 15% prompt differences, using no temperature variations (deterministic inference).  
- `full_results.csv`: Combined results from all prompt differences, indicating whether each model contradicted itself.  
- `results_merged.csv`: Merged results from deterministic experiments across open source models, after executing them on kaggle.  
- `temperature_exp/`: Model results under different temperature settings and roles.  
  - `invalid_results/`: Contains results from excluded models (e.g., too small for reliable analysis).  

### Plots
- Various plots visualizing experimental results:  
  - `model-switching-matrix.png` / `model-switching-matrixiverted.png`  
  - `resistance_curve.png`  
  - `temperature_comparison_matrix.png`  
  - `temperature_variation_barplots.png`  

### Notebooks
- `llm-self-preservation.ipynb`: Code used to gather model answers for deterministic and temperature experiments.  
- `plots.ipynb`: Generates plots from gathered data.  

### Other Files
- `prompts.py`: Script for generating queries/prompts.  

- `self_preservation_gathered_data.xlsx`: Excel file with sheets for 2%, 10%, 15%, combined results, and query details.  
- `README.md`: This file.  

# Execution Details
Open-source models were evaluated using an automated pipeline on Hugging Face, running on Kaggle’s free **P100 GPU**. This environment was sufficient for generating model responses efficiently. Each model was queried in a stateless manner (new context for every prompt) with controlled generation parameters (temperature, top-p, max tokens) to ensure reproducibility.