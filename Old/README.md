# Pediatric Sleep Apnea Project

This repository contains the research, data, and code for my master's thesis on **pediatric sleep apnea**, focusing on the application of machine learning to assist in the diagnosis of sleep disorders in children and adolescents (0–18 years).

## 📚 Project Overview

The goal of this thesis is to explore how machine learning models can be trained using **polysomnography reports** and **patient history** to improve diagnostic accuracy of **pediatric obstructive sleep apnea (OSA)** and related sleep disorders.

## 🧠 Key Objectives

- Collect and pre-process clinical and polysomnographic data.
- Perform literature review on existing diagnostic tools and ML approaches.
- Develop and evaluate predictive models using Python (e.g., scikit-learn, XGBoost).
- Analyze model performance and clinical relevance.
- Discuss ethical and privacy implications.

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Jupyter Notebooks

## 📁 Repository Structure

```bash
├── data/               # Sample or synthetic data (if applicable)
├── notebooks/          # Jupyter notebooks for exploration and modeling
├── src/                # Python scripts and model code
├── results/            # Outputs, figures, evaluation metrics
├── references/         # Papers, articles, and sources used
└── README.md           # This file
# Pediatric Sleep Apnea Project

This repository contains the research, data, and code for my master's thesis on **pediatric sleep apnea**, focusing on the application of machine learning to assist in the diagnosis of sleep disorders in children and adolescents (0–18 years).

## 📚 Project Overview

The goal of this thesis is to explore how machine learning models can be trained using **polysomnography reports** and **patient history** to improve diagnostic accuracy of **pediatric obstructive sleep apnea (OSA)** and related sleep disorders.

## 🧠 Key Objectives

- Collect and pre-process clinical and polysomnographic data.
- Perform literature review on existing diagnostic tools and ML approaches.
- Develop and evaluate predictive models using Python (e.g., scikit-learn, XGBoost).
- Analyze model performance and clinical relevance.
- Discuss ethical and privacy implications.

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Jupyter Notebooks

## 📁 Repository Structure

```bash
├── data/               # Sample or synthetic data (if applicable)
├── notebooks/          # Jupyter notebooks for exploration and modeling
├── src/                # Python scripts and model code
├── results/            # Outputs, figures, evaluation metrics
├── references/         # Papers, articles, and sources used
└── README.md           # This file
# pediatric-sleep-apnea-ds
ISCTE Data Science academic project that studies the application of ML to support pediatric sleep apneas diagnosis.

# Study Related Information

## Channels mapping in Polysomnography

| Group                    | Example Names (in EDF)                                                        | Main Clinical Function                                         | Used in Sleep Apnea Diagnosis                               |
| ------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------------------- |
| **EEG**                  | `C3-M2`, `C4-M1`, `F3-M2`, `F4-M1`, `O1-M2`, `O2-M1`, `CZ-O1`                 | Sleep staging (NREM, REM), micro-arousals, arousals            | Indirect (sleep classification and arousals, not in AHI)    |
| **EOG**                  | `LOC-M2`, `ROC-M1`                                                            | Eye movements → identify REM                                   | Indirect (required for REM staging)                         |
| **EMG**                  | `Chin1-Chin2`, `EMG Chin`, `LLEG`, `RLEG`                                     | Muscle tone (chin for REM, legs for PLMS)                      | Indirect (PLMS, sleep fragmentation criterion)              |
| **ECG**                  | `ECG LA-RA`, `ECG EKG2-EKG`                                                   | Heart rhythm, HR, HRV                                          | Complementary (tachycardia/bradycardia associated with apneas) |
| **Respiration – flow**   | `Resp Flow`, `Resp Airflow`, `Flow_DR`, `XFlow`, `PTAF`, `C-Flow`, `Nasal`    | Nasal/oral airflow (apnea/hypopnea detection)                  | ✅ Central to diagnosis                                      |
| **Respiration – effort** | `Resp Chest`, `Resp Thoracic`, `Resp Abdomen`, `Resp Abdominal`, `C-Pressure` | Respiratory effort (differentiate obstructive vs central apnea)| ✅ Central                                                   |
| **Snore**                | `Snore`, `Snore_DR`                                                           | Presence of snoring                                            | Complementary (supports obstructive apnea phenotype)         |
| **SpO₂ (oximetry)**      | `SpO2`, `SaO2`, `Osat`, `O2Sat`, `Oximetry`, `Oximeter`                       | Blood oxygen saturation, desaturations                         | ✅ Essential (ODI, T90, desaturations)                       |
| **CO₂**                  | `EtCO2`, `TcCO2`, `Capno`                                                     | Capnography (end-tidal) and transcutaneous CO₂ → hypoventilation | Complementary (especially in pediatrics)                     |
| **Other**                | `Patient Event`, `Rate`, `Tidal Vol`                                          | Manual events or respiratory volumes                           | Support                                                     |
