# CryptoFinance: Bitcoin Market Regime Prediction

## Project Overview
This repository hosts the group project for the Data Sciences Institute (University of Toronto) Certificate & Microcredential program. We created an end-to-end machine learning workflow that classifies whether Bitcoin's next closing price will finish at least 1% higher than today (Bullish) or not (Bearish) so that retail investors, quantitative analysts, and financial content creators can react to rapid price swings with greater confidence. The work is documented in the notebooks [`notebooks/btc_price_prediction.ipynb`](notebooks/btc_price_prediction.ipynb) and [`notebooks/mlflow_databricks.ipynb`](notebooks/mlflow_databricks.ipynb).

---

## Identifying the Business Issue
* **Stakeholders** – Individual investors, algorithmic traders, crypto research desks, and financial media outlets seeking objective guidance for volatile Bitcoin markets.
* **Business Value** – Early identification of bullish or bearish regimes supports better risk management, automated trading, and content scheduling by surfacing actionable signals instead of raw price movements.

### Guiding Questions
* **Who is the intended audience for your project?** – Team members completing the Data Sciences Institute capstone, retail-focused investors testing systematic entries, and instructors or mentors reviewing the workflow end-to-end.
* **What is the question you will answer with your analysis?** – Given today’s engineered market state, will Bitcoin’s next closing price exceed the current close by at least 1%?【F:notebooks/btc_price_prediction.ipynb†L1088-L1111】
* **What are the key variables and attributes in your dataset?** – Daily OHLCV inputs (`Open`, `High`, `Low`, `Close`, `Volume`) plus derived spreads (`Open-Close`, `Low-High`) and technical indicators such as multiple moving averages, MACD variants, RSI, Bollinger Bands, rate-of-change metrics, volatility windows, stochastic oscillators, OBV, momentum, and volume change features.【F:notebooks/btc_price_prediction.ipynb†L1181-L1231】
* **Do you need to clean your data, and if so what is the best strategy?** – Rolling indicators and target shifts introduce leading NaNs; the notebook drops those rows before modeling and fills the OBV seed with zero while leaving raw prices untouched.【F:notebooks/btc_price_prediction.ipynb†L2040-L2056】【F:notebooks/btc_price_prediction.ipynb†L1221-L1228】
* **How can you explore the relationships between different variables?** – The notebook plots histograms for core OHLCV fields and computes a correlation heatmap of prices, returns, and indicator columns to examine co-movements prior to modeling.【F:notebooks/btc_price_prediction.ipynb†L521-L531】【F:notebooks/btc_price_prediction.ipynb†L1908-L1919】
* **What types of patterns or trends are in your data?** – The 1% sensitivity label yields an imbalanced target with roughly one-third bullish observations, and correlation inspection confirms highly coupled OHLC prices with more dispersed volume behaviour.【F:notebooks/btc_price_prediction.ipynb†L1088-L1111】【F:notebooks/btc_price_prediction.ipynb†L1908-L1919】
* **Are there any specific libraries or frameworks that are well-suited to your project requirements?** – pandas, NumPy, seaborn, and matplotlib support preprocessing and EDA; scikit-learn, XGBoost, and TensorFlow/Keras power the Logistic Regression, Random Forest, XGBoost, and LSTM models; MLflow integrations in the Databricks notebook capture experiments for reproducibility.【F:notebooks/btc_price_prediction.ipynb†L2029-L2063】【F:notebooks/mlflow_databricks.ipynb†L158-L307】

## Formulating the ML Problem
* **Prediction Task** – Binary classification of the following day’s market direction (Bullish vs. Bearish) using a 1% upside threshold.【F:notebooks/btc_price_prediction.ipynb†L1088-L1111】
* **Success Criteria** – Surpass a naive directional baseline by tracking accuracy, ROC-AUC, and class recall on the hold-out split (recent runs reach 0.72 accuracy with modest ROC-AUC).【F:notebooks/btc_price_prediction.ipynb†L1951-L1983】【F:notebooks/btc_price_prediction.ipynb†L2550-L2566】
* **Operational Constraints** – Daily batch inference must complete within minutes and produce MLflow-tracked artefacts deployable from Databricks.

## Gathering Data
1. **Historical Pricing** – Daily OHLCV Bitcoin prices (Nov 2020 – Nov 2025) from `data/raw/dataset.csv`.
2. **Weekly On-chain Sentiment** – Curated dataset (`data/raw/weekly_on_chain_sentiment.csv`) derived from web-scraped on-chain analysis blogs. Each summary was scored as Bullish/Bearish using ChatGPT with advanced/few-shot prompting to standardize sentiment labels.
3. **Technical References** – Supplementary research links captured in `links.md` to guide feature ideation and validation.

## Merging and Preparing Data
* Loaded the historical BTC-USD prices into a pandas DataFrame indexed by trading day.
* Engineered technical analysis features from the price/volume history and computed spreads prior to modeling.
* Created a binary target where **Bullish** indicates the next close is at least 1% higher than today’s close (otherwise **Bearish**).【F:notebooks/btc_price_prediction.ipynb†L1088-L1111】
* Retained the additional weekly sentiment CSV for future integration; it is not yet merged into the modeling pipeline.
* Split the data chronologically with `shuffle=False` to avoid look-ahead leakage in subsequent experiments.【F:notebooks/btc_price_prediction.ipynb†L2050-L2063】

## Data Analysis and Visualization
* Conducted exploratory analysis within `notebooks/btc_price_prediction.ipynb` to examine price/volume distributions and indicator correlations before modeling.【F:notebooks/btc_price_prediction.ipynb†L521-L531】【F:notebooks/btc_price_prediction.ipynb†L1908-L1919】
* Visualized correlations between engineered indicators and returns to identify dominant drivers.【F:notebooks/btc_price_prediction.ipynb†L1908-L1919】
* Reported target class balance to show the bullish minority class that models must account for.【F:notebooks/btc_price_prediction.ipynb†L1088-L1111】

## Expanding the Dataset
* Stored a supplementary `weekly_on_chain_sentiment.csv` file derived from on-chain commentary analysis for future experimentation.
* Documented prompt-engineering and scraping resources in the project notes while deferring automated ingestion to later work.
* Ensured the additional dataset follows the same date conventions so it can be merged on `Week_Start_Date` when modeling expands.

## Feature Engineering
* Generated technical analysis metrics including moving averages, RSI, MACD, Bollinger Bands, and volume-derived oscillators.【F:notebooks/btc_price_prediction.ipynb†L1181-L1231】
* Lagged returns and volatility estimates to capture momentum and mean-reversion effects, and standardized features prior to modeling.【F:notebooks/btc_price_prediction.ipynb†L1181-L1209】【F:notebooks/btc_price_prediction.ipynb†L2058-L2061】
* Left sentiment augmentations for follow-up work once the weekly dataset is joined to the primary table.
* Persisted the scaler alongside trained estimators inside each modeling cell for reproducible inference.【F:notebooks/btc_price_prediction.ipynb†L2058-L2061】

### Machine Learning Guiding Questions
* **What are the specific objectives and success criteria for your machine learning model?** – The objective is to flag next-day price jumps of at least 1%; success is judged by hold-out accuracy, ROC-AUC, and class-level precision/recall recorded in the notebooks (e.g., XGBoost reached 0.7178 accuracy with ROC-AUC 0.5866 on the reserved test window, while Logistic Regression achieved 0.6963 accuracy).【F:notebooks/btc_price_prediction.ipynb†L1951-L1983】【F:notebooks/btc_price_prediction.ipynb†L2550-L2566】
* **How can you select the most relevant features for training?** – The current workflow relies on domain-driven feature lists of spreads, momentum, and volatility indicators, then inspects model coefficients/feature importances logged through MLflow to decide which engineered metrics warrant retention.【F:notebooks/btc_price_prediction.ipynb†L1181-L1231】【F:notebooks/mlflow_databricks.ipynb†L241-L307】
* **Are there any missing values or outliers that need to be addressed through preprocessing?** – Rolling calculations and the shifted target create leading NaNs that are removed with `dropna`, and the OBV seed is filled with zero; no additional outlier clipping is applied in the current experiments.【F:notebooks/btc_price_prediction.ipynb†L2040-L2056】【F:notebooks/btc_price_prediction.ipynb†L1221-L1228】
* **Which machine learning algorithms are suitable for the problem domain?** – Baseline Logistic Regression, Random Forest, and XGBoost classifiers capture tabular signal interactions, while an LSTM sequence model ingests 10-day windows to learn temporal dependencies.【F:notebooks/btc_price_prediction.ipynb†L2029-L2067】【F:notebooks/btc_price_prediction.ipynb†L2624-L2673】
* **What techniques are available to validate and tune the hyperparameters?** – Models are evaluated on a chronological 80/20 split with MLflow autologging so manual hyperparameter adjustments are tracked; the LSTM leverages a 20% validation split with early stopping to prevent overfitting, and Databricks experiments can be rerun with altered settings for comparative analysis.【F:notebooks/btc_price_prediction.ipynb†L2050-L2067】【F:notebooks/btc_price_prediction.ipynb†L2624-L2668】【F:notebooks/mlflow_databricks.ipynb†L241-L307】
* **How should the data be split into training, validation, and test sets?** – Use `train_test_split` with `shuffle=False` to hold out the most recent 20% of observations for testing, and rely on the Keras `validation_split=0.2` argument during LSTM training to carve out an in-sample validation fold for early stopping.【F:notebooks/btc_price_prediction.ipynb†L2050-L2063】【F:notebooks/btc_price_prediction.ipynb†L2662-L2668】
* **Are there any ethical implications or biases associated with the machine learning model?** – The model only observes historical market data, so there is no personal information, but it can still propagate optimism bias if traders over-trust signals with low bullish recall; documentation highlights the class imbalance (32% bullish) to discourage overconfident deployment.【F:notebooks/btc_price_prediction.ipynb†L1088-L1111】【F:notebooks/btc_price_prediction.ipynb†L1951-L1983】
* **How can you document the machine learning pipeline and model architecture for future reference?** – Version the notebooks alongside helper utilities, capture each training run with MLflow autologging on Databricks, and store scaler/estimator artifacts so that future contributors can trace parameters, metrics, and model summaries directly from the tracking UI.【F:notebooks/mlflow_databricks.ipynb†L199-L307】【F:notebooks/btc_price_prediction.ipynb†L2058-L2061】

## Training the Model and Tuning Parameters
* Baseline and advanced models implemented: Logistic Regression, Random Forest, XGBoost, and an LSTM neural network for sequential learning.【F:notebooks/btc_price_prediction.ipynb†L2029-L2067】【F:notebooks/btc_price_prediction.ipynb†L2624-L2673】
* Hyperparameters are presently adjusted manually and monitored via MLflow runs, while the LSTM employs early stopping and dropout to manage overfitting.【F:notebooks/btc_price_prediction.ipynb†L2624-L2668】【F:notebooks/mlflow_databricks.ipynb†L241-L307】
* Enabled MLflow autologging on Databricks to capture parameters, metrics, model artefacts, and lineage for every experiment run.【F:notebooks/mlflow_databricks.ipynb†L199-L307】

## Assessing the Model
* Evaluated models on the chronological hold-out split using accuracy, precision/recall, ROC-AUC, and classification reports to understand trade-offs (e.g., Random Forest underperformed at 0.5184 accuracy while XGBoost achieved 0.7178).【F:notebooks/btc_price_prediction.ipynb†L1951-L1983】
* Inspected logged metrics and feature weights through MLflow to determine which engineered indicators contribute most; SHAP analysis is a noted extension for future work.【F:notebooks/mlflow_databricks.ipynb†L241-L307】
* Selected the best-performing configuration by comparing recorded MLflow runs rather than a fixed accuracy threshold.【F:notebooks/mlflow_databricks.ipynb†L241-L307】

## Making Predictions
* Combined trained estimators with their fitted scalers to support local inference workflows in the notebook environment.【F:notebooks/btc_price_prediction.ipynb†L2058-L2061】
* Printed probability outputs and classification reports inside the notebooks to validate bullish/bearish predictions before any deployment step.【F:notebooks/btc_price_prediction.ipynb†L1951-L1983】

## Deploying the Model
* Configured MLflow to use the Databricks tracking URI and experiment workspace, enabling hosted comparison of training runs.【F:notebooks/mlflow_databricks.ipynb†L199-L245】
* Documented next steps for packaging batch jobs or REST endpoints once model performance goals are satisfied; registry promotion is still to be implemented.

## Monitoring and Rectification
* Relied on MLflow run metadata (parameters, metrics, and artifacts) to monitor how successive experiments perform over time.【F:notebooks/mlflow_databricks.ipynb†L199-L307】
* Plan to schedule retraining when tracked accuracy dips or market conditions change, noting the bullish class imbalance that can erode recall.【F:notebooks/btc_price_prediction.ipynb†L1088-L1111】【F:notebooks/btc_price_prediction.ipynb†L1951-L1983】
* Preserved notebook outputs and logs to aid root-cause analysis and rollback discussions during future monitoring phases.【F:notebooks/mlflow_databricks.ipynb†L199-L307】

---

## Repository Structure
```
├── data/
│   └── raw/
│       ├── dataset.csv
│       └── weekly_on_chain_sentiment.csv
├── notebooks/
│   ├── btc_price_prediction.ipynb
│   └── mlflow_databricks.ipynb
├── src/
│   └── ... (utility code for feature engineering, modeling, and evaluation)
├── environment.yaml
└── README.md
```

## Getting Started
1. Create the conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate cryptofinance
   ```
2. Launch Jupyter Lab/Notebook and open the analysis notebooks:
   ```bash
   jupyter lab
   ```
3. Configure Databricks credentials (workspace URL, access token) as environment variables to run the MLflow-managed experiments.

## Acknowledgements
* Project completed by Vincent, Julian, Kirti, and Juan for the Data Sciences Institute team project requirement.
* We thank the maintainers of the open-source libraries (NumPy, pandas, scikit-learn, XGBoost, TensorFlow/Keras, MLflow) used in this analysis.
