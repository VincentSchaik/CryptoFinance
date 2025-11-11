# CryptoFinance: Bitcoin Market Regime Prediction

## Project Overview
This repository hosts the group project for the Data Sciences Institute (University of Toronto) Certificate & Microcredential program. We created an end-to-end machine learning workflow that classifies the next-day Bitcoin market regime (Bullish or Bearish) so that retail investors, quantitative analysts, and financial content creators can react to rapid price swings with greater confidence. The work is documented in the notebooks [`notebooks/btc_price_prediction.ipynb`](notebooks/btc_price_prediction.ipynb) and [`notebooks/mlflow_databricks.ipynb`](notebooks/mlflow_databricks.ipynb).

---

## Identifying the Business Issue
* **Stakeholders** – Individual investors, algorithmic traders, crypto research desks, and financial media outlets seeking objective guidance for volatile Bitcoin markets.
* **Business Value** – Early identification of bullish or bearish regimes supports better risk management, automated trading, and content scheduling by surfacing actionable signals instead of raw price movements.

### Guiding Questions
* **Who is the intended audience?** – Retail investors refining discretionary entries, quantitative teams maintaining automated strategies, and market commentators needing evidence-backed narratives.
* **What question does the analysis answer?** – “Given today’s market state, will Bitcoin close higher or lower tomorrow?”
* **Key variables and attributes?** – Daily OHLCV fields (open, high, low, close, volume), engineered technical indicators (moving averages, RSI, MACD, Bollinger Band widths, volatility lags), and weekly qualitative sentiment scores aligned to each trading day.
* **Do we need to clean the data?** – Yes. We remove duplicate timestamps, forward-fill small sentiment gaps, validate numeric ranges, and winsorize extreme outliers before scaling.
* **How do we explore relationships?** – Time-series plots, rolling correlation heatmaps, pairwise scatterplots, SHAP-based feature attributions, and confusion matrix diagnostics across temporal slices.
* **What patterns or trends appear?** – Momentum bursts, volatility clustering, sentiment-driven divergences, and regime shifts around macro events that motivate frequent retraining.
* **Libraries/frameworks best suited?** – pandas for wrangling, NumPy for numerical routines, scikit-learn for classical models and preprocessing, XGBoost for gradient boosting, TensorFlow/Keras for LSTM modeling, MLflow for experiment tracking, and Databricks for scalable execution.

## Formulating the ML Problem
* **Prediction Task** – Binary classification of the following day’s market direction (Bullish vs. Bearish).
* **Success Criteria** – Achieve at least 65% directional accuracy on hold-out data while maintaining interpretable drivers of the prediction.
* **Operational Constraints** – Daily batch inference must complete within minutes and produce MLflow-tracked artefacts deployable from Databricks.

## Gathering Data
1. **Historical Pricing** – Daily OHLCV Bitcoin prices (Nov 2020 – Nov 2025) from `data/raw/dataset.csv`.
2. **Weekly On-chain Sentiment** – Curated dataset (`data/raw/weekly_on_chain_sentiment.csv`) derived from web-scraped on-chain analysis blogs. Each summary was scored as Bullish/Bearish using ChatGPT with advanced/few-shot prompting to standardize sentiment labels.
3. **Technical References** – Supplementary research links captured in `links.md` to guide feature ideation and validation.

## Merging and Preparing Data
* Converted raw files into a unified time series indexed by trading day.
* Forward-filled weekly sentiment to align with daily observations, then merged with technical indicators.
* Created a binary target where **Bullish** indicates the closing price exceeds the previous day’s close, otherwise **Bearish**.
* Split the data into training, validation, and test sets using chronological order to prevent look-ahead bias.

## Data Analysis and Visualization
* Conducted exploratory analysis within `notebooks/btc_price_prediction.ipynb` to understand price trends, volatility clusters, and volume regimes.
* Visualized correlations between engineered indicators, sentiment, and returns to identify dominant drivers.
* Plotted class balance and temporal drift to confirm the necessity of regular retraining.

## Expanding the Dataset
* Applied web scraping scripts (documented in the notebooks) to ingest weekly on-chain commentaries from 2020–2025.
* Employed ChatGPT-driven summarization and sentiment scoring prompts to generate consistent Bullish/Bearish labels.
* Validated sentiment coverage against the price timeline to ensure minimal gaps before merging into the master dataset.

## Feature Engineering
* Generated technical analysis metrics including moving averages, RSI, MACD, Bollinger Bands, and volume-derived oscillators.
* Lagged returns and volatility estimates to capture momentum and mean-reversion effects.
* One-hot encoded sentiment labels and constructed interaction terms between sentiment and price momentum.
* Scaled numerical features with standardization where appropriate and persisted preprocessing parameters for reuse.

### Machine Learning Guiding Questions
* **Objectives and success criteria?** – Deliver >65% directional accuracy with stable precision/recall across bullish and bearish classes while keeping inference latency under several minutes.
* **Selecting relevant features?** – Start from domain-informed technical indicators, apply mutual information and permutation importance screening, and prune collinear predictors before modeling.
* **Handling missing values/outliers?** – Forward-fill weekly sentiment, interpolate short-lived price gaps, drop extended missing periods, and winsorize abnormal return spikes to protect tree-based models.
* **Which algorithms suit the problem?** – Logistic Regression for interpretability, Random Forest and XGBoost for non-linear relationships, and LSTM to capture temporal dependencies in sequential features.
* **Validation and tuning techniques?** – Time-series cross-validation (expanding window), grid/random search for classical models, Bayesian optimization experiments on Databricks, and early stopping with learning-rate schedules for the LSTM.
* **Data splitting strategy?** – Chronological split into training (2020–2023), validation (2024), and test (2025) cohorts, with walk-forward evaluation for robustness checks.
* **Ethical implications or biases?** – Recognize crypto market manipulation risk, avoid over-reliance on sentiment sourced from limited communities, and document limitations so users do not overfit investment decisions.
* **Documenting pipeline and architecture?** – Store preprocessing/feature scripts in `src/`, version notebooks, capture MLflow run metadata, and register champion models with lineage notes in the Databricks Model Registry.

## Training the Model and Tuning Parameters
* Baseline and advanced models implemented: Logistic Regression, Random Forest, XGBoost, and an LSTM neural network for sequential learning.
* Hyperparameters tuned through cross-validated grid searches (tree-based models) and learning-rate scheduling/early stopping (LSTM).
* Enabled MLflow autologging on Databricks to capture parameters, metrics, model artefacts, and lineage for every experiment run.

## Assessing the Model
* Evaluated models on the chronological validation set using accuracy, precision/recall, ROC-AUC, and confusion matrices to understand trade-offs.
* Compared feature importances (tree-based models) and SHAP value explanations to assess the impact of sentiment vs. technical indicators.
* Selected the best-performing model based on validation accuracy exceeding the 65% target while maintaining stable performance on the test set.

## Making Predictions
* Packaged the champion model with its preprocessing pipeline to support daily inference jobs on new market data.
* Demonstrated prediction generation and visualization of bullish/bearish probabilities inside the notebooks.

## Deploying the Model
* Registered the champion model to the Databricks Model Registry via MLflow for controlled promotion between Staging and Production stages.
* Documented deployment steps for batch scoring jobs and potential REST API serving using Databricks serving endpoints.

## Monitoring and Rectification
* Established MLflow model versions and run metrics to monitor drift, latency, and accuracy over time.
* Proposed weekly retraining triggered by significant drops in directional accuracy or major market regime shifts.
* Logged experiment metadata to facilitate root-cause analysis and rapid rollback if model performance degrades.

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
