# AI Recruitment Pipeline — End-to-End Machine Learning Evaluation

An end-to-end Machine Learning project that merges the Kaggle **AI Recruitment Pipeline Dataset** (`yaswanthkumary/ai-recruitment-pipeline-dataset`) with primary survey responses collected via Google Forms (`google_form_recruitment_dirty.xlsx`), cleans real-world data impurities, and implements supervised, unsupervised, association rule mining, and reinforcement learning techniques.

---

## Dataset & Impurity Cleaning Pipeline

1. **External Benchmark (`yaswanthkumary/ai-recruitment-pipeline-dataset`):** 10,174 candidate records containing `ID`, `Name`, `Role`, `Transcript`, `Resume`, `decision`, `Reason_for_decision`, and `Job_Description`.
2. **Primary Survey Data (`google_form_recruitment_dirty.xlsx`):** 300 student responses with `Timestamp` (`Application_Date`), institutional/personal emails (`@chitkara.edu.in` & `@gmail.com`), and natural survey quirks (missing fields, duplicate submissions, inconsistent casing, whitespace, and placeholder strings).
3. **Preprocessing & Feature Engineering:**
   - Removed duplicates, standardized text/decision labels (`select` vs. `reject`), and dropped incomplete records.
   - Engineered length features (`word_count`, `resume_length`, `job_length`) scaled with `StandardScaler` and `MinMaxScaler`.
   - Computed lexical skill overlap (`relevance_score`) between candidate profiles and job descriptions.
   - Extracted top 30 n-gram features via `TfidfVectorizer` and ANOVA F-test (`SelectKBest`).

---

## Models & Algorithms Implemented

| Evaluation Unit | Techniques & Models | Metrics & Visualizations |
| :--- | :--- | :--- |
| **Supervised Regression** | Simple Linear, Multiple Linear, Polynomial Regression (deg=2) | MAE, MSE, RMSE, $R^2$ Score, Actual vs. Predicted Scatter |
| **Supervised Classification** | Logistic Regression, KNN, Decision Tree, Random Forest, Naïve Bayes, SVM | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Model Validation** | Stratified Train-Test Split (80/20), 5-Fold Cross-Validation | Bias-Variance Trade-off Curve (Underfitting vs. Overfitting) |
| **Unsupervised Clustering** | K-Means, Hierarchical (Agglomerative), DBSCAN | Silhouette Score, PCA 2D Cluster Scatter Plot |
| **Dimensionality Reduction** | Principal Component Analysis (PCA) | Explained Variance Ratio (`77.98%` in 2 components) |
| **Association & Recommender** | Apriori, FP-Growth, Cosine Similarity | Support, Confidence, Lift, Top-K Similar Candidates |
| **RL & Ethical AI** | Q-Learning Adaptive Interview Simulation, Role Fairness Audit | Learned Q-Table, Selection Rate Parity Bar Chart |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt