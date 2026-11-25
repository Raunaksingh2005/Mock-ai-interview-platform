# Mock AI Interview Platform — End-to-End ML Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Raunaksingh2005/Mock-ai-interview-platform/blob/main/Mock_Ai_Interview_Platfrom_project.ipynb)

An end-to-end Machine Learning recruitment evaluation pipeline that integrates the Kaggle **AI Recruitment Pipeline Dataset** (`yaswanthkumary/ai-recruitment-pipeline-dataset`) with primary student survey data (`google_form_recruitment_dirty.xlsx`), cleans real-world data impurities, and benchmarks regression, classification, clustering, association rule mining, reinforcement learning, and ethical bias auditing.

---

## Dataset & Preprocessing Pipeline

1. **Data Integration:** Merged 10,174 benchmark candidate records with 300 primary Google Form responses (`@chitkara.edu.in` & `@gmail.com`), aligning schemas (`ID`, `Name`, `Role`, `Transcript`, `Resume`, `decision`, `Reason_for_decision`, `Job_Description`, `Application_Date`).
2. **Data Cleaning:** Removed duplicate submissions, standardized inconsistent casing/labels (`Select`, `SELECT `, ` selected` → `select`), replaced placeholder junk (`N/A`, `---`, `?`), and dropped incomplete entries, yielding **10,334 clean records** (`50.1%` selected, `49.9%` rejected).
3. **Feature Engineering & Selection:**
   - Scaled length features (`word_count`, `resume_length`, `job_length`) using `StandardScaler` and `MinMaxScaler`.
   - Engineered a lexical skill-overlap `relevance_score` comparing candidate transcripts/resumes against job descriptions.
   - Extracted top 30 n-gram features via `TfidfVectorizer(ngram_range=(1, 2))` and ANOVA F-test (`SelectKBest`), producing a **34-feature matrix**.

---

## Experimental Results

### 1. Regression Models (Predicting `relevance_score`)
| Model | MAE | MSE | RMSE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| Simple Linear Regression | 0.0987 | 0.0163 | 0.1277 | 0.0219 |
| Multiple Linear Regression | 0.0834 | 0.0122 | 0.1105 | 0.2683 |
| **Polynomial Regression (deg=2)** | **0.0816** | **0.0117** | **0.1083** | **0.2971** |

### 2. Classification Models & 5-Fold Cross-Validation
| Model | Test Accuracy | Precision | Recall | F1-Score | 5-Fold CV |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **63.8%** | **0.64** | **0.65** | **0.64** | **59.0%** |
| SVM (RBF) | 61.9% | 0.59 | 0.78 | 0.67 | 56.3% |
| Logistic Regression | 61.0% | 0.60 | 0.67 | 0.63 | 58.8% |
| Naïve Bayes | 60.1% | 0.60 | 0.60 | 0.60 | 54.2% |
| Decision Tree (depth=5) | 58.7% | 0.56 | 0.84 | 0.67 | 55.5% |
| KNN (k=5) | 58.2% | 0.58 | 0.58 | 0.58 | 53.1% |

### 3. Unsupervised Learning, Association Rules & Reinforcement Learning
- **PCA & Clustering:** 2 principal components retain **77.98%** of total variance; both K-Means and Hierarchical Clustering achieve a **0.644 Silhouette Score**.
- **Association Rule Mining (FP-Growth & Apriori):** High-confidence keyword co-occurrence rules such as `{great} → {experience}` (`Support = 0.954`, `Confidence = 0.9997`).
- **Cosine Similarity Recommender:** Retrieves the top-3 closest candidate profiles for peer benchmarking.
- **Reinforcement Learning (Q-Learning) & Ethics:** Adaptive interview difficulty agent learns optimal progression policies (`Level Up` in Easy/Medium states, `Stay` in Hard state) alongside a role-wise selection rate fairness audit.

---

## Team Contributions
- **Data Collection, Cleaning & Missing Values:** Pratham Maini (2410990661)
- **Data Integration, Transformation & Reduction:** Ridhi Jain (2410990687)
- **Feature Scaling, Engineering, Extraction & Selection:** Raunak Singh (2410990679)
- **Model Training, Evaluation & Validation:** Sukhamrit Singh (2410990719)
