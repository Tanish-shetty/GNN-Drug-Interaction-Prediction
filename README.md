# 💊 GNN-Based Drug–Drug Interaction (DDI) Prediction

An interactive **Streamlit dashboard** that predicts **Drug–Drug Interactions (DDIs)** using **Graph Neural Networks (GNNs)**. This project demonstrates how graph-based deep learning models outperform traditional machine learning approaches for biomedical interaction prediction through an intuitive analytics dashboard.

---

## 📖 Overview

Drug–Drug Interaction (DDI) prediction is an essential task in healthcare, helping identify potentially harmful interactions before medications are prescribed. Since drugs naturally form interconnected networks, **Graph Neural Networks (GNNs)** provide an effective way to model these complex relationships.

This project compares traditional machine learning algorithms with graph-based deep learning models and presents the results through an interactive Streamlit dashboard.

---

## ✨ Features

* 🎨 Interactive Streamlit dashboard
* 🌙 Modern dark-themed interface
* 📊 Interactive charts and performance visualizations
* 🧠 Comparison of classical ML and GNN models
* 📈 Confusion matrix and evaluation metrics
* 🏆 Highlights the best-performing model
* 📱 Clean and responsive UI

---

## 🎯 Objectives

* Predict Drug–Drug Interactions using graph learning.
* Compare traditional machine learning algorithms with Graph Neural Networks.
* Visualize model performance through interactive analytics.
* Demonstrate the effectiveness of attention-based graph learning in biomedical applications.

---

## 🧠 Methodology

### 1. Graph Construction

* Drugs are represented as **nodes**
* Drug interactions are represented as **edges**
* The interaction network forms a biomedical graph

### 2. Feature Learning

Graph Neural Networks learn node embeddings by aggregating information from neighboring drugs.

### 3. Models Used

| Model                             | Description                                                                       |
| --------------------------------- | --------------------------------------------------------------------------------- |
| **KNN**                           | Distance-based classification baseline                                            |
| **K-Means**                       | Unsupervised clustering baseline                                                  |
| **GraphSAGE**                     | Graph Neural Network using neighborhood aggregation                               |
| **Graph Attention Network (GAT)** | Attention-based Graph Neural Network that assigns importance to neighboring nodes |

### 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

# 🏗️ System Architecture

```text
                           ┌──────────────────────┐
                           │  Drug Interaction    │
                           │      Dataset         │
                           └──────────┬───────────┘
                                      │
                                      ▼
                        ┌──────────────────────────┐
                        │ Data Preprocessing       │
                        │ • Data Cleaning          │
                        │ • Feature Engineering    │
                        │ • Feature Encoding       │
                        └──────────┬───────────────┘
                                   │
                                   ▼
                      ┌────────────────────────────┐
                      │ Graph Construction         │
                      │ • Drugs → Nodes            │
                      │ • Interactions → Edges     │
                      └──────────┬─────────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────────┐
                  │ Model Training                  │
                  │                                 │
                  │ • KNN                           │
                  │ • K-Means                       │
                  │ • GraphSAGE                     │
                  │ • GAT                           │
                  └──────────┬──────────────────────┘
                             │
                             ▼
                  ┌────────────────────────────┐
                  │ Model Evaluation           │
                  │ • Accuracy                 │
                  │ • Precision                │
                  │ • Recall                   │
                  │ • F1-Score                 │
                  │ • Confusion Matrix         │
                  └──────────┬─────────────────┘
                             │
                             ▼
                ┌──────────────────────────────────┐
                │ Streamlit Interactive Dashboard  │
                │                                  │
                │ • KPI Cards                      │
                │ • Charts & Graphs                │
                │ • Model Comparison               │
                │ • Performance Insights           │
                └──────────────────────────────────┘
```

---

# 🔄 Project Workflow

```text
Dataset
   │
   ▼
Data Cleaning & Feature Extraction
   │
   ▼
Graph Construction
(Nodes = Drugs, Edges = Drug Interactions)
   │
   ▼
Graph Representation Learning
(GraphSAGE / GAT)
   │
   ▼
Model Training
   │
   ▼
Prediction
   │
   ▼
Performance Evaluation
   │
   ▼
Interactive Streamlit Dashboard
```

### Workflow Steps

1. **Dataset Collection**

   * Load the Drug–Drug Interaction dataset.
   * Identify drugs and their known interactions.

2. **Data Preprocessing**

   * Handle missing values.
   * Encode and normalize features.
   * Prepare data for graph construction.

3. **Graph Construction**

   * Represent drugs as nodes.
   * Represent interactions as edges.
   * Build the biomedical interaction graph.

4. **Model Training**

   * Train baseline models (**KNN** and **K-Means**).
   * Train graph neural networks (**GraphSAGE** and **GAT**).
   * Learn graph embeddings.

5. **Prediction**

   * Predict whether two drugs interact.
   * Generate interaction probabilities.

6. **Evaluation**

   * Compare all models using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

7. **Visualization**

   * Present results through the Streamlit dashboard using interactive charts and model comparisons.

---

## 📊 Dashboard Sections

### 🏠 Overview

* Project introduction
* Drug interaction problem statement
* Dataset summary
* Key performance indicators

### 🤖 Models

* Traditional Machine Learning models
* Graph Neural Network concepts
* GraphSAGE and GAT overview

### 📈 Results

* Accuracy comparison
* Multi-metric comparison
* Confusion matrix
* Model-wise analysis
* Best model insights

### 🔮 Future Scope

* Hybrid GNN + Transformer architectures
* Larger biomedical datasets
* Explainable AI (XAI)
* Clinical decision support integration
* Cloud deployment

---

## 📊 Results Summary

| Model                             | Performance         |
| --------------------------------- | ------------------- |
| KNN                               | Baseline            |
| K-Means                           | Baseline            |
| GraphSAGE                         | High Performance    |
| **Graph Attention Network (GAT)** | **~91% Accuracy ⭐** |

### Key Findings

* Graph-based models outperform traditional machine learning approaches.
* GAT captures complex relationships more effectively through its attention mechanism.
* Attention-based learning improves prediction accuracy by prioritizing important neighboring nodes.
* **Graph Attention Network (GAT)** achieved the highest overall performance with approximately **91% accuracy**.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Framework

* Streamlit

### Machine Learning

* Scikit-learn
* PyTorch
* PyTorch Geometric

### Data Processing

* NumPy
* Pandas

### Visualization

* Matplotlib
* Seaborn

---

## 📂 Project Structure

```text
GNN-DDI-Prediction/
│
├── app.py                  # Main Streamlit application
├── data/                   # Dataset files
├── models/                 # ML & GNN implementations
├── notebooks/              # Experiments
├── results/                # Outputs & visualizations
├── assets/                 # Images and resources
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/GNN-DDI-Prediction.git
```

Navigate to the project directory:

```bash
cd GNN-DDI-Prediction
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Once launched, the dashboard will open in your default web browser.

---

## 🚀 Future Enhancements

* Integration with larger biomedical datasets
* Multi-class DDI prediction
* Explainable AI (XAI) visualizations
* REST API support
* Docker and Streamlit Cloud deployment
* Real-time prediction system
* Transformer-GNN hybrid models
* Drug recommendation system

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new feature branch.

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push the branch.

```bash
git push origin feature/YourFeature
```

5. Open a Pull Request.
