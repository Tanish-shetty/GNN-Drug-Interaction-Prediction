# 💊 GNN-Based Drug–Drug Interaction (DDI) Prediction

An interactive **Streamlit dashboard** that predicts **Drug–Drug Interactions (DDIs)** using **Graph Neural Networks (GNNs)**. The project demonstrates how graph-based deep learning models outperform traditional machine learning techniques for biomedical interaction prediction through an intuitive analytics dashboard.

---

## 📖 Overview

Drug–Drug Interaction (DDI) prediction plays a crucial role in reducing adverse drug reactions and improving patient safety. Since drugs naturally form interconnected networks through their interactions, **Graph Neural Networks (GNNs)** provide an effective way to learn these complex relationships.

This project compares traditional machine learning approaches with graph-based deep learning models and presents the results through an interactive Streamlit application.

---

## ✨ Features

* 🎨 Interactive **Streamlit** dashboard
* 🌙 Modern dark-themed UI
* 📊 Interactive charts and performance visualizations
* 🧠 Comparison of classical ML and GNN models
* 📈 Confusion matrix and evaluation metrics
* 🏆 Best-performing model highlighted with detailed insights
* 📱 Responsive and easy-to-use interface

---

## 🎯 Objectives

* Predict Drug–Drug Interactions using graph learning.
* Compare traditional machine learning algorithms with Graph Neural Networks.
* Visualize model performance through interactive analytics.
* Demonstrate the advantages of attention-based graph learning in biomedical applications.

---

## 🧠 Methodology

### 1. Graph Construction

* Drugs are represented as **nodes**
* Drug interactions are represented as **edges**
* The interaction network forms a biomedical graph

### 2. Feature Learning

Graph Neural Networks generate embeddings by aggregating information from neighboring drugs.

### 3. Models Evaluated

| Model                             | Description                                                         |
| --------------------------------- | ------------------------------------------------------------------- |
| **KNN**                           | Distance-based classification baseline                              |
| **K-Means**                       | Unsupervised clustering baseline                                    |
| **GraphSAGE**                     | Neighborhood aggregation Graph Neural Network                       |
| **Graph Attention Network (GAT)** | Attention-based GNN for learning important neighboring interactions |

### 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 📊 Dashboard Overview

### 🏠 Home

* Project overview
* Drug interaction problem statement
* Dataset summary
* Key performance indicators

### 🤖 Models

* Introduction to traditional ML models
* Graph Neural Network concepts
* Working principles of GraphSAGE and GAT

### 📈 Results

* Model comparison charts
* Accuracy trends
* Confusion matrix
* Precision, Recall and F1-score comparison
* Best model analysis

### 🔮 Future Scope

* Larger biomedical datasets
* Explainable AI for healthcare
* Transformer + GNN hybrid architectures
* Cloud deployment
* Clinical decision support integration

---

## 📊 Results

| Model                             | Accuracy         |
| --------------------------------- | ---------------- |
| KNN                               | Baseline         |
| K-Means                           | Baseline         |
| GraphSAGE                         | High Performance |
| **Graph Attention Network (GAT)** | **~91%** ⭐       |

### Key Findings

* ✅ Graph-based learning significantly outperforms traditional machine learning.
* ✅ Graph Attention Networks capture complex drug relationships more effectively.
* ✅ Attention mechanisms improve prediction accuracy by focusing on important neighboring nodes.
* ✅ GAT achieved the best overall performance with approximately **91% accuracy**.

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
├── models/                 # ML & GNN model implementations
├── notebooks/              # Experiments and training notebooks
├── results/                # Saved outputs and visualizations
├── assets/                 # Images and dashboard resources
├── requirements.txt        # Project dependencies
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

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will start locally and open in your default web browser.

---

## 🚀 Future Enhancements

* Integrate larger public biomedical datasets
* Support multi-class DDI prediction
* Add Explainable AI (XAI) visualizations
* Deploy using Docker and Streamlit Cloud
* REST API integration
* Real-time prediction interface
* Transformer-GNN hybrid architectures
* Drug recommendation system

---

## 🤝 Contributing

Contributions are welcome!

If you would like to improve the project:

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/YourFeature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature/YourFeature
```

5. Open a Pull Request

---

## 📜 License

This project is intended for educational and research purposes. You may modify and use it under the appropriate open-source license of your choice.

---

## 👨‍💻 Author

**Tanish Shetty**

If you found this project helpful, consider giving the repository a ⭐ on GitHub!
