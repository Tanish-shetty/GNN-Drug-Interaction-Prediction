# GNN for Drug–Drug Interaction (DDI) Prediction

## 📌 Overview

An interactive **Streamlit dashboard** for predicting Drug–Drug Interactions (DDIs) using **Graph Neural Networks (GNNs)**. The app visualizes model performance, compares classical ML vs GNN models, and highlights insights using graph-based biomedical data.

## 🎯 Key Highlights

* Interactive UI built with **Streamlit**
* Dark-themed analytics dashboard
* Comparison of **KNN, K-Means, GraphSAGE, and GAT**
* Best model: **Graph Attention Network (GAT)** with ~91% accuracy
* Visualizations: accuracy trends, confusion matrix, multi-metric plots

## 🧠 Methodology

1. **Graph Construction**: Drugs → nodes, interactions → edges
2. **Feature Learning**: Node embeddings via GNNs
3. **Models Used**:

   * KNN (baseline)
   * K-Means (clustering)
   * GraphSAGE (graph aggregation)
   * GAT (attention-based learning)
4. **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score

## 📊 Dashboard Sections

* **Overview**: KPIs, dataset info, problem statement
* **Models**: Explanation of ML & GNN models
* **Results**: Graphs, confusion matrix, performance comparison
* **Future Scope**: Improvements and applications

## 🛠️ Tech Stack

* Python
* Streamlit
* NumPy, Pandas
* Matplotlib, Seaborn
* Scikit-learn
* PyTorch / PyTorch Geometric

## 📂 Project Structure

```
├── app.py               # Main Streamlit dashboard
├── data/                # Dataset files
├── models/              # Model implementations
├── notebooks/           # Experiments
├── results/             # Outputs & visualizations
├── requirements.txt
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app.py
```

## 📈 Results Summary

* **GAT** achieves highest performance (~91% accuracy)
* Graph-based models outperform traditional ML
* Attention mechanism improves interaction prediction

## 🔮 Future Scope

* Larger biomedical datasets
* Real-time healthcare integration
* Hybrid models (GNN + Transformers)
* Web/mobile deployment

## 🤝 Contributing

Open to contributions. Fork the repo and create a pull request.




