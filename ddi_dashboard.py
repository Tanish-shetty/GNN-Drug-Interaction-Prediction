import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GNN DDI Prediction | Tanish Shetty",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --red:   #C0392B;
    --dark:  #1A1A1A;
    --card:  #111111;
    --mid:   #222222;
    --border:#333333;
    --text:  #E8E8E8;
    --muted: #888888;
    --accent:#E74C3C;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark);
    color: var(--text);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0D0D0D;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metric cards */
.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--red);
    border-radius: 4px;
    padding: 20px 24px;
    margin-bottom: 4px;
}
.metric-card .label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-card .value {
    font-family: 'Space Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-card .sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 6px;
}

/* Section header */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--red);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 32px 0 20px;
}

/* Tag pills */
.tag {
    display: inline-block;
    background: #2A0A08;
    border: 1px solid var(--red);
    color: var(--accent);
    border-radius: 2px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    padding: 3px 10px;
    margin: 3px;
    letter-spacing: 1px;
}

/* Info box */
.info-box {
    background: #0D0D0D;
    border-left: 3px solid var(--red);
    border-radius: 0 4px 4px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.7;
    color: #CCCCCC;
}

/* Model comparison table */
.model-row {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    gap: 16px;
}
.model-row:last-child { border-bottom: none; }
.model-name {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    font-weight: 700;
    width: 120px;
    flex-shrink: 0;
}
.bar-wrap { flex: 1; background: #1E1E1E; height: 8px; border-radius: 2px; }
.bar-fill { height: 8px; border-radius: 2px; background: var(--red); }
.bar-fill.best { background: #27AE60; }
.acc-val {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #1A0A09 0%, #0D0D0D 60%, #0A1A0D 100%);
    border: 1px solid var(--border);
    border-top: 4px solid var(--red);
    border-radius: 4px;
    padding: 36px 40px;
    margin-bottom: 24px;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #FFFFFF;
    margin: 0 0 8px;
    line-height: 1.3;
}
.hero .subtitle {
    font-size: 14px;
    color: var(--muted);
    margin: 0;
}
.hero .meta {
    margin-top: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 1px;
}
.hero .meta span { color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 10px;'>
        <div style='font-family:Space Mono,monospace;font-size:11px;color:#C0392B;
                    letter-spacing:3px;text-transform:uppercase;margin-bottom:4px;'>
            Mini Project
        </div>
        <div style='font-family:Space Mono,monospace;font-size:16px;font-weight:700;
                    color:#FFFFFF;line-height:1.4;'>
            GNN DDI<br>Prediction
        </div>
    </div>
    <hr style='border-color:#333;margin:12px 0 20px;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔬 Models", "📈 Results", "🔭 Future Scope"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#333;margin:24px 0 16px;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px;color:#888;line-height:1.8;'>

    </div>
    """, unsafe_allow_html=True)

# ─── Helper: Dark matplotlib theme ────────────────────────────────────────────
def dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#1A1A1A")
    for spine in ax.spines.values():
        spine.set_color("#333333")
    ax.tick_params(colors="#888888")
    ax.xaxis.label.set_color("#888888")
    ax.yaxis.label.set_color("#888888")
    ax.title.set_color("#CCCCCC")
    return fig, ax

# ─── Data ─────────────────────────────────────────────────────────────────────
models   = ["KNN", "K-Means", "GraphSAGE", "GAT"]
accuracy  = [0.72, 0.65, 0.84, 0.91]
precision = [0.70, 0.62, 0.83, 0.90]
recall    = [0.71, 0.64, 0.82, 0.92]
f1        = [0.70, 0.63, 0.82, 0.91]

gat_epochs = list(range(1, 11))
gat_acc    = [0.58, 0.64, 0.70, 0.74, 0.78, 0.82, 0.85, 0.87, 0.89, 0.91]

# Confusion matrix (GAT)
cm = np.array([[90, 10], [8, 92]])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("""
    <div class='hero'>
        <h1>Graph Neural Network for<br>Drug–Drug Interaction Prediction</h1>
        <p class='subtitle'>
            A data-driven GNN approach to predicting DDIs using graph-structured biomedical data.
        </p>
        <div class='meta'>
            Dataset: <span>DrugBank DDI (Kaggle)</span> &nbsp;·&nbsp;
            Best Model: <span>Graph Attention Network (GAT)</span> &nbsp;·&nbsp;
            Best Accuracy: <span>91%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Accuracy",  "0.91", "GAT model"),
        ("Precision", "0.90", "GAT model"),
        ("Recall",    "0.92", "GAT model"),
        ("F1-Score",  "0.91", "GAT model"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4], kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value'>{val}</div>
                <div class='sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Problem Statement</div>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        <div class='info-box'>
            Drug–Drug Interactions (DDIs) can cause <b>severe side effects</b>, reduced drug
            effectiveness, and life-threatening conditions. Traditional DDI detection requires
            costly clinical trials that are time-consuming and not scalable for large drug libraries.
            <br><br>
            This project proposes a <b>Graph Neural Network-based model</b> to predict DDIs
            efficiently using structural and relational drug data — offering a faster alternative
            to laboratory testing.
        </div>
        """, unsafe_allow_html=True)
    with col_r:
        st.markdown("""
        <div class='info-box'>
            <b>Objectives:</b><br>
            • Analyze drug relationships using graph representation<br>
            • Compare traditional ML (KNN, K-Means) with GNN models<br>
            • Improve prediction accuracy using GraphSAGE and GAT<br>
            • Provide a scalable solution for healthcare applications
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.markdown("""
        <div class='metric-card'>
            <div class='label'>Source</div>
            <div style='font-family:Space Mono,monospace;font-size:18px;font-weight:700;
                        color:#E74C3C;margin-top:6px;'>DrugBank DDI</div>
            <div class='sub'>Kaggle Dataset</div>
        </div>""", unsafe_allow_html=True)
    with d2:
        st.markdown("""
        <div class='metric-card'>
            <div class='label'>Graph Nodes</div>
            <div style='font-family:Space Mono,monospace;font-size:18px;font-weight:700;
                        color:#E74C3C;margin-top:6px;'>Individual Drugs</div>
            <div class='sub'>Each drug = one node</div>
        </div>""", unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div class='metric-card'>
            <div class='label'>Graph Edges</div>
            <div style='font-family:Space Mono,monospace;font-size:18px;font-weight:700;
                        color:#E74C3C;margin-top:6px;'>Interactions</div>
            <div class='sub'>Known DDI pairs</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Tech Stack</div>", unsafe_allow_html=True)
    tags = ["Python", "NumPy", "Pandas", "Scikit-learn", "PyTorch", "PyG (PyTorch Geometric)",
            "GraphSAGE", "GAT", "Matplotlib", "Seaborn"]
    st.markdown(" ".join(f"<span class='tag'>{t}</span>" for t in tags), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Models":
    st.markdown("<div class='section-header'>Machine Learning Models Used</div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    model_cards = [
        ("KNN", "K-Nearest Neighbors",
         "Classifies drug pairs based on feature similarity. Acts as the baseline comparison model. Simple yet effective for establishing a performance floor.",
         "Baseline · Similarity-based · Supervised"),
        ("K-Means", "K-Means Clustering",
         "Groups similar drugs into clusters to help identify interaction patterns. Unsupervised approach — useful for exploratory analysis but lower accuracy on classification tasks.",
         "Unsupervised · Clustering · Exploratory"),
        ("GraphSAGE", "Graph Sample and Aggregate",
         "Aggregates neighborhood information from neighboring drug nodes to learn meaningful graph representations. Scalable to large drug interaction graphs.",
         "Inductive · Graph-based · Scalable"),
        ("GAT", "Graph Attention Network",
         "Uses attention mechanisms to assign different importance weights to neighboring nodes. Captures critical drug relationships and achieves the highest prediction accuracy.",
         "Attention · Best Model · 91% Accuracy"),
    ]

    for i, (short, full, desc, tags_str) in enumerate(model_cards):
        col = m1 if i % 2 == 0 else m2
        border_color = "#27AE60" if short == "GAT" else "#C0392B"
        with col:
            st.markdown(f"""
            <div style='background:#111;border:1px solid #333;border-top:3px solid {border_color};
                        border-radius:4px;padding:20px;margin-bottom:16px;'>
                <div style='display:flex;align-items:center;gap:12px;margin-bottom:12px;'>
                    <div style='font-family:Space Mono,monospace;font-size:20px;font-weight:700;
                                color:{border_color};'>{short}</div>
                    <div style='font-size:13px;color:#888;'>{full}</div>
                </div>
                <div style='font-size:13px;color:#CCC;line-height:1.7;margin-bottom:12px;'>{desc}</div>
                <div>{''.join(f"<span class='tag'>{t.strip()}</span>" for t in tags_str.split('·'))}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>GAT Working Mechanism</div>", unsafe_allow_html=True)
    steps = [
        ("01", "Feature Gathering", "Each drug node collects feature vectors from its neighboring nodes in the graph."),
        ("02", "Attention Score Calc", "Attention scores are computed between each pair of connected nodes using a learned function."),
        ("03", "Weight Assignment", "Higher attention weights are assigned to more important or influential neighboring nodes."),
        ("04", "Feature Aggregation", "Weighted neighbor features are aggregated into an updated node representation."),
        ("05", "DDI Prediction", "Aggregated representations are passed through a classifier to predict interaction presence."),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;align-items:flex-start;gap:16px;padding:14px 0;
                    border-bottom:1px solid #222;'>
            <div style='font-family:Space Mono,monospace;font-size:20px;font-weight:700;
                        color:#C0392B;flex-shrink:0;width:40px;'>{num}</div>
            <div>
                <div style='font-family:Space Mono,monospace;font-size:12px;color:#EEE;
                            margin-bottom:4px;'>{title}</div>
                <div style='font-size:13px;color:#888;'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Workflow Pipeline</div>", unsafe_allow_html=True)
    pipeline = ["Data Collection", "Data Preprocessing", "Graph Construction",
                "Feature Extraction", "Model Training", "Model Evaluation", "DDI Prediction"]
    cols = st.columns(len(pipeline))
    for i, (col, step) in enumerate(zip(cols, pipeline)):
        with col:
            st.markdown(f"""
            <div style='text-align:center;'>
                <div style='background:#C0392B;border-radius:50%;width:32px;height:32px;
                            line-height:32px;margin:0 auto 8px;font-family:Space Mono,monospace;
                            font-size:11px;font-weight:700;color:#fff;'>{i+1}</div>
                <div style='font-size:10px;color:#888;text-align:center;line-height:1.4;'>{step}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Results":
    st.markdown("<div class='section-header'>Model Performance Comparison</div>", unsafe_allow_html=True)

    # Bar chart - model accuracy
    fig, ax = dark_fig(9, 4)
    colors = ["#C0392B", "#922B21", "#1A5276", "#27AE60"]
    bars = ax.bar(models, accuracy, color=colors, width=0.5, edgecolor="#333", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Model Accuracy Comparison", fontsize=13, pad=15)
    for bar, val in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}", ha='center', va='bottom',
                fontsize=11, fontweight='bold', color='#CCCCCC',
                fontfamily='monospace')
    ax.axhline(y=0.90, color='#27AE60', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(3.4, 0.91, 'Best', color='#27AE60', fontsize=9, fontfamily='monospace')
    ax.grid(axis='y', color='#2A2A2A', linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("<div class='section-header'>All Metrics Comparison</div>", unsafe_allow_html=True)

    # Multi-metric line chart
    fig2, ax2 = dark_fig(9, 4)
    metrics_data = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1}
    palette = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    x = np.arange(len(models))
    for (metric, vals), color in zip(metrics_data.items(), palette):
        ax2.plot(x, vals, marker='o', color=color, label=metric,
                 linewidth=2, markersize=7, markerfacecolor=color)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontfamily='monospace', fontsize=11)
    ax2.set_ylim(0.5, 1.0)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Evaluation Metrics Across Models", fontsize=13, pad=15)
    ax2.legend(facecolor='#1A1A1A', edgecolor='#333', labelcolor='#CCC', fontsize=10)
    ax2.grid(color='#2A2A2A', linewidth=0.5)
    ax2.set_axisbelow(True)
    fig2.tight_layout()
    st.pyplot(fig2)

    col_cm, col_epoch = st.columns(2)

    with col_cm:
        st.markdown("<div class='section-header'>Confusion Matrix (GAT)</div>", unsafe_allow_html=True)
        fig3, ax3 = dark_fig(5, 4)
        im = ax3.imshow(cm, cmap='RdYlGn', vmin=0, vmax=100)
        ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
        ax3.set_xticklabels(["No Interaction", "Interaction"], color='#AAA', fontsize=10)
        ax3.set_yticklabels(["No Interaction", "Interaction"], color='#AAA', fontsize=10)
        ax3.set_xlabel("Predicted", fontsize=11)
        ax3.set_ylabel("Actual", fontsize=11)
        ax3.set_title("GAT Confusion Matrix", fontsize=12)
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, str(cm[i,j]), ha='center', va='center',
                         fontsize=18, fontweight='bold', color='#fff', fontfamily='monospace')
        fig3.tight_layout()
        st.pyplot(fig3)

    with col_epoch:
        st.markdown("<div class='section-header'>GAT Training Accuracy</div>", unsafe_allow_html=True)
        fig4, ax4 = dark_fig(5, 4)
        ax4.plot(gat_epochs, gat_acc, color='#C0392B', linewidth=2.5,
                 marker='o', markersize=7, markerfacecolor='#E74C3C',
                 markeredgecolor='#111', markeredgewidth=1.5)
        ax4.fill_between(gat_epochs, gat_acc, alpha=0.15, color='#C0392B')
        ax4.set_xlabel("Epoch", fontsize=11)
        ax4.set_ylabel("Accuracy", fontsize=11)
        ax4.set_title("GAT Training Accuracy Over Epochs", fontsize=12)
        ax4.set_ylim(0.5, 1.0)
        ax4.grid(color='#2A2A2A', linewidth=0.5)
        ax4.set_axisbelow(True)
        fig4.tight_layout()
        st.pyplot(fig4)

    # ── GAT Architecture Diagrams ──────────────────────────────────────────────
    st.markdown("<div class='section-header'>GAT Architecture Visualisation</div>", unsafe_allow_html=True)

    col_gat1, col_gat2 = st.columns(2)

    # ── Plot 1: Multi-head attention (like the top diagram in the image) ──────
    with col_gat1:
        fig_gat1, ax_gat1 = plt.subplots(figsize=(6, 5))
        fig_gat1.patch.set_facecolor("#111111")
        ax_gat1.set_facecolor("#111111")
        ax_gat1.set_aspect('equal')
        ax_gat1.axis('off')
        ax_gat1.set_title("Multi-Head Attention in GAT", fontsize=12,
                           color='#CCCCCC', pad=14, fontfamily='monospace')

        # Central node h1
        center = np.array([0.0, 0.0])
        # Neighbor positions (h2..h6)
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        neighbor_pos = [(1.8 * np.cos(a), 1.8 * np.sin(a)) for a in angles]
        neighbor_labels = [r'$\vec{h}_2$', r'$\vec{h}_3$', r'$\vec{h}_4$',
                           r'$\vec{h}_5$', r'$\vec{h}_6$', r'$\vec{h}_1$']
        # Attention weights (alpha values — vary them for visual interest)
        attn_weights = [0.35, 0.20, 0.10, 0.12, 0.08, 0.15]
        head_colors  = ['#E74C3C', '#3498DB', '#2ECC71']

        # Draw edges with colour per attention head
        for idx, (nx_, ny_) in enumerate(neighbor_pos):
            for hi, hc in enumerate(head_colors):
                offset = (hi - 1) * 0.06
                ax_gat1.annotate("",
                    xy=(center[0] + offset, center[1] + offset),
                    xytext=(nx_, ny_),
                    arrowprops=dict(
                        arrowstyle="->" if hi == 0 else "-",
                        color=hc,
                        lw=1.4 + attn_weights[idx] * 3,
                        alpha=0.6,
                        connectionstyle=f"arc3,rad={0.05*(hi-1)}"
                    )
                )
            # Attention label
            mid_x = (nx_ + center[0]) / 2
            mid_y = (ny_ + center[1]) / 2
            ax_gat1.text(mid_x, mid_y,
                         f'α={attn_weights[idx]:.2f}',
                         fontsize=7, color='#AAAAAA',
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.15', fc='#1A1A1A', ec='#333', lw=0.5))

        # Draw neighbor nodes
        for (nx_, ny_), lbl in zip(neighbor_pos, neighbor_labels):
            circ = plt.Circle((nx_, ny_), 0.30, color='#1A3A5C', ec='#3498DB', lw=1.5, zorder=4)
            ax_gat1.add_patch(circ)
            ax_gat1.text(nx_, ny_, lbl, ha='center', va='center',
                         fontsize=9, color='#AADCFF', zorder=5)

        # Draw central node
        circ_c = plt.Circle(center, 0.35, color='#5B1313', ec='#E74C3C', lw=2.5, zorder=5)
        ax_gat1.add_patch(circ_c)
        ax_gat1.text(0, 0, r'$\vec{h}_1$', ha='center', va='center',
                     fontsize=10, color='#FF8888', fontweight='bold', zorder=6)

        # Output arrow + label
        ax_gat1.annotate("", xy=(3.2, 0), xytext=(0.35, 0),
                          arrowprops=dict(arrowstyle="->", color='#3498DB', lw=2))
        ax_gat1.text(2.8, 0.18, "concat/avg", fontsize=8,
                     color='#3498DB', ha='center', fontfamily='monospace')
        # h1' output node
        circ_out = plt.Circle((3.6, 0), 0.30, color='#0D3321', ec='#2ECC71', lw=1.5, zorder=4)
        ax_gat1.add_patch(circ_out)
        ax_gat1.text(3.6, 0, r"$\vec{h}_1'$", ha='center', va='center',
                     fontsize=9, color='#88FFB0', zorder=5)

        # Legend for attention heads
        for hi, (hc, hlbl) in enumerate(zip(head_colors, ['Head 1','Head 2','Head 3'])):
            ax_gat1.plot([], [], color=hc, lw=2, label=hlbl)
        ax_gat1.legend(loc='lower left', facecolor='#1A1A1A', edgecolor='#333',
                       labelcolor='#CCC', fontsize=8)

        ax_gat1.set_xlim(-2.5, 4.3)
        ax_gat1.set_ylim(-2.5, 2.5)
        fig_gat1.tight_layout()
        st.pyplot(fig_gat1)
        st.markdown("""<div class='info-box' style='font-size:12px;'>
            <b style='color:#E74C3C;'>Multi-Head Attention</b> — Each coloured line represents a
            different attention head. The central node <b>h₁</b> aggregates weighted information
            from all neighbours. Multiple heads capture different relationship aspects simultaneously.
        </div>""", unsafe_allow_html=True)

    # ── Plot 2: Softmax attention mechanism (like the bottom diagram) ─────────
    with col_gat2:
        fig_gat2, ax_gat2 = plt.subplots(figsize=(6, 5))
        fig_gat2.patch.set_facecolor("#111111")
        ax_gat2.set_facecolor("#111111")
        ax_gat2.set_aspect('equal')
        ax_gat2.axis('off')
        ax_gat2.set_title("Attention Score Computation (Softmax)", fontsize=12,
                           color='#CCCCCC', pad=14, fontfamily='monospace')

        # Source node hi (left)
        src = np.array([0.5, 3.0])
        circ_src = plt.Circle(src, 0.30, color='#1A3A5C', ec='#3498DB', lw=1.5, zorder=4)
        ax_gat2.add_patch(circ_src)
        ax_gat2.text(*src, r'$h_i$', ha='center', va='center',
                     fontsize=10, color='#AADCFF', fontweight='bold', zorder=5)

        # Neighbour nodes hj1..hj4 (left column)
        nb_y = [4.5, 3.5, 2.5, 1.5]
        nb_labels = [r'$h_{j_1}$', r'$h_{j_2}$', r'$h_{j_3}$', r'$h_{j_4}$']
        attn_raw = [0.50, 0.25, 0.15, 0.10]   # softmax outputs
        nb_pos = []
        for y, lbl in zip(nb_y, nb_labels):
            pos = np.array([0.5, y])
            nb_pos.append(pos)
            circ_nb = plt.Circle(pos, 0.25, color='#1A3A5C', ec='#3498DB', lw=1.2, zorder=4)
            ax_gat2.add_patch(circ_nb)
            ax_gat2.text(*pos, lbl, ha='center', va='center',
                         fontsize=8, color='#AADCFF', zorder=5)
            # Line from neighbour to concatenation box
            ax_gat2.plot([pos[0]+0.25, 2.0], [pos[1], pos[1]],
                         color='#555', lw=1.0, zorder=2)
            # Line from hi to concat box
            ax_gat2.plot([src[0]+0.25, 2.0], [src[1], pos[1]],
                         color='#444', lw=0.8, linestyle='--', zorder=2)

        # Concatenation / linear transform box (yellow in original)
        rect = plt.Rectangle((2.0, 1.1), 0.5, 3.6,
                              color='#7D6608', ec='#F1C40F', lw=1.5, zorder=3)
        ax_gat2.add_patch(rect)
        ax_gat2.text(2.25, 3.0, r'$e_{ij}$', ha='center', va='center',
                     fontsize=9, color='#F1C40F', fontweight='bold',
                     rotation=90, zorder=4)

        # Arrow to softmax circle
        ax_gat2.annotate("", xy=(3.2, 3.0), xytext=(2.5, 3.0),
                          arrowprops=dict(arrowstyle="->", color='#F39C12', lw=1.5))
        # Softmax circle
        circ_sm = plt.Circle((3.55, 3.0), 0.35, color='#2C1810', ec='#E67E22', lw=1.5, zorder=4)
        ax_gat2.add_patch(circ_sm)
        ax_gat2.text(3.55, 3.0, 'Soft\nmax', ha='center', va='center',
                     fontsize=7, color='#F39C12', fontweight='bold', zorder=5)

        # Output alpha nodes (right)
        alpha_labels = [r'$α_{ij_1}$', r'$α_{ij_2}$', r'$α_{ij_3}$', r'$α_{ij_4}$']
        alpha_colors  = ['#E74C3C', '#C0392B', '#922B21', '#7B241C']
        out_x = 5.0
        for aidx, (ay, albl, acol, aw) in enumerate(zip(nb_y, alpha_labels, alpha_colors, attn_raw)):
            circ_a = plt.Circle((out_x, ay), 0.28, color='#2A0A08', ec=acol, lw=1.5, zorder=4)
            ax_gat2.add_patch(circ_a)
            ax_gat2.text(out_x, ay, albl, ha='center', va='center',
                         fontsize=7.5, color=acol, zorder=5)
            ax_gat2.text(out_x + 0.45, ay, f'{aw:.2f}',
                         ha='left', va='center', fontsize=7,
                         color='#888', fontfamily='monospace')
            # Arrow from softmax to alpha
            ax_gat2.annotate("", xy=(out_x - 0.28, ay), xytext=(3.9, 3.0),
                              arrowprops=dict(arrowstyle="->", color=acol,
                                             lw=1.0 + aw*2, alpha=0.7,
                                             connectionstyle="arc3,rad=0.1"))

        ax_gat2.set_xlim(-0.3, 6.0)
        ax_gat2.set_ylim(0.8, 5.2)
        fig_gat2.tight_layout()
        st.pyplot(fig_gat2)
        st.markdown("""<div class='info-box' style='font-size:12px;'>
            <b style='color:#E74C3C;'>Softmax Attention Scoring</b> — Raw attention scores
            <b>e_ij</b> are computed via a linear transform of concatenated node features,
            then normalised with Softmax to produce weights <b>α_ij</b> that sum to 1.
            Thicker arrows = higher attention weight assigned to that neighbour.
        </div>""", unsafe_allow_html=True)

    # Summary table
    st.markdown("<div class='section-header'>Performance Summary Table</div>", unsafe_allow_html=True)
    df = pd.DataFrame({
        "Model":     models,
        "Accuracy":  accuracy,
        "Precision": precision,
        "Recall":    recall,
        "F1-Score":  f1,
    })
    def highlight_best(s):
        is_max = s == s.max()
        return ['background-color:#0D2B0D; color:#27AE60; font-weight:bold' if v else '' for v in is_max]

    styled = df.style\
        .apply(highlight_best, subset=["Accuracy","Precision","Recall","F1-Score"])\
        .format({"Accuracy":"{:.2f}","Precision":"{:.2f}","Recall":"{:.2f}","F1-Score":"{:.2f}"})\
        .set_properties(**{"background-color":"#111","color":"#CCC","border":"1px solid #333"})\
        .set_table_styles([
            {"selector":"thead th","props":[("background-color","#1A1A1A"),("color","#C0392B"),
                                             ("font-family","Space Mono, monospace"),("font-size","11px"),
                                             ("letter-spacing","1px"),("text-transform","uppercase"),
                                             ("border","1px solid #333")]},
        ])\
        .hide(axis='index')
    st.dataframe(styled, use_container_width=True)

    # Insight callouts
    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
    insights = [
        ("GAT achieves 91% accuracy", "The attention mechanism lets GAT focus on the most influential neighboring drug nodes, yielding the highest performance across all metrics."),
        ("GraphSAGE outperforms classical ML", "By aggregating neighborhood information, GraphSAGE (84%) substantially improves over KNN (72%) and K-Means (65%)."),
        ("K-Means is weakest for classification", "Clustering is useful for exploration but unsuitable for direct binary DDI classification — reflected in its 65% accuracy."),
    ]
    for title, body in insights:
        st.markdown(f"""
        <div class='info-box'>
            <b style='color:#E74C3C;'>{title}</b><br>{body}
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FUTURE SCOPE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔭 Future Scope":
    st.markdown("<div class='section-header'>Conclusion</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        This project demonstrates that <b>Graph Neural Networks are highly effective</b> for predicting
        Drug–Drug Interactions. Among all models tested, the <b>Graph Attention Network (GAT)</b>
        provided the best performance — achieving <b>91% accuracy, 90% precision, 92% recall, and
        0.91 F1-score</b> — due to its ability to capture complex drug relationships and prioritize
        important interactions via the attention mechanism.<br><br>
        The study confirms that graph-based learning approaches significantly outperform traditional
        machine learning models like KNN and K-Means in handling relational biomedical data,
        making GNNs a promising solution for scalable DDI prediction systems.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Future Scope</div>", unsafe_allow_html=True)
    future_items = [
        ("📂", "Larger Datasets", "Use larger and more diverse biomedical drug databases beyond DrugBank to improve model generalizability."),
        ("🏥", "Real-time Integration", "Integrate the trained model into real-time hospital and pharmacy management systems for on-the-fly DDI alerts."),
        ("🧠", "Deep Learning Hybrids", "Combine GNNs with Transformers or CNNs to capture both structural and sequential drug properties simultaneously."),
        ("🌐", "Web/Mobile Deployment", "Deploy as a web-based or mobile medical tool accessible to clinicians and pharmacists."),
        ("🔧", "Hyperparameter Tuning", "Perform systematic hyperparameter optimization (learning rate, layers, heads) to further boost GAT performance."),
        ("🗣️", "NLP Integration", "Combine GNN with NLP to extract DDI signals from medical literature and drug package inserts automatically."),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(future_items):
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:#111;border:1px solid #333;border-radius:4px;
                        padding:20px;margin-bottom:16px;display:flex;gap:16px;align-items:flex-start;'>
                <div style='font-size:24px;flex-shrink:0;'>{icon}</div>
                <div>
                    <div style='font-family:Space Mono,monospace;font-size:13px;font-weight:700;
                                color:#E8E8E8;margin-bottom:6px;'>{title}</div>
                    <div style='font-size:13px;color:#888;line-height:1.6;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>References</div>", unsafe_allow_html=True)
    refs = [
        ("DrugBank DDI Dataset", "Kaggle", "https://www.kaggle.com/datasets/tnglmng/drugbank-ddi"),
        ("Graph Neural Networks for DDI Prediction", "Nature Scientific Reports (2025)", "https://www.nature.com/articles/s41598-025-12936-1"),
        ("Inductive Representation Learning on Large Graphs (GraphSAGE)", "Hamilton et al., 2017", "#"),
        ("Graph Attention Networks (GAT)", "Veličković et al., 2018", "#"),
        ("Scikit-learn: Machine Learning in Python", "Pedregosa et al., 2011", "#"),
    ]
    for title, source, url in refs:
        st.markdown(f"""
        <div style='padding:10px 0;border-bottom:1px solid #222;display:flex;
                    justify-content:space-between;align-items:center;'>
            <div>
                <div style='font-size:13px;color:#CCC;'>{title}</div>
                <div style='font-size:11px;color:#666;margin-top:2px;'>{source}</div>
            </div>
            {"<a href='" + url + "' target='_blank' style='font-family:Space Mono,monospace;font-size:10px;color:#C0392B;text-decoration:none;letter-spacing:1px;'>LINK →</a>" if url != "#" else ""}
        </div>""", unsafe_allow_html=True)
