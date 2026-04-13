# ============================================================
# DIAMOND PRICE ANALYSIS — STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Diamond Price Analysis",
    page_icon="💎",
    layout="wide"
)

import os
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'diamonds.csv')

# ============================================================
# REFERENCE GUIDES
# ============================================================

COLOR_GUIDE = {
    'D': 'D — Colorless (Best)',
    'E': 'E — Colorless',
    'F': 'F — Near Colorless',
    'G': 'G — Near Colorless (Most Common)',
    'H': 'H — Near Colorless',
    'I': 'I — Slightly Yellow',
    'J': 'J — Light Yellow (Lowest)'
}

CLARITY_GUIDE = {
    'IF':   'IF  — Internally Flawless (Best)',
    'VVS1': 'VVS1 — Very Very Slightly Included',
    'VVS2': 'VVS2 — Very Very Slightly Included',
    'VS1':  'VS1  — Very Slightly Included',
    'VS2':  'VS2  — Very Slightly Included (Common)',
    'SI1':  'SI1  — Slightly Included (Common)',
    'SI2':  'SI2  — Slightly Included',
    'I1':   'I1   — Included (Lowest)'
}

CUT_GUIDE = {
    'Fair':      'Fair      — Lowest quality cut',
    'Good':      'Good      — Below average cut',
    'Very Good': 'Very Good — Above average cut',
    'Premium':   'Premium   — Excellent cut',
    'Ideal':     'Ideal     — Best possible cut'
}

# ============================================================
# LOAD & PREPARE DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    df = df.drop_duplicates()
    df = df[(df['x'] > 0) & (df['y'] > 0) & (df['z'] > 0)]
    df = df[(df['y'] < 20) & (df['z'] < 20)].copy()
    df['log_price'] = np.log(df['price'])

    cut_order     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_order   = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    df['cut_encoded']     = df['cut'].map({v: i for i, v in enumerate(cut_order)})
    df['color_encoded']   = df['color'].map({v: i for i, v in enumerate(color_order)})
    df['clarity_encoded'] = df['clarity'].map({v: i for i, v in enumerate(clarity_order)})

    scale_cols = ['carat', 'depth', 'table', 'x', 'y', 'z',
                  'cut_encoded', 'color_encoded', 'clarity_encoded']
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df, df_scaled, scaler

df, df_scaled, scaler = load_data()

# ============================================================
# TRAIN MODELS ONCE
# ============================================================

@st.cache_resource
def train_all_models(_df, _df_scaled):
    scale_cols = ['carat', 'depth', 'table', 'x', 'y', 'z',
                  'cut_encoded', 'color_encoded', 'clarity_encoded']

    # Clustering
    X_cluster = _df_scaled[scale_cols].values
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels_arr = km.fit_predict(X_cluster)

    # Regression
    features_reg = ['carat', 'depth', 'table', 'x', 'y', 'z',
                    'cut_encoded', 'color_encoded', 'clarity_encoded']
    X_reg  = _df_scaled[features_reg].values
    y_reg  = _df['log_price'].values
    poly   = PolynomialFeatures(degree=2, include_bias=False)
    ridge  = Ridge(alpha=1.0)
    X_poly = poly.fit_transform(X_reg)
    ridge.fit(X_poly, y_reg)

    # Anomaly Detection
    features_anom = ['carat', 'depth', 'table', 'x', 'y', 'z',
                     'cut_encoded', 'color_encoded', 'clarity_encoded', 'log_price']
    X_anom_raw  = _df[features_anom].values
    scaler_anom = StandardScaler()
    X_anom      = scaler_anom.fit_transform(X_anom_raw)
    pca         = PCA(n_components=2, random_state=42)
    X_pca       = pca.fit_transform(X_anom)
    ocsvm       = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    ocsvm.fit(X_anom)
    pred_anom   = ocsvm.predict(X_anom)

    # Classification
    features_clf = ['carat', 'depth', 'table', 'x', 'y', 'z',
                    'color_encoded', 'clarity_encoded']
    X_clf = _df_scaled[features_clf].values
    y_clf = _df['cut_encoded'].values
    rf    = RandomForestClassifier(n_estimators=500, max_depth=20,
                                   min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_clf, y_clf)

    return km, cluster_labels_arr, poly, ridge, X_pca, pred_anom, rf

km, cluster_labels_arr, poly, ridge, X_pca, pred_anom, rf = train_all_models(df, df_scaled)

# Add results to df
df = df.copy()
df['cluster'] = cluster_labels_arr
cluster_map   = {0: 'Mid-range', 1: 'Budget', 2: 'Luxury', 3: 'Upper Mid-range'}
df['segment'] = df['cluster'].map(cluster_map)
df['anomaly'] = pred_anom
df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💎 Diamond Filters")
st.sidebar.markdown("Use these filters to explore the **Overview** tab.")
st.sidebar.markdown("---")

cut_filter = st.sidebar.multiselect(
    "✂️ Cut Quality",
    options=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
    default=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
)

price_filter = st.sidebar.slider(
    "💰 Price Range ($)",
    min_value=int(df['price'].min()),
    max_value=int(df['price'].max()),
    value=(int(df['price'].min()), int(df['price'].max()))
)

carat_filter = st.sidebar.slider(
    "⚖️ Carat Range",
    min_value=float(df['carat'].min()),
    max_value=float(df['carat'].max()),
    value=(float(df['carat'].min()), float(df['carat'].max()))
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 Quick Reference")
st.sidebar.markdown("""
**Cut (Best → Worst):**
Ideal → Premium → Very Good → Good → Fair

**Color (Best → Worst):**
D → E → F → G → H → I → J

**Clarity (Best → Worst):**
IF → VVS1 → VVS2 → VS1 → VS2 → SI1 → SI2 → I1
""")

df_filtered = df[
    (df['cut'].isin(cut_filter)) &
    (df['price'].between(price_filter[0], price_filter[1])) &
    (df['carat'].between(carat_filter[0], carat_filter[1]))
]

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔵 Clustering",
    "📈 Price Predictor",
    "🔴 Anomaly Detection",
    "🌲 Cut Predictor",
    "💡 Business Insights"
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:
    st.title("💎 Diamond Price Analysis Dashboard")
    st.markdown("**Data Mining Final Project | CRISP-DM Methodology | 53,772 Diamonds**")
    st.markdown("---")

    st.info("""
    👋 **Welcome! Here is how to use this dashboard:**

    • Use the **filters on the left sidebar** to explore specific diamonds
    • Click any **tab above** to switch between analyses
    • 📊 **Overview** — Dataset summary and key charts
    • 🔵 **Clustering** — How diamonds group into market segments
    • 📈 **Price Predictor** — Enter any diamond details to get an estimated price
    • 🔴 **Anomaly Detection** — Diamonds with unusual or suspicious pricing
    • 🌲 **Cut Predictor** — Predict the cut grade from physical measurements
    • 💡 **Business Insights** — Key findings and recommendations
    """)

    st.markdown("---")
    st.markdown("### 📊 Current Selection Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💎 Total Diamonds",  f"{len(df_filtered):,}")
    col2.metric("💰 Average Price",   f"${df_filtered['price'].mean():,.0f}")
    col3.metric("⚖️ Average Carat",   f"{df_filtered['carat'].mean():.2f}")
    col4.metric("🏆 Maximum Price",   f"${df_filtered['price'].max():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Price Distribution")
        fig = px.histogram(df_filtered, x='price', nbins=50,
                           color_discrete_sequence=['#1E88E5'])
        fig.update_layout(xaxis_title="Price (USD)",
                          yaxis_title="Number of Diamonds",
                          showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **What this chart shows:**
        Most diamonds are priced **under $2,500**.
        Very few diamonds are extremely expensive.
        This uneven distribution is called **right-skewed** — common in pricing data.
        """)

    with col2:
        st.markdown("#### Price vs Carat Weight")
        fig = px.scatter(df_filtered, x='carat', y='price',
                         color='cut', opacity=0.3,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(xaxis_title="Carat (Diamond Weight)",
                          yaxis_title="Price (USD)",
                          legend_title="Cut Quality",
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **What this chart shows:**
        Heavier diamonds (more carats) = higher price.
        Notice **clusters at 0.5, 1.0, 1.5, 2.0 carats** — buyers prefer round numbers!
        Each color = a different cut quality grade.
        """)

    st.markdown("---")
    st.markdown("### 📖 Diamond Features Guide")
    st.markdown("Not sure what Cut, Color, or Clarity mean? Here is a full explanation:")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**✂️ Cut Quality (Best → Worst)**")
        for k, v in CUT_GUIDE.items():
            st.markdown(f"• {v}")
        st.markdown("*Cut affects how well a diamond reflects light.*")

    with col2:
        st.markdown("**🎨 Color Grade (Best → Worst)**")
        for k, v in COLOR_GUIDE.items():
            st.markdown(f"• {v}")
        st.markdown("*D is completely colorless. J has a visible yellow tint.*")

    with col3:
        st.markdown("**🔍 Clarity Grade (Best → Worst)**")
        for k, v in CLARITY_GUIDE.items():
            st.markdown(f"• {v}")
        st.markdown("*IF is flawless. I1 has visible inclusions (imperfections).*")

# ============================================================
# TAB 2 — CLUSTERING
# ============================================================

with tab2:
    st.header("🔵 K-Means Clustering — Market Segments")
    st.markdown("**Goal:** Automatically group all 53,772 diamonds into natural market segments.")
    st.markdown("---")

    st.info("""
    **What is Clustering?**
    Clustering is a technique that groups similar items together — with no human labels needed.
    We used the **K-Means** algorithm which found **4 natural diamond groups**.
    Each group represents a different type of customer in the diamond market.

    **Features used:** Carat, Depth, Table, X, Y, Z dimensions, Cut, Color, Clarity
    **Algorithm:** K-Means with k=4 | **Silhouette Score: 0.2182**
    """)

    profile = df.groupby('segment').agg(
        Count     = ('price', 'size'),
        Avg_Price = ('price', 'mean'),
        Avg_Carat = ('carat', 'mean')
    ).round(2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💙 Budget",          f"${profile.loc['Budget','Avg_Price']:,.0f} avg",
                f"{profile.loc['Budget','Avg_Carat']:.2f} carat | {profile.loc['Budget','Count']:,} diamonds")
    col2.metric("🧡 Mid-range",       f"${profile.loc['Mid-range','Avg_Price']:,.0f} avg",
                f"{profile.loc['Mid-range','Avg_Carat']:.2f} carat | {profile.loc['Mid-range','Count']:,} diamonds")
    col3.metric("💚 Upper Mid-range", f"${profile.loc['Upper Mid-range','Avg_Price']:,.0f} avg",
                f"{profile.loc['Upper Mid-range','Avg_Carat']:.2f} carat | {profile.loc['Upper Mid-range','Count']:,} diamonds")
    col4.metric("💗 Luxury",          f"${profile.loc['Luxury','Avg_Price']:,.0f} avg",
                f"{profile.loc['Luxury','Avg_Carat']:.2f} carat | {profile.loc['Luxury','Count']:,} diamonds")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Clusters: Carat vs Price")
        fig = px.scatter(df, x='carat', y='price', color='segment',
                         opacity=0.3,
                         color_discrete_map={
                             'Budget':          '#2196F3',
                             'Mid-range':       '#FF9800',
                             'Upper Mid-range': '#4CAF50',
                             'Luxury':          '#E91E63'
                         })
        fig.update_layout(xaxis_title="Carat (Diamond Weight)",
                          yaxis_title="Price (USD)",
                          legend_title="Market Segment",
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **What this chart shows:**
        Each color = one market segment.
        **Carat weight clearly separates the 4 groups.**
        Bigger diamonds automatically fall into higher-value segments.
        This confirms that carat is the dominant pricing factor.
        """)

    with col2:
        st.markdown("#### Average Price per Market Segment")
        fig = px.bar(profile, x=profile.index, y='Avg_Price',
                     color=profile.index,
                     color_discrete_map={
                         'Budget':          '#2196F3',
                         'Mid-range':       '#FF9800',
                         'Upper Mid-range': '#4CAF50',
                         'Luxury':          '#E91E63'
                     })
        fig.update_layout(xaxis_title="Market Segment",
                          yaxis_title="Average Price (USD)",
                          showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **What this chart shows:**
        The Luxury segment is worth roughly **10x more** than the Budget segment.
        The Budget segment is by far the largest group.
        Mid-range and Upper Mid-range target similar buyers.
        """)

    st.markdown("---")
    st.markdown("#### Detailed Segment Summary Table")
    display_profile = profile.copy()
    display_profile.columns = ['Number of Diamonds', 'Average Price ($)', 'Average Carat']
    display_profile['Average Price ($)'] = display_profile['Average Price ($)'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display_profile, use_container_width=True)
    st.markdown("**Key finding:** Carat size is the main factor that separates all 4 segments.")

# ============================================================
# TAB 3 — REGRESSION / PRICE PREDICTOR
# ============================================================

with tab3:
    st.header("📈 Diamond Price Predictor")
    st.markdown("**Goal:** Estimate the fair market price of any diamond.")
    st.markdown("---")

    st.info("""
    **How to use this tool:**
    1. Set the diamond **Carat weight** using the slider — this is the most important factor!
    2. Select the **Cut, Color, and Clarity** grades from the dropdowns
    3. Expand **Advanced Settings** to adjust Depth and Table if known
    4. The estimated price updates **instantly** as you change any value

    **Model:** Polynomial Regression + Ridge | **R² = 0.9858 (98.58% accuracy)**
    Trained on 53,772 real diamonds from the Kaggle Diamonds dataset.
    """)

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Enter Diamond Details")

        carat_input = st.slider("⚖️ Carat (Diamond Weight)", 0.2, 5.0, 1.0, 0.01,
                                help="Weight of the diamond. 1 carat = 0.2 grams. Biggest impact on price!")

        cut_input = st.selectbox("✂️ Cut Quality",
                                 ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
                                 index=4,
                                 format_func=lambda x: CUT_GUIDE[x],
                                 help="Ideal is the best cut. Fair is the lowest quality.")

        color_input = st.selectbox("🎨 Color Grade",
                                   ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                                   index=3,
                                   format_func=lambda x: COLOR_GUIDE[x],
                                   help="D is colorless (best). J has the most yellow color.")

        clarity_input = st.selectbox("🔍 Clarity Grade",
                                     ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
                                     index=3,
                                     format_func=lambda x: CLARITY_GUIDE[x],
                                     help="IF is flawless (best). I1 has visible imperfections.")

        with st.expander("⚙️ Advanced Settings — Depth & Table (optional)"):
            st.markdown("""
            **Depth %** — Height of the diamond as a percentage of its width.
            Ideal range: **59% to 63%**

            **Table %** — Width of the flat top surface as a percentage of total width.
            Ideal range: **53% to 58%**

            If you are not sure, leave these at the default values.
            """)
            depth_input = st.slider("📏 Depth %", 43.0, 79.0, 61.7, 0.1)
            table_input = st.slider("📐 Table %", 43.0, 95.0, 57.5, 0.1)

        cut_map     = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
        color_map   = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
        clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

        avg_x = df[df['carat'].between(carat_input - 0.1, carat_input + 0.1)]['x'].mean()
        avg_y = df[df['carat'].between(carat_input - 0.1, carat_input + 0.1)]['y'].mean()
        avg_z = df[df['carat'].between(carat_input - 0.1, carat_input + 0.1)]['z'].mean()

        input_raw = pd.DataFrame([{
            'carat':           carat_input,
            'depth':           depth_input,
            'table':           table_input,
            'x':               avg_x if not np.isnan(avg_x) else 5.0,
            'y':               avg_y if not np.isnan(avg_y) else 5.0,
            'z':               avg_z if not np.isnan(avg_z) else 3.0,
            'cut_encoded':     cut_map[cut_input],
            'color_encoded':   color_map[color_input],
            'clarity_encoded': clarity_map[clarity_input]
        }])

        input_scaled = scaler.transform(input_raw)
        input_poly   = poly.transform(input_scaled)
        price_pred   = np.exp(ridge.predict(input_poly)[0])

        st.markdown("---")
        st.success(f"💰 Estimated Price: **${price_pred:,.0f}**")
        st.caption("Based on Polynomial + Ridge Regression | R² = 0.9858 | Trained on 53,772 diamonds")

    with col2:
        st.markdown("#### Price Reference Chart")
        fig = px.scatter(df, x='carat', y='price',
                         color='cut', opacity=0.15,
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig.add_trace(go.Scatter(
            x=[carat_input], y=[price_pred],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Your Diamond'
        ))
        fig.update_layout(xaxis_title="Carat (Diamond Weight)",
                          yaxis_title="Price (USD)",
                          legend_title="Cut Quality",
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **What this chart shows:**
        The **red star ⭐** marks your diamond's predicted position.
        Compare it to the cloud of real diamonds around it.
        If the star sits within the cloud — the price estimate is realistic.
        If it sits above — the diamond may be overpriced.
        If below — it may be a good deal.
        """)

# ============================================================
# TAB 4 — ANOMALY DETECTION
# ============================================================

with tab4:
    st.header("🔴 Anomaly Detection — Unusual Diamonds")
    st.markdown("**Goal:** Find diamonds whose price does not match their physical characteristics.")
    st.markdown("---")

    st.info("""
    **What is an Anomaly?**
    An anomalous diamond behaves very differently from all other diamonds.

    **Possible reasons:**
    - Diamond is **underpriced** for its quality → possible pricing error or great deal
    - Diamond is **overpriced** for its size → data error or premium branding
    - Diamond has **unusual dimensions** → rare gem or measurement error

    **Method used:** One-Class SVM + PCA
    **Result: 538 diamonds (1%) flagged for manual review**
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("✅ Normal Diamonds",    f"{(pred_anom == 1).sum():,}",  "99.00% of total")
    col2.metric("⚠️ Anomalous Diamonds", f"{(pred_anom == -1).sum():,}", "1.00% of total")
    col3.metric("📊 PCA Variance",        "66.0%", "Explained by 2 components")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### PCA View — Where Are the Anomalies?")
        plot_df = pd.DataFrame({
            'PC1':  X_pca[:, 0],
            'PC2':  X_pca[:, 1],
            'Type': ['Anomaly' if p == -1 else 'Normal' for p in pred_anom]
        })
        fig = px.scatter(plot_df, x='PC1', y='PC2', color='Type',
                         opacity=0.4,
                         color_discrete_map={'Normal': '#1E88E5', 'Anomaly': '#E53935'})
        fig.update_layout(legend_title="Diamond Type",
                          xaxis_title="Component 1 (summary of all features)",
                          yaxis_title="Component 2",
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **What this chart shows:**
        Each dot = one diamond. **Blue = normal, Red = anomaly.**
        Anomalies appear at the **outer edges** — far from the main cluster.
        The X and Y axes are mathematical summaries of all 10 features combined.
        Diamonds far from the center are statistically unusual.
        """)

    with col2:
        st.markdown("#### Anomalies in Real Terms — Price vs Carat")
        fig = px.scatter(df, x='carat', y='price',
                         color='anomaly_label',
                         opacity=0.3,
                         color_discrete_map={'Normal': '#1E88E5', 'Anomaly': '#E53935'})
        fig.update_layout(xaxis_title="Carat (Diamond Weight)",
                          yaxis_title="Price (USD)",
                          legend_title="Diamond Type",
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **What this chart shows:**
        Red dots = anomalous diamonds shown in real carat vs price space.
        These diamonds have a price that **does not match** expectations for their size.
        Some are suspiciously cheap — others are unusually expensive.
        """)

    st.markdown("---")
    st.markdown("#### ⚠️ Sample Anomalous Diamonds — Flagged for Review")

    anomaly_df = df[df['anomaly'] == -1][
        ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']
    ].head(10).copy().reset_index(drop=True)
    anomaly_df.index   = anomaly_df.index + 1
    anomaly_df['color']   = anomaly_df['color'].map(COLOR_GUIDE)
    anomaly_df['clarity'] = anomaly_df['clarity'].map(CLARITY_GUIDE)
    anomaly_df['price']   = anomaly_df['price'].apply(lambda x: f"${x:,}")
    anomaly_df.columns    = ['Carat', 'Cut', 'Color', 'Clarity', 'Depth %', 'Table %', 'Price']

    st.dataframe(anomaly_df, use_container_width=True)
    st.markdown("""
    **How to read this table:**
    - **Carat** — Diamond weight
    - **Cut / Color / Clarity** — Quality grades (see Overview tab for full guide)
    - **Depth % / Table %** — Physical dimension ratios
    - **Price** — Current listed price

    **Example — Row 2:** Good cut, VS1 clarity diamond priced at only $327.
    A VS1 clarity diamond with those features should normally cost much more.
    This suspicious pricing is why it was flagged as an anomaly.
    """)

# ============================================================
# TAB 5 — CLASSIFICATION / CUT PREDICTOR
# ============================================================

with tab5:
    st.header("🌲 Cut Grade Predictor")
    st.markdown("**Goal:** Predict the cut grade of a diamond from its physical measurements.")
    st.markdown("---")

    st.info("""
    **How to use this tool:**
    1. Enter the diamond's physical measurements using the sliders below
    2. Select the Color and Clarity grades
    3. The model instantly predicts the **Cut Grade**
    4. The confidence chart shows how certain the model is for each grade

    **Model:** Random Forest (500 trees) | **Accuracy: 78.11%**

    **Why not 100% accurate?**
    Cut grade is not fully determined by measurements alone.
    It also depends on the cutter's individual skill and company standards.
    Very Good and Premium grades have nearly identical dimensions — even experts sometimes disagree!
    """)

    st.markdown("---")

    features_clf = ['carat', 'depth', 'table', 'x', 'y', 'z', 'color_encoded', 'clarity_encoded']
    cut_labels   = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Enter Diamond Measurements")

        carat_c = st.slider("⚖️ Carat (Weight)", 0.2, 5.0, 1.0, 0.01,
                            help="Diamond weight in carats")
        depth_c = st.slider("📏 Depth %", 43.0, 79.0, 61.7, 0.1,
                            help="Height ÷ average width × 100. Ideal range: 59% to 63%")
        table_c = st.slider("📐 Table %", 43.0, 95.0, 57.5, 0.1,
                            help="Flat top width ÷ average width × 100. Ideal range: 53% to 58%")
        color_c = st.selectbox("🎨 Color Grade", ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                               format_func=lambda x: COLOR_GUIDE[x],
                               help="D = colorless (best). J = most yellow (lowest).")
        clarity_c = st.selectbox("🔍 Clarity Grade",
                                 ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'],
                                 format_func=lambda x: CLARITY_GUIDE[x],
                                 help="IF = flawless (best). I1 = visible inclusions (lowest).")

        color_map2   = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
        clarity_map2 = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

        avg_x2 = df[df['carat'].between(carat_c - 0.1, carat_c + 0.1)]['x'].mean()
        avg_y2 = df[df['carat'].between(carat_c - 0.1, carat_c + 0.1)]['y'].mean()
        avg_z2 = df[df['carat'].between(carat_c - 0.1, carat_c + 0.1)]['z'].mean()

        input_c = pd.DataFrame([{
            'carat':           carat_c,
            'depth':           depth_c,
            'table':           table_c,
            'x':               avg_x2 if not np.isnan(avg_x2) else 5.0,
            'y':               avg_y2 if not np.isnan(avg_y2) else 5.0,
            'z':               avg_z2 if not np.isnan(avg_z2) else 3.0,
            'color_encoded':   color_map2[color_c],
            'clarity_encoded': clarity_map2[clarity_c]
        }])

        input_c_scaled = scaler.transform(
            input_c.assign(cut_encoded=0)[
                ['carat', 'depth', 'table', 'x', 'y', 'z',
                 'cut_encoded', 'color_encoded', 'clarity_encoded']
            ]
        )[:, [0, 1, 2, 3, 4, 5, 7, 8]]

        pred_cut       = rf.predict(input_c_scaled)[0]
        prob_cut       = rf.predict_proba(input_c_scaled)[0]
        predicted_label = cut_labels[pred_cut]

        st.markdown("---")
        st.success(f"✂️ Predicted Cut Grade: **{predicted_label}**")
        st.markdown(f"*{CUT_GUIDE[predicted_label]}*")

    with col2:
        st.markdown("#### Model Confidence per Cut Grade")
        prob_df = pd.DataFrame({'Cut Grade': cut_labels, 'Confidence': prob_cut})
        fig = px.bar(prob_df, x='Cut Grade', y='Confidence',
                     color='Cut Grade',
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(yaxis_tickformat='.0%',
                          yaxis_title="Confidence (%)",
                          xaxis_title="Cut Grade",
                          showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **How to read this chart:**
        The **tallest bar** = the model's predicted cut grade.
        A bar close to 100% = the model is very confident.
        Multiple bars at similar height = the model is uncertain between those grades.
        Very Good and Premium often show similar confidence — their dimensions overlap.
        """)

        st.markdown("#### What Features Drive the Cut Prediction?")
        importances = pd.Series(rf.feature_importances_, index=features_clf)
        feature_names = {
            'carat':           '⚖️ Carat',
            'depth':           '📏 Depth %',
            'table':           '📐 Table %',
            'x':               '📍 X (Length)',
            'y':               '📍 Y (Width)',
            'z':               '📍 Z (Height)',
            'color_encoded':   '🎨 Color',
            'clarity_encoded': '🔍 Clarity'
        }
        importances.index = [feature_names[f] for f in importances.index]
        fig = px.bar(importances.sort_values(), orientation='h',
                     color_discrete_sequence=['#1E88E5'])
        fig.update_layout(xaxis_title="Importance Score",
                          yaxis_title="",
                          showlegend=False,
                          plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Table % and Depth %** are the most important features for predicting cut grade.
        They directly define the geometry of how a diamond is cut.
        Color and Clarity have almost no impact on cut prediction — as expected.
        """)

# ============================================================
# TAB 6 — BUSINESS INSIGHTS
# ============================================================

with tab6:
    st.header("💡 Business Insights & Recommendations")
    st.markdown("**Top 3 actionable findings from the complete data mining analysis**")
    st.markdown("---")

    st.markdown("### 🏆 Finding 1: Carat Weight Drives Everything")

    col1, col2, col3 = st.columns(3)
    col1.metric("Correlation with Price", "0.92",   "Strongest of all features")
    col2.metric("Regression R²",          "0.9858", "98.58% price accuracy")
    col3.metric("Top Feature",            "Carat",  "Dominant in all 4 models")

    st.markdown("""
    Carat weight is the **single most powerful predictor** of diamond price.
    - A 1-carat diamond costs roughly **5x more** than a 0.5-carat diamond
    - Cut, Color, and Clarity do matter — but carat dominates above all
    - Our regression model predicts price with **98.58% accuracy**

    **Recommendation for retailers:**
    Price primarily by carat weight, then fine-tune based on cut, color, and clarity.
    """)

    st.markdown("---")
    st.markdown("### 🎯 Finding 2: 4 Clear Customer Segments")

    seg_data = df.groupby('segment').agg(
        Count     = ('price', 'size'),
        Avg_Price = ('price', 'mean'),
        Avg_Carat = ('carat', 'mean')
    ).round(2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💙 Budget",
                f"${seg_data.loc['Budget','Avg_Price']:,.0f} avg",
                f"{seg_data.loc['Budget','Count']:,} diamonds ({seg_data.loc['Budget','Count']/len(df)*100:.0f}%)")
    col2.metric("🧡 Mid-range",
                f"${seg_data.loc['Mid-range','Avg_Price']:,.0f} avg",
                f"{seg_data.loc['Mid-range','Count']:,} diamonds ({seg_data.loc['Mid-range','Count']/len(df)*100:.0f}%)")
    col3.metric("💚 Upper Mid-range",
                f"${seg_data.loc['Upper Mid-range','Avg_Price']:,.0f} avg",
                f"{seg_data.loc['Upper Mid-range','Count']:,} diamonds ({seg_data.loc['Upper Mid-range','Count']/len(df)*100:.0f}%)")
    col4.metric("💗 Luxury",
                f"${seg_data.loc['Luxury','Avg_Price']:,.0f} avg",
                f"{seg_data.loc['Luxury','Count']:,} diamonds ({seg_data.loc['Luxury','Count']/len(df)*100:.0f}%)")

    st.markdown("""
    K-Means clustering found **4 natural customer segments** in the diamond market.
    - Budget segment is the largest — most buyers want affordable small diamonds
    - Luxury segment has the highest margin per diamond
    - Carat size is the main driver separating all segments

    **Recommendation:**
    Design separate marketing, pricing, and inventory strategies for each segment.
    Focus on volume for Budget, and on premium service for Luxury.
    """)

    st.markdown("---")
    st.markdown("### ⚠️ Finding 3: 538 Diamonds Need Immediate Review")

    col1, col2 = st.columns(2)
    col1.metric("Diamonds Flagged",  "538",                 "1% of total dataset")
    col2.metric("Detection Method",  "One-Class SVM + PCA", "Anomaly Detection")

    st.markdown("""
    Our anomaly detection model flagged **538 diamonds (1%)** with unusual patterns.
    - Some are **underpriced** for their quality → possible pricing errors to correct
    - Some have **extreme or unusual dimensions** → possible data entry errors
    - Some may be **rare gems** → worth highlighting as premium items

    **Recommendation:**
    Have a gemologist manually review all 538 flagged diamonds.
    Correcting pricing errors alone could recover significant revenue.
    """)

    st.markdown("---")
    st.markdown("### 📊 Full Model Performance Summary")

    results_df = pd.DataFrame({
        'Technique':      ['Clustering',       'Regression',         'Anomaly Detection',    'Classification'],
        'Algorithm':      ['K-Means (k=4)',    'Polynomial + Ridge', 'One-Class SVM + PCA',  'Random Forest (500 trees)'],
        'Key Metric':     ['Silhouette 0.2182','R² = 0.9858',        '538 anomalies (1%)',   'Accuracy 78.11%'],
        'Business Goal':  ['4 segments ✅',    'R² > 0.90 ✅',       'Detect anomalies ✅',  'Accuracy > 85% ❌']
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Note on Classification (78.11%):**
    Cut grade depends on cutter skill, not just measurements.
    Very Good and Premium grades overlap in feature space — even experts sometimes disagree.
    78.11% is considered strong performance for this specific problem.
    """)

    st.markdown("---")
    st.success("💎 Diamond Price Analysis Dashboard | Data Mining Final Project | Python & Streamlit")