import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="Career Intelligence Dashboard")

# 2. Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv("employee_data.csv")
    # Feature Engineering
    df['Promotion_Gap_Ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Role_Stagnation_Index'] = df['YearsInCurrentRole'] / (df['YearsAtCompany'] + 1)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'employee_data.csv' not found. Please ensure it is in the same folder as app.py.")
    st.stop()

# 3. Sidebar Filtering
st.sidebar.header("Dashboard Filters")
all_roles = df['JobRole'].unique().tolist()
selected_roles = st.sidebar.multiselect("Select Job Roles", all_roles, default=all_roles)

# Apply filter
df_filtered = df[df['JobRole'].isin(selected_roles)].copy()

# Safety Check: Prevent clustering on empty data
if df_filtered.empty:
    st.warning("No data available for the selected job roles. Please select at least one role.")
    st.stop()

# 4. Refined Clustering & Labeling
features = ['Promotion_Gap_Ratio', 'Role_Stagnation_Index', 'PerformanceRating']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_filtered[features])

kmeans = KMeans(n_clusters=4, random_state=42)
df_filtered['Cluster_ID'] = kmeans.fit_predict(scaled_data)

def map_career_labels(row):
    if row['PerformanceRating'] >= 3.5 and row['Promotion_Gap_Ratio'] < 0.3:
        return "Career Accelerator"
    elif row['PerformanceRating'] >= 3.5 and row['Promotion_Gap_Ratio'] > 0.6:
        return "Growth-Stagnated High-Potency"
    elif row['Role_Stagnation_Index'] > 0.7:
        return "Role-Siloed Expert"
    else:
        return "Core Institutional Anchor"

df_filtered['Career_Intelligence_Label'] = df_filtered.apply(map_career_labels, axis=1)

# 5. Dashboard Layout
st.title("🚀 Career Intelligence Dashboard: Palo Alto Networks")

tab1, tab2, tab3 = st.tabs(["Diagnostic Overview", "Promotion Gap Monitor", "Intervention Priorities"])

with tab1:
    st.subheader("Workforce Diagnostic: Career Intelligence Segments")
    fig = px.scatter(df_filtered, x="YearsAtCompany", y="MonthlyIncome", 
                     color="Career_Intelligence_Label", 
                     hover_data=['JobRole'],
                     title="Income vs Tenure by Career Profile")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("High Promotion Gap Alerts")
    threshold = st.slider("Set Promotion Gap Threshold", 0.0, 1.0, 0.6)
    stagnant_df = df_filtered[df_filtered['Promotion_Gap_Ratio'] > threshold]
    st.dataframe(stagnant_df[['JobRole', 'Department', 'Promotion_Gap_Ratio', 'PerformanceRating']])

with tab3:
    st.subheader("Retention Priorities (Intervention Needed)")
    priority_list = df_filtered[df_filtered['Career_Intelligence_Label'] == "Growth-Stagnated High-Potency"]
    if not priority_list.empty:
        st.write("Suggested Action: Immediate LEAP Program Review / Career Discussion.")
        st.table(priority_list[['JobRole', 'Department', 'PerformanceRating']])
    else:
        st.info("No high-priority retention cases found in the current selection.")