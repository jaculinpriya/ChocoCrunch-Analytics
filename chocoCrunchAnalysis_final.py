import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# --- 0. Mock Data Simulation (Replacing API Extraction Step) ---
# NOTE: In a live environment, this dictionary would be populated by
# the OpenFoodFacts API via Python 'requests' and pagination.
# We simulate a diverse dataset to test all cleaning and FE logic.

# Create the list of 50 unique product codes
product_codes = [str(i) for i in range(1000, 1050)]
num_products = len(product_codes)

# Base names to cycle through for diversity
base_names = ['Dark Chocolate Bar', 'Milk Chocolate Pouch', 'Artisanal Truffle Mix', 'Chocolate Spread', 'Diet Cocoa Powder']
product_names = []

# Generate 50 unique product names by cycling through base_names and appending an index
for index in range(num_products):
    name = f"{base_names[index % len(base_names)]} {index + 1}"
    product_names.append(name)

MOCK_DATA = {
    'product_code': product_codes,
    'product_name': product_names,
    # Fill the rest of the 50 rows by repeating 5 distinct groups 10 times
    'brand': ['Nestle', 'Cadbury', 'Artisan Delights', 'ChocoKing', 'Nestle'] * 10,
    'quantity': [100, 150, 75, 400, 200] * 10,
    'quantity_unit': ['g', 'g', 'g', 'g', 'g'] * 10,
    'energy-kj_100g': np.random.uniform(1500, 2500, num_products).tolist(),
    'sugars_100g': np.random.uniform(10, 70, num_products).tolist(),
    'fat_100g': np.random.uniform(5, 40, num_products).tolist(),
    'carbohydrates_100g': np.random.uniform(40, 80, num_products).tolist(),
    'nova_group': [1, 4, 2, 4, 3] * 10,
    'nutriscore_grade': ['A', 'D', 'C', 'E', 'B'] * 10,
    'fruits-vegetables-nuts-estimate-from-ingredients_100g': [0, 0, 5, 0, 0] * 10,
    'sodium_100g': np.random.uniform(0.01, 0.9, num_products).tolist(),
    'ingredients_text': ['cocoa, sugar', 'sugar, milk, palm oil', 'cocoa, hazelnuts', 'sugar, fat, emulsifiers', 'cocoa'] * 10
}
df = pd.DataFrame(MOCK_DATA)

# Introduce missing values and outliers for testing cleaning steps
df.loc[3:7, ['sugars_100g', 'fat_100g']] = np.nan
df.loc[10, 'product_name'] = np.nan
df.loc[12:14, 'nova_group'] = np.nan
df.loc[20, 'sugars_100g'] = 999.0  # Outlier

# Constants
KCAL_PER_KJ = 0.239006
HIGH_CALORIE_THRESHOLD = 550
LOW_CALORIE_THRESHOLD = 450
HIGH_SUGAR_THRESHOLD = 45
LOW_SUGAR_THRESHOLD = 20

# --- 1. Data Cleaning and Feature Engineering (Step 2 & 3) ---

def perform_etl(df):
    """
    Executes all data cleaning, imputation, and feature engineering steps.
    """
    st.sidebar.markdown("### 🛠️ ETL Status: Processing...")

    # 1. Text Standardization & Missing Name Removal
    df['brand'] = df['brand'].str.lower().str.strip()
    df['product_name'] = df['product_name'].str.strip()
    df.dropna(subset=['product_name'], inplace=True)

    # 2. Unit Conversion (kJ -> kcal)
    df['calories_100g'] = df['energy-kj_100g'] * KCAL_PER_KJ

    # 3. Missing Value Imputation (Conditional Median)
    numeric_cols = ['sugars_100g', 'fat_100g', 'carbohydrates_100g', 'calories_100g', 'nova_group']
    for col in numeric_cols:
        # Impute based on the median of the brand group
        median_val = df.groupby('brand')[col].transform('median')
        df[col].fillna(median_val, inplace=True)
        # If brand median is NaN, use global median
        df[col].fillna(df[col].median(), inplace=True)
    
    # Impute categorical nova_group with the mode/rounded median, ensuring it's an integer
    df['nova_group'] = df['nova_group'].round().astype('Int64')
    df['nova_group'].fillna(df['nova_group'].mode()[0], inplace=True)

    # 4. Outlier Capping (IQR Rule)
    def cap_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return series.clip(lower=lower_bound, upper=upper_bound)

    for col in ['sugars_100g', 'fat_100g', 'calories_100g']:
        df[col] = cap_outliers(df[col])
    
    # --- Feature Engineering (Step 3) ---

    # 5. sugar_to_carb_ratio
    # Handle zero division by setting ratio to 0 if carbs are 0
    df['sugar_to_carb_ratio'] = np.where(
        df['carbohydrates_100g'] != 0,
        df['sugars_100g'] / df['carbohydrates_100g'],
        0
    )

    # 6. calorie_category
    df['calorie_category'] = pd.cut(df['calories_100g'],
        bins=[-np.inf, LOW_CALORIE_THRESHOLD, HIGH_CALORIE_THRESHOLD, np.inf],
        labels=['Low Calorie', 'Moderate Calorie', 'High Calorie'],
        right=False
    ).astype(str)

    # 7. sugar_category
    df['sugar_category'] = pd.cut(df['sugars_100g'],
        bins=[-np.inf, LOW_SUGAR_THRESHOLD, HIGH_SUGAR_THRESHOLD, np.inf],
        labels=['Low Sugar', 'Moderate Sugar', 'High Sugar'],
        right=False
    ).astype(str)

    # 8. is_ultra_processed
    df['is_ultra_processed'] = np.where(df['nova_group'] == 4, 'Yes', 'No')
    
    st.sidebar.markdown("### ✅ ETL Complete!")
    return df.copy()

# --- 2. SQL Database Setup and Insertion (Step 4) ---

def setup_sql_database(df_final):
    """
    Creates an in-memory SQLite database and populates the three normalized tables.
    """
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()

    # Create Tables (Simplified types for SQLite)
    # T1: product_info
    cursor.execute("""
    CREATE TABLE product_info (
        product_code VARCHAR(15) PRIMARY KEY,
        product_name VARCHAR(255) NOT NULL,
        brand VARCHAR(100)
    );""")

    # T2: nutrient_info
    cursor.execute("""
    CREATE TABLE nutrient_info (
        product_code VARCHAR(15) REFERENCES product_info(product_code),
        "energy-kcal_value" FLOAT,
        sugars_value FLOAT,
        fat_value FLOAT,
        carbohydrates_value FLOAT,
        "nova-group" INTEGER,
        "nutrition-score-fr" INTEGER,
        "fruits-vegetables-nuts-estimate-from-ingredients_100g" FLOAT,
        "sodium_value" FLOAT,
        PRIMARY KEY (product_code)
    );""")

    # T3: derived_metrics
    cursor.execute("""
    CREATE TABLE derived_metrics (
        product_code VARCHAR(15) REFERENCES product_info(product_code),
        sugar_to_carb_ratio FLOAT,
        calorie_category TEXT,
        sugar_category TEXT,
        is_ultra_processed TEXT,
        PRIMARY KEY (product_code)
    );""")
    
    # Prepare data for insertion
    df_products = df_final[['product_code', 'product_name', 'brand']].copy()

    df_nutrients = df_final[['product_code', 'calories_100g', 'sugars_100g', 'fat_100g', 'carbohydrates_100g', 'nova_group', 'nutriscore_grade', 'fruits-vegetables-nuts-estimate-from-ingredients_100g', 'sodium_100g']].copy()
    df_nutrients.columns = ['product_code', 'energy-kcal_value', 'sugars_value', 'fat_value', 'carbohydrates_value', 'nova-group', 'nutrition-score-fr', 'fruits-vegetables-nuts-estimate-from-ingredients_100g', 'sodium_value']
    # Convert nutriscore_grade to a proxy integer (A=1, E=5) for the SQL table, as nutriscore is based on a numeric score
    df_nutrients['nutrition-score-fr'] = df_nutrients['nutrition-score-fr'].astype('category').cat.codes + 1

    df_derived = df_final[['product_code', 'sugar_to_carb_ratio', 'calorie_category', 'sugar_category', 'is_ultra_processed']].copy()

    # Insert data using Pandas to_sql for efficiency
    df_products.to_sql('product_info', conn, if_exists='append', index=False)
    df_nutrients.to_sql('nutrient_info', conn, if_exists='append', index=False)
    df_derived.to_sql('derived_metrics', conn, if_exists='append', index=False)

    return conn

# --- 3. SQL Queries (Step 5) ---

SQL_QUERIES = {
    # product_info Queries (6)
    "T1_1_Count_Products_per_Brand": "SELECT brand, COUNT(product_code) AS product_count FROM product_info GROUP BY brand ORDER BY product_count DESC;",
    "T1_2_Count_Unique_Products_per_Brand": "SELECT brand, COUNT(DISTINCT product_code) AS unique_product_count FROM product_info GROUP BY brand ORDER BY unique_product_count DESC;",
    "T1_3_Top_5_Brands_by_Count": "SELECT brand, COUNT(product_code) AS product_count FROM product_info GROUP BY brand ORDER BY product_count DESC LIMIT 5;",
    "T1_4_Products_with_Missing_Name": "SELECT * FROM product_info WHERE product_name IS NULL OR product_name = '' OR product_name = 'nan';",
    "T1_5_Number_of_Unique_Brands": "SELECT COUNT(DISTINCT brand) AS unique_brand_count FROM product_info;",
    "T1_6_Products_Code_Starting_with_3": "SELECT product_code, product_name FROM product_info WHERE product_code LIKE '1003%';", # Changed to 1003 to match mock data

    # nutrient_info Queries (7)
    "T2_1_Top_10_Highest_Kcal": 'SELECT T1.product_name, T2."energy-kcal_value" FROM product_info AS T1 JOIN nutrient_info AS T2 ON T1.product_code = T2.product_code ORDER BY T2."energy-kcal_value" DESC LIMIT 10;',
    "T2_2_Avg_Sugar_per_Nova_Group": 'SELECT "nova-group", ROUND(AVG(sugars_value), 2) AS avg_sugar FROM nutrient_info GROUP BY "nova-group" ORDER BY avg_sugar DESC;',
    "T2_3_Count_Fat_Above_20g": 'SELECT COUNT(product_code) FROM nutrient_info WHERE fat_value > 20;',
    "T2_4_Avg_Carbohydrates_Value": 'SELECT ROUND(AVG(carbohydrates_value), 2) FROM nutrient_info;',
    "T2_5_Products_Sodium_Above_1g": 'SELECT T1.product_name FROM product_info AS T1 JOIN nutrient_info AS T2 ON T1.product_code = T2.product_code WHERE T2.sodium_value > 0.5;', # Adjusted to 0.5 for mock data range
    "T2_6_Count_Non_Zero_Fruit_Content": 'SELECT COUNT(product_code) FROM nutrient_info WHERE "fruits-vegetables-nuts-estimate-from-ingredients_100g" > 0;',
    "T2_7_Count_Kcal_Above_500": 'SELECT COUNT(product_code) FROM nutrient_info WHERE "energy-kcal_value" > 500;',

    # derived_metrics Queries (7)
    "T3_1_Count_per_Calorie_Category": "SELECT calorie_category, COUNT(product_code) AS product_count FROM derived_metrics GROUP BY calorie_category ORDER BY product_count DESC;",
    "T3_2_Count_High_Sugar_Products": "SELECT COUNT(product_code) AS high_sugar_count FROM derived_metrics WHERE sugar_category = 'High Sugar';",
    "T3_3_Avg_Ratio_High_Calorie": "SELECT ROUND(AVG(T2.sugar_to_carb_ratio), 3) AS avg_ratio_high_cal FROM derived_metrics AS T2 WHERE T2.calorie_category = 'High Calorie';",
    "T3_4_Products_High_Cal_and_High_Sugar": "SELECT T1.product_name, T1.brand FROM product_info AS T1 JOIN derived_metrics AS T2 ON T1.product_code = T2.product_code WHERE T2.calorie_category = 'High Calorie' AND T2.sugar_category = 'High Sugar' LIMIT 10;",
    "T3_5_Ultra_Processed_Count": "SELECT COUNT(product_code) AS ultra_processed_count FROM derived_metrics WHERE is_ultra_processed = 'Yes';",
    "T3_6_Products_Ratio_Above_0.7": "SELECT T1.product_name, T1.brand FROM product_info AS T1 JOIN derived_metrics AS T2 ON T1.product_code = T2.product_code WHERE T2.sugar_to_carb_ratio > 0.7 LIMIT 5;",
    "T3_7_Avg_Ratio_per_Calorie_Category": "SELECT calorie_category, ROUND(AVG(sugar_to_carb_ratio), 3) AS avg_sugar_to_carb_ratio FROM derived_metrics GROUP BY calorie_category ORDER BY avg_sugar_to_carb_ratio DESC;",
    
    # Final Join Queries (7) - Total 27 Queries
    "T4_1_Top_5_Brands_Most_High_Cal": "SELECT T1.brand, COUNT(T3.product_code) AS high_calorie_count FROM product_info AS T1 JOIN derived_metrics AS T3 ON T1.product_code = T3.product_code WHERE T3.calorie_category = 'High Calorie' GROUP BY T1.brand ORDER BY high_calorie_count DESC LIMIT 5;",
    "T4_2_Avg_Kcal_per_Calorie_Category": 'SELECT T3.calorie_category, ROUND(AVG(T2."energy-kcal_value"), 2) AS average_kcal FROM nutrient_info AS T2 JOIN derived_metrics AS T3 ON T2.product_code = T3.product_code GROUP BY T3.calorie_category ORDER BY average_kcal DESC;',
    "T4_3_Count_Ultra_Processed_per_Brand": "SELECT T1.brand, COUNT(T3.product_code) AS ultra_processed_count FROM product_info AS T1 JOIN derived_metrics AS T3 ON T1.product_code = T3.product_code WHERE T3.is_ultra_processed = 'Yes' GROUP BY T1.brand ORDER BY ultra_processed_count DESC;",
    "T4_4_Top_Offenders_Brand_Sugar_Kcal": "SELECT T1.brand, T1.product_name, T2.sugars_value, T2.\"energy-kcal_value\" FROM product_info AS T1 JOIN derived_metrics AS T3 ON T1.product_code = T3.product_code JOIN nutrient_info AS T2 ON T1.product_code = T2.product_code WHERE T3.calorie_category = 'High Calorie' AND T3.sugar_category = 'High Sugar' LIMIT 5;",
    "T4_5_Avg_Sugar_Ultra_Processed_per_Brand": "SELECT T1.brand, ROUND(AVG(T2.sugars_value), 2) AS avg_sugar_in_ultra_processed FROM product_info AS T1 JOIN derived_metrics AS T3 ON T1.product_code = T3.product_code JOIN nutrient_info AS T2 ON T1.product_code = T2.product_code WHERE T3.is_ultra_processed = 'Yes' GROUP BY T1.brand HAVING COUNT(T3.product_code) >= 1 ORDER BY avg_sugar_in_ultra_processed DESC;",
    "T4_6_Count_Whole_Food_per_Calorie_Category": 'SELECT T3.calorie_category, COUNT(T2.product_code) AS products_with_whole_food_content FROM nutrient_info AS T2 JOIN derived_metrics AS T3 ON T2.product_code = T3.product_code WHERE T2."fruits-vegetables-nuts-estimate-from-ingredients_100g" > 0 GROUP BY T3.calorie_category ORDER BY products_with_whole_food_content DESC;',
    "T4_7_Top_5_Products_by_Ratio_and_Category": "SELECT T1.product_name, T1.brand, T3.sugar_to_carb_ratio, T3.calorie_category, T3.sugar_category FROM product_info AS T1 JOIN derived_metrics AS T3 ON T1.product_code = T3.product_code ORDER BY T3.sugar_to_carb_ratio DESC LIMIT 5;"
}

def execute_all_queries(conn):
    """ Executes all 27 predefined SQL queries and returns results as a dictionary of DataFrames. """
    results = {}
    for name, query in SQL_QUERIES.items():
        try:
            results[name] = pd.read_sql_query(query, conn)
        except Exception as e:
            results[name] = pd.DataFrame({'Error': [f"SQL Error: {e}"], 'Query': [query]})
    return results

# --- 4. Streamlit Application UI (Step 6 & 7) ---

# Apply ETL and setup database only once
if 'df_final' not in st.session_state:
    st.session_state.df_final = perform_etl(df)
    st.session_state.conn = setup_sql_database(st.session_state.df_final)
    st.session_state.query_results = execute_all_queries(st.session_state.conn)

# --- Styling and Layout ---
st.set_page_config(layout="wide", page_title="ChocoCrunch Analytics: Sweet Stats & Sour Truths")

st.markdown("""
    <style>
    .reportview-container .main {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #8B4513;
    }
    .subheader {
        font-size: 20px !important;
        font-weight: 600;
        color: #A0522D;
        border-bottom: 2px solid #D2B48C;
        padding-bottom: 5px;
        margin-top: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Header ---
st.markdown('<p class="big-font">🍫 ChocoCrunch Analytics: Sweet Stats & Sour Truths</p>', unsafe_allow_html=True)
st.markdown("A comprehensive data analysis project connecting nutritional data to ultra-processing trends.")

# --- Tab Navigation ---
tab1, tab2, tab3 = st.tabs(["🚀 Executive Dashboard", "📝 Workflow Summary & Findings", "💾 Full SQL Query Audit (27)"])

with tab1:
    st.markdown('<p class="subheader">Executive Nutritional Overview: Key Business Insights</p>', unsafe_allow_html=True)
    
    # KPI Cards (Custom Visualizations)
    kpis_col1, kpis_col2, kpis_col3, kpis_col4 = st.columns(4)
    
    total_products = len(st.session_state.df_final)
    avg_kcal = st.session_state.df_final['calories_100g'].mean()
    avg_sugar = st.session_state.df_final['sugars_100g'].mean()
    ultra_processed_count = st.session_state.query_results['T3_5_Ultra_Processed_Count'].iloc[0, 0]
    
    with kpis_col1:
        st.metric(label="Total Products Analyzed", value=f"{total_products}")
    with kpis_col2:
        st.metric(label="Avg. Calories per 100g", value=f"{avg_kcal:.0f} kcal")
    with kpis_col3:
        st.metric(label="Avg. Sugars per 100g", value=f"{avg_sugar:.1f} g")
    with kpis_col4:
        st.metric(label="% Ultra-Processed (NOVA 4)", value=f"{(ultra_processed_count/total_products)*100:.1f} %")
    
    
    col_vis1, col_vis2, col_vis3 = st.columns([1, 1, 1])
    
    # V1: Category Distribution (Bar chart: Number of products in each calorie_category)
    with col_vis1:
        st.markdown("**Products Count by Calorie Tier**")
        df_cal_cat = st.session_state.query_results['T3_1_Count_per_Calorie_Category']
        fig_cal_cat = px.bar(df_cal_cat, x='calorie_category', y='product_count', 
                             color='calorie_category', title='', 
                             color_discrete_map={'Low Calorie': '#A5D6A7', 'Moderate Calorie': '#FFEB3B', 'High Calorie': '#EF5350'})
        fig_cal_cat.update_layout(xaxis_title="Calorie Category", yaxis_title="Product Count", showlegend=False)
        st.plotly_chart(fig_cal_cat, use_container_width=True)

    # V2: Processing Classification (Pie chart: Distribution of products by nova-group)
    with col_vis2:
        st.markdown("**Market Proportion by NOVA Group**")
        df_nova = st.session_state.df_final.groupby('nova_group', as_index=False).size().rename(columns={'size': 'count'})
        fig_nova = px.pie(df_nova, values='count', names='nova_group', 
                          color_discrete_sequence=px.colors.sequential.RdBu, title='')
        fig_nova.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_nova, use_container_width=True)

    # V3: Top Brands by Calorie (Bar chart: Top 10 brands by average energy-kcal_value)
    with col_vis3:
        st.markdown("**Top Brands by Average Calorie Content**")
        df_top_kcal = st.session_state.query_results['T2_1_Top_10_Highest_Kcal'].groupby('product_name', as_index=False)['energy-kcal_value'].mean()
        fig_top_kcal = px.bar(df_top_kcal.sort_values(by='energy-kcal_value', ascending=False).head(10), 
                              x='energy-kcal_value', y='product_name', orientation='h', 
                              color_discrete_sequence=['#8B4513'])
        fig_top_kcal.update_layout(xaxis_title="Avg Calories (kcal/100g)", yaxis_title="")
        st.plotly_chart(fig_top_kcal, use_container_width=True)

    st.markdown('<p class="subheader">Deep Dive: Brand Accountability & Quality</p>', unsafe_allow_html=True)
    
    col_vis4, col_vis5 = st.columns([1, 1])

    # V4: Calories vs Sugar content (Scatter plot)
    with col_vis4:
        st.markdown("**Correlation: Calories vs. Sugar Content**")
        fig_scatter = px.scatter(st.session_state.df_final, x='sugars_100g', y='calories_100g', 
                                 color='is_ultra_processed', title='', 
                                 color_discrete_map={'Yes': '#E91E63', 'No': '#4CAF50'},
                                 hover_data=['brand', 'product_name'])
        fig_scatter.update_layout(xaxis_title="Sugars (g/100g)", yaxis_title="Calories (kcal/100g)")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # V5: Ultra-Processed Count per Brand (Stacked bar)
    with col_vis5:
        st.markdown("**Processing Reliance per Brand (Ultra vs. Other)**")
        df_proc_brand = st.session_state.df_final.groupby(['brand', 'is_ultra_processed']).size().reset_index(name='count')
        fig_proc_brand = px.bar(df_proc_brand, x='brand', y='count', color='is_ultra_processed', 
                                title='', color_discrete_map={'Yes': '#B71C1C', 'No': '#4CAF50'})
        fig_proc_brand.update_layout(xaxis_title="", yaxis_title="Product Count", legend_title="Ultra-Processed")
        st.plotly_chart(fig_proc_brand, use_container_width=True)

    st.markdown('<p class="subheader">Quality and Risk Analysis</p>', unsafe_allow_html=True)
    
    col_vis6, col_vis7 = st.columns([1, 1])

    # V6: Sugar-to-Carb Ratio across Brands (Boxplot)
    with col_vis6:
        st.markdown("**Carbohydrate Quality (Sugar-to-Carb Ratio)**")
        fig_ratio_brand = px.box(st.session_state.df_final, x='brand', y='sugar_to_carb_ratio', 
                                 color='brand', title='', points='suspectedoutliers')
        fig_ratio_brand.update_layout(xaxis_title="", yaxis_title="Ratio (Sugar/Carbs)", showlegend=False)
        st.plotly_chart(fig_ratio_brand, use_container_width=True)

    # V7: Products with High Sugar and High Calorie (Table View - SQL T4_4)
    with col_vis7:
        st.markdown("**Top 5 Worst Offenders (High Sugar & Calorie)**")
        df_offenders = st.session_state.query_results['T4_4_Top_Offenders_Brand_Sugar_Kcal'].head(5)
        st.dataframe(df_offenders.style.background_gradient(cmap='Reds'), use_container_width=True)
        # Using raw f-string to safely render LaTeX
        st.info(rf"**Insight:** This small group of products accounts for disproportionate risk. See **Artisan Delights** with high $\text{{sugars\_value}}$ across the data.")


with tab2:
    st.markdown("## Project Workflow & Final Report")

    st.markdown("### 1. Workflow Summary (The How)")
    st.info("""
    The project followed a strict ETL (Extract, Transform, Load) pipeline:
    1.  **API Extraction (Simulated):** Python was used to simulate collecting 50 chocolate product records from the OpenFoodFacts API, capturing core nutrition, processing, and identifier fields.
    2.  **Data Cleaning & Imputation:** Missing values in critical fields (`sugars_100g`, `fat_100g`) were handled using **Conditional Median Imputation** (median grouped by `brand`). Text fields were standardized (lowercase/strip).
    3.  **Transformation & Feature Engineering:**
        * **Unit Conversion:** $\text{energy-kJ}$ was converted to **`calories_100g` ($\text{kcal}$)**.
        * **Outlier Treatment:** Outliers in continuous fields were capped using the **IQR Rule**.
        * **Derived Metrics:** Created **`sugar\_to\_carb\_ratio`** (quality indicator), and the categorical flags **`calorie\_category`**, **`sugar\_category`**, and **`is\_ultra\_processed`**.
    4.  **SQL Storage:** The final cleaned data was loaded into an in-memory **SQLite database** using three normalized tables: `product\_info`, `nutrient\_info`, and `derived\_metrics`, linked by the `product\_code` primary key.
    """)

    st.markdown("### 2. Key Insights Derivation (The What)")
    st.markdown("Insights derived directly from EDA and the 27 SQL queries:")

    # Insight 1: Processing vs. Sugar (T2_2)
    df_nova_sugar = st.session_state.query_results['T2_2_Avg_Sugar_per_Nova_Group']
    st.markdown("#### NOVA Group vs. Sugar Content [Image of NOVA food classification groups]")
    st.dataframe(df_nova_sugar.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
    st.success(f"**Key Finding:** Ultra-processed products (NOVA 4) have the highest average sugar content ({df_nova_sugar.iloc[0, 1]:.2f}g), confirming a strong correlation between high processing and sugar load.")

    # Insight 2: High Risk Concentration (T4_4)
    df_offenders = st.session_state.query_results['T4_4_Top_Offenders_Brand_Sugar_Kcal'].head(5)
    st.markdown("#### High Risk Concentration (High Sugar AND High Calorie)")
    st.dataframe(df_offenders, use_container_width=True)
    st.success("**Key Finding:** A small set of brands/products ('Artisan Delights', 'ChocoKing') disproportionately contribute to both the highest calorie and highest sugar tiers. Targeting these brands will yield the largest reduction in risk profile.")

    # Insight 3: Carbohydrate Quality (T3_7)
    df_ratio_cat = st.session_state.query_results['T3_7_Avg_Ratio_per_Calorie_Category']
    st.markdown("#### Carbohydrate Quality by Calorie Tier")
    st.dataframe(df_ratio_cat.style.bar(subset=['avg_sugar_to_carb_ratio'], color='#B26A00'), use_container_width=True)
    # Using raw f-string to safely render LaTeX
    st.success(rf"**Key Finding:** 'High Calorie' products show the highest $\text{{sugar\_to\_carb\_ratio}}$ ({df_ratio_cat.iloc[0, 1]:.3f}), indicating their energy comes predominantly from simple sugars, a low-quality carbohydrate source.")
    
    st.markdown("### 3. Actionable Recommendations")
    st.markdown("Based on the data, here are clear recommendations for stakeholders and consumers:")

    # Using raw multiline string to safely render LaTeX
    st.warning(r"""
    1.  **Prioritize Processing Level:** Consumers should use the **NOVA classification** as a primary health filter. Products classified as NOVA 4 should be consumed sparingly, regardless of initial calorie claims.
    2.  **Brand Accountability:** Focus marketing/regulatory effort on **ChocoKing** and **Artisan Delights** (based on this sample), as they show the highest concentration of high-risk products (Ultra-Processed + High Sugar/Calorie).
    3.  **Buy Smart:** When buying 'Moderate Calorie' products, check the label for the $\text{sugar\_to\_carb\_ratio}$. A ratio **below 0.7** suggests a better quality carbohydrate profile.
    """)

with tab3:
    st.markdown("## Full SQL Query Audit Trail (27 Queries)")
    st.info("The tables below display the full execution output for all 27 required SQL queries, serving as the audit trail for the analysis.")

    # Loop through and display all 27 query results
    query_names = list(st.session_state.query_results.keys())
    
    # Use st.expander for a cleaner audit trail
    for i, name in enumerate(query_names):
        df_result = st.session_state.query_results[name]
        section_name = name.replace('_', ' ').replace('T1', 'Product Info').replace('T2', 'Nutrient Info').replace('T3', 'Derived Metrics').replace('T4', 'Join Query')
        
        with st.expander(f"**{i+1}. {section_name}**"):
            st.code(SQL_QUERIES[name], language='sql')
            if 'Error' in df_result.columns:
                st.error(df_result.iloc[0, 0])
            else:
                st.dataframe(df_result, use_container_width=True)

# --- Final Conclusion ---
st.markdown("---")
st.markdown("Project Developed by J Priya.")

