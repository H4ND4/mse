import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="MSE Companies Evaluation", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS 
st.markdown("""
<style>
    /* Improved Tab Styling - Make titles visible */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: #1f77b4 !important;
    }
    
    /* Investment Picks - Make company names stand out */
    .highlight-box h3 {
        color: #2c3e50 !important;
        margin-top: 0;
        font-size: 1.3rem;
    }
    
    /* Hover effects for highlight boxes */
    .highlight-box {
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .highlight-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    
    /* Improved table highlighting */
    .dataframe td.highlight {
        font-weight: bold;
    }
    
    /* Green highlight for max values */
    .dataframe td.highlight.max {
        background-color: #d4edda !important;
        color: #155724 !important;
    }
    
    /* Red highlight for min values */
    .dataframe td.highlight.min {
        background-color: #f8d7da !important;
        color: #721c24 !important;
    }
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
        }
        
        .highlight-box h3 {
            color: #ffffff !important;
        }
        
        /* Dark mode table highlights */
        .dataframe td.highlight.max {
            background-color: #1e3b1e !important;
            color: #a3d8a3 !important;
        }
        
        .dataframe td.highlight.min {
            background-color: #3b1e1e !important;
            color: #ffb3b3 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# SPLIT DATA for price adjustments
SPLIT_DATA = {
    "APU": [
        {"date": "2016-06-20", "ratio": 10},  # 10:1 split (new shares per old share)
        {"date": "2008-07-15", "ratio": 100}  # 100:1 split
    ],
    "MNDL": [
        {"date": "2019-11-25", "ratio": 100}  # 100:1 split
    ]
}

@st.cache_data
def load_data(filename):
    company_code = os.path.basename(filename).split('.')[0]
    df = pd.read_csv(filename)
    df['Арилжигдсан Огноо'] = pd.to_datetime(df['Арилжигдсан Огноо'])
    
    numeric_cols = ['Ханш Дээд', 'Ханш Доод', 'Ханш Нээлт', 'Ханш Хаалт',
                    'Арилжигдсан Ширхэг', 'Арилжигдсан Үнийн дүн']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Sort by date ascending for proper split adjustment
    df = df.sort_values("Арилжигдсан Огноо")
 
    # Only filter out rows where ALL price columns are zero
    price_cols = ['Ханш Дээд', 'Ханш Доод', 'Ханш Нээлт', 'Ханш Хаалт']
    df = df[~(df[price_cols] == 0).all(axis=1)]

    # Calculate daily return only for rows with valid closing prices
    valid_close = df['Ханш Хаалт'] > 0
    df.loc[valid_close, 'Өдрийн өгөөж'] = df.loc[valid_close, 'Ханш Хаалт'].pct_change() * 100
    
    df['Price Range'] = df['Ханш Дээд'] - df['Ханш Доод']
    df['Year'] = df['Арилжигдсан Огноо'].dt.year
    df['Month'] = df['Арилжигдсан Огноо'].dt.month
    return df

# def calculate_advanced_metrics(df, year=None):
#     df_filtered = df[df["Year"] == year] if year else df
#     if len(df_filtered) < 2:
#         return None
    
#     # Filter out days with zero or negative closing prices
#     valid_prices = (df_filtered['Ханш Хаалт'] > 0)
#     df_valid = df_filtered[valid_prices].copy()
    
#     if len(df_valid) < 2:
#         return {
#             "Total Return (%)": 0,
#             "Daily Volatility (%)": 0,
#             "Total Volume (Million)": 0,
#             "Avg Daily Volume (Million)": 0,
#             "Current Price": 0,
#             "Max Price": 0,
#             "Min Price": 0,
#             "Average Price": 0,
#             "Avg Price Range": 0,
#             "Trading Days": len(df_filtered),
#             "Valid Trading Days": 0
#         }
    
#     # Calculate daily returns only on valid days
#     df_valid['Daily Return'] = df_valid['Ханш Хаалт'].pct_change()
    
#     # Calculate volatility (skip first day which has NaN return)
#     volatility = df_valid['Daily Return'].iloc[1:].std() * 100  # as percentage
    
#     # Calculate total return properly
#     first_valid_price = df_valid['Ханш Хаалт'].iloc[0]
#     last_valid_price = df_valid['Ханш Хаалт'].iloc[-1]
#     total_return = ((last_valid_price - first_valid_price) / first_valid_price) * 100

#     # Volume metrics
#     avg_volume = df_filtered['Арилжигдсан Ширхэг'].mean()
#     total_volume = df_filtered["Арилжигдсан Ширхэг"].sum() / 1e6
#     avg_daily_volume = avg_volume / 1e6
    
#     # Price metrics
#     max_price = df_valid["Ханш Хаалт"].max()
#     min_price = df_valid["Ханш Хаалт"].min()
#     avg_price = df_valid["Ханш Хаалт"].mean()
#     current_price = df_valid["Ханш Хаалт"].iloc[-1]
#     avg_price_range = df_filtered['Price Range'].mean()
    
#     return {
#         "Total Return (%)": round(total_return, 2),
#         "Daily Volatility (%)": round(volatility, 2),
#         "Total Volume (Million)": round(total_volume, 2),
#         "Avg Daily Volume (Million)": round(avg_daily_volume, 2),
#         "Current Price": round(current_price, 2),
#         "Max Price": round(max_price, 2),
#         "Min Price": round(min_price, 2),
#         "Average Price": round(avg_price, 2),
#         "Avg Price Range": round(avg_price_range, 2),
#         "Trading Days": len(df_filtered),
#         "Valid Trading Days": len(df_valid)
#     }

def calculate_advanced_metrics(df, year=None, start_year=None):
    if start_year:
        df_filtered = df[df["Year"] >= start_year]
    elif year:
        df_filtered = df[df["Year"] == year]
    else:
        df_filtered = df
        
    if len(df_filtered) < 2:
        return None
    
    # Filter out days with zero or negative closing prices
    valid_prices = (df_filtered['Ханш Хаалт'] > 0)
    df_valid = df_filtered[valid_prices].copy()
    
    if len(df_valid) < 2:
        return {
            "Total Return (%)": 0,
            "Daily Volatility (%)": 0,
            "Total Volume (Million)": 0,
            "Avg Daily Volume (Million)": 0,
            "Current Price": 0,
            "Max Price": 0,
            "Min Price": 0,
            "Average Price": 0,
            "Avg Price Range": 0,
            "Trading Days": len(df_filtered),
            "Valid Trading Days": 0,
            "Return Since 2020 (%)": 0  # Add this new field
        }
    
    # Calculate daily returns only on valid days
    df_valid['Daily Return'] = df_valid['Ханш Хаалт'].pct_change()
    
    # Calculate volatility (skip first day which has NaN return)
    volatility = df_valid['Daily Return'].iloc[1:].std() * 100  # as percentage
    
    # Calculate total return properly
    first_valid_price = df_valid['Ханш Хаалт'].iloc[0]
    last_valid_price = df_valid['Ханш Хаалт'].iloc[-1]
    total_return = ((last_valid_price - first_valid_price) / first_valid_price) * 100
    
    # Calculate return since 2020 if start_year is specified
    return_since_2020 = 0
    if start_year:
        return_since_2020 = total_return
    elif year is None:  # For all years case
        df_since_2020 = df[df["Year"] >= 2020]
        if len(df_since_2020) >= 2:
            valid_prices_2020 = (df_since_2020['Ханш Хаалт'] > 0)
            df_valid_2020 = df_since_2020[valid_prices_2020].copy()
            if len(df_valid_2020) >= 2:
                first_price_2020 = df_valid_2020['Ханш Хаалт'].iloc[0]
                last_price_2020 = df_valid_2020['Ханш Хаалт'].iloc[-1]
                return_since_2020 = ((last_price_2020 - first_price_2020) / first_price_2020) * 100

    # Volume metrics
    avg_volume = df_filtered['Арилжигдсан Ширхэг'].mean()
    total_volume = df_filtered["Арилжигдсан Ширхэг"].sum() / 1e6
    avg_daily_volume = avg_volume / 1e6
    
    # Price metrics
    max_price = df_valid["Ханш Хаалт"].max()
    min_price = df_valid["Ханш Хаалт"].min()
    avg_price = df_valid["Ханш Хаалт"].mean()
    current_price = df_valid["Ханш Хаалт"].iloc[-1]
    avg_price_range = df_filtered['Price Range'].mean()
    
    return {
        "Total Return (%)": round(total_return, 2),
        "Daily Volatility (%)": round(volatility, 2),
        "Total Volume (Million)": round(total_volume, 2),
        "Avg Daily Volume (Million)": round(avg_daily_volume, 2),
        "Current Price": round(current_price, 2),
        "Max Price": round(max_price, 2),
        "Min Price": round(min_price, 2),
        "Average Price": round(avg_price, 2),
        "Avg Price Range": round(avg_price_range, 2),
        "Trading Days": len(df_filtered),
        "Valid Trading Days": len(df_valid),
        "Return Since 2020 (%)": round(return_since_2020, 2)  # Add this new field
    }
    
def create_empty_metrics():
    """Create empty metrics dictionary for error cases"""
    return {
        "Total Return (%)": 0,
        "Daily Volatility (%)": 0,
        "Total Volume (Million)": 0,
        "Avg Daily Volume (Million)": 0,
        "Current Price": 0,
        "Max Price": 0,
        "Min Price": 0,
        "Average Price": 0,
        "Avg Price Range": 0,
        "Max Drawdown (%)": 0,
        "Trading Days": 0,
        "Valid Trading Days": 0,
        "Data Quality": "No Data"
    }

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    try:
        if len(prices) < 2:
            return 0
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown) if not np.isnan(max_drawdown) else 0
    except:
        return 0

#new shdee
def create_yearly_summary_table(df):
    # Filter out years with no valid closing prices
    valid_years = df[df['Ханш Хаалт'] > 0].groupby('Year').filter(lambda x: len(x) > 1)
    
    if len(valid_years) == 0:
        return pd.DataFrame()  # Return empty dataframe if no valid data
    
    yearly_summary = valid_years.groupby('Year').agg({
        'Ханш Хаалт': ['first', 'last', 'max', 'min', 'mean'],
        'Арилжигдсан Ширхэг': 'sum',
        'Өдрийн өгөөж': ['std', 'mean']
    }).round(2)
    
    # Calculate yearly return only for years with valid first and last prices
    valid_returns = (yearly_summary['Ханш Хаалт']['last'] > 0) & (yearly_summary['Ханш Хаалт']['first'] > 0)
    yearly_summary['Yearly Return %'] = 0  # Initialize with zeros
    yearly_summary.loc[valid_returns, 'Yearly Return %'] = (
        (yearly_summary['Ханш Хаалт']['last'] - yearly_summary['Ханш Хаалт']['first']) /
        yearly_summary['Ханш Хаалт']['first']
    ) * 100
    
    yearly_summary.columns = ['Open Price', 'Close Price', 'High', 'Low', 'Avg Price',
                            'Total Volume', 'Volatility', 'Avg Daily Return', 'Yearly Return %']
    return yearly_summary
    
def create_price_chart(df, company_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Арилжигдсан Огноо"],
        y=df["Ханш Хаалт"],
        mode='lines+markers',
        name='Хаалтын ханш',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4)
    ))
    fig.update_layout(
        title=f"{company_name} - Хувьцааны ханшийн түүх",
        xaxis_title="Огноо",
        yaxis_title="Ханш",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig
    
    yearly_summary['Yearly Return %'] = ((yearly_summary['Ханш Хаалт']['last'] - yearly_summary['Ханш Хаалт']['first']) /
                                         yearly_summary['Ханш Хаалт']['first']) * 100
    
    yearly_summary.columns = ['Open Price', 'Close Price', 'High', 'Low', 'Avg Price',
                              'Total Volume', 'Volatility', 'Avg Daily Return', 'Yearly Return %']
    return yearly_summary

def create_monthly_bar_chart(df):
    # Group by Year and Month, calculate mean closing price
    monthly_avg = df.groupby(['Year', 'Month'])['Ханш Хаалт'].mean().reset_index()
    
    # If only one year is selected, create a simple bar chart
    if len(monthly_avg['Year'].unique()) == 1:
        fig = px.bar(
            monthly_avg,
            x='Month',
            y='Ханш Хаалт',
            title="Сарын дундаж хаалтын ханш",
            labels={'Ханш Хаалт': 'Дундаж ханш', 'Month': 'Сар'},
            text_auto='.1f'
        )
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis_title="Дундаж ханш",
            hovermode='x unified',
            template='plotly_white'
        )
    else:
        # For multiple years, create grouped bars
        fig = px.bar(
            monthly_avg,
            x='Month',
            y='Ханш Хаалт',
            color='Year',
            barmode='group',
            title="Сарын дундаж хаалтын ханш",
            labels={'Ханш Хаалт': 'Дундаж ханш', 'Month': 'Сар'},
            text_auto='.1f'
        )
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            yaxis_title="Дундаж ханш",
            legend_title="Он",
            hovermode='x unified',
            template='plotly_white'
        )
    
    return fig

# Main App Layout
st.markdown('<h1 class="main-header">MSE Analysis</h1>', unsafe_allow_html=True)
st.markdown("### Анализ хувьцаат компани 1-р ангилал")

# Company files dictionary (adjust paths as needed)
DATA_DIR = "data"  # Adjust this to your data directory
company_files = {
    "AARD": "AARD.csv",
    "ADB": "ADB.csv", 
    "AIC": "AIC.csv",
    "APU": "APU.csv",
    "BODI": "BODI.csv",
    "GAZR": "GAZR.csv",
    "GLMT": "GLMT.csv",
    "GOV": "GOV.csv",
    "KHAN": "KHAN.csv",
    "MFC": "MFC.csv",
    "MIK": "MIK.csv", 
    "MMX": "MMX.csv",
    "MNDL": "MNDL.csv",
    "MNP": "MNP.csv",
    "MSE": "MSE.csv",
    "SBM": "SBM.csv",
    "TDB": "TDB.csv",
    "XAC": "XAC.csv",
}

# Create tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Company Analysis", "Compare All"])

with tab1:
    st.subheader("Executive Summary")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **Mongolian Stock Exchange Companies Analysis Report**
        
        This comprehensive analysis covers 18 companies listed on the Mongolian Stock Exchange (MSE) 
        from the "Хувьцаат компани 1-р ангилал" category. Our analysis focuses on:
        
        - Growth Analysis: Stock price performance over time
        - Volatility Assessment: Risk measurement through price fluctuations  
        - Trading Volume: Market liquidity and investor interest
        - Temporal Analysis: Year-over-year performance tracking
        """)
    
    with col2:
        st.markdown("""
        <div class="highlight-box">
            <strong>Key Metrics Analyzed:</strong>
            <ul>
                <li>Total Return</li>
                <li>Daily Volatility</li>
                <li>Trading Volume</li>
                <li>Price Range Analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Available Companies")
    
    # Display companies in a nice grid
    cols = st.columns(6)
    for i, company in enumerate(company_files.keys()):
        with cols[i % 6]:
            st.button(company, key=f"overview_{company}", disabled=True)

with tab2:
    st.subheader("Individual Company Analysis")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_company = st.selectbox("Select Company", list(company_files.keys()))
        
        # Load data using your method
        try:
            df = load_data(os.path.join(DATA_DIR, company_files[selected_company]))
            available_years = sorted(df["Year"].unique(), reverse=True)
            selected_year = st.selectbox("Select Year for Analysis", ["All Years"] + list(available_years))
            
            analysis_year = None if selected_year == "All Years" else selected_year
            
        except Exception as e:
            st.error(f"Error loading data for {selected_company}: {str(e)}")
            st.stop()
    
    with col2:
        st.markdown(f"### {selected_company} Analysis")
        
        # Display basic info
        period_text = f"Year {selected_year}" if selected_year != "All Years" else "All Available Data"
        st.markdown(f"<div class='highlight-box'><strong>Analysis Period:</strong> {period_text}</div>", unsafe_allow_html=True)
    
    # Display split information if applicable
    if selected_company in SPLIT_DATA:
        st.markdown("#### Note: Splits!")
        splits = SPLIT_DATA[selected_company]
        for split in splits:
            st.write(f"- {split['date']}: {split['ratio']}x split")

    # Calculate metrics using your method
    metrics = calculate_advanced_metrics(df, analysis_year)
    
    if metrics:
        # Key metrics using your format
        st.subheader("Гол үзүүлэлтүүд")
        col1, col2, col3 = st.columns(3)
        col1.metric("Өдөр тутмын савлагаа", f"{metrics['Daily Volatility (%)']}%")
        col2.metric("Дундаж өдөр тутмын ширхэг", f"{metrics['Avg Daily Volume (Million)']*1e6:,.0f}")
        col3.metric("Дундаж ханшийн хэлбэлзэл", f"{metrics['Avg Price Range']:.2f}")
        
        # Additional metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{metrics['Total Return (%)']}%")
        with col2:
            st.metric("Current Price", f"{metrics['Current Price']}")
        with col3:
            st.metric("Total Volume", f"{metrics['Total Volume (Million)']}M")
    
    # Yearly Summary Table using your method
    st.subheader("Жилийн Дүн")
    yearly_summary = create_yearly_summary_table(df)
    st.dataframe(yearly_summary, use_container_width=True)
    
    # Price History Chart
    st.subheader("Хаалтын ханш график")
    price_chart = create_price_chart(df if selected_year == "All Years" else df[df["Year"] == analysis_year], selected_company)
    st.plotly_chart(price_chart, use_container_width=True)

    # Monthly Trend Chart
    st.subheader("Сарын дундаж хаалтын ханш")
    bar_fig  = create_monthly_bar_chart(df if selected_year == "All Years" else df[df["Year"] == analysis_year])
    st.plotly_chart(bar_fig , use_container_width=True)
    
    # Max & Min Price Days using your method
    df_filtered = df if selected_year == "All Years" else df[df["Year"] == analysis_year]
    max_close = df_filtered.loc[df_filtered['Ханш Хаалт'].idxmax()]
    min_close = df_filtered.loc[df_filtered['Ханш Хаалт'].idxmin()]
    
    st.subheader("Хамгийн өндөр ба бага ханштай өдрүүд")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Хамгийн өндөр:")
        st.write(f"**Огноо:** {max_close['Арилжигдсан Огноо'].date()}")
        st.write(f"**Ханш:** {max_close['Ханш Хаалт']}")
        st.write(f"**Ширхэг:** {max_close['Арилжигдсан Ширхэг']:,}")
    with col2:
        st.markdown("#### Хамгийн бага:")
        st.write(f"**Огноо:** {min_close['Арилжигдсан Огноо'].date()}")
        st.write(f"**Ханш:** {min_close['Ханш Хаалт']}")
        st.write(f"**Ширхэг:** {min_close['Арилжигдсан Ширхэг']:,}")

# with tab3:    
#     # Load all company data and calculate metrics
#     with st.spinner("Loading and analyzing all companies..."):
#         comparison_data = []
        
#         for company, filename in company_files.items():
#             try:
#                 df = load_data(os.path.join(DATA_DIR, filename))
#                 metrics = calculate_advanced_metrics(df)
#                 if metrics:
#                     metrics["Company"] = company
#                     comparison_data.append(metrics)
#             except:
#                 continue
    
#     if comparison_data:
#         comparison_df = pd.DataFrame(comparison_data).set_index("Company")
        
#         # Display sortable comparison table
#         st.subheader("Comprehensive Comparison Table")
#         st.dataframe(
#             comparison_df,
#             use_container_width=True
#         )
        
#         # Interactive charts
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Return vs Volatility scatter plot
#             fig_scatter = px.scatter(
#                 comparison_df.reset_index(),
#                 x="Daily Volatility (%)",
#                 y="Total Return (%)",
#                 size="Total Volume (Million)",
#                 hover_name="Company",
#                 title="Risk vs Return Analysis",
#                 template='plotly_white'
#             )
#             st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    
    # Load all company data and calculate metrics
    with st.spinner("Loading and analyzing all companies..."):
        comparison_data = []
        
        for company, filename in company_files.items():
            try:
                df = load_data(os.path.join(DATA_DIR, filename))
                # Calculate metrics for all years and since 2020
                metrics = calculate_advanced_metrics(df)
                metrics_2020 = calculate_advanced_metrics(df, start_year=2020)
                
                if metrics and metrics_2020:
                    metrics["Company"] = company
                    metrics["Return Since 2020 (%)"] = metrics_2020["Total Return (%)"]
                    comparison_data.append(metrics)
            except:
                continue
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data).set_index("Company")
        
        # Reorder columns to put the new column where you want it
        column_order = [
            "Return Since 2020 (%)",  # New column first
            "Total Return (%)",
            "Daily Volatility (%)",
            "Current Price",
            "Max Price",
            "Min Price",
            "Average Price",
            "Avg Price Range",
            "Total Volume (Million)",
            "Avg Daily Volume (Million)",
            "Trading Days",
            "Valid Trading Days"
        ]
        comparison_df = comparison_df[column_order]
        
        # Display sortable comparison table
        st.subheader("Comprehensive Comparison Table")
        st.dataframe(
            comparison_df,
            use_container_width=True
        )
        
        # Interactive charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Return vs Volatility scatter plot
            fig_scatter = px.scatter(
                comparison_df.reset_index(),
                x="Daily Volatility (%)",
                y="Return Since 2020 (%)",  # Changed to use the new column
                size="Total Volume (Million)",
                hover_name="Company",
                title="Risk vs Return Since 2020",
                template='plotly_white'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
# Footer
st.markdown("---")
st.markdown("*MSE Companies Analysis Tool*")
