"""
ITS Playground - Real-time Traffic Analytics Dashboard
Streamlit application for monitoring traffic patterns, revenue, and system health
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ITS Traffic Analytics",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-success {
        padding: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .alert-warning {
        padding: 0.5rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .alert-danger {
        padding: 0.5rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_its_data():
    """Load ITS datasets with caching"""
    try:
        passage_df = pd.read_csv('passage_logs.csv')
        transaction_df = pd.read_csv('toll_transactions.csv')
        anpr_df = pd.read_csv('anpr_data.csv')
        ev_df = pd.read_csv('ev_charging.csv')
        safety_df = pd.read_csv('road_safety_incidents.csv')
        
        # Convert timestamps
        passage_df['timestamp'] = pd.to_datetime(passage_df['timestamp'])
        transaction_df['timestamp'] = pd.to_datetime(transaction_df['timestamp'])
        anpr_df['timestamp'] = pd.to_datetime(anpr_df['timestamp'])
        ev_df['start_time'] = pd.to_datetime(ev_df['start_time'])
        safety_df['timestamp'] = pd.to_datetime(safety_df['timestamp'])
        
        return passage_df, transaction_df, anpr_df, ev_df, safety_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_api_data():
    """Fetch data from API if available"""
    try:
        response = requests.get('http://localhost:5000/api/kpis', timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def calculate_kpis(passage_df, transaction_df):
    """Calculate key performance indicators"""
    if passage_df is None or transaction_df is None:
        return {}
    
    successful_transactions = transaction_df[transaction_df['transaction_status'] == 'Success']
    
    # Time range
    date_range = (passage_df['timestamp'].max() - passage_df['timestamp'].min()).days + 1
    
    kpis = {
        'total_vehicles': len(passage_df),
        'daily_avg_vehicles': len(passage_df) / date_range if date_range > 0 else 0,
        'total_revenue': successful_transactions['toll_amount'].sum(),
        'daily_avg_revenue': successful_transactions['toll_amount'].sum() / date_range if date_range > 0 else 0,
        'avg_speed': passage_df['speed_kmh'].mean(),
        'tag_detection_rate': passage_df['tag_detected'].mean(),
        'transaction_success_rate': (transaction_df['transaction_status'] == 'Success').mean(),
        'avg_processing_time': successful_transactions['processing_time_sec'].mean() if len(successful_transactions) > 0 else 0
    }
    
    return kpis

def create_traffic_flow_chart(passage_df):
    """Create hourly traffic flow visualization"""
    if passage_df is None or passage_df.empty:
        return go.Figure()
    
    # Hourly traffic volume
    hourly_data = passage_df.groupby(passage_df['timestamp'].dt.hour).size().reset_index()
    hourly_data.columns = ['hour', 'vehicle_count']
    
    fig = px.line(
        hourly_data, 
        x='hour', 
        y='vehicle_count',
        title='Hourly Traffic Volume',
        labels={'hour': 'Hour of Day', 'vehicle_count': 'Vehicle Count'}
    )
    
    fig.update_traces(line_color='#1f77b4', line_width=3)
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Vehicle Count",
        hovermode='x unified'
    )
    
    return fig

def create_lane_utilization_chart(passage_df):
    """Create lane utilization visualization"""
    if passage_df is None or passage_df.empty:
        return go.Figure()
    
    lane_data = passage_df['lane_number'].value_counts().sort_index()
    
    fig = px.bar(
        x=lane_data.index,
        y=lane_data.values,
        title='Lane Utilization',
        labels={'x': 'Lane Number', 'y': 'Vehicle Count'},
        color=lane_data.values,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Lane Number",
        yaxis_title="Vehicle Count",
        showlegend=False
    )
    
    return fig

def create_revenue_chart(transaction_df):
    """Create revenue analysis visualization"""
    if transaction_df is None or transaction_df.empty:
        return go.Figure()
    
    successful = transaction_df[transaction_df['transaction_status'] == 'Success']
    payment_revenue = successful.groupby('payment_method')['toll_amount'].sum()
    
    fig = px.pie(
        values=payment_revenue.values,
        names=payment_revenue.index,
        title='Revenue by Payment Method'
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_speed_distribution_chart(passage_df):
    """Create speed distribution visualization"""
    if passage_df is None or passage_df.empty:
        return go.Figure()
    
    fig = px.box(
        passage_df,
        x='vehicle_class',
        y='speed_kmh',
        title='Speed Distribution by Vehicle Class',
        color='vehicle_class'
    )
    
    fig.update_layout(
        xaxis_title="Vehicle Class",
        yaxis_title="Speed (km/h)",
        showlegend=False
    )
    
    return fig

def detect_anomalies(passage_df, transaction_df):
    """Detect system anomalies"""
    if passage_df is None or transaction_df is None:
        return {}
    
    anomalies = {}
    
    # Speed violations
    speed_violations = passage_df[passage_df['speed_kmh'] > 100]
    anomalies['speed_violations'] = len(speed_violations)
    
    # Missing transactions for tagged vehicles
    tagged_passages = passage_df[passage_df['tag_detected'] == True]
    passages_with_transactions = passage_df[passage_df['passage_id'].isin(transaction_df['passage_id'])]
    missing_transactions = tagged_passages[~tagged_passages['passage_id'].isin(passages_with_transactions['passage_id'])]
    anomalies['revenue_leakage'] = len(missing_transactions)
    anomalies['potential_loss'] = len(missing_transactions) * 5.50  # Average toll
    
    # Failed transactions
    failed_transactions = transaction_df[transaction_df['transaction_status'] == 'Failed']
    anomalies['failed_transactions'] = len(failed_transactions)
    
    return anomalies

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚦 ITS Traffic Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        
        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Time range selector
        st.subheader("📅 Time Range")
        time_range = st.selectbox(
            "Select range:",
            ["Last Hour", "Last 24 Hours", "Last 7 Days", "All Time"]
        )
        
        # System status
        st.subheader("🖥️ System Status")
        api_data = get_api_data()
        if api_data:
            st.markdown('<div class="alert-success">✅ API Online</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-warning">⚠️ API Offline (Using cached data)</div>', unsafe_allow_html=True)
    
    # Load data
    passage_df, transaction_df, anpr_df, ev_df, safety_df = load_its_data()
    
    if passage_df is None:
        st.error("❌ Unable to load data. Please ensure datasets are available.")
        st.info("💡 Run the data generator script to create sample datasets.")
        return
    
    # Filter data based on time range
    now = passage_df['timestamp'].max()
    if time_range == "Last Hour":
        cutoff = now - timedelta(hours=1)
    elif time_range == "Last 24 Hours":
        cutoff = now - timedelta(days=1)
    elif time_range == "Last 7 Days":
        cutoff = now - timedelta(days=7)
    else:
        cutoff = passage_df['timestamp'].min()
    
    filtered_passage_df = passage_df[passage_df['timestamp'] >= cutoff]
    filtered_transaction_df = transaction_df[transaction_df['timestamp'] >= cutoff]
    
    # Key Performance Indicators
    st.header("📊 Key Performance Indicators")
    
    kpis = calculate_kpis(filtered_passage_df, filtered_transaction_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Vehicles",
            f"{kpis.get('total_vehicles', 0):,}",
            delta=f"{kpis.get('daily_avg_vehicles', 0):.0f}/day"
        )
    
    with col2:
        st.metric(
            "Total Revenue",
            f"${kpis.get('total_revenue', 0):,.2f}",
            delta=f"${kpis.get('daily_avg_revenue', 0):.0f}/day"
        )
    
    with col3:
        st.metric(
            "Average Speed",
            f"{kpis.get('avg_speed', 0):.1f} km/h",
            delta=f"{kpis.get('avg_speed', 0) - 75:.1f} vs limit"
        )
    
    with col4:
        st.metric(
            "Tag Detection Rate",
            f"{kpis.get('tag_detection_rate', 0):.1%}",
            delta=f"{kpis.get('transaction_success_rate', 0):.1%} success"
        )
    
    # Traffic Analysis Charts
    st.header("🚦 Traffic Flow Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        traffic_chart = create_traffic_flow_chart(filtered_passage_df)
        st.plotly_chart(traffic_chart, use_container_width=True)
    
    with col2:
        lane_chart = create_lane_utilization_chart(filtered_passage_df)
        st.plotly_chart(lane_chart, use_container_width=True)
    
    # Revenue Analysis
    st.header("💰 Revenue Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        revenue_chart = create_revenue_chart(filtered_transaction_df)
        st.plotly_chart(revenue_chart, use_container_width=True)
    
    with col2:
        speed_chart = create_speed_distribution_chart(filtered_passage_df)
        st.plotly_chart(speed_chart, use_container_width=True)
    
    # Anomaly Detection
    st.header("⚠️ Anomaly Detection")
    
    anomalies = detect_anomalies(filtered_passage_df, filtered_transaction_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        speed_violations = anomalies.get('speed_violations', 0)
        if speed_violations > 0:
            st.markdown(f'<div class="alert-warning">⚠️ <strong>{speed_violations}</strong> speed violations detected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success">✅ No speed violations</div>', unsafe_allow_html=True)
    
    with col2:
        revenue_leakage = anomalies.get('revenue_leakage', 0)
        if revenue_leakage > 0:
            potential_loss = anomalies.get('potential_loss', 0)
            st.markdown(f'<div class="alert-danger">🚨 <strong>{revenue_leakage}</strong> missing transactions<br>Potential loss: <strong>${potential_loss:.2f}</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success">✅ No revenue leakage detected</div>', unsafe_allow_html=True)
    
    with col3:
        failed_transactions = anomalies.get('failed_transactions', 0)
        if failed_transactions > 0:
            st.markdown(f'<div class="alert-warning">⚠️ <strong>{failed_transactions}</strong> failed transactions</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success">✅ All transactions successful</div>', unsafe_allow_html=True)
    
    # Real-time Data Table
    st.header("📋 Recent Activity")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚗 Recent Passages", "💳 Recent Transactions", "📷 ANPR Data", "⚡ EV Charging"])
    
    with tab1:
        if not filtered_passage_df.empty:
            recent_passages = filtered_passage_df.nlargest(10, 'timestamp')[
                ['timestamp', 'vehicle_id', 'vehicle_class', 'speed_kmh', 'lane_number', 'tag_detected']
            ]
            st.dataframe(recent_passages, use_container_width=True)
        else:
            st.info("No recent passages in selected time range")
    
    with tab2:
        if not filtered_transaction_df.empty:
            recent_transactions = filtered_transaction_df.nlargest(10, 'timestamp')[
                ['timestamp', 'vehicle_id', 'toll_amount', 'payment_method', 'transaction_status', 'processing_time_sec']
            ]
            st.dataframe(recent_transactions, use_container_width=True)
        else:
            st.info("No recent transactions in selected time range")
    
    with tab3:
        if anpr_df is not None and not anpr_df.empty:
            recent_anpr = anpr_df.nlargest(10, 'timestamp')[
                ['timestamp', 'license_plate', 'state', 'ocr_confidence', 'weather_condition', 'vehicle_make', 'vehicle_color']
            ]
            st.dataframe(recent_anpr, use_container_width=True)
        else:
            st.info("No ANPR data available")
    
    with tab4:
        if ev_df is not None and not ev_df.empty:
            recent_ev = ev_df.nlargest(10, 'start_time')[
                ['start_time', 'station_id', 'duration_minutes', 'energy_delivered_kwh', 'charging_cost', 'vehicle_type']
            ]
            st.dataframe(recent_ev, use_container_width=True)
        else:
            st.info("No EV charging data available")
    
    # Advanced Analytics Section
    with st.expander("🔬 Advanced Analytics", expanded=False):
        st.subheader("Traffic Pattern Analysis")
        
        if not filtered_passage_df.empty:
            # Heatmap of traffic by hour and day
            filtered_passage_df['hour'] = filtered_passage_df['timestamp'].dt.hour
            filtered_passage_df['day_name'] = filtered_passage_df['timestamp'].dt.day_name()
            
            heatmap_data = filtered_passage_df.groupby(['day_name', 'hour']).size().reset_index(name='count')
            
            if not heatmap_data.empty:
                heatmap_pivot = heatmap_data.pivot(index='day_name', columns='hour', values='count').fillna(0)
                
                fig_heatmap = px.imshow(
                    heatmap_pivot.values,
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    title="Traffic Intensity Heatmap (Hour vs Day of Week)",
                    labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Vehicle Count'},
                    color_continuous_scale='YlOrRd'
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Vehicle class distribution
        st.subheader("Vehicle Classification Analysis")
        if not filtered_passage_df.empty:
            class_dist = filtered_passage_df['vehicle_class'].value_counts()
            
            fig_class = px.treemap(
                names=class_dist.index,
                values=class_dist.values,
                title="Vehicle Class Distribution (Treemap)"
            )
            
            st.plotly_chart(fig_class, use_container_width=True)
    
    # System Monitoring
    with st.expander("🖥️ System Monitoring", expanded=False):
        st.subheader("Data Quality Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if anpr_df is not None and not anpr_df.empty:
                avg_ocr_confidence = anpr_df['ocr_confidence'].mean()
                st.metric("Average OCR Confidence", f"{avg_ocr_confidence:.1%}")
            else:
                st.metric("Average OCR Confidence", "N/A")
        
        with col2:
            if not filtered_transaction_df.empty:
                avg_processing_time = filtered_transaction_df[filtered_transaction_df['transaction_status'] == 'Success']['processing_time_sec'].mean()
                st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
            else:
                st.metric("Avg Processing Time", "N/A")
        
        with col3:
            if not filtered_passage_df.empty:
                image_capture_rate = filtered_passage_df['image_captured'].mean()
                st.metric("Image Capture Rate", f"{image_capture_rate:.1%}")
            else:
                st.metric("Image Capture Rate", "N/A")
        
        # System alerts
        st.subheader("System Alerts")
        
        alerts = []
        
        # Check for data freshness
        if not filtered_passage_df.empty:
            latest_data = filtered_passage_df['timestamp'].max()
            time_since_update = (datetime.now() - latest_data.to_pydatetime()).total_seconds() / 3600
            
            if time_since_update > 1:
                alerts.append(f"⚠️ Data is {time_since_update:.1f} hours old")
        
        # Check for anomalies
        if anomalies.get('speed_violations', 0) > 10:
            alerts.append(f"🚨 High number of speed violations: {anomalies['speed_violations']}")
        
        if anomalies.get('revenue_leakage', 0) > 5:
            alerts.append(f"💰 Significant revenue leakage detected: ${anomalies.get('potential_loss', 0):.2f}")
        
        # Check ANPR performance
        if anpr_df is not None and not anpr_df.empty:
            low_confidence_rate = (anpr_df['ocr_confidence'] < 0.8).mean()
            if low_confidence_rate > 0.1:
                alerts.append(f"📷 High ANPR error rate: {low_confidence_rate:.1%}")
        
        if alerts:
            for alert in alerts:
                st.markdown(f'<div class="alert-warning">{alert}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-success">✅ All systems operating normally</div>', unsafe_allow_html=True)
    
    # Export Data Section
    with st.expander("📤 Export Data", expanded=False):
        st.subheader("Download Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if not filtered_passage_df.empty:
                csv_passages = filtered_passage_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Passage Data (CSV)",
                    data=csv_passages,
                    file_name=f"passage_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if not filtered_transaction_df.empty:
                csv_transactions = filtered_transaction_df.to_csv(index=False)
                st.download_button(
                    label="💳 Download Transaction Data (CSV)",
                    data=csv_transactions,
                    file_name=f"transaction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            # Generate summary report
            report_data = {
                "generated_at": datetime.now().isoformat(),
                "time_range": time_range,
                "kpis": kpis,
                "anomalies": anomalies,
                "data_quality": {
                    "total_records": len(filtered_passage_df),
                    "avg_ocr_confidence": anpr_df['ocr_confidence'].mean() if anpr_df is not None and not anpr_df.empty else None,
                    "image_capture_rate": filtered_passage_df['image_captured'].mean() if not filtered_passage_df.empty else None
                }
            }
            
            json_report = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                label="📋 Download Summary Report (JSON)",
                data=json_report,
                file_name=f"its_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer with last update time
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Run the application
if __name__ == "__main__":
    main()

# Additional utility functions for the dashboard

def create_ml_predictions_section(passage_df):
    """Create machine learning predictions section"""
    if passage_df is None or passage_df.empty:
        return
    
    st.subheader("🤖 ML Predictions")
    
    # Simple traffic prediction for next hour
    current_hour = datetime.now().hour
    next_hour = (current_hour + 1) % 24
    
    # Get historical average for next hour
    historical_avg = passage_df.groupby(passage_df['timestamp'].dt.hour).size()
    
    if next_hour in historical_avg.index:
        predicted_volume = historical_avg[next_hour]
        st.info(f"🔮 Predicted traffic volume for {next_hour}:00 - {next_hour + 1}:00: **{predicted_volume} vehicles**")
    
    # Anomaly prediction based on current patterns
    recent_data = passage_df.tail(100)  # Last 100 records
    avg_speed = recent_data['speed_kmh'].mean()
    
    if avg_speed > 90:
        st.warning("⚠️ **Traffic flowing faster than usual** - Monitor for speed violations")
    elif avg_speed < 40:
        st.warning("⚠️ **Slow traffic detected** - Possible congestion or incidents")
    else:
        st.success("✅ **Traffic flowing normally**")

def create_weather_impact_analysis(anpr_df):
    """Analyze weather impact on system performance"""
    if anpr_df is None or anpr_df.empty:
        return
    
    st.subheader("🌤️ Weather Impact Analysis")
    
    weather_performance = anpr_df.groupby('weather_condition').agg({
        'ocr_confidence': ['mean', 'count'],
        'image_quality': lambda x: (x == 'High').mean()
    }).round(3)
    
    weather_performance.columns = ['Avg OCR Confidence', 'Record Count', 'High Quality Rate']
    
    st.dataframe(weather_performance)
    
    # Weather impact visualization
    fig_weather = px.box(
        anpr_df,
        x='weather_condition',
        y='ocr_confidence',
        title='OCR Performance by Weather Condition',
        color='weather_condition'
    )
    
    st.plotly_chart(fig_weather, use_container_width=True)

def create_revenue_forecast(transaction_df):
    """Create revenue forecasting based on historical data"""
    if transaction_df is None or transaction_df.empty:
        return
    
    st.subheader("📈 Revenue Forecasting")
    
    # Daily revenue trend
    successful_transactions = transaction_df[transaction_df['transaction_status'] == 'Success']
    daily_revenue = successful_transactions.groupby(successful_transactions['timestamp'].dt.date)['toll_amount'].sum()
    
    if len(daily_revenue) >= 7:  # Need at least 7 days for trend analysis
        # Simple linear trend
        x = np.arange(len(daily_revenue))
        y = daily_revenue.values
        
        # Linear regression
        z = np.polyfit(x, y, 1)
        trend_line = np.poly1d(z)
        
        # Forecast next 7 days
        future_x = np.arange(len(daily_revenue), len(daily_revenue) + 7)
        forecast = trend_line(future_x)
        
        st.info(f"📊 **7-day Revenue Forecast:** ${forecast.sum():,.2f}")
        st.info(f"📈 **Daily Trend:** ${z[0]:+.2f} per day")
        
        # Visualization
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=list(daily_revenue.index),
            y=daily_revenue.values,
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color='blue')
        ))
        
        # Trend line
        fig_forecast.add_trace(go.Scatter(
            x=list(daily_revenue.index),
            y=trend_line(x),
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash')
        ))
        
        # Forecast
        forecast_dates = [daily_revenue.index[-1] + timedelta(days=i) for i in range(1, 8)]
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='green', dash='dot')
        ))
        
        fig_forecast.update_layout(
            title='Revenue Trend and 7-Day Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)

# Add these sections to the main dashboard by calling them in the appropriate places