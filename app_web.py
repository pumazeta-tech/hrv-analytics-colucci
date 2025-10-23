import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from io import BytesIO

# Configurazione pagina
st.set_page_config(
    page_title="HRV Analytics - di Roberto",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header epico
st.markdown('<h1 class="main-header">🏥 HRV ANALYTICS PLATFORM</h1>', unsafe_allow_html=True)
st.markdown("### di **Roberto** - Analisi Variabilità Cardiaca Professionale")

# Sidebar - Navigazione
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Navigazione")
    page = st.radio("Scegli sezione:", ["📊 Dashboard", "📈 Analisi Dettagliata", "👤 Profilo", "⚙️ Impostazioni"])
    
    st.markdown("---")
    st.info("**Nuova funzionalità:** Carica i tuoi file IBI direttamente dal Bodyguard 2!")

# Pagina Dashboard
if page == "📊 Dashboard":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("SDNN Medio", "85.2 ms", "↑ 12%")
        st.metric("Coerenza Cardiaca", "68%", "↑ 5%")
        
    with col2:
        st.metric("RMSSD", "52.1 ms", "↑ 8%")
        st.metric("Freq. Cardiaca", "64 bpm", "↓ 3%")
        
    with col3:
        st.metric("Stress Index", "1.2", "↓ 15%")
        st.metric("Sonno Qualità", "92%", "↑ 7%")
    
    # Grafico interattivo
    st.subheader("📈 Andamento Settimanale")
    
    # Dati di esempio
    dates = pd.date_range(start='2024-01-01', periods=7, freq='D')
    data = {
        'Data': dates,
        'SDNN': np.random.normal(80, 15, 7),
        'RMSSD': np.random.normal(50, 10, 7),
        'Coerenza': np.random.normal(65, 8, 7)
    }
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Data'], y=df['SDNN'], name='SDNN', line=dict(color='#e74c3c')))
    fig.add_trace(go.Scatter(x=df['Data'], y=df['RMSSD'], name='RMSSD', line=dict(color='#3498db')))
    fig.add_trace(go.Scatter(x=df['Data'], y=df['Coerenza'], name='Coerenza', line=dict(color='#2ecc71')))
    
    fig.update_layout(
        title="Trend Settimanale Parametri HRV",
        xaxis_title="Data",
        yaxis_title="Valori",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Pagina Analisi Dettagliata
elif page == "📈 Analisi Dettagliata":
    st.header("🔍 Analisi Dettagliata File IBI")
    
    # Upload file
    uploaded_file = st.file_uploader("📤 Carica il tuo file IBI dal Bodyguard 2", type=['txt', 'ibi', 'csv'])
    
    if uploaded_file is not None:
        # Simulazione analisi (useremo le tue funzioni dopo)
        st.success(f"✅ File {uploaded_file.name} caricato con successo!")
        
        # Metriche simulate
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("SDNN", "163.1 ms", "Alto")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("RMSSD", "219.2 ms", "Molto Alto")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Coerenza", "31.3%", "Moderata")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Bilancio LF/HF", "1.1", "Ottimale")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Grafico Poincaré interattivo
        st.subheader("🔄 Poincaré Plot Interattivo")
        
        # Simulazione dati Poincaré
        np.random.seed(42)
        n_points = 200
        mean_rr = 950
        rr_n = np.random.normal(mean_rr, 50, n_points)
        rr_n1 = rr_n + np.random.normal(0, 30, n_points)
        
        fig_poincare = px.scatter(
            x=rr_n, y=rr_n1, 
            title="Poincaré Plot - Variabilità Cardiaca",
            labels={'x': 'RRn (ms)', 'y': 'RRn+1 (ms)'}
        )
        
        # Aggiungi linea identità
        fig_poincare.add_trace(
            go.Scatter(x=[800, 1100], y=[800, 1100], 
                      mode='lines', name='Linea Identità',
                      line=dict(dash='dash', color='red'))
        )
        
        st.plotly_chart(fig_poincare, use_container_width=True)
        
        # Bottone download report
        if st.button("📄 Genera Report PDF Completo"):
            st.balloons()
            st.success("Report generato! (Integreremo le tue funzioni PDF qui)")

# Pagina Profilo
elif page == "👤 Profilo":
    st.header("👤 Il Tuo Profilo Personale")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            nome = st.text_input("Nome", "Roberto")
            eta = st.slider("Età", 18, 80, 45)
            altezza = st.number_input("Altezza (cm)", 150, 200, 175)
            
        with col2:
            sesso = st.selectbox("Sesso", ["Maschio", "Femmina"])
            peso = st.number_input("Peso (kg)", 50, 150, 75)
            attivita = st.selectbox("Livello Attività", 
                                  ["Sedentario", "Moderato", "Attivo", "Atleta"])
        
        submitted = st.form_submit_button("💾 Salva Profilo")
        if submitted:
            st.success("Profilo aggiornato con successo!")

# Pagina Impostazioni
elif page == "⚙️ Impostazioni":
    st.header("⚙️ Impostazioni e Preferenze")
    
    st.subheader("Notifiche")
    not_email = st.checkbox("Email di riepilogo settimanale", True)
    not_alert = st.checkbox("Alert parametri anomali", True)
    
    st.subheader("Esportazione Dati")
    exp_format = st.radio("Formato esportazione", ["CSV", "JSON", "Excel"])
    
    if st.button("💾 Salva Impostazioni"):
        st.success("Impostazioni salvate!")

# Footer
st.markdown("---")
st.markdown("**HRV Analytics Platform** - Creato con ❤️ da Roberto e DeepSeek")