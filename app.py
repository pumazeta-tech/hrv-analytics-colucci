import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configurazione pagina
st.set_page_config(page_title="HRV Analytics", page_icon="❤️", layout="wide")

st.title("🏥 HRV ANALYTICS")
st.markdown("Analisi della Variabilità Cardiaca")

# =============================================================================
# DIARIO ATTIVITÀ SEMPLIFICATO
# =============================================================================

st.sidebar.header("📝 Diario Attività")

# Inizializza session state
if 'activities' not in st.session_state:
    st.session_state.activities = []

# Aggiungi attività
with st.sidebar.expander("➕ Aggiungi Attività"):
    activity_name = st.text_input("Nome attività")
    
    col1, col2 = st.columns(2)
    with col1:
        activity_date = st.date_input("Data")
    with col2:
        activity_time = st.time_input("Ora")
    
    activity_color = st.color_picker("Colore", "#3498db")
    
    if st.button("💾 Salva Attività"):
        if activity_name:
            start_time = datetime.combine(activity_date, activity_time)
            activity = {
                'name': activity_name,
                'start': start_time,
                'color': activity_color
            }
            st.session_state.activities.append(activity)
            st.success("Attività salvata!")

# Mostra attività salvate
if st.session_state.activities:
    st.sidebar.subheader("Attività Salvate")
    for i, activity in enumerate(st.session_state.activities):
        st.sidebar.write(f"**{activity['name']}** - {activity['start'].strftime('%H:%M')}")
        if st.sidebar.button(f"Elimina", key=f"del_{i}"):
            st.session_state.activities.pop(i)
            st.rerun()

# =============================================================================
# ANALISI SIMULATA SEMPLIFICATA
# =============================================================================

st.sidebar.header("⚙️ Analisi Simulata")

# Input base
recording_hours = st.sidebar.slider("Durata (ore)", 1.0, 24.0, 8.0)
health_factor = st.sidebar.slider("Profilo Salute", 0.1, 1.0, 0.5)

if st.sidebar.button("🚀 AVVIA ANALISI", type="primary"):
    
    # Metriche simulate semplici
    metrics = {
        'sdnn': 50 + (100 * health_factor),
        'rmssd': 30 + (80 * health_factor),
        'hr_mean': 65 - (10 * health_factor),
        'recording_hours': recording_hours
    }
    
    # Mostra risultati
    st.success("✅ Analisi completata!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("SDNN", f"{metrics['sdnn']:.1f} ms")
    with col2:
        st.metric("RMSSD", f"{metrics['rmssd']:.1f} ms")
    with col3:
        st.metric("HR Medio", f"{metrics['hr_mean']:.1f} bpm")
    
    # Timeline con attività
    st.header("📊 Timeline con Attività")
    
    # Genera dati timeline
    time_points = 100
    times = [datetime.now() + timedelta(hours=i * recording_hours / time_points) for i in range(time_points)]
    sdnn_values = metrics['sdnn'] + 10 * np.sin(np.linspace(0, 4*np.pi, time_points))
    
    # Crea grafico
    fig = go.Figure()
    
    # Aggiungi linea SDNN
    fig.add_trace(go.Scatter(
        x=times, y=sdnn_values,
        mode='lines',
        name='SDNN',
        line=dict(color='blue', width=3)
    ))
    
    # Aggiungi attività
    if st.session_state.activities:
        for activity in st.session_state.activities:
            fig.add_vline(
                x=activity['start'],
                line_dash="dash",
                line_color=activity['color'],
                annotation_text=activity['name']
            )
    
    fig.update_layout(
        title="Timeline HRV con Attività",
        xaxis_title="Tempo",
        yaxis_title="SDNN (ms)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabella attività
    if st.session_state.activities:
        st.subheader("📋 Attività Registrate")
        activity_data = []
        for activity in st.session_state.activities:
            activity_data.append({
                'Attività': activity['name'],
                'Ora': activity['start'].strftime('%H:%M'),
                'Colore': activity['color']
            })
        
        df_activities = pd.DataFrame(activity_data)
        st.dataframe(df_activities, use_container_width=True)

else:
    # Schermata iniziale
    st.info("👆 Usa la sidebar per aggiungere attività e avviare l'analisi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Come usare:")
        st.markdown("""
        1. 📝 Aggiungi attività nel diario
        2. ⚙️ Imposta parametri analisi
        3. 🚀 Avvia analisi
        4. 📊 Visualizza risultati con attività
        """)
    
    with col2:
        st.subheader("Funzionalità:")
        st.markdown("""
        - Diario attività personalizzato
        - Timeline interattiva
        - Analisi HRV simulata
        - Integrazione attività nei grafici
        """)

# Footer
st.markdown("---")
st.markdown("Creato con ❤️ | HRV Analytics")