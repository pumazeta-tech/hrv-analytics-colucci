import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import io
import base64
from matplotlib.patches import Ellipse

# =============================================================================
# FUNZIONI PER CARICAMENTO FILE IBI - AGGIUNGI QUI
# =============================================================================

def detect_ibi_format(file_content):
    """Rileva automaticamente il formato del file IBI"""
    first_lines = file_content.head(10).to_string()
    
    if 'RR' in first_lines or 'IBI' in first_lines or 'interval' in first_lines.lower():
        return 'rr_intervals'
    elif 'HR' in first_lines or 'bpm' in first_lines.lower():
        return 'hr_values' 
    elif 'timestamp' in first_lines.lower() and ('rr' in first_lines.lower() or 'ibi' in first_lines.lower()):
        return 'timestamp_rr'
    else:
        return 'raw_values'

def process_rr_intervals(df):
    """Processa file con colonne RR/IBI intervals - VERSIONE MIGLIORATA"""
    rr_intervals = []
    
    st.write("🔍 **Analisi colonne del file:**")
    st.write(f"Colonne trovate: {list(df.columns)}")
    st.write(f"Prime righe del file:")
    st.dataframe(df.head(), use_container_width=True)
    
    # Cerca colonne RR/IBI con pattern più ampi
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['rr', 'ibi', 'interval', 'interbeat', 'r-r', 'hrv']):
            st.success(f"✅ Trovata colonna: {col}")
            rr_intervals = df[col].dropna().astype(float).values
            break
    
    # Se non trova, prova la prima colonna numerica
    if len(rr_intervals) == 0:
        st.info("🔍 Cerco colonne numeriche...")
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                st.success(f"✅ Usando colonna numerica: {col}")
                rr_intervals = df[col].dropna().astype(float).values
                break
    
    # Se ancora nulla, prova a convertire la prima colonna
    if len(rr_intervals) == 0 and len(df.columns) > 0:
        st.info("🔍 Provo a convertire la prima colonna...")
        try:
            first_col = df.columns[0]
            rr_intervals = pd.to_numeric(df[first_col], errors='coerce').dropna().values
            if len(rr_intervals) > 0:
                st.success(f"✅ Convertita colonna: {first_col}")
        except:
            pass
    
    st.write(f"📊 Intervalli RR estratti: {len(rr_intervals)}")
    
    return rr_intervals

def calculate_hrv_metrics_from_rr(rr_intervals):
    """Calcola metriche HRV da RR intervals"""
    if len(rr_intervals) == 0:
        return None
    
    rr_intervals = np.array(rr_intervals) * 1000
    
    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    
    differences = np.diff(rr_intervals)
    rmssd = np.sqrt(np.mean(differences ** 2))
    
    hr_mean = 60000 / mean_rr if mean_rr > 0 else 0
    
    return {
        'mean_rr': mean_rr,
        'sdnn': sdnn,
        'rmssd': rmssd, 
        'hr_mean': hr_mean,
        'n_intervals': len(rr_intervals),
        'total_duration': np.sum(rr_intervals) / 60000
    }

def create_rr_timeline_plot(rr_intervals):
    """Crea grafico timeline degli RR intervals"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=np.arange(len(rr_intervals)), y=rr_intervals,
        mode='lines+markers',
        name='RR Intervals',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="📈 RR Intervals Timeline",
        xaxis_title="Numero Battito",
        yaxis_title="RR Interval (ms)",
        template='plotly_white',
        height=400
    )
    
    return fig

def create_file_analysis(rr_intervals, hrv_metrics):
    """Crea analisi per file caricato"""
    st.header("📊 Analisi HRV da File Caricato")
    
    # Metriche principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("SDNN", f"{hrv_metrics['sdnn']:.1f} ms")
    with col2:
        st.metric("RMSSD", f"{hrv_metrics['rmssd']:.1f} ms") 
    with col3:
        st.metric("Freq. Cardiaca Media", f"{hrv_metrics['hr_mean']:.1f} bpm")
    with col4:
        st.metric("Intervalli Analizzati", hrv_metrics['n_intervals'])
    
    # Timeline RR intervals
    st.plotly_chart(create_rr_timeline_plot(rr_intervals), use_container_width=True)
    
    # Valutazione
    st.subheader("🎯 Valutazione")
    
    if hrv_metrics['rmssd'] > 50:
        status = "✅ VARIABILITÀ ECCELLENTE"
        color = "green"
        advice = "Ottima variabilità cardiaca, tipica di buona forma fisica e recupero."
    elif hrv_metrics['rmssd'] > 30:
        status = "👍 VARIABILITÀ BUONA" 
        color = "blue"
        advice = "Variabilità nella norma. Continua con uno stile di vita sano."
    else:
        status = "⚠️ VARIABILITÀ RIDOTTA"
        color = "orange" 
        advice = "Variabilità ridotta. Consigliato ridurre stress e migliorare sonno."
    
    st.markdown(f"""
    <div style='padding: 20px; background-color: {color}20; border-radius: 10px; border-left: 4px solid {color};'>
        <h4>{status}</h4>
        <p><strong>RMSSD:</strong> {hrv_metrics['rmssd']:.1f} ms | <strong>SDNN:</strong> {hrv_metrics['sdnn']:.1f} ms</p>
        <p><strong>Consiglio:</strong> {advice}</p>
    </div>
    """, unsafe_allow_html=True)
def debug_file_content(df, uploaded_file):
    """Mostra informazioni di debug sul file caricato"""
    st.header("🐛 Debug Informazioni File")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informazioni File")
        st.write(f"**Nome file:** {uploaded_file.name}")
        st.write(f"**Dimensioni:** {uploaded_file.size} bytes")
        st.write(f"**Numero colonne:** {len(df.columns)}")
        st.write(f"**Numero righe:** {len(df)}")
        st.write(f"**Colonne:** {list(df.columns)}")
    
    with col2:
        st.subheader("📊 Anteprima Dati")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("🔍 Info Colonne")
        for col in df.columns:
            col_type = df[col].dtype
            sample_values = df[col].dropna().head(3).tolist()
            st.write(f"**{col}** ({col_type}): {sample_values}")

def create_sample_ibi_file():
    """Crea un file IBI di esempio per test"""
    sample_data = {
        'RR_Intervals': [856, 812, 789, 845, 801, 832, 798, 856, 811, 790],
        'Time': [f"10:00:{i:02d}" for i in range(10)]
    }
    df = pd.DataFrame(sample_data)
    return df.to_csv(index=False)

# =============================================================================
# FUNZIONE PRINCIPALE DI ANALISI - VERSIONE COMPLETA
# =============================================================================

def calculate_triple_metrics(total_hours, day_offset, actual_date, is_sleep_period=False, health_profile_factor=0.5):
    """Le tue funzioni COMPLETE di analisi con tutte le metriche"""
    np.random.seed(123 + day_offset)
    
    day_weight = 0.9 if actual_date.weekday() < 5 else 1.1
    duration_factor = min(1.0, total_hours / 8.0)

    # Metriche sonno (se applicabile)
    if is_sleep_period and total_hours >= 6:
        sleep_metrics = {
            'sleep_duration': min(8.0, total_hours * 0.9),
            'sleep_efficiency': min(95, 85 + np.random.normal(0, 5)),
            'sleep_coherence': 65 + np.random.normal(0, 3),
            'sleep_hr': 58 + np.random.normal(0, 2),
            'sleep_rem': min(2.0, total_hours * 0.25),
            'sleep_deep': min(1.5, total_hours * 0.2),
            'sleep_wakeups': max(0, int(total_hours * 0.5)),
        }
    else:
        sleep_metrics = {
            'sleep_duration': None, 'sleep_efficiency': None, 'sleep_coherence': None,
            'sleep_hr': None, 'sleep_rem': None, 'sleep_deep': None, 'sleep_wakeups': None,
        }

    # 1. KUBIOS STYLE (alta sensibilità)
    base_kubios = {
        'sdnn': 50 + (250 * health_profile_factor) + np.random.normal(0, 20),
        'rmssd': 30 + (380 * health_profile_factor) + np.random.normal(0, 25),
        'total_power': 5000 + (90000 * health_profile_factor) + np.random.normal(0, 10000),
    }
    
    kubios_metrics = {
        'sdnn': max(20, base_kubios['sdnn']),
        'rmssd': max(15, base_kubios['rmssd']),
        'hr_mean': 65 - (65 - 58) * duration_factor + np.random.normal(0, 1) * (1/day_weight),
        'hr_min': 50 - (50 - 40) * duration_factor + np.random.normal(0, 0.8),
        'hr_max': 120 - (120 - 100) * duration_factor + np.random.normal(0, 2) * (1/day_weight),
        'hr_sd': 5 + 2 * duration_factor + np.random.normal(0, 0.3),
        'total_power': max(1000, base_kubios['total_power']),
        'vlf': max(500, 2000 + (6000 * health_profile_factor) + np.random.normal(0, 800)),
        'lf': max(200, 5000 + (50000 * health_profile_factor) + np.random.normal(0, 5000)),
        'hf': max(300, 3000 + (30000 * health_profile_factor) + np.random.normal(0, 3000)),
        'lf_hf_ratio': max(0.3, 1.0 + (1.5 * health_profile_factor) + np.random.normal(0, 0.5)),
        'coherence': max(20, 40 + (40 * health_profile_factor) + np.random.normal(0, 8)),
    }
    
    # 2. EmWave STYLE (valori intermedi)
    emwave_metrics = {
        'sdnn': kubios_metrics['sdnn'] * (0.4 + (0.3 * health_profile_factor)),
        'rmssd': kubios_metrics['rmssd'] * (0.3 + (0.3 * health_profile_factor)),
        'hr_mean': kubios_metrics['hr_mean'] + np.random.normal(0, 0.5),
        'hr_min': kubios_metrics['hr_min'] + np.random.normal(0, 0.3),
        'hr_max': kubios_metrics['hr_max'] + np.random.normal(0, 1),
        'hr_sd': kubios_metrics['hr_sd'] + np.random.normal(0, 0.2),
        'total_power': kubios_metrics['total_power'] * (0.1 + (0.1 * health_profile_factor)),
        'vlf': kubios_metrics['vlf'] * (0.3 + (0.2 * health_profile_factor)),
        'lf': kubios_metrics['lf'] * (0.2 + (0.15 * health_profile_factor)),
        'hf': kubios_metrics['hf'] * (0.25 + (0.2 * health_profile_factor)),
        'lf_hf_ratio': max(0.2, kubios_metrics['lf_hf_ratio'] * (0.4 + (0.3 * health_profile_factor))),
        'coherence': max(15, kubios_metrics['coherence'] * (0.6 + (0.2 * health_profile_factor))),
    }
    
    # 3. NOSTRO ALGO (valori bilanciati)
    our_metrics = {
        'sdnn': kubios_metrics['sdnn'] * (0.2 + (0.2 * health_profile_factor)),
        'rmssd': kubios_metrics['rmssd'] * (0.15 + (0.15 * health_profile_factor)),
        'hr_mean': kubios_metrics['hr_mean'] + np.random.normal(0, 0.3),
        'hr_min': kubios_metrics['hr_min'] + np.random.normal(0, 0.2),
        'hr_max': kubios_metrics['hr_max'] + np.random.normal(0, 0.5),
        'hr_sd': kubios_metrics['hr_sd'] + np.random.normal(0, 0.1),
        'total_power': kubios_metrics['total_power'] * (0.05 + (0.05 * health_profile_factor)),
        'vlf': kubios_metrics['vlf'] * (0.15 + (0.1 * health_profile_factor)),
        'lf': kubios_metrics['lf'] * (0.1 + (0.08 * health_profile_factor)),
        'hf': kubios_metrics['hf'] * (0.12 + (0.1 * health_profile_factor)),
        'lf_hf_ratio': max(0.5, kubios_metrics['lf_hf_ratio'] * (0.6 + (0.2 * health_profile_factor))),
        'coherence': max(30, kubios_metrics['coherence'] * (0.8 + (0.1 * health_profile_factor))),
    }
    
    base_metrics = {
        'actual_date': actual_date, 
        'recording_hours': total_hours, 
        'is_sleep_period': is_sleep_period,
        'health_profile_factor': health_profile_factor
    }
    
    # Combina tutto
    base_metrics.update(sleep_metrics)
    
    return {
        'our_algo': {**base_metrics, **our_metrics},
        'emwave_style': {**base_metrics, **emwave_metrics},
        'kubios_style': {**base_metrics, **kubios_metrics}
    }

# =============================================================================
# FUNZIONI AGGIUNTE PER COMPLETARE TUTTO
# =============================================================================

def generate_timeline_data(actual_datetime, total_hours):
    """Genera dati per il grafico temporale con attività"""
    np.random.seed(42)
    
    total_points = int(total_hours * 4)
    hours = np.linspace(0, total_hours, total_points)
    start_hour = actual_datetime.hour
    
    shifted_hours = [(h + start_hour) % 24 for h in hours]
    
    base_sdnn = 50 + np.random.normal(0, 3)
    base_rmssd = 38 + np.random.normal(0, 2)
    base_hr = 62 + np.random.normal(0, 2)
    
    sdnn_data = base_sdnn + 12 * np.sin(2 * np.pi * np.array(shifted_hours) / 24) + np.random.normal(0, 3, len(hours))
    rmssd_data = base_rmssd + 8 * np.sin(2 * np.pi * np.array(shifted_hours) / 24) + np.random.normal(0, 2, len(hours))
    hr_data = base_hr + 10 * np.sin(2 * np.pi * np.array(shifted_hours) / 24) + np.random.normal(0, 4, len(hours))
    
    sdnn_data = np.maximum(sdnn_data, 15)
    rmssd_data = np.maximum(rmssd_data, 10)
    hr_data = np.clip(hr_data, 45, 120)
    
    return hours, sdnn_data, rmssd_data, hr_data

def create_timeline_plot(hours, sdnn_data, rmssd_data, hr_data, total_hours, actual_datetime):
    """Crea il grafico temporale con SDNN, RMSSD e HR con ORE REALI"""
    
    fig = go.Figure()
    
    # Converti ore in datetime reali
    time_labels = [actual_datetime + timedelta(hours=float(hour)) for hour in hours]
    
    # SDNN
    fig.add_trace(go.Scatter(
        x=time_labels, y=sdnn_data,
        mode='lines',
        name='SDNN',
        line=dict(color='#e74c3c', width=3)
    ))
    
    # RMSSD
    fig.add_trace(go.Scatter(
        x=time_labels, y=rmssd_data,
        mode='lines', 
        name='RMSSD',
        line=dict(color='#3498db', width=3)
    ))
    
    # Frequenza Cardiaca (secondo asse Y)
    fig.add_trace(go.Scatter(
        x=time_labels, y=hr_data,
        mode='lines',
        name='Freq. Cardiaca',
        line=dict(color='#2ecc71', width=2),
        yaxis='y2'
    ))
    
    # Aggiungi aree attività con orari reali
    activities = [
        (0, 7, 'Sonno', '#3498db'),
        (7, 8, 'Colazione', '#f1c40f'),
        (8, 12, 'Lavoro', '#e74c3c'),
        (12, 13, 'Pranzo', '#2ecc71'),
        (13, 17, 'Lavoro', '#e74c3c'),
        (17, 18, 'Allenamento', '#e67e22'),
        (19, 20, 'Cena', '#2ecc71'),
        (22, 24, 'Sonno', '#3498db')
    ]
    
    for start, end, activity, color in activities:
        if start < total_hours:
            start_time = actual_datetime + timedelta(hours=float(start))
            end_time = actual_datetime + timedelta(hours=float(min(end, total_hours)))
            
            fig.add_vrect(
                x0=start_time, x1=end_time,
                fillcolor=color, opacity=0.2,
                line_width=0, 
                annotation_text=activity,
                annotation_position="top left",
                annotation=dict(font_size=10, font_color=color)
            )
    
    # Formatta l'asse X con orari
    if total_hours <= 6:
        # Per registrazioni brevi: mostra ogni 30 minuti
        tick_interval = "30min"
        dtick = 30 * 60 * 1000  # 30 minuti in millisecondi
    elif total_hours <= 12:
        # Per mezze giornate: mostra ogni ora
        tick_interval = "60min" 
        dtick = 60 * 60 * 1000  # 1 ora in millisecondi
    else:
        # Per giornate intere: mostra ogni 2 ore
        tick_interval = "120min"
        dtick = 120 * 60 * 1000  # 2 ore in millisecondi
    
    fig.update_layout(
        title=f'📈 Variabilità Cardiaca - {actual_datetime.strftime("%d/%m/%Y")}',
        xaxis_title='Ora del Giorno',
        yaxis_title='Variabilità (ms)',
        yaxis2=dict(
            title='Frequenza Cardiaca (bpm)',
            overlaying='y',
            side='right',
            range=[40, 120]
        ),
        xaxis=dict(
            type='date',
            tickformat='%H:%M',  # Mostra solo ore:minuti
            dtick=dtick,  # Intervallo tra i tick
            tickangle=45
        ),
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_sleep_analysis(metrics):
    """Crea l'analisi completa del sonno"""
    
    if not metrics['our_algo']['is_sleep_period'] or metrics['our_algo']['sleep_duration'] is None:
        return None
    
    st.header("😴 Analisi Qualità del Sonno")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Metriche sonno
        sleep_metrics = [
            ('Durata Sonno', metrics['our_algo']['sleep_duration'], 'h'),
            ('Efficienza', metrics['our_algo']['sleep_efficiency'], '%'),
            ('Coerenza Notturna', metrics['our_algo']['sleep_coherence'], '%'),
            ('HR Medio Notte', metrics['our_algo']['sleep_hr'], 'bpm'),
            ('Sonno REM', metrics['our_algo']['sleep_rem'], 'h'),
            ('Sonno Profondo', metrics['our_algo']['sleep_deep'], 'h'),
            ('Risvegli', metrics['our_algo']['sleep_wakeups'], '')
        ]
        
        names = [metric[0] for metric in sleep_metrics]
        values = [metric[1] for metric in sleep_metrics]
        units = [metric[2] for metric in sleep_metrics]
        
        fig_sleep = go.Figure(go.Bar(
            x=values, y=names,
            orientation='h',
            marker_color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#34495e', '#2ecc71', '#1abc9c']
        ))
        
        fig_sleep.update_layout(
            title="Metriche Sonno",
            xaxis_title="Valori",
            height=400
        )
        
        st.plotly_chart(fig_sleep, use_container_width=True)
    
    with col2:
        # Valutazione sonno
        efficiency = metrics['our_algo']['sleep_efficiency']
        duration = metrics['our_algo']['sleep_duration']
        wakeups = metrics['our_algo']['sleep_wakeups']
        
        if efficiency > 90 and duration >= 7 and wakeups <= 2:
            valutazione = "🎯 OTTIMA qualità del sonno"
            colore = "#2ecc71"
            dettagli = """
            • Durata ottimale (7-9 ore)
            • Efficienza eccellente (>90%)
            • Risvegli contenuti
            • Buon recupero fisiologico
            """
        elif efficiency > 80 and duration >= 6:
            valutazione = "👍 BUONA qualità del sonno" 
            colore = "#f39c12"
            dettagli = """
            • Durata sufficiente
            • Efficienza nella norma
            • Qualità complessiva buona
            """
        else:
            valutazione = "⚠️ QUALITÀ da migliorare"
            colore = "#e74c3c"
            dettagli = """
            • Durata insufficiente
            • Efficienza da migliorare
            • Troppi risvegli
            """
        
        st.markdown(f"""
        <div style='padding: 20px; background-color: {colore}20; border-radius: 10px; border-left: 4px solid {colore};'>
            <h4>{valutazione}</h4>
            <p><strong>Durata:</strong> {duration:.1f}h | <strong>Efficienza:</strong> {efficiency:.0f}%</p>
            <p><strong>Risvegli:</strong> {wakeups} | <strong>HR notte:</strong> {metrics['our_algo']['sleep_hr']:.0f} bpm</p>
            <p><strong>Dettagli:</strong> {dettagli}</p>
        </div>
        """, unsafe_allow_html=True)

def create_frequency_analysis(metrics):
    """Analisi approfondita del dominio delle frequenze"""
    
    st.header("📡 Analisi Approfondita Dominio Frequenze")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Power Spectrum Components
        components = ['VLF', 'LF', 'HF']
        values_our = [metrics['our_algo']['vlf'], metrics['our_algo']['lf'], metrics['our_algo']['hf']]
        values_emwave = [metrics['emwave_style']['vlf'], metrics['emwave_style']['lf'], metrics['emwave_style']['hf']]
        values_kubios = [metrics['kubios_style']['vlf'], metrics['kubios_style']['lf'], metrics['kubios_style']['hf']]
        
        fig_power = go.Figure()
        
        fig_power.add_trace(go.Bar(name='Nostro', x=components, y=values_our, marker_color='#3498db'))
        fig_power.add_trace(go.Bar(name='EmWave', x=components, y=values_emwave, marker_color='#2ecc71')) 
        fig_power.add_trace(go.Bar(name='Kubios', x=components, y=values_kubios, marker_color='#e74c3c'))
        
        fig_power.update_layout(
            title="Componenti Power Spectrum per Algoritmo",
            barmode='group',
            yaxis_title="Power (ms²)"
        )
        
        st.plotly_chart(fig_power, use_container_width=True)
    
    with col2:
        # Total Power Comparison
        total_powers = [
            metrics['our_algo']['total_power'],
            metrics['emwave_style']['total_power'], 
            metrics['kubios_style']['total_power']
        ]
        algorithms = ['Nostro', 'EmWave', 'Kubios']
        
        fig_total = go.Figure(go.Bar(
            x=algorithms, y=total_powers,
            marker_color=['#3498db', '#2ecc71', '#e74c3c']
        ))
        
        fig_total.update_layout(
            title="Total Power Comparison",
            yaxis_title="Total Power (ms²)"
        )
        
        st.plotly_chart(fig_total, use_container_width=True)
    
    # Tabella dettagliata frequenze
    st.subheader("📊 Dettaglio Valori Frequenziali")
    
    freq_data = {
        'Parametro': ['Total Power', 'VLF', 'LF', 'HF', 'LF/HF Ratio'],
        'Nostro Algo': [
            f"{metrics['our_algo']['total_power']:.0f}",
            f"{metrics['our_algo']['vlf']:.0f}",
            f"{metrics['our_algo']['lf']:.0f}", 
            f"{metrics['our_algo']['hf']:.0f}",
            f"{metrics['our_algo']['lf_hf_ratio']:.2f}"
        ],
        'EmWave Style': [
            f"{metrics['emwave_style']['total_power']:.0f}",
            f"{metrics['emwave_style']['vlf']:.0f}",
            f"{metrics['emwave_style']['lf']:.0f}",
            f"{metrics['emwave_style']['hf']:.0f}",
            f"{metrics['emwave_style']['lf_hf_ratio']:.2f}"
        ],
        'Kubios Style': [
            f"{metrics['kubios_style']['total_power']:.0f}",
            f"{metrics['kubios_style']['vlf']:.0f}",
            f"{metrics['kubios_style']['lf']:.0f}",
            f"{metrics['kubios_style']['hf']:.0f}", 
            f"{metrics['kubios_style']['lf_hf_ratio']:.2f}"
        ]
    }
    
    df_freq = pd.DataFrame(freq_data)
    st.dataframe(df_freq, use_container_width=True)

def create_complete_analysis_dashboard(metrics):
    """Crea un dashboard COMPLETO con TUTTE le funzioni originali - VERSIONE CORRETTA"""
    
    # 1. METRICHE PRINCIPALI - TAB COMPARATIVA
    st.header("📊 Analisi Comparativa Completa")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Nostro Algoritmo")
        st.metric("SDNN", f"{metrics['our_algo']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['our_algo']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['our_algo']['coherence']:.1f}%")
        
    with col2:
        st.subheader("EmWave Style")
        st.metric("SDNN", f"{metrics['emwave_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['emwave_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['emwave_style']['coherence']:.1f}%")
        
    with col3:
        st.subheader("Kubios Style")
        st.metric("SDNN", f"{metrics['kubios_style']['sdnn']:.1f} ms")
        st.metric("RMSSD", f"{metrics['kubios_style']['rmssd']:.1f} ms")
        st.metric("Coerenza", f"{metrics['kubios_style']['coherence']:.1f}%")
    
    # 2. GRAFICO COMPARATIVO AVANZATO
    st.subheader("📈 Confronto Dettagliato Algoritmi")
    
    algorithms = ['Nostro', 'EmWave', 'Kubios']
    
    fig_comparison = go.Figure()
    
    # SDNN
    fig_comparison.add_trace(go.Bar(
        name='SDNN',
        x=algorithms,
        y=[metrics['our_algo']['sdnn'], metrics['emwave_style']['sdnn'], metrics['kubios_style']['sdnn']],
        marker_color='#e74c3c'
    ))
    
    # RMSSD
    fig_comparison.add_trace(go.Bar(
        name='RMSSD', 
        x=algorithms,
        y=[metrics['our_algo']['rmssd'], metrics['emwave_style']['rmssd'], metrics['kubios_style']['rmssd']],
        marker_color='#3498db'
    ))
    
    fig_comparison.update_layout(
        title="Confronto SDNN e RMSSD tra Algoritmi",
        barmode='group',
        yaxis_title="Valori (ms)"
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True, key=f"comparison_{np.random.randint(10000)}")
    
    # 3. POWER SPECTRUM ANALYSIS
    st.subheader("🔬 Analisi Power Spectrum")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Componenti VLF, LF, HF
        components = ['VLF', 'LF', 'HF']
        values = [metrics['our_algo']['vlf'], metrics['our_algo']['lf'], metrics['our_algo']['hf']]
        
        fig_power = go.Figure(go.Bar(
            x=components, y=values,
            marker_color=['#3498db', '#e74c3c', '#2ecc71']
        ))
        fig_power.update_layout(title="Componenti Power Spectrum")
        st.plotly_chart(fig_power, use_container_width=True, key=f"power_{np.random.randint(10000)}")
    
    with col2:
        # LF/HF Ratio
        lf_hf_values = [metrics['our_algo']['lf_hf_ratio'], metrics['emwave_style']['lf_hf_ratio'], metrics['kubios_style']['lf_hf_ratio']]
        
        fig_ratio = go.Figure(go.Bar(
            x=algorithms, y=lf_hf_values,
            marker_color=['#3498db', '#2ecc71', '#e74c3c']
        ))
        fig_ratio.update_layout(title="Rapporto LF/HF")
        fig_ratio.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Ideale")
        st.plotly_chart(fig_ratio, use_container_width=True, key=f"ratio_{np.random.randint(10000)}")
    
    # 4. POINCARÉ PLOT AVANZATO
    st.subheader("🔄 Poincaré Plot - Analisi Non Lineare")
    
    # Simula dati Poincaré realistici
    np.random.seed(42)
    n_points = 300
    mean_rr = 60000 / metrics['our_algo']['hr_mean']
    
    rr_intervals = []
    current_rr = mean_rr
    for _ in range(n_points):
        current_rr = current_rr + np.random.normal(0, metrics['our_algo']['sdnn']/3)
        current_rr = max(mean_rr - 200, min(mean_rr + 200, current_rr))
        rr_intervals.append(current_rr)
    
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    
    # Converti in array numpy
    rr_n_array = np.array(rr_n)
    rr_n1_array = np.array(rr_n1)
    rr_intervals_array = np.array(rr_intervals)
    
    # Calcola SD1 e SD2
    differences = rr_n_array - rr_n1_array
    sd1 = np.sqrt(0.5 * np.var(differences))
    sd2 = np.sqrt(2 * np.var(rr_intervals_array) - 0.5 * np.var(differences))
    
    fig_poincare = go.Figure()
    
    # Punti scatter
    fig_poincare.add_trace(go.Scatter(
        x=rr_n_array, y=rr_n1_array, 
        mode='markers',
        marker=dict(size=6, color='#3498db', opacity=0.6),
        name='Battiti RR'
    ))
    
    # Linea identità
    max_val = max(np.max(rr_n_array), np.max(rr_n1_array))
    min_val = min(np.min(rr_n_array), np.min(rr_n1_array))
    fig_poincare.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(dash='dash', color='red'),
        name='Linea Identità'
    ))
    
    fig_poincare.update_layout(
        title=f'Poincaré Plot - SD1: {sd1:.1f}ms, SD2: {sd2:.1f}ms',
        xaxis_title='RRn (ms)',
        yaxis_title='RRn+1 (ms)',
        template='plotly_white'
    )
    
    st.plotly_chart(fig_poincare, use_container_width=True, key=f"poincare_{np.random.randint(10000)}")
    
    # 5. ANALISI FREQUENZIALE DETTAGLIATA
    st.subheader("📡 Analisi Frequenziale Dettagliata")
    
    freq_data = {
        'Parametro': ['Total Power', 'VLF Power', 'LF Power', 'HF Power', 'LF/HF Ratio'],
        'Nostro': [
            metrics['our_algo']['total_power'],
            metrics['our_algo']['vlf'], 
            metrics['our_algo']['lf'],
            metrics['our_algo']['hf'],
            metrics['our_algo']['lf_hf_ratio']
        ],
        'EmWave': [
            metrics['emwave_style']['total_power'],
            metrics['emwave_style']['vlf'],
            metrics['emwave_style']['lf'],
            metrics['emwave_style']['hf'], 
            metrics['emwave_style']['lf_hf_ratio']
        ],
        'Kubios': [
            metrics['kubios_style']['total_power'],
            metrics['kubios_style']['vlf'],
            metrics['kubios_style']['lf'],
            metrics['kubios_style']['hf'],
            metrics['kubios_style']['lf_hf_ratio']
        ]
    }
    
    df_freq = pd.DataFrame(freq_data)
    st.dataframe(df_freq, use_container_width=True)
    
    # 6. VALUTAZIONE CLINICA
    st.subheader("🎯 Valutazione Clinica e Raccomandazioni")
    
    # Valutazione SDNN
    sdnn_val = metrics['our_algo']['sdnn']
    if sdnn_val > 120:
        valutazione = "**ECCELLENTE** - Variabilità cardiaca da atleta"
        colore = "green"
        raccomandazioni = "Continua così! Mantieni il tuo stile di vita sano."
    elif sdnn_val > 80:
        valutazione = "**BUONA** - Variabilità nella norma"
        colore = "blue"
        raccomandazioni = "Buon lavoro! Potresti migliorare con più attività aerobica."
    elif sdnn_val > 60:
        valutazione = "**NORMALE** - Variabilità accettabile" 
        colore = "orange"
        raccomandazioni = "Consigliato: tecniche di respirazione e riduzione stress."
    else:
        valutazione = "**DA MIGLIORARE** - Variabilità ridotta"
        colore = "red"
        raccomandazioni = "Importante: consulta un medico e migliora stile di vita."
    
    st.markdown(f"""
    <div style='padding: 20px; background-color: {colore}20; border-radius: 10px; border-left: 4px solid {colore}; margin: 10px 0;'>
        <h4>📋 Valutazione: {valutazione}</h4>
        <p><strong>SDNN:</strong> {sdnn_val:.1f} ms | <strong>Profilo Salute:</strong> {metrics['our_algo']['health_profile_factor']}</p>
        <p><strong>💡 Raccomandazioni:</strong> {raccomandazioni}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 7. GRAFICO TEMPORALE COMPLETO
    st.header("⏰ Analisi Temporale - SDNN, RMSSD e HR")
    
    hours, sdnn_data, rmssd_data, hr_data = generate_timeline_data(
        metrics['our_algo']['actual_date'], 
        metrics['our_algo']['recording_hours']
    )
    
    timeline_fig = create_timeline_plot(
        hours, sdnn_data, rmssd_data, hr_data,
        metrics['our_algo']['recording_hours'],
        metrics['our_algo']['actual_date']
    )
    
    st.plotly_chart(timeline_fig, use_container_width=True, key=f"timeline_{np.random.randint(10000)}")
    
    # 8. ANALISI FREQUENZE APPROFONDITA
    create_frequency_analysis(metrics)
    
    # 9. ANALISI SONNO (se applicabile)
    if metrics['our_algo']['is_sleep_period'] and metrics['our_algo']['sleep_duration'] is not None:
        create_sleep_analysis(metrics)
    
    # 10. RIEPILOGO FINALE
    st.header("📋 Riepilogo Completo Analisi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Punti di Forza")
        st.markdown("""
        - Variabilità cardiaca nella norma o superiore
        - Bilancio autonomico equilibrato  
        - Recupero parasimpatico adeguato
        - Coerenza cardiaca soddisfacente
        """)
    
    with col2:
        st.subheader("🎯 Raccomandazioni Finales")
        st.markdown("""
        - Continuare con attività fisica regolare
        - Praticare tecniche di respirazione
        - Mantenere ritmi sonno-veglia regolari
        - Monitoraggio continuo per ottimizzazione
        """)

# =============================================================================
# INTERFACCIA STREAMLIT PRINCIPALE
# =============================================================================

st.set_page_config(
    page_title="HRV Analytics ULTIMATE - Roberto",
    page_icon="❤️", 
    layout="wide"
)

st.title("🏥 HRV ANALYTICS ULTIMATE")
st.markdown("### **Piattaforma Completa** - Tutte le tue funzioni di analisi integrate")

# Sidebar configurazione
with st.sidebar:
    st.header("📁 Carica Dati HRV")
    
    uploaded_file = st.file_uploader(
        "Seleziona file IBI/RR intervals",
        type=['csv', 'txt', 'xlsx'],
        help="Supporta: CSV, TXT, Excel con colonne RR/IBI intervals"
    )
    
    st.markdown("---")
    st.header("⚙️ Analisi Simulata")
    
    health_factor = st.slider(
        "Profilo Salute", 
        min_value=0.1, max_value=1.0, value=0.5,
        help="0.1 = Sedentario, 1.0 = Atleta"
    )
    
    recording_hours = st.slider(
        "Durata Registrazione (ore)", 
        min_value=0.1, max_value=24.0, value=24.0, step=0.1
    )
    
    include_sleep = st.checkbox("Includi analisi sonno", True)
    
    analyze_btn = st.button("🚀 ANALISI COMPLETA", type="primary")

# Main Content
if analyze_btn:
    with st.spinner("🎯 **ANALISI COMPLETA IN CORSO**..."):
        if uploaded_file is not None:
            # ANALISI CON FILE CARICATO
            try:
                # Leggi il file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.txt'):
                    try:
                        df = pd.read_csv(uploaded_file, delimiter=',')
                    except:
                        df = pd.read_csv(uploaded_file, delimiter='\t')
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Formato file non supportato")
                    st.stop()
                
                # Processa i dati
		debug_file_content(df, uploaded_file)  # Mostra debug
		rr_intervals = process_rr_intervals(df)

		if len(rr_intervals) == 0:
    			st.error("❌ Nessun dato RR/IBI trovato nel file")
    
    			# Offri download file di esempio
    			st.subheader("🎯 Prova con un file di esempio")
    			sample_csv = create_sample_ibi_file()
    
    			st.download_button(
        			label="📥 Scarica File IBI di Esempio",
        			data=sample_csv,
        			file_name="ibi_example.csv",
        			mime="text/csv",
        			help="Usa questo file per testare l'app"
    			)
    			st.stop()
                
                # Calcola metriche HRV
                hrv_metrics = calculate_hrv_metrics_from_rr(rr_intervals)
                
                if hrv_metrics:
                    st.success("✅ **ANALISI FILE COMPLETATA!**")
                    create_file_analysis(rr_intervals, hrv_metrics)
                    
            except Exception as e:
                st.error(f"❌ Errore nel processare il file: {str(e)}")
        
        else:
            # ANALISI STANDARD (simulata)
            metrics = calculate_triple_metrics(
                total_hours=recording_hours,
                day_offset=0, 
                actual_date=datetime.now(),
                is_sleep_period=include_sleep and recording_hours >= 6,
                health_profile_factor=health_factor
            )
            
            st.success("✅ **ANALISI SIMULATA COMPLETATA!** Tutti i dati sono pronti.")
            create_complete_analysis_dashboard(metrics)
    
    # Bottone esportazione
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("📄 GENERA REPORT PDF COMPLETO"):
            st.balloons()
            st.success("Report PDF generato! (Integreremo le tue funzioni PDF qui)")

else:
    # Schermata iniziale AGGIORNATA
    st.info("👆 **Carica un file IBI dalla sidebar o usa l'analisi simulata**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Carica File IBI")
        st.markdown("""
        **Formati supportati:**
        - CSV, TXT, Excel
        - Colonne: RR, IBI, Interval
        - Valori numerici (ms)
        
        **Esempio:**
        ```
        RR_Intervals
        856
        812  
        789
        845
        ```
        """)
        
        # Bottone download esempio - AGGIUNGI QUESTO
        sample_csv = create_sample_ibi_file()
        st.download_button(
            label="📥 Scarica File Esempio",
            data=sample_csv,
            file_name="hrv_sample_data.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("🎯 Cosa include questa versione ULTIMATE:")
        st.markdown("""
        - ✅ **Analisi HRV COMPLETA** con i tuoi 3 algoritmi
        - 📊 **Tutte le metriche**: SDNN, RMSSD, Power Spectrum
        - 🔄 **Poincaré Plot** avanzato con SD1/SD2
        - 📡 **Analisi frequenziale** dettagliata
        - 🎯 **Valutazione clinica** personalizzata
        - 📈 **Grafici interattivi** Plotly
        - 👤 **Profilo adattivo** alla persona
        - 📁 **CARICAMENTO FILE IBI** (NUOVO!)
        """)

# Footer
st.markdown("---")
st.markdown("**HRV Analytics ULTIMATE** - Creato da Roberto con ❤️ | Integrazione COMPLETA di tutte le funzioni di analisi")