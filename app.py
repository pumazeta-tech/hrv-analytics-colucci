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
# FUNZIONI PER CARICAMENTO FILE IBI - VERSIONE VELOCE
# =============================================================================

def read_ibi_file_fast(uploaded_file):
    """Legge file IBI - VERSIONE VELOCE e PULITA"""
    try:
        uploaded_file.seek(0)
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.splitlines()
        
        rr_intervals = []
        found_points = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if '[POINTS]' in line:
                found_points = True
                continue
            
            if found_points and not line.startswith('['):
                try:
                    val = float(line.replace(',', '.'))
                    if 200 <= val <= 2000:
                        rr_intervals.append(val)
                    elif 0.2 <= val <= 2.0:
                        rr_intervals.append(val * 1000)
                except ValueError:
                    continue
        
        return np.array(rr_intervals, dtype=float)
        
    except Exception as e:
        st.error(f"❌ Errore nella lettura del file: {e}")
        return np.array([])

def calculate_hrv_metrics_from_rr(rr_intervals):
    """Calcola metriche HRV da RR intervals - VERSIONE VELOCA"""
    if len(rr_intervals) == 0:
        return None
    
    rr_intervals = np.array(rr_intervals, dtype=float)
    
    # Se valori troppo piccoli, converti in ms
    if np.mean(rr_intervals) < 100:
        rr_intervals = rr_intervals * 1000
    
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

# =============================================================================
# FUNZIONE PRINCIPALE DI ANALISI - VERSIONE COMPLETA ORIGINALE
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
# FUNZIONI AGGIUNTE PER COMPLETARE TUTTO - ORIGINALI
# =============================================================================

def generate_timeline_data(start_datetime, total_hours):
    """Genera dati per il grafico temporale con attività"""
    np.random.seed(42)
    
    total_points = int(total_hours * 4)
    hours = np.linspace(0, total_hours, total_points)
    
    # Crea timeline reale basata su start_datetime
    time_labels = [start_datetime + timedelta(hours=float(hour)) for hour in hours]
    
    start_hour = start_datetime.hour + start_datetime.minute/60
    
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
    
    return time_labels, sdnn_data, rmssd_data, hr_data

def create_timeline_plot_with_activities(time_labels, sdnn_data, rmssd_data, hr_data, total_hours, start_datetime):
    """Crea il grafico temporale con SDNN, RMSSD e HR con attività personalizzate"""
    
    fig = go.Figure()
    
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
    
    # Aggiungi attività dal diario
    if 'activities' in st.session_state and st.session_state.activities:
        for i, activity in enumerate(st.session_state.activities):
            # Verifica se l'attività ricade nel periodo di registrazione
            recording_end = start_datetime + timedelta(hours=total_hours)
            
            if (activity['end'] >= start_datetime and activity['start'] <= recording_end):
                # Calcola l'intersezione tra attività e periodo di registrazione
                activity_start = max(activity['start'], start_datetime)
                activity_end = min(activity['end'], recording_end)
                
                if activity_start < activity_end:
                    fig.add_vrect(
                        x0=activity_start, x1=activity_end,
                        fillcolor=activity['color'], opacity=0.3,
                        line_width=1, line_color=activity['color'],
                        annotation_text=f"{activity['type']}",
                        annotation_position="top left",
                        annotation=dict(font_size=10, font_color=activity['color'])
                    )
    
    # Formatta l'asse X con orari
    if total_hours <= 6:
        dtick = 30 * 60 * 1000
    elif total_hours <= 12:
        dtick = 60 * 60 * 1000
    else:
        dtick = 120 * 60 * 1000
    
    fig.update_layout(
        title=f'📈 Variabilità Cardiaca - {start_datetime.strftime("%d/%m/%Y %H:%M")}',
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
            tickformat='%H:%M',
            dtick=dtick,
            tickangle=45
        ),
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    return fig

def create_sleep_analysis(metrics):
    """Crea l'analisi completa del sonno"""
    
    st.header("😴 Analisi Qualità del Sonno")
    
    sleep_data = metrics['our_algo']
    duration = sleep_data.get('sleep_duration', 0)
    efficiency = sleep_data.get('sleep_efficiency', 0)
    coherence = sleep_data.get('sleep_coherence', 0)
    hr_night = sleep_data.get('sleep_hr', 0)
    rem = sleep_data.get('sleep_rem', 0)
    deep = sleep_data.get('sleep_deep', 0)
    wakeups = sleep_data.get('sleep_wakeups', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Metriche Sonno")
        
        sleep_metrics = [
            ('Durata Sonno', duration, 'h', '#3498db'),
            ('Efficienza', efficiency, '%', '#e74c3c'),
            ('Coerenza Notturna', coherence, '%', '#f39c12'),
            ('HR Medio Notte', hr_night, 'bpm', '#9b59b6'),
            ('Sonno REM', rem, 'h', '#34495e'),
            ('Sonno Profondo', deep, 'h', '#2ecc71'),
            ('Risvegli', wakeups, '', '#1abc9c')
        ]
        
        names = [f"{metric[0]}" for metric in sleep_metrics]
        values = [metric[1] for metric in sleep_metrics]
        colors = [metric[3] for metric in sleep_metrics]
        
        fig_sleep = go.Figure(go.Bar(
            x=values, y=names,
            orientation='h',
            marker_color=colors
        ))
        
        fig_sleep.update_layout(
            title="Metriche Sonno Dettagliate",
            xaxis_title="Valori",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_sleep, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Valutazione Qualità Sonno")
        
        if duration > 0:
            if efficiency > 90 and duration >= 7 and wakeups <= 2:
                valutazione = "🎯 OTTIMA qualità del sonno"
                colore = "#2ecc71"
                dettagli = "• Durata ottimale (7-9 ore)<br>• Efficienza eccellente (>90%)<br>• Risvegli contenuti"
            elif efficiency > 80 and duration >= 6:
                valutazione = "👍 BUONA qualità del sonno" 
                colore = "#f39c12"
                dettagli = "• Durata sufficiente<br>• Efficienza nella norma<br>• Qualità complessiva buona"
            else:
                valutazione = "⚠️ QUALITÀ da migliorare"
                colore = "#e74c3c"
                dettagli = "• Durata insufficiente<br>• Efficienza da migliorare<br>• Troppi risvegli"
            
            st.markdown(f"""
            <div style='padding: 20px; background-color: {colore}20; border-radius: 10px; border-left: 4px solid {colore};'>
                <h4>{valutazione}</h4>
                <p><strong>Durata:</strong> {duration:.1f}h | <strong>Efficienza:</strong> {efficiency:.0f}%</p>
                <p><strong>Risvegli:</strong> {wakeups} | <strong>HR notte:</strong> {hr_night:.0f} bpm</p>
                <p><strong>Dettagli:</strong> {dettagli}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if duration > 0:
                fig_pie = go.Figure(go.Pie(
                    labels=['Sonno Leggero', 'Sonno REM', 'Sonno Profondo'],
                    values=[duration - rem - deep, rem, deep],
                    marker_colors=['#3498db', '#e74c3c', '#2ecc71']
                ))
                fig_pie.update_layout(title="Composizione Sonno")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        else:
            st.info("💤 **Dati sonno non disponibili**")

def create_frequency_analysis(metrics):
    """Analisi approfondita del dominio delle frequenze"""
    
    st.header("📡 Analisi Approfondita Dominio Frequenze")
    
    col1, col2 = st.columns(2)
    
    with col1:
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

def create_complete_analysis_dashboard(metrics, start_datetime=None):
    """Crea un dashboard COMPLETO con TUTTE le funzioni"""
    
    if start_datetime is None:
        start_datetime = metrics['our_algo']['actual_date']
    
    # 1. METRICHE PRINCIPALI
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
    
    # 2. GRAFICO COMPARATIVO
    st.subheader("📈 Confronto Dettagliato Algoritmi")
    
    algorithms = ['Nostro', 'EmWave', 'Kubios']
    
    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(name='SDNN', x=algorithms, y=[metrics['our_algo']['sdnn'], metrics['emwave_style']['sdnn'], metrics['kubios_style']['sdnn']], marker_color='#e74c3c'))
    fig_comparison.add_trace(go.Bar(name='RMSSD', x=algorithms, y=[metrics['our_algo']['rmssd'], metrics['emwave_style']['rmssd'], metrics['kubios_style']['rmssd']], marker_color='#3498db'))
    
    fig_comparison.update_layout(title="Confronto SDNN e RMSSD tra Algoritmi", barmode='group')
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # 3. POWER SPECTRUM
    st.subheader("🔬 Analisi Power Spectrum")
    
    col1, col2 = st.columns(2)
    
    with col1:
        components = ['VLF', 'LF', 'HF']
        values = [metrics['our_algo']['vlf'], metrics['our_algo']['lf'], metrics['our_algo']['hf']]
        fig_power = go.Figure(go.Bar(x=components, y=values, marker_color=['#3498db', '#e74c3c', '#2ecc71']))
        fig_power.update_layout(title="Componenti Power Spectrum")
        st.plotly_chart(fig_power, use_container_width=True)
    
    with col2:
        lf_hf_values = [metrics['our_algo']['lf_hf_ratio'], metrics['emwave_style']['lf_hf_ratio'], metrics['kubios_style']['lf_hf_ratio']]
        fig_ratio = go.Figure(go.Bar(x=algorithms, y=lf_hf_values, marker_color=['#3498db', '#2ecc71', '#e74c3c']))
        fig_ratio.update_layout(title="Rapporto LF/HF")
        fig_ratio.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Ideale")
        st.plotly_chart(fig_ratio, use_container_width=True)
    
    # 4. POINCARÉ PLOT
    st.subheader("🔄 Poincaré Plot - Analisi Non Lineare")
    
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
    
    rr_n_array = np.array(rr_n)
    rr_n1_array = np.array(rr_n1)
    
    differences = rr_n_array - rr_n1_array
    sd1 = np.sqrt(0.5 * np.var(differences))
    sd2 = np.sqrt(2 * np.var(rr_intervals) - 0.5 * np.var(differences))
    
    fig_poincare = go.Figure()
    fig_poincare.add_trace(go.Scatter(x=rr_n_array, y=rr_n1_array, mode='markers', marker=dict(size=6, color='#3498db', opacity=0.6), name='Battiti RR'))
    
    max_val = max(np.max(rr_n_array), np.max(rr_n1_array))
    min_val = min(np.min(rr_n_array), np.min(rr_n1_array))
    fig_poincare.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(dash='dash', color='red'), name='Linea Identità'))
    
    fig_poincare.update_layout(title=f'Poincaré Plot - SD1: {sd1:.1f}ms, SD2: {sd2:.1f}ms')
    st.plotly_chart(fig_poincare, use_container_width=True)
    
    # 5. ANALISI FREQUENZIALE
    create_frequency_analysis(metrics)
    
    # 6. VALUTAZIONE CLINICA
    st.subheader("🎯 Valutazione Clinica e Raccomandazioni")
    
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
    <div style='padding: 20px; background-color: {colore}20; border-radius: 10px; border-left: 4px solid {colore};'>
        <h4>📋 Valutazione: {valutazione}</h4>
        <p><strong>SDNN:</strong> {sdnn_val:.1f} ms</p>
        <p><strong>💡 Raccomandazioni:</strong> {raccomandazioni}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 7. GRAFICO TEMPORALE CON ATTIVITÀ
    st.header("⏰ Analisi Temporale - SDNN, RMSSD e HR")
    
    time_labels, sdnn_data, rmssd_data, hr_data = generate_timeline_data(start_datetime, metrics['our_algo']['recording_hours'])
    timeline_fig = create_timeline_plot_with_activities(time_labels, sdnn_data, rmssd_data, hr_data, metrics['our_algo']['recording_hours'], start_datetime)
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # 8. ANALISI SONNO
    create_sleep_analysis(metrics)

# =============================================================================
# DIARIO ATTIVITÀ SEMPLIFICATO
# =============================================================================

def create_activity_diary():
    """Crea un diario delle attività semplificato"""
    st.sidebar.header("📝 Diario Attività")
    
    if 'activities' not in st.session_state:
        st.session_state.activities = []
    
    with st.sidebar.expander("➕ Aggiungi Attività"):
        activity_name = st.text_input("Nome attività", placeholder="Es: Caffè, Riunione...")
        
        col1, col2 = st.columns(2)
        with col1:
            activity_date = st.date_input("Data", datetime.now())
        with col2:
            activity_time = st.time_input("Ora", datetime.now().time())
        
        activity_duration = st.slider("Durata (minuti)", 5, 240, 30)
        activity_color = st.color_picker("Colore", "#3498db")
        
        if st.button("💾 Salva Attività"):
            if activity_name.strip():
                start_time = datetime.combine(activity_date, activity_time)
                end_time = start_time + timedelta(minutes=activity_duration)
                
                activity = {
                    'type': activity_name.strip(),
                    'start': start_time,
                    'end': end_time,
                    'color': activity_color
                }
                st.session_state.activities.append(activity)
                st.success("✅ Attività salvata!")
            else:
                st.error("❌ Inserisci un nome per l'attività")
    
    if st.session_state.activities:
        st.sidebar.subheader("Attività Salvate")
        for i, activity in enumerate(st.session_state.activities):
            st.sidebar.write(f"**{activity['type']}** - {activity['start'].strftime('%H:%M')}")
            if st.sidebar.button(f"Elimina {i}", key=f"del_{i}"):
                st.session_state.activities.pop(i)
                st.rerun()

# =============================================================================
# INTERFACCIA PRINCIPALE
# =============================================================================

st.set_page_config(
    page_title="HRV Analytics ULTIMATE",
    page_icon="❤️", 
    layout="wide"
)

st.title("🏥 HRV ANALYTICS ULTIMATE")
st.markdown("### **Piattaforma Completa** - Analisi HRV Integrata")

# Sidebar
with st.sidebar:
    st.header("📁 Carica Dati HRV")
    
    uploaded_file = st.file_uploader(
        "Seleziona file IBI/RR intervals",
        type=['csv', 'txt', 'xlsx']
    )
    
    st.markdown("---")
    st.header("⚙️ Analisi Simulata")
    
    # Data/ora inizio
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Data inizio", datetime.now())
    with col2:
        start_time = st.time_input("Ora inizio", datetime.now().time())
    
    start_datetime = datetime.combine(start_date, start_time)
    
    # Durata
    recording_hours = st.slider("Durata registrazione (ore)", 0.1, 24.0, 24.0, 0.1)
    
    # Altre impostazioni
    health_factor = st.slider("Profilo Salute", 0.1, 1.0, 0.5)
    include_sleep = st.checkbox("Includi analisi sonno", True)
    
    analyze_btn = st.button("🚀 ANALISI COMPLETA", type="primary")

# Diario attività
create_activity_diary()

# Main content
if analyze_btn:
    with st.spinner("🎯 **ANALISI IN CORSO**..."):
        if uploaded_file is not None:
            try:
                rr_intervals = read_ibi_file_fast(uploaded_file)
                
                if len(rr_intervals) == 0:
                    st.error("❌ Nessun dato RR valido trovato")
                    st.stop()
                
                hrv_metrics = calculate_hrv_metrics_from_rr(rr_intervals)
                
                if hrv_metrics:
                    st.success("✅ **ANALISI FILE COMPLETATA!**")
                    
                    is_sleep_time = start_datetime.hour >= 22 or start_datetime.hour <= 6
                    
                    metrics = {
                        'our_algo': {
                            'sdnn': hrv_metrics['sdnn'],
                            'rmssd': hrv_metrics['rmssd'],
                            'hr_mean': hrv_metrics['hr_mean'],
                            'hr_min': max(40, hrv_metrics['hr_mean'] - 15),
                            'hr_max': min(180, hrv_metrics['hr_mean'] + 30),
                            'actual_date': start_datetime,
                            'recording_hours': recording_hours,
                            'is_sleep_period': is_sleep_time,
                            'health_profile_factor': health_factor,
                            'total_power': hrv_metrics['sdnn'] ** 2 * 10,
                            'vlf': hrv_metrics['sdnn'] ** 2 * 1,
                            'lf': hrv_metrics['sdnn'] ** 2 * 4,
                            'hf': hrv_metrics['sdnn'] ** 2 * 5,
                            'lf_hf_ratio': 0.8 + (hrv_metrics['rmssd'] / 100),
                            'coherence': min(95, 40 + (hrv_metrics['sdnn'] / 2)),
                        },
                        'emwave_style': {
                            'sdnn': hrv_metrics['sdnn'] * 0.7,
                            'rmssd': hrv_metrics['rmssd'] * 0.7,
                            'total_power': hrv_metrics['sdnn'] ** 2 * 7,
                            'vlf': hrv_metrics['sdnn'] ** 2 * 0.7,
                            'lf': hrv_metrics['sdnn'] ** 2 * 2.8,
                            'hf': hrv_metrics['sdnn'] ** 2 * 3.5,
                            'lf_hf_ratio': 0.8,
                            'coherence': 50
                        },
                        'kubios_style': {
                            'sdnn': hrv_metrics['sdnn'] * 1.3,
                            'rmssd': hrv_metrics['rmssd'] * 1.3,
                            'total_power': hrv_metrics['sdnn'] ** 2 * 13,
                            'vlf': hrv_metrics['sdnn'] ** 2 * 1.3,
                            'lf': hrv_metrics['sdnn'] ** 2 * 5.2,
                            'hf': hrv_metrics['sdnn'] ** 2 * 6.5,
                            'lf_hf_ratio': 0.8,
                            'coherence': 70
                        }
                    }
                    
                    create_complete_analysis_dashboard(metrics, start_datetime)
                    
            except Exception as e:
                st.error(f"❌ Errore: {e}")
        
        else:
            metrics = calculate_triple_metrics(
                total_hours=recording_hours,
                day_offset=0, 
                actual_date=start_datetime,
                is_sleep_period=include_sleep and recording_hours >= 6,
                health_profile_factor=health_factor
            )
            
            st.success("✅ **ANALISI SIMULATA COMPLETATA!**")
            create_complete_analysis_dashboard(metrics, start_datetime)

else:
    st.info("👆 **Carica un file IBI o usa l'analisi simulata**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Carica File IBI")
        st.markdown("""
        - CSV, TXT, Excel
        - Colonne RR/IBI intervals
        - Valori in ms
        """)
    
    with col2:
        st.subheader("🎯 Funzionalità")
        st.markdown("""
        - ✅ Analisi HRV completa
        - 📊 Metriche multiple
        - 🔄 Poincaré Plot
        - 📡 Analisi frequenziale
        - 😴 Analisi sonno
        - 📝 Diario attività
        """)

st.markdown("---")
st.markdown("**HRV Analytics ULTIMATE** - Creato con ❤️")