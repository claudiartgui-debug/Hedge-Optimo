import sys 
import os 
import pandas as pd 
import calendar
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import streamlit as st 
import requests
import io

plt.rcParams.update({
    "axes.titlesize": 14,     
    "axes.titleweight": "bold",
    "axes.labelsize": 10, 
    "grid.linestyle": "--",
    "grid.alpha": 0.3
})

# ==== FUNCTIONS ===== #

@st.cache_data
def tables_cmg_margin(name: str , year:int, list_scenario : list) -> pd.DataFrame:
    df= pd.read_excel(name)
    df = df[(df['year'] == year) & (df['sample'] != 'Mean') & df['scenario'].isin(list_scenario)].astype({'sample': int})
    return df

@st.cache_data
def stats_capacity( df: pd.DataFrame, price: float) -> pd.DataFrame:
    cost = 1.5
    capacidad = list(range(1,400,1))
    all_stats = []
    df1 = df.copy()
    df1['days_in_month'] = df1.apply(lambda row: calendar.monthrange(row['year'], row['month'])[1], axis=1)
    for c in capacidad:
        delta = (c * df1['days_in_month'] * 24) * (price - cost - df1['value_cmg'])
        df_temp = df1.copy()
        df_temp[f'EEP'] = df_temp['value_musd_EEP'] + delta / 1e6
        df_temp[f'EEP_withoutchilca2'] = df_temp['value_musd_withoutchilca2'] + delta / 1e6
        
        df_grouped = df_temp.groupby(['simulation', 'scenario', 'sample', 'year'], as_index=False).agg(
            {'EEP':'sum', f'EEP_withoutchilca2':'sum'})
        
        stats = metric_statistics(df_grouped, 4) 
        for var, metrics in stats.items():
            metrics['capacity'] = c
            metrics['variable'] = var
            all_stats.append(metrics)
            
    df_stats = pd.DataFrame(all_stats)
    df_stats = df_stats.sort_values(['variable', 'capacity']).reset_index(drop=True)
    return df_stats

def plot_hist(df: pd.DataFrame, dict_labels:dict, year:int ) -> None:
    
    hists = {}
    scatters = {}
    agg_dict = {col: agg_func for col, (_, agg_func) in dict_labels.items()}
    df_grouped = df.groupby(['simulation', 'scenario', 'sample', 'year'], as_index=False).agg(agg_dict)
    
    # histogramas
    for col in dict_labels.keys():
        hists[col] = hist(df_grouped,col, dict_labels, year) 
        
    ## scatters 
    keys = list(dict_labels.keys())
    if len(keys) >= 2:
        colx = keys[0]
        for coly in keys[1:]:
            scatters[f"{colx}_vs_{coly}"] = scatter_plot(df_grouped, colx, coly , year, dict_labels)
    else: print("No hay suficientes columnas para plotear")
    
    return hists , scatters

def hist(df: pd.DataFrame,col:str, dict_labels:dict, year:int ) -> None:    
        data = df[col].dropna().values
        bins = 10
        counts, bin_edges = np.histogram(data, bins=bins)
        fig, ax = plt.subplots(figsize= (10,6)) 
        bars = ax.bar(range(len(counts)), counts, width=0.95, align='center')
        labels = [f"[{int(bin_edges[i])}, {int(bin_edges[i+1])}]" for i in range(len(bin_edges)-1)]
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, count, str(count),
                        ha='center', va='bottom', fontsize=10) 
                
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xlabel(dict_labels[col][0] , fontsize=12)
        ax.set_title(f"{dict_labels[col][0]} - {year}", fontsize=14)
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', alpha=0.2)
        fig.tight_layout()
        return fig

def scatter_plot(df: pd.DataFrame, colx: str, coly: str , year:int , dict_labels) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))  
    ax.scatter(df[colx], df[coly], alpha=0.6, color='teal')
    x = df[colx].dropna().values
    y = df[coly].dropna().values

    if len(x) > 1: 
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color='blue', alpha=0.6, linewidth=1, label='Tendencia')
        ax.text(
            x.min(), y.max(),
            f'y = {m:.2f}x + {b:.2f}', fontsize=12, color='blue', ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    
    ax.set_xlabel(f'{dict_labels[colx][0]}')
    ax.set_ylabel(f'{dict_labels[coly][0]}')
    ax.set_title(f'{dict_labels[coly][0]} vs {dict_labels[colx][0]} - {year}')
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', alpha=0.2)
    fig.tight_layout()
    return fig   

def metric_statistics (df: pd.DataFrame, num_col: int) -> dict:
    stats = {}
    cols = df.columns[num_col:]
    for col in cols:
        serie = df[col].dropna().values
        std_val = np.std(serie)
        p5 = np.percentile(serie, 95)
        p10 = np.percentile(serie, 10)
        p50 = np.percentile(serie, 50)
        p90 = np.percentile(serie, 90)
        p95 = np.percentile(serie, 5)
        min_val = np.min(serie)
        max_val = np.max(serie)
        downside = np.mean(serie[serie < p10])
        upside = np.mean(serie[serie > p90])
        spread = upside - downside
        at_risk = p50 - downside
        stats[col] = {
            'Mean': p50,
            'SD': std_val,
            'P5': p5,
            'P95': p95,
            'Min': min_val,
            'Max': max_val,
            #'P95-Min': p95 - min_val,
            #'P5-P95': p95 - p5,
            #'Max-P5': max_val - p5,
            'Downside': downside,
            'Upside': upside,
            'Spread': spread,
            'At Risk@P90': at_risk}  
    return stats

def stats_graphic(df_stats: pd.DataFrame, stat: str, year: int, tittle:str) -> plt.Figure: 
    
    fig, ax = plt.subplots(figsize=(8,5))  
    x = df_stats["capacity"].values
    y = df_stats[stat].values

    if len(x) > 1: 
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = make_interp_spline(x, y)(x_smooth)
        ax.plot(x_smooth, y_smooth, color='teal', linewidth=2)
    else:
        ax.plot(x, y, 'o-', color='teal')

    ax.set_title(f"{tittle} - {stat} - {year}", fontweight='bold')
    ax.set_xlabel("Capacity (MW)")
    ax.set_ylabel(f"S-P {stat} (MUSD)")
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.minorticks_on()
    ax.grid(which='minor', linestyle=':', alpha=0.2)

    if len(y) > 0:
        ax.set_ylim(0, max(y)*1.1) 

    ax.legend([stat], loc="lower right")
    fig.tight_layout()
    return fig









