import sys 
import os 
import streamlit as st
import pandas as pd
import plotly.express as px
from hedge_optimo_excel import tables_cmg_margin, stats_capacity, plot_hist, scatter_plot, stats_graphic, metric_statistics
import matplotlib.pyplot as plt
import numpy as np
import requests
import io

# ==== CONFIGURACIÓN ===== # 

st.set_page_config(layout="wide")

# inputs
name = "hedge_optimo_streamlit_JUL.xlsx"
fecha = "JUL-25"

ppa_prices = list(np.arange(38, 42.5, 0.25).round(2))
ppa_price  = st.sidebar.selectbox("PPA price ($/MWh)", ppa_prices, index=0)

years = [2025, 2026, 2027, 2028, 2029, 2030]
year = st.sidebar.selectbox("Year", years, index=0)

scenarios = [ 'ALL', 'Normal' ,'Without Chilca1']
scenario = st.sidebar.selectbox("Scenario", scenarios, index=0)

dict_sens  = {
    'Normal': ['base', 'high', 'low'],
    'Without Chilca1': ['basewithoutchilca1', 'highwithoutchilca1', 'lowwithoutchilca1'],
    'ALL': ['basewithoutchilca1', 'highwithoutchilca1', 'lowwithoutchilca1','base', 'high', 'low' ]
    }

list_scenario = dict_sens[scenario]

df_inputs = tables_cmg_margin(name , year, list_scenario) 

# ========== Graphics ========== #
dict_labels ={
    'value_cmg' : ["Spot Price ($/MWh)" , 'mean'],
    'value_musd_EEP': ["Margin EEP (MUSD)", 'sum'],
    'value_musd_withoutchilca2': ["Margin EEP without Chilca2 (MUSD)", 'sum']
}

st.title(f"Hedge Optimo - {fecha}")
st.markdown(f"#### Year: {year}, Scenarios: {scenario}, PPA price ($/MWh): {ppa_price}")
df_inputs[ list(dict_labels.keys())] = df_inputs[ list(dict_labels.keys())] .round(2)

st.header("Histograms")
col1, col2, col3 = st.columns(3)

hists , scatters = plot_hist(df_inputs, dict_labels, year)
hists_keys = list(hists .keys())
with col1: st.pyplot(hists [hists_keys[0]])
with col2: st.pyplot(hists [hists_keys[1]])
with col3: st.pyplot(hists [hists_keys[2]])

st.header("Summary Inputs")
st.subheader("Data")
df_inputs[df_inputs.columns[5:]] = df_inputs[df_inputs.columns[5:]].round(1)
st.dataframe(df_inputs)

st.subheader("Statistics")
stats = metric_statistics (df_inputs, 5)
df_stats = pd.DataFrame(stats).T.reset_index().rename(columns={'index': 'Metric'})
df_stats[df_stats.columns[1:]] = df_stats[df_stats.columns[1:]].round(1)
st.dataframe(df_stats)


st.header("Scatters")
col4, col5 = st.columns(2)
scatters_keys = list(scatters.keys())
with col4: st.pyplot(scatters[scatters_keys [0]])
with col5: st.pyplot(scatters[scatters_keys [1]])

st.header("EEP + PPA Deal per capacity in MW")
df_stats_capacity = stats_capacity(df_inputs, ppa_price)
cols = [ 'capacity','At Risk@P90','Mean', 'SD', 'P5', 'P95', 'Min', 'Max', 'Downside', 'Upside', 'Spread']
df_stats_capacity[cols[1:]] = df_stats_capacity[cols[1:]].round(1)

col8, col9 = st.columns(2)
with col8:
    tittle = 'EEP'
    fig1 = stats_graphic(df_stats_capacity[df_stats_capacity['variable']== tittle][['capacity','At Risk@P90']],"At Risk@P90", year, tittle ) 
    st.pyplot(fig1)
with col9:
    tittle = 'EEP_withoutchilca2'
    fig2 = stats_graphic(df_stats_capacity[df_stats_capacity['variable']== tittle][['capacity','At Risk@P90']],"At Risk@P90", year, tittle) 
    st.pyplot(fig2)

st.header("Statistics")
col6, col7 = st.columns(2)
with col6:
    st.subheader("EEP ALL assets")
    st.dataframe(df_stats_capacity[df_stats_capacity['variable']=='EEP'][cols])
with col7:
    st.subheader("EEP without Chilca2")
    st.dataframe(df_stats_capacity[df_stats_capacity['variable']=='EEP_withoutchilca2'][cols])

# ========== To run in local ========== #
#cd "C:\Users\ZJ6638\OneDrive - ENGIE\MC\Streamlit\JUL_Hedge_Optimo-main"
#python -m streamlit run main_hedge_excel.py
# 38 - 42.5  : steps = 0.25











