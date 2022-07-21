## Libraries
import streamlit as st
# Lib system
import sys
import os
from os.path import join, dirname
from IPython.display import Image
import pandas as pd
# Lib data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


st.set_page_config(
    page_title="Dashboard Castorama",
    #page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Liaison entre le fichier main.py et style.css
lien_css='./dashboard/style.css'

with open(lien_css) as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# import des données
df = pd.read_csv('./data/data_traite.csv',header=0)
df_t_negatif = pd.read_csv('./data/df_t_negatif.csv',header=0)
df_negatif = pd.read_csv('./data/df_negatif.csv',header=0)
df_neutre = pd.read_csv('./data/df_neutre.csv',header=0)
df_positif = pd.read_csv('./data/df_positif.csv',header=0)
df_t_positif = pd.read_csv('./data/df_t_positif.csv',header=0)

##df = df.drop(['Group Id','Business Id','Timezone','Timezone','Rating','Response date','Response'],axis=1)

# Création de la variable colors avec la liste des codes couleurs utilisés pour les graphiques
# 0 : bleu, 1 : jaune, 2 : orange, 3 : rouge, 4 : violet
colors = ['#0075C4', '#F8DE18', '#FE9A05', '#F04343', '#794BDA']

# Affichage de la sidebar a gauche
st.sidebar.image('./img/logo.jpeg',use_column_width=True)
magasin = df['City'].unique()
magasin = np.insert(magasin,0,'Général')
choix_magasin = st.sidebar.selectbox("Sélection d'un magasin :", options=magasin, on_change=None)

#---- Récupération des dataframe spécifiques aux magasins
df_magasin = df.loc[df['City']== choix_magasin]
df_t_negatif_magasin = df_t_negatif.loc[df['City']== choix_magasin]
df_negatif_magasin = df_negatif.loc[df['City']== choix_magasin]
df_neutre_magasin = df_neutre.loc[df['City']== choix_magasin]
df_positif_magasin = df_positif.loc[df['City']== choix_magasin]
df_positif_magasin = df_positif.loc[df['City']== choix_magasin]
df_t_positif_magasin = df_t_positif.loc[df['City']== choix_magasin]
