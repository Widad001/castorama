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



# Partie tableau de bord Générale 
if choix_magasin == 'Général':
    tab1, tab2, tab3 = st.tabs(['Tableau de bord','Analyse des commentaires','Comparaison des magasins'])
    
    # Première page : Tableau de bord général
    with tab1:

        # Titre de la page
        st.title("Tableau de bord général")

        col1, col2  = st.columns(2)

        # Première colonne
        with col1:
            
            # GRAPHIQUE NOTES DES MAGASINS
            
            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Notes des magasins</p>', unsafe_allow_html=True)

                # Comptage du nombre de note selon la note (ex : nombre de 5 étoiles)
                nombres_notes = df['Rating'].value_counts()

                # Remise dans l'ordre des notes (de 5 à 1)
                nombres_notes = nombres_notes.sort_index(axis = 0, ascending = False)

                # Attribution des labels
                labels = ['5 étoiles', '4 étoiles', '3 étoiles', '2 étoiles', '1 étoile']
                
                # Création du diagramme circulaire
                ax = plt.figure(figsize = (5, 5))
                plt.pie(nombres_notes, labels=None, colors=colors, startangle=90)
                plt.legend(bbox_to_anchor=(-0.5, 0.5), loc='center left', labels=labels, frameon=False)

                st.pyplot(ax) 

            
            
            # GRAPHIQUE TEMPS DE RÉPONSE

            with st.container():

                # Suppression des lignes ou il n'y a pas de réponse
                df_time = df[df['Response date'].notna()]

                # Conversion des colonnes 'Creation date' et 'Response date' en colonne date
                df_time['Creation date'] = pd.to_datetime(df_time['Creation date'])
                df_time['Response date'] = pd.to_datetime(df_time['Response date'])
                
                # Calcule de la différence entre la colonne'Creation date' et 'Response date'
                delta=[]
                for i in df_time.index:
                    time1 = df_time['Creation date'][i]
                    time2 = df_time['Response date'][i]
                    if time2 < time1: # Pour éviter d'avoir des temps de réponse négatif, car cela n'a pas de sens
                        result=''
                    else:
                        result = time2-time1
                    delta.append(result)

                # Ajout d'une colonne Time_delta qui contient les résultats des différences
                df_time=df_time.assign(Time_delta=delta)

                # Calcule du temps de réponse moyen
                df_time['Time_delta'] = df_time['Time_delta'].dt.total_seconds()
                temps_reponse=df_time['Time_delta'].mean()
                temps_reponse = pd.to_timedelta(temps_reponse, unit='s')
                temps_reponse=str(temps_reponse)

                # Conversion du mot days en jours
                temps_reponse = temps_reponse.replace("days", "jours")

                # Récupération de la position avant le mot jours (pos1) et après (pos2)
                pos1 = temps_reponse.find('jours')
                pos2 = pos1 + len('jours')
                
                # Création de 3 variables avec le temps moyen pour la mise en page
                nombre_jours=temps_reponse[0:pos1]
                jours=temps_reponse[pos1:pos2]
                nombre_heures=temps_reponse[pos2:len(temps_reponse)]

                # Affichage des résultats
                st.markdown(f'<p>Temps de réponse moyen</p>', unsafe_allow_html=True)
                st.markdown(f'<h2>{nombre_jours}</h2>', unsafe_allow_html=True)
                st.markdown(f'<p>{jours}</p>', unsafe_allow_html=True)
                st.markdown(f'<p>{nombre_heures}</p>', unsafe_allow_html=True)



        # Deuxième colonne
        with col2:
            
            # GRAPHIQUE PROVENANCE DES COMMENTAIRES

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Provenance des commentaires</p>', unsafe_allow_html=True)

                # Création d'un dataframe groupé par platforme comprennant seulement la plateform et le nombre de group id
                nbr2 = df[['Platform','Group Id']].groupby('Platform').count().sort_values(by='Group Id', ascending=False)

                # Remise à 0 et 1 des index au lieu de Google my buisness et Facebook
                nbr2.reset_index(0, inplace=True)

                # Renommage de la colonne Group Id
                nbr2.rename(columns={'Group Id':'Nombre de commentaires'}, inplace=True)

                # Renommage des labels
                labels=['Google My Business', 'Facebook']

                # Création du graphique
                figure2 = plt.figure(figsize=(10,6))
                plt.bar(labels,nbr2['Nombre de commentaires'],color = colors)
                plt.tight_layout()

                # Affichage du graphique
                st.pyplot(figure2) 



            # GRAPHIQUE NOMBRE DE COMMENTAIRES PAR SENTIMENTS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Nombre de commentaires par sentiments</p>', unsafe_allow_html=True)

                # Création d'un dataframe groupé par comp_score comprennant seulement le comp_score (sentiment) et le nombre de group id
                nbr = df[['comp_score','Group Id']].groupby('comp_score').count().sort_values(by='Group Id', ascending=False)

                # Remise à 0 et 1 des index au lieu des sentiments
                nbr.reset_index(0, inplace=True)

                # Renommage de la colonne Group Id
                nbr.rename(columns={'Group Id':'Nombre de commentaires'}, inplace=True)

                # Remise en ordre des lignes
                order = ['tres positif', 'positif', 'neutre', 'negatif', 'tres negatif']
                nbr = nbr.reindex(nbr['comp_score'].map(dict(zip(order, range(len(order))))).sort_values().index)

                # Remise dans l'ordre des index
                nbr.reset_index(0, inplace=True)

                # Suppression de la nouvelle colonne index
                nbr = nbr.drop(['index'], axis=1)

                # Renommage des labels
                labels=['Très positif', 'Positif', 'Neutre', 'Négatif', 'Très négatif']

                # Remise dans l'ordre des couleurs
                colors_graph = [colors[4], colors[0], colors[1], colors[2], colors[3]]

                # Création du graphique
                figure = plt.figure(figsize=(6.5,5))
                plt.bar(labels,nbr['Nombre de commentaires'],color=colors_graph)
                plt.tight_layout()

                # Affichage du graphique
                st.pyplot(figure) 

    