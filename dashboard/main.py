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

    
    # Deuxième page : Analyse des commentaires
    with tab2:

        # Titre de la page
        st.title("Analyse des commentaires")

        col1, col2  = st.columns(2)
       
       # Première colonne
        with col1:
            
            # GRAPHIQUE NOMBRE DE COMMENTAIRE SELON LES RESSENTIS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Nombre de commentaires selon les ressentis</p>', unsafe_allow_html=True)

                # Création d'une colonne commentaire avec pour valeur 1
                df=df.assign(Nombre_commentaire=1)
                df['Creation date'] = pd.to_datetime(df['Creation date'])
                
                # Modification du jour et de l'heure pour faire un trie par mois
                def insert_time(row):
                    return row['Creation date'].replace(day=1, hour=0, minute=0, second=0)

                df['Creation date'] = df.apply(lambda r: insert_time(r), axis=1)


                # Création de 5 dataframe selon le sentiment (où le comp_score=sentiment_voulu)
                df_tres_negatif=df[df['comp_score']=='tres negatif']
                df_negatif=df[df['comp_score']=='negatif']
                df_neutre=df[df['comp_score']=='neutre']
                df_positif=df[df['comp_score']=='positif']
                df_tres_positif=df[df['comp_score']=='tres positif']
                
                # Groupement de dataframe par date
                somme_tres_negatif = df_tres_negatif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_negatif = df_negatif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_neutre = df_neutre.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_positif = df_positif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_tres_positif = df_tres_positif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()

                figure_courbes=plt.figure(figsize =[10,10])

                # Titre des courbes
                labels=['Très négatif', 'Négatif', 'Neutre', 'Positif', 'Très positif']

                # Affichage des courbes
                plt.plot(somme_tres_negatif.index, somme_tres_negatif['Nombre_commentaire'], color= colors[3])
                plt.plot(somme_negatif.index, somme_negatif['Nombre_commentaire'], color= colors[2])
                plt.plot(somme_neutre.index, somme_neutre['Nombre_commentaire'], color= colors[1])
                plt.plot(somme_positif.index, somme_positif['Nombre_commentaire'], color= colors[0])
                plt.plot(somme_tres_positif.index, somme_tres_positif['Nombre_commentaire'], color= colors[4])

                plt.legend(bbox_to_anchor=(-0.25, 0.5), loc='center left', labels=labels, frameon=False)
                plt.show()

                st.pyplot(figure_courbes)
                # Fin graphique du nombre de commentaire selon les ressentis



            # GRAPHIQUES TOPICS QUI RESSORTENT LE PLUS POUR LES SENTIMENTS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Sujets les plus récurents dans les commentaires très positifs</p>', unsafe_allow_html=True)

                # Récupération des valeurs
                topics = df_t_positif['topic label'].value_counts()
                labels = ["Choix", "Personnel", "Félicitation"]

                # Affichage du graphique
                t_positif = plt.figure(figsize = (5, 5))
                plt.pie(topics, labels=None, colors=colors, startangle=90)
                plt.legend(bbox_to_anchor=(-0.5, 0.5), loc='center left', labels=labels, frameon=False)
                st.pyplot(t_positif)



        # Deuxième colonne
        with col2:

            # GRAPHIQUES TOPICS QUI RESSORTENT LE PLUS POUR LES SENTIMENTS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Sujets les plus récurents dans les commentaires</p>', unsafe_allow_html=True)

                # Récupération des valeurs
                topics_n = df['topic label'].value_counts()
                labels = df['topic label'].unique()

                # Affichage du graphique
                topic = plt.figure(figsize = (5, 5))
                plt.pie(topics_n, labels=None, colors=colors, startangle=90)
                plt.legend(bbox_to_anchor=(-0.5, 0.5), loc='center left', labels=labels, frameon=False)
                st.pyplot(topic)



            # GRAPHIQUES TOPICS QUI RESSORTENT LE PLUS POUR LES SENTIMENTS TRÈS NÉGATIFS
            
            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Sujets les plus récurents dans les commentaires très négatif</p>', unsafe_allow_html=True)

                # Récupération des valeurs
                topics_tn = df_t_negatif['topic label'].value_counts()
                labels = df_t_negatif['topic label'].unique()

                # Affichage du graphique
                t_negatif = plt.figure(figsize = (5, 5))
                plt.pie(topics_tn, labels=None, colors=colors, startangle=90)
                plt.legend(bbox_to_anchor=(-0.5, 0.5), loc='center left', labels=labels, frameon=False)
                st.pyplot(t_negatif)



    # Troisième page : Comparaison des magasins
    with tab3:

        # Titre de la page
        st.title("Comparaison des magasins")

        col1, col2  = st.columns(2)
       
       # Première colonne
        with col1:

            # GRAPHIQUE TOP 5 DES MAGASINS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Top 5 des magasins les mieux classés</p>', unsafe_allow_html=True)

                # Création d'un dataframe groupé par city faisant la moyenne
                df_city=df.groupby(by=['City']).mean().groupby(level=[0]).cumsum()

                # Trie dans l'ordre dessendant et récupération des 5 premières colonnes
                top5 = df_city.sort_values('compound', ascending = False).head(5)

                # Création du graphique
                figure_top5 = plt.figure(figsize = (6.5, 5))
                plt.bar(top5.index,top5['compound'],color = colors)
                plt.show()

                # Affichage dy graphique
                st.pyplot(figure_top5) 



        # Deuxième colonne
        with col2:

            # GRAPHIQUE TOP 5 DES MAGASINS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Top 5 des magasins les moins bien classés</p>', unsafe_allow_html=True)

                # Création d'un dataframe groupé par city faisant la moyenne
                df_city=df.groupby(by=['City']).mean().groupby(level=[0]).cumsum()

                # Trie dans l'ordre assendant et récupération des 5 premières colonnes
                bottom5 = df_city.sort_values('compound', ascending = True).head(5)

                # Création du graphique
                figure_bottom5 = plt.figure(figsize = (6.5, 5))
                plt.bar(bottom5.index,bottom5['compound'],color = colors)
                plt.show()
                
                # Affichage dy graphique
                st.pyplot(figure_bottom5) 
           



# Partie tableau de bord magasin 

else :
    tab1, tab2 = st.tabs(['Tableau de bord','Analyse des commentaires'])

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
                nombres_notes = df_magasin['Rating'].value_counts()

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
                df_time = df_magasin[df_magasin['Response date'].notna()]

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
                nbr2 = df_magasin[['Platform','Group Id']].groupby('Platform').count().sort_values(by='Group Id', ascending=False)

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
                nbr = df_magasin[['comp_score','Group Id']].groupby('comp_score').count().sort_values(by='Group Id', ascending=False)

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



    # Deuxième page : Analyse des commentaires
    with tab2:

        # Titre de la page
        st.title("Analyse des commentaires")

        col1, col2  = st.columns(2)
       
       # Première colonne
        with col1:

            # GRAPHIQUE NOMBRE DE COMMENTAIRE SELON LES RESSENTIS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Nombre de commentaires selon les ressentis</p>', unsafe_allow_html=True)

                # Création d'une colonne commentaire avec pour valeur 1
                df_magasin=df_magasin.assign(Nombre_commentaire=1)
                df_magasin['Creation date'] = pd.to_datetime(df['Creation date'])
                
                # Modification du jour et de l'heure pour faire un trie par mois
                def insert_time(row):
                    return row['Creation date'].replace(day=1, hour=0, minute=0, second=0)

                df_magasin['Creation date'] = df_magasin.apply(lambda r: insert_time(r), axis=1)


                # Création de 5 dataframe selon le sentiment (où le comp_score=sentiment_voulu)
                df_tres_negatif=df_magasin[df_magasin['comp_score']=='tres negatif']
                df_negatif=df_magasin[df_magasin['comp_score']=='negatif']
                df_neutre=df_magasin[df_magasin['comp_score']=='neutre']
                df_positif=df_magasin[df_magasin['comp_score']=='positif']
                df_tres_positif=df_magasin[df_magasin['comp_score']=='tres positif']
                
                # Groupement de dataframe par date
                somme_tres_negatif = df_tres_negatif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_negatif = df_negatif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_neutre = df_neutre.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_positif = df_positif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()
                somme_tres_positif = df_tres_positif.groupby(by=['Creation date']).sum().groupby(level=[0]).cumsum()

                figure_courbes=plt.figure(figsize =[10,10])

                # Titre des courbes
                labels=['Très négatif', 'Négatif', 'Neutre', 'Positif', 'Très positif']

                # Affichage des courbes
                plt.plot(somme_tres_negatif.index, somme_tres_negatif['Nombre_commentaire'], color= colors[3])
                plt.plot(somme_negatif.index, somme_negatif['Nombre_commentaire'], color= colors[2])
                plt.plot(somme_neutre.index, somme_neutre['Nombre_commentaire'], color= colors[1])
                plt.plot(somme_positif.index, somme_positif['Nombre_commentaire'], color= colors[0])
                plt.plot(somme_tres_positif.index, somme_tres_positif['Nombre_commentaire'], color= colors[4])

                plt.legend(bbox_to_anchor=(-0.25, 0.5), loc='center left', labels=labels, frameon=False)
                plt.show()

                st.pyplot(figure_courbes)
                # Fin graphique du nombre de commentaire selon les ressentis



            # GRAPHIQUES TOPICS QUI RESSORTENT LE PLUS POUR LES SENTIMENTS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Sujets les plus récurents dans les commentaires très positifs</p>', unsafe_allow_html=True)

                # Récupération des valeurs
                topics = df_t_positif_magasin['topic label'].value_counts()

                # Convertion de topics en dataframe
                topics= pd.DataFrame(topics, columns=['topic label'])

                # Création d'un index
                topics.reset_index(0, inplace=True)

                # Remise en ordre des lignes
                order = ['choix', 'personnel', 'félicitation']
                topics = topics.reindex(topics['index'].map(dict(zip(order, range(len(order))))).sort_values().index)
                
                # Renommage des labels
                labels = ["Choix", "Personnel", "Félicitation"]

                # Affichage du graphique
                t_positif = plt.figure(figsize = (5, 5))
                plt.pie(topics['topic label'], labels=None, colors=colors, startangle=90)
                plt.legend(bbox_to_anchor=(-0.5, 0.5), loc='center left', labels=labels, frameon=False)
                st.pyplot(t_positif)



        # Deuxième colonne
        with col2:

            # GRAPHIQUES TOPICS QUI RESSORTENT LE PLUS POUR LES SENTIMENTS

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Sujets les plus récurents dans les commentaires</p>', unsafe_allow_html=True)

                # Récupération des valeurs
                topics = df_magasin['topic label'].value_counts()

                # Convertion de topics en dataframe
                topics= pd.DataFrame(topics, columns=['topic label'])

                # Création d'un index
                topics.reset_index(0, inplace=True)

                # Remise en ordre des lignes
                order = ['Vendeurs', 'Produits', 'Magasin']
                topics = topics.reindex(topics['index'].map(dict(zip(order, range(len(order))))).sort_values().index)

                # Affichage du graphique
                topic = plt.figure(figsize = (5, 5))
                plt.pie(topics['topic label'], labels=None, colors=colors, startangle=90)
                plt.legend(bbox_to_anchor=(-0.5, 0.5), loc='center left', labels=order, frameon=False)
                st.pyplot(topic)



            # GRAPHIQUES TOPICS QUI RESSORTENT LE PLUS POUR LES SENTIMENTS TRÈS NÉGATIFS            

            with st.container():

                # Titre du graphique
                st.markdown(f'<p>Sujets les plus récurents dans les commentaires très négatif</p>', unsafe_allow_html=True)

                # Récupération des valeurs
                topics = df_t_negatif_magasin['topic label'].value_counts()

                # Convertion de topics en dataframe
                topics= pd.DataFrame(topics, columns=['topic label'])

                # Création d'un index
                topics.reset_index(0, inplace=True)

                # Remise en ordre des lignes
                order = ['Vendeurs', 'Produits', 'Magasin']
                topics = topics.reindex(topics['index'].map(dict(zip(order, range(len(order))))).sort_values().index)

                # Affichage du graphique
                t_negatif = plt.figure(figsize = (5, 5))
                plt.pie(topics['topic label'], labels=None, colors=colors, startangle=90)
                plt.legend(bbox_to_anchor=(-0.5, 0.5), loc='center left', labels=order, frameon=False)
                st.pyplot(t_negatif)