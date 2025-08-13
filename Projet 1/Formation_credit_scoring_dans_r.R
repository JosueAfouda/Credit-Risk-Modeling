# # <center> **MACHINE LEARNING POUR LA MODELISATION DU RISQUE DE CREDIT DANS R (CREDIT SCORING)**# Par [Josué AFOUDA](https://afouda-datascience.com/)# [Suivre le cours](https://afouda-datascience.com/cours/machine-learning-pour-la-modelisation-du-risque-de-credit-credit-scoring-dans-r/)# # <font color=green> Compréhension de la problématique business# ![jinyun-xBuu23uxarU-unsplash.jpg](attachment:jinyun-xBuu23uxarU-unsplash.jpg)# Lorsqu'une banque prête de l'argent à une personne, elle prend le risque que cette dernière ne rembourse pas cet argent dans le délai convenu. Ce risque est appelé **Risque de Crédit**. Alors avant d'octroyer un crédit, les banques vérifient si le client (ou la cliente) qui demandent un prêt sera capable ou pas de le rembourser. Cette vérification se fait grâce à l'analyse de plusieurs paramètres tels que les revenus, les biens, les dépenses actuelles du client, etc. Cette analyse est encore effectuée manuellement par plusieurs banques. Ainsi, elle est très consommatrice en temps et en ressources financières. 
# 
# Grâce au **Machine Learning**, il est possible d'automatiser cette tâche et de pouvoir prédire avec plus de précision les clients qui seront en défaut de paiement. 
# 
# Dans ce projet, nous allons construire un algorithme capable de prédire si une personne sera en défaut de paiement ou pas (1 : défaut, 0 : non-défaut). Il s'agit donc d'un problème de classification car nous voulons prédire une variable discrète (binaire pour être précis).
# # <font color=green> Importation des librairies et des données# Importation des librairieslibrary(tidyverse)library(ggthemes)library(ROSE)library(pROC)# Paramètres du rendu des graphiquesoptions(repr.plot.width=6, repr.plot.height=4)theme_set(theme_minimal())# En ce qui concerne les données, il s'agit des informations collectées sur d'anciens clients ayant contracté des prêts qui sont utilisées pour prédire le comportement des nouveaux clients. 
# 
# Deux (02) types de données peuvent être utilisés pour modéliser la probabilité de défaut de
# paiement :
# 
# • **Données liées à la demande de crédit** ;
# 
# • **Données comportementales décrivant le bénéficiaire du prêt**.
# 
# Dans la pratique, les banques utilisent un mélange de ces deux types de données pour construire
# leur **modèle de scoring** appliqué à la gestion du risque de crédit.
# 
# Importation des donnéesurl = 'https://github.com/JosueAfouda/Credit-Risk-Modeling/raw/master/data_credit.txt'df <- read.csv(url)head(df)# Structure de la dataframestr(df)# Les données proviennent de [Kaggle](https://www.kaggle.com/laotse/credit-risk-dataset) qui la plus célèbre plateforme de compétitions en Data Science.
# 
# L'ensemble des données compte 12 variables et 32581 observations (lignes) historiques. Chaque observation correspond à une personne ayant contracté un prêt. On a des variables qui décrivent le prêt (montant, statut, taux d'intérêt, etc.) et d'autres variables qui décrivent la personne ayant ontracté ce prêt (age, revenu, etc.). Nous allons donc utiliser ces données historiques afin de construire le modèle de *scoring* qui va prédire le statut des nouveaux candidats à un crédit.               
# Il est très important de comprendre les variables de notre jeu de données :    
# * ***person_age*** : variable indiquant l'âge de la personne ;           
# * ***person_income*** : variable indiquant le revenu annuel (ou encore le salaire) de la personne ;             
# * ***person_home_ownership*** : variable indiquant le statut de la personne par rapport à son lieu d'habitation (popriétaire, locataire, etc.) ;        
# * ***person_emp_length*** : variable indiquant la durée (en mois) depuis laquelle la personne est en activité professionnelle ;           
# * ***loan_intent*** : variable indiquant le motif du crédit ;          
# * ***loan_grade*** : Notation de la solvabilité du client. classes de A à G avec A indiquant la classe de solvabilité la plus élevée et G la plus basse ;           
# * ***loan_amnt*** : variable indiquant le montant du prêt ;                 
# * ***loan_int_rate*** : variable indiquant le taux d'intérêt du crédit ;       
# * ***loan_status*** : c'est la variable d'intérêt. Elle indique si la personne est en défaut de paiement (1) ou pas (0) ;       
# * ***loan_percent_income*** : variable indiquant le pourcentage du crédit par rapport au revenu (ratio dette / revenu) ;          
# * ***cb_person_default_on_file*** : variable indiquant si la personne a été en défaut de paiement ou pas dans le passé                  
# * ***cb_person_cred_hist_length*** : variable indiquant la durée des antécédents de crédits.     
# 
# Passons à présent à l'analyse exploratoire des données qui nous permettra de mieux les comprendre.# # <font color=green> Analyse exploratoire des données# Résumé statistiquesummary(df)# * Nous remarquons que les moyennes et les écart-types sont très différents d'une variable à une autre. Cela indique que les données ne sont pas à la même échelle. Il faudra probablement normaliser les données avant de les modéliser. en effet, certains algoritmes de Machine Learning nécessitent une normalisation des données pour un meilleur résultat.     
# 
# * De plus, il y a des valeurs manquantes au niveau des variables *person_emp_length* et *loan_int_rate*. Ces deux variables contiennent probablement des valeurs aberantes vu la différence très importante entre leur 3è quartile et leur maximum. Les valeurs aberantes sont très distantes des autres valeurs et ne sont donc pas représentatives de l'ensemble des données. Elles peuvent causer d'importatantes erreurs de modélisation.
# 
# Dans la partie consacrée au nettoyage des données, nous aurons principalement à traiter les valeurs aberrantes (*outliers*) et les valeurs manquantes.# Passons maintenat à l'analyse de la variable cible (*loan_status*).# Transformation de la variable cible en variable catégorielledf$loan_status <- as.factor(df$loan_status)# Table de fréquence de la variable cible ('loan_status')print(prop.table(table(df$loan_status)))# Diagramme à barre de la variable 'loan_status'plot(df$loan_status, main = 