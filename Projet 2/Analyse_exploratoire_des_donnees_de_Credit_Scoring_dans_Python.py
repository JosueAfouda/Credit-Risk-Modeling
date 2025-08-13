# # <center> **ANALYSE EXPLORATOIRE DES DONNEES DE CREDIT SCORING DANS PYTHON**
# 
# Par [Josué AFOUDA](https://afouda-datascience.com/)

# [Suivre le cours](https://afouda-datascience.com/cours/analyse-exploratoire-des-donnees-de-credit-scoring-dans-python/)

# # <font color=green> Compréhension de la problématique business

# ![jinyun-xBuu23uxarU-unsplash.jpg](attachment:jinyun-xBuu23uxarU-unsplash.jpg)

# Lorsqu'une banque prête de l'argent à une personne, elle prend le risque que cette dernière ne rembourse pas cet argent dans le délai convenu. Ce risque est appelé **Risque de Crédit**. Alors avant d'octroyer un crédit, les banques vérifient si le client (ou la cliente) qui demandent un prêt sera capable ou pas de le rembourser. Cette vérification se fait grâce à l'analyse de plusieurs paramètres tels que les revenus, les biens, les dépenses actuelles du client, etc. Cette analyse est encore effectuée manuellement par plusieurs banques. Ainsi, elle est très consommatrice en temps et en ressources financières. 
# 
# Grâce au **Machine Learning**, il est possible d'automatiser cette tâche et de pouvoir prédire avec plus de précision les clients qui seront en défaut de paiement. 
# 
# Dans ce projet, nous allons construire un algorithme capable de prédire si une personne sera en défaut de paiement ou pas (1 : défaut, 0 : non-défaut). Il s'agit donc d'un problème de classification car nous voulons prédire une variable discrète (binaire pour être précis).

# # <font color=green> Importation des librairies et des données

# %%
# Importation des librairies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Importation des données

df = pd.read_csv('https://github.com/JosueAfouda/Credit-Risk-Modeling/raw/master/credit_risk_dataset.csv')

df.head()

# %%
df.info()

# %%
# Les données proviennent de [Kaggle](https://www.kaggle.com/laotse/credit-risk-dataset) qui la plus célèbre plateforme de compétitions en Data Science.
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
# Passons à présent à l'analyse exploratoire des données qui nous permettra de mieux les comprendre.

# # <font color=green> Analyse exploratoire des données

# %%
df.describe()

# %%
# Les moyennes et les écart-types sont très différents d'une variable à une autre. Cela indique que les données ne sont pas à la même échelle. Il faudra probablement normaliser les données avant de les modéliser. en effet, certains algoritmes de Machine Learning nécessitent une normalisation des données pour un meilleur résultat.     
# 
# De plus, il y a des valeurs manquantes au niveau des variables *person_emp_length* et *loan_int_rate*. Ces deux variables contiennent probablement des valeurs aberantes vu la différence très importante entre leur 3è quartile et leur maximum. Les valeurs aberantes sont très distantes des autres valeurs et ne sont donc pas représentatives de l'ensemble des données. Elles peuvent causer d'importatantes erreurs de modélisation.
# 
# Dans la partie consacrée au nettoyage des données, nous aurons principalement à traiter les valeurs aberrantes (*outliers*) et les valeurs manquantes.

# %%
# Analyse de la variable cible ('loan_status')

df['loan_status'].value_counts()

# %%
df['loan_status'].value_counts(normalize = True)

# %%
# Les résultats montrent qu'il y un déséquilibre de classe très importants dans les données. En effet, seulement environ 22% de clients sont en défaut de paiement contre un peu plus de 78% de bons clients. 
# 
# Le déséquilibre de classe est souvent observé dans les données de crédit. la majorité des demandeurs de crédit sont incités à ne pas être en défaut de paiement car plus ils remboursent le crédit dans les délais, plus leurs côtes de crédit augmentent et donc ils peuvent à nouveau emprunter pour effectuer d'autres investissements.
# 
# Si le déséquilibre observé ici est tout à fait normal, il n'en demeure moins que cela représente un grand défi de classification pour les algorithmes de Machine Learning.
# 
# Dans la partie préparation des données pour la modélisation, il va falloir résoudre ce problème.

# %%
# Visualisation de la variable cible

sns.countplot(x = 'loan_status', data = df)
plt.title('Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'person_age'

sns.boxplot(x = 'loan_status', y = 'person_age', data = df)
plt.title('Age vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'person_income'

sns.boxplot(x = 'loan_status', y = 'person_income', data = df)
plt.title('Revenu annuel vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'person_home_ownership'

sns.countplot(x = 'person_home_ownership', hue = 'loan_status', data = df)
plt.title('Statut de propriété du logement vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'person_emp_length'

sns.boxplot(x = 'loan_status', y = 'person_emp_length', data = df)
plt.title('Durée d\'emploi vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'loan_intent'

sns.countplot(x = 'loan_intent', hue = 'loan_status', data = df)
plt.title('Motif du prêt vs Statut de crédit')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# Analyse de la variable 'loan_grade'

sns.countplot(x = 'loan_grade', hue = 'loan_status', data = df, order = sorted(df['loan_grade'].unique()))
plt.title('Grade du prêt vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'loan_amnt'

sns.boxplot(x = 'loan_status', y = 'loan_amnt', data = df)
plt.title('Montant du prêt vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'loan_int_rate'

sns.boxplot(x = 'loan_status', y = 'loan_int_rate', data = df)
plt.title('Taux d\'intérêt vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'loan_percent_income'

sns.boxplot(x = 'loan_status', y = 'loan_percent_income', data = df)
plt.title('Pourcentage du revenu vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'cb_person_default_on_file'

sns.countplot(x = 'cb_person_default_on_file', hue = 'loan_status', data = df)
plt.title('Défaut antérieur vs Statut de crédit')
plt.show()

# %%
# Analyse de la variable 'cb_person_cred_hist_length'

sns.boxplot(x = 'loan_status', y = 'cb_person_cred_hist_length', data = df)
plt.title('Durée de l\'historique de crédit vs Statut de crédit')
plt.show()

# %%
# # <font color=green> Nettoyage des données

# %%
# Création d'une copie de la dataframe df
df_clean = df.copy()

# %%
# Traitement des valeurs aberrantes

# person_age
df_clean = df_clean[df_clean['person_age'] < 100]

# person_income
df_clean = df_clean[df_clean['person_income'] < 2000000]

# person_emp_length
df_clean = df_clean[df_clean['person_emp_length'] < 60]

# loan_amnt
df_clean = df_clean[df_clean['loan_amnt'] < 35000]

# loan_int_rate
df_clean = df_clean[df_clean['loan_int_rate'] < 25]

# loan_percent_income
df_clean = df_clean[df_clean['loan_percent_income'] < 0.8]

# cb_person_cred_hist_length
df_clean = df_clean[df_clean['cb_person_cred_hist_length'] < 30]

# %%
# Traitement des valeurs manquantes

# Imputation par la médiane pour les variables numériques
for col in ['person_emp_length', 'loan_int_rate']:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)

# %%
# Vérification des valeurs manquantes
df_clean.isnull().sum()

# %%
# # <font color=green> Préparation des données pour la modélisation

# %%
# Encodage des variables catégorielles
df_clean = pd.get_dummies(df_clean, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], drop_first=True)

# %%
df_clean.head()

# %%
# Séparation des variables explicatives (X) et de la variable cible (y)
X = df_clean.drop('loan_status', axis=1)
y = df_clean['loan_status']

# %%
# Division des données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
# Normalisation des variables numériques
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.select_dtypes(include=np.number))
X_test_scaled = scaler.transform(X_test.select_dtypes(include=np.number))

# %%
# Reconstitution des dataframes avec les variables normalisées et catégorielles
X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.select_dtypes(include=np.number).columns, index=X_train.index)
X_train_final = pd.concat([X_train_final, X_train.select_dtypes(include='uint8')], axis=1)

X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.select_dtypes(include=np.number).columns, index=X_test.index)
X_test_final = pd.concat([X_test_final, X_test.select_dtypes(include='uint8')], axis=1)

# %%
X_train_final.head()

# %%
X_test_final.head()

# %%
# # <font color=green> Modélisation

# %%
# Importation des modèles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV

# %%
# Importation des métriques d'évaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# %%
# Définition des modèles à tester
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVC': SVC(random_state=42, probability=True),
    'K-Neighbors': KNeighborsClassifier(),
    'MLP Classifier': MLPClassifier(random_state=42, max_iter=1000),
    'Gaussian Naive Bayes': GaussianNB(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Gaussian Process': GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42)
}

# %%
# Entraînement et évaluation des modèles
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    y_proba = model.predict_proba(X_test_final)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'Confusion Matrix': cm
    }
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, ROC AUC: {roc_auc if isinstance(roc_auc, float) else 'N/A':.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(cm)
    print("\n"+"="*50+"\n")

# %%
# Affichage des résultats sous forme de tableau
results_df = pd.DataFrame(results).T
results_df

# %%
# Visualisation des performances ROC AUC
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df['ROC AUC'])
plt.title('ROC AUC des Modèles')
plt.ylabel('ROC AUC')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %%
# # <font color=green> Conclusion

# %%
# FELICITATIONS pour avoir suivi cette formation jusqu'à la fin. Vous êtes à maintenant capables :
# 
# * **d'analyser, de nettoyer et de préparer les données pour modéliser la probabilité de défaut de paiement** ; 
# 
# * **d'analyser la performance des différents modèles construits et de déterminer et de déterminer le seuil optimal pour la prédiction des résultats de la variable cible** ;
# 
# * **de comparer plusieurs modèles en utilisant une métrique comme l'AUC** ;
# 
# * **de bien structurer votre code R en créant des fonctions qui rendent votre flux de travail beaucoup plus clair et digeste**. 
# 
# Ces compétences vous aideront dans d'autres tâches de mdoélisation par des algorithmes de Machine Learning. 
# 
# La modélisation du risque de crédit par les méthodes d'apprentissage automatique est un domaine passionnant et il reste encore beaucoup de choses à apprendre. C'est pour cela, vous ne devez pas vous arrêter en si bon chemin

# %%
# ![mon_logo.jpg](attachment:mon_logo.jpg)
