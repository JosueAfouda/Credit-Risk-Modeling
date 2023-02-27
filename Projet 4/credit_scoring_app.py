
import streamlit
import urllib.request
import json
import os
import ssl

def run():

  # En-tête
  streamlit.title("Application Machine Learning pour la prédiction du risque de défaut de paiement d'un crédit bancaire 💰")
  streamlit.subheader("Auteur : Josué AFOUDA")
  streamlit.markdown("Cette application simule l'utilisation d'un service prédictif qui dérive d'un modèle Machine Learning déployé sur Microsoft Azure." \
              "Le modèle a été construit avec la fonctionnalité Azure Machine Learning Automatisé et a un AUC pondéré de 0,95077." \
              "Il prédit si un demandeur de crédit sera éligible (classe 0) ou non éligible (classe 1) au prêt bancaire 💰.")


  # Variables d'entrées renseignées par l'utilisateur
  var1=streamlit.text_input("Age du demandeur de prêt")
  var2=streamlit.text_input("Revenu annuel du demandeur de prêt")
  var3=streamlit.selectbox('Propriété immobilière : (MORTGAGE : hypothèque, OWN : propriétaire, RENT : Locataire, OTHER : Autres cas):',('MORTGAGE', 'OWN', 'RENT', 'OTHER'))
  var4=streamlit.text_input("Depuis quand le demandeur est-il en activité professionnelle ? (Durée en nombre d'années)")
  var5=streamlit.selectbox("Motif du prêt : (DEBTCONSOLIDATION : Rachat d'un crédit, HOMEIMPROVEMENT : Travaux de rénovation immobilière, VENTURE : Business):",
                             ('DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'))
  var6=streamlit.selectbox('Catégorie du crédit :',('A', 'B', 'C', 'D', 'E', 'F', 'G'))
  var7=streamlit.text_input("Montant du crédit")
  var8=streamlit.text_input("Taux d'intérêt du crédit (en %)")
  var9=streamlit.text_input("Ratio Dette/Revenu du demandeur de crédit (valeur décimale entre 0 et 1)")
  var10=streamlit.selectbox('Est-ce que le demandeur de prêt est à découvert bancaire ? : (True : Oui, False : Non):',('True', 'False'))
  var11=streamlit.text_input("Echéance des crédits antécédents (en nombre d'années)")
  
  # Code du service prédictif
  
  def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
      ssl._create_default_https_context = ssl._create_unverified_context

  allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.


  def return_prediction():
      # Request data goes here
      data = {
          "data":
          [
              {
                  'person_age': var1,
                  'person_income': var2,
                  'person_home_ownership': var3,
                  'person_emp_length': var4,
                  'loan_intent': var5,
                  'loan_grade': var6,
                  'loan_amnt': var7,
                  'loan_int_rate': var8,
                  'loan_percent_income': var9,
                  'cb_person_default_on_file': var10,
                  'cb_person_cred_hist_length': var11,
              },
          ],
      }

      body = str.encode(json.dumps(data))

      url = 'http://27b31c8a-6099-4c9c-a881-3c7cb3db768e.francecentral.azurecontainer.io/score'
      api_key = '9rpviU6YUITGlwa7DQn4Tgaaf1TXUvEJ' # Replace this with the API key for the web service
      headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

      req = urllib.request.Request(url, body, headers)

      try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return result
        #print(result)
      
      except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(json.loads(error.read().decode("utf8", 'ignore')))

  #################################

  if streamlit.button("Predict"):
    resultat = return_prediction()
    streamlit.success("La classe prédite par le modèle est {}".format(resultat))
    
if __name__=='__main__':
    run()

