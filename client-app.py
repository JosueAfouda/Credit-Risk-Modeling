name = input("Quel est votre nom ?")

print("Salut", name, ". Veuillez renseignez les informations ci-dessous :")

person_age = int(input("Age :"))
person_income = int(input("Revenu annuel :"))
person_home_ownership = input('Propriété immobilière : (MORTGAGE : hypothèque, OWN : propriétaire, RENT : Locataire, OTHER : Autres cas):')
person_emp_length = int(input("Depuis quand le demandeur est-il en activité professionnelle ? (Durée en nombre d'années)"))
loan_intent = input("Motif du prêt : (DEBTCONSOLIDATION : Rachat d'un crédit, HOMEIMPROVEMENT : Travaux de rénovation immobilière, VENTURE : Business):")
loan_grade = input('Catégorie du crédit (A, B, C, D, E, F, G) :')
loan_amnt = int(input("Montant du crédit"))
loan_int_rate = float(input("Taux d'intérêt du crédit (en %)"))
loan_percent_income = float(input("Ratio Dette/Revenu du demandeur de crédit (valeur décimale entre 0 et 1)"))
cb_person_default_on_file = input('Est-ce que le demandeur de prêt est à découvert bancaire ? (true, false)')
cb_person_cred_hist_length = int(input("Echéance des crédits antécédents (en nombre d'années)"))

import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data =  {
  "Inputs": {
    "data": [
      {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
      }
    ]
  },
  "GlobalParameters": {
    "method": "predict"
  }
}

body = str.encode(json.dumps(data))

url = 'http://4e40e8d8-feaf-42b1-b116-c38548887f1b.eastus2.azurecontainer.io/score'


headers = {'Content-Type':'application/json'}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))