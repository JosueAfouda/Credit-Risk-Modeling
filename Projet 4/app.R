# Integration d'un modele de Machine Learning dans R Shiny : Cas de la modelisation du risque de credit

library(shiny)
library(shinydashboard)

model <- readRDS('credit_scoring_rf_model.rds')

ui <- dashboardPage(
  dashboardHeader(
    title = "Credit Scoring",
    dropdownMenu(
      type = "messages",
      messageItem(
        from = "Josué AFOUDA",
        message = "Cours sur la modélisation du Risque de crédit",
        href = "https://afouda-datascience.com/cours/machine-learning-pour-la-modelisation-du-risque-de-credit-credit-scoring-dans-r/"
      ),
      # Autre message
      messageItem(
        from = "Josué AFOUDA",
        message = "Formation sur R Shiny",
        href = "https://afouda-datascience.com/cours/super-r-shiny-course/"
      )
    )
  ), 
  
  dashboardSidebar(), 
  
  dashboardBody(
    
    tabItem(
      tabName = "features",
      fluidRow(box(valueBoxOutput("score_prediction")),
               box(numericInput("var1", label = "Age du demandeur de credit", 
                                value = 20, min = 18))),
      
      fluidRow(box(numericInput("var2", label = "Revenu annuel demandeur de credit", 
                                value = 10000, min = 0)),
               box(selectInput("var3", 
                               label = "Propriété immobilière : (MORTGAGE : hypothèque, OWN : propriétaire, RENT : Locataire, OTHER : Autres cas)", 
                               choices = c('MORTGAGE', 'OWN', 'RENT', 'OTHER')))),
      
      fluidRow(box(numericInput("var4", 
                                label = "Depuis quand le demandeur est-il en activité professionnelle ? (Durée en nombre d'années)", 
                                value = 3, min = 0)),
               box(selectInput("var5", 
                               label = "Motif du prêt : (DEBTCONSOLIDATION : Rachat d'un crédit, HOMEIMPROVEMENT : Travaux de rénovation immobilière, VENTURE : Business)", 
                               choices = c('DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE'))),),
      fluidRow(box(selectInput("var6", 
                               label = "Catégorie du crédit", 
                               choices = c('A', 'B', 'C', 'D', 'E', 'F', 'G'))),
               box(numericInput("var7", 
                                label = "Montant du crédit", 
                                value = 2000, min = 0))),
      
      fluidRow(box(numericInput("var8", 
                                label = "Taux d'intéret du crédit (en %)", 
                                value = 3.5, min = 0)),
               box(numericInput("var9", 
                                label = "Ratio Dette/Revenu du demandeur de crédit (valeur décimale entre 0 et 1)", 
                                value = 0.1, min = 0, max = 1))),
      
      fluidRow(box(selectInput("var10", 
                               label = "Est-ce que le demandeur de credit est à découvert bancaire ? : (Y : Oui, N : Non):", 
                               choices = c('Y', 'N'))),
               box(numericInput("var11", 
                                label = "Echéance des crédits en cours (en nombre d'années)", 
                                value = 5, min = 0)))
      
    )
    
  )
  
)

server <- function(input, output) {
  
  prediction <- reactive({
    predict(
      model,
      data.frame(
        "person_age" = input$var1,
        "person_income" = input$var2,
        "person_home_ownership" = input$var3,
        "person_emp_length" = input$var4,
        "loan_intent" = input$var5,
        "loan_grade" = input$var6,
        "loan_amnt" = input$var7,
        "loan_int_rate" = input$var8,
        "loan_percent_income" = input$var9,
        "cb_person_default_on_file" = input$var10,
        "cb_person_cred_hist_length" = input$var11
      ),
      type = 'raw'
    )
  })
  
  prediction_label <- reactive({
    ifelse(prediction() == "0", "Eligible au Crédit", "Non Eligible au Crédit")
  })
  
  prediction_prob <- reactive({
    predict(
      model,
      data.frame(
        "person_age" = input$var1,
        "person_income" = input$var2,
        "person_home_ownership" = input$var3,
        "person_emp_length" = input$var4,
        "loan_intent" = input$var5,
        "loan_grade" = input$var6,
        "loan_amnt" = input$var7,
        "loan_int_rate" = input$var8,
        "loan_percent_income" = input$var9,
        "cb_person_default_on_file" = input$var10,
        "cb_person_cred_hist_length" = input$var11
      ),
      type = "prob"
    ) 
  })
  
  prediction_color <- reactive({
    ifelse(prediction() == "0", "green", "red")
  })
  
  output$score_prediction <- renderValueBox({
    
    valueBox(
      value = paste(round(100*prediction_prob()$`1`, 0), "%"),
      subtitle = prediction_label(),
      color = prediction_color(),
      icon = icon("hand-holding-usd")
    )                     
    
  })
  
}

shinyApp(ui, server)