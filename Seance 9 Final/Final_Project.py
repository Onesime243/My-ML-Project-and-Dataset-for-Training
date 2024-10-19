import pickle
import streamlitApp as st

model = pickle.load(open("C:/Users/Onesime/Documents/Formation en Machine Learning avec Scikit_Learn/Seance 9 Final/classifier_XGBC_final.pkl", 'rb'))

def main():
    import streamlit as st
    st.title('Solution Machine Learning pour la prediction de désabonnement des clients')
    

    #Input Variables
    CreditScore = st.text_input("CreditScore")
    Geography = st.text_input("Geography")
    Gender = st.text_input("Gender")
    Age = st.text_input("Age")
    Tenure = st.text_input("Tenure")
    Balance = st.text_input("Balance")
    NumOfProducts = st.text_input("NumOfProducts")
    HasCrCard = st.text_input("HasCrCard")
    IsActiveMember = st.text_input("IsActiveMember")
    EstimatedSalary = st.text_input("EstimatedSalary")


    #Prediction Code
    if st.button("Predict"):
        makeprediction = model.predict([[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])
        output = round(makeprediction [0],2)
        st.success("La valeur Predicte est {}".format(output))
        
if __name__=='__main__':
    main()


        