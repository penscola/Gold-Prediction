import numpy as np
import pandas as pd
import pickle
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Load the saved full pipeline from the file
model_file = '/media/penscola/Penscola@Tech/Projects/Gold-Prediction/model/Random-Forest-Regressor.pkl'

with open(model_file, 'rb') as f_in:
    scaler, model = pickle.load(f_in)

# Define the predict function
def predict(SPX, USO, SLV, EUR_USD):
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'SPX': [SPX] if SPX is not None else [0],  # Replace None with default value
        'USO': [USO] if USO is not None else [0],  # Replace None with default value
        'SLV': [SLV] if SLV is not None else [0],  # Replace None with default value
        'EUR_USD': [EUR_USD] if EUR_USD is not None else [0],  # Replace None with default value
    })


        # Make predictions using the loaded logistic regression model
        #predict probabilities
    predictions = model.predict(input_data)
    #take the index of the maximum probability


    #return predictions[0]
    return(f'[Info] Predicted probabilities{predictions}')
    
# Setting Gradio App Interface
with gr.Blocks(css=".gradio-container {background-color:grey }",theme=gr.themes.Base(primary_hue='blue'),title='Uriel') as demo:
    gr.Markdown("# Gold Price prediction #\n*This App allows the user to predict the price of Gold.*")
    
    # Receiving ALL Input Data here
    gr.Markdown("**Demographic Data**")
    with gr.Row():
        gender = gr.Number(label="Standard & Poor's Index")
        SeniorCitizen = gr.Number(label="United State Oil Fund")
        Partner = gr.Number(label="Silver Price")
        Dependents = gr.Number(label="EURO_Dollar Exchange")


    # Output Prediction
    output = gr.Text(label="Outcome")
    submit_button = gr.Button("Predict")
    
    submit_button.click(fn= predict,
                        outputs= output,
                        inputs=[gender, SeniorCitizen, Partner, Dependents],
    
    ),
    
    # Add the reset and flag buttons
    def clear():
        output.value = ""
        return 'Predicted values have been reset'
         
    clear_btn = gr.Button("Reset", variant="primary")
    clear_btn.click(fn=clear, inputs=None, outputs=output)
        
 
demo.launch(inbrowser = True)