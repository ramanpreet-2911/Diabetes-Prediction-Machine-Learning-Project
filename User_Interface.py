import tkinter as tk
from tkinter import ttk
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('diabetes_model.pkl')  

# Load the previously fitted StandardScaler
scaler = joblib.load('scaler.pkl')  

#create a function to make predictions


def make_prediction():
    try:
        # Collect input data from the user
        preg = int(entry_pregnancies.get())
        gluc = int(entry_glucose.get())
        bp = int(entry_blood_pressure.get())
        sthick = int(entry_skin_thickness.get())
        insulin = int(entry_insulin.get())
        bmi = float(entry_bmi.get())
        dpf = float(entry_dpf.get())
        age = int(entry_age.get())
        
        
        unknown = [preg, gluc, bp, sthick, insulin, bmi, dpf, age]
        final = np.asarray(unknown)
        final_and_reshaped = final.reshape(1,-1)
        standardised = scaler .transform(final_and_reshaped)
        

        #predicting the result
        prediction = model.predict(standardised)

        # Display the prediction
        if prediction[0] == 1:
            result_label.config(text="Diabetic", foreground="red")
        else:
            result_label.config(text="Not Diabetic", foreground="green")

    except ValueError:
        result_label.config(text="Invalid input. Please enter valid numeric values.", foreground="black")

# Initialize the Tkinter window
root = tk.Tk()
root.title("Diabetes Prediction App")

# Create labels and entry widgets for user input
labels = ["Number of Pregnancies:", "Glucose Level:", "Blood Pressure:", "Skin Thickness:", "Insulin Level:",
          "BMI:", "Diabetes Pedigree Function:", "Age:"]

for i, label_text in enumerate(labels):
    label = ttk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5, sticky=tk.W)

entry_pregnancies = ttk.Entry(root)
entry_glucose = ttk.Entry(root)
entry_blood_pressure = ttk.Entry(root)
entry_skin_thickness = ttk.Entry(root)
entry_insulin = ttk.Entry(root)
entry_bmi = ttk.Entry(root)
entry_dpf = ttk.Entry(root)
entry_age = ttk.Entry(root)

entry_pregnancies.grid(row=0, column=1, padx=10, pady=5)
entry_glucose.grid(row=1, column=1, padx=10, pady=5)
entry_blood_pressure.grid(row=2, column=1, padx=10, pady=5)
entry_skin_thickness.grid(row=3, column=1, padx=10, pady=5)
entry_insulin.grid(row=4, column=1, padx=10, pady=5)
entry_bmi.grid(row=5, column=1, padx=10, pady=5)
entry_dpf.grid(row=6, column=1, padx=10, pady=5)
entry_age.grid(row=7, column=1, padx=10, pady=5)

# Create a button to make predictions
predict_button = ttk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=8, column=0, columnspan=2, pady=10)

# Create a label to display the prediction result
result_label = ttk.Label(root, text="", font=("Helvetica", 16))
result_label.grid(row=9, column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
