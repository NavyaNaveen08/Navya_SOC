### **README: TimeSynth AI - Logic Depth Predictor**  

---

## **Project Overview**  
This project predicts **logic depth in RTL designs** using **machine learning models**. It provides a simple **web interface** where users can enter RTL parameters and obtain predictions.  

---

 **Installation and Setup**  

**1. Install Required Libraries**  
Run the following command in a Google Colab cell to install dependencies:  
```python
!pip install gradio joblib numpy pandas
```

---

**2. Load the Trained Model from Google Drive**  
Ensure that the trained model is saved in Google Drive as `optimized_timing_prediction_model.pkl`, then mount Google Drive:  
```python
from google.colab import drive
drive.mount('/content/drive')
```

---

##**3. Run the Web Interface**  
Copy and execute the following script in a Google Colab cell:  
```python
import gradio as gr
import joblib
import numpy as np

# Load trained model from Google Drive
model = joblib.load("/content/drive/My Drive/optimized_timing_prediction_model.pkl")

# Define prediction function
def predict_logic_depth(fan_in, fan_out, num_gates, num_logic_levels, path_delay, FOC, SGPA):
    features = np.array([[fan_in, fan_out, num_gates, num_logic_levels, path_delay, FOC, SGPA]])
    prediction = model.predict(features)
    return f"Predicted Logic Depth: {prediction[0]:.2f}"

# Create Gradio UI
interface = gr.Interface(
    fn=predict_logic_depth,
    inputs=[
        gr.Number(label="Fan In"),
        gr.Number(label="Fan Out"),
        gr.Number(label="Number of Gates"),
        gr.Number(label="Number of Logic Levels"),
        gr.Number(label="Path Delay"),
        gr.Number(label="FOC"),
        gr.Number(label="SGPA")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Logic Depth Predictor"
)

# Launch Web App
interface.launch(share=True)

4. Access the Web Interface 
After running the script, a **public link** (e.g., `https://abcdef.gradio.live`) will be generated.  
1. Click the link to open the web interface.  
2. Enter the RTL parameters in the input fields.  
3. Click **Predict** to obtain the predicted logic depth.  

---

Additional Notes 
- Ensure that the trained model file (`optimized_timing_prediction_model.pkl`) is stored in Google Drive before running the script.  
- If the link does not work, restart Google Colab and rerun the script.  
