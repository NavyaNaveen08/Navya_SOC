### **README: TimeSynth AI - Logic Depth Predictor**  

---

## **Table of Contents**  
1. Project Overview  
2. Problem Statement  
3. Solution Approach  
4. System Architecture  
5. Algorithm Details  
6. Dataset Used  
7. Model Training & Optimization  
8. Installation & Setup  
9. Running the Model & Web Interface  
10. API & Web UI Usage  
11. Future Improvements  
12. Contributors  
13. License  

---

## **1. Project Overview**  
TimeSynth AI - Logic Depth Predictor is a machine learning-based solution designed to predict combinational logic depth in RTL designs.  
It helps designers estimate timing violations early without requiring full synthesis, reducing computational overhead and improving design iteration speed.  

---

## **2. Problem Statement**  
Traditional timing analysis in FPGA and SoC design is computationally expensive and slow, requiring synthesis-based methods to estimate logic depth.  
Early detection of timing violations can optimize the design cycle, reduce computational costs, and improve RTL development workflows.  

---

## **3. Solution Approach**  
The solution utilizes machine learning models trained on RTL design parameters to predict logic depth.  
Users provide RTL design inputs, and the trained model estimates the expected logic depth without requiring synthesis.  
The results are displayed through a web-based interface, allowing for quick and efficient analysis.  

---

## **4. System Architecture**  
- Input: User provides RTL parameters such as fan-in, fan-out, number of gates, logic levels, path delay, etc.  
- Processing: The system preprocesses the data and feeds it into the trained machine learning model.  
- Output: The predicted logic depth is displayed via a web UI or API.  

---

## **5. Algorithm Details**  
- The model is trained on a dataset containing RTL design parameters.  
- Feature selection is applied to choose the most relevant variables.  
- Machine learning techniques, including Gradient Boosting and Random Forest, are used to improve prediction accuracy.  
- The trained model is optimized for inference speed and efficiency.  

---

## **6. Dataset Used**  
The dataset includes RTL design parameters relevant to logic depth prediction. Features include:  
- Fan In  
- Fan Out  
- Number of Gates  
- Number of Logic Levels  
- Path Delay  
- Fan-Out Complexity (FOC)  
- Signal Gate Propagation Analysis (SGPA)  

The dataset is stored in CSV format and used for training the model.  

---

## **7. Model Training & Optimization**  
- The dataset is split into training and testing sets.  
- Hyperparameter tuning is performed to improve accuracy.  
- Multiple machine learning models are evaluated, and the best-performing model is selected.  
- The trained model is saved for deployment.  

---

## **8. Installation & Setup**  

### **Running in Google Colab**  
1. Open Google Colab.  
2. Upload the trained model to Google Drive.  
3. Mount Google Drive in Colab.  
4. Install the necessary dependencies.  

### **Dependencies Installation**  
Run the following command in Google Colab:  
```python
!pip install gradio joblib numpy pandas
```

---

## **9. Running the Model & Web Interface**  
To launch the web-based UI, run the following script in Google Colab:  
```python
import gradio as gr
import joblib
import numpy as np

model = joblib.load("/content/drive/My Drive/optimized_timing_prediction_model.pkl")

def predict_logic_depth(fan_in, fan_out, num_gates, num_logic_levels, path_delay, FOC, SGPA):
    features = np.array([[fan_in, fan_out, num_gates, num_logic_levels, path_delay, FOC, SGPA]])
    prediction = model.predict(features)
    return f"Predicted Logic Depth: {prediction[0]:.2f}"

interface = gr.Interface(
    fn=predict_logic_depth,
    inputs=[gr.Number(label="Fan In"), gr.Number(label="Fan Out"), gr.Number(label="Number of Gates"), 
            gr.Number(label="Number of Logic Levels"), gr.Number(label="Path Delay"), 
            gr.Number(label="FOC"), gr.Number(label="SGPA")],
    outputs=gr.Textbox(label="Prediction"),
    title="Logic Depth Predictor"
)

interface.launch(share=True)
```

After running the script, a public link will be generated. Clicking the link will open the web UI, where users can enter RTL parameters and get predictions.  

---

## **10. API & Web UI Usage**  

### **Web UI**
- Enter the RTL parameters.  
- Click "Predict" to get the predicted logic depth.  

### **API Endpoint (Optional)**
The trained model can also be accessed via an API by sending a POST request with the required parameters.  
Example request:  
```python
import requests

url = "https://abcdef.gradio.live/api/predict"  # Replace with actual URL
data = {"fan_in": 5, "fan_out": 10, "num_gates": 50, "num_logic_levels": 8, "path_delay": 1.2, "FOC": 3, "SGPA": 2}

response = requests.post(url, json=data)
print(response.json())
```

---

## **11. Future Improvements**  
- Improve model accuracy using advanced deep learning techniques.  
- Deploy the model on a cloud-based platform for scalability.  
- Enhance the web interface for better usability.  
- Expand API functionality for batch predictions.  

---
