# Predictive-Maintenance
Predictive Maintenance using Machine Learning 🛠️🤖
Overview
This project focuses on building a Predictive Maintenance model to predict machine failures before they occur. By analyzing sensor data such as temperature, torque, and rotational speed, the model can identify potential risks, helping industries reduce downtime and maintenance costs.

This project was developed as part of the AI Training Program at the British University in Egypt (BUE).

📊 Dataset Description
The dataset used is the Predictive Maintenance Dataset, which contains 10,000 data points with the following features:


Air Temperature [K]: The surrounding temperature. 


Process Temperature [K]: The temperature of the manufacturing process. 


Rotational Speed [rpm]: The speed of the machine's rotation. 


Torque [Nm]: The torque applied to the machine. 


Tool Wear [min]: The duration the tool has been in use. 


Failure Types: Includes Power Failure, Overstrain Failure, Tool Wear Failure, and more. 

🛠️ Key Steps & Methodology
Data Preprocessing: Handling missing values, normalizing features, and analyzing class distribution.

Exploratory Data Analysis (EDA): Visualizing correlations between features and identifying key failure triggers.

Model Building: Implementing supervised learning algorithms (like Random Forest or Decision Trees).

Evaluation: Using performance metrics such as Accuracy, Confusion Matrix, and Feature Importance.

📈 Results & Visualizations
1. Confusion Matrix
The model shows high precision in classifying different failure types, especially in identifying "No Failure" cases.

2. Feature Importance
Based on the analysis, Torque and Rotational Speed were found to be the most significant factors contributing to machine failure.

🎓 Acknowledgments
Special thanks to the British University in Egypt (BUE) for providing the training and resources to complete this project.
