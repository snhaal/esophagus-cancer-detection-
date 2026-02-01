Here is the formatted content for your README.md without emojis. You can copy and paste this directly into your file.

Esophagus Cancer Detection using EfficientNet with CLAHE and Fourier Transform
Project Description
This project implements an AI-powered diagnostic tool for the automated classification of esophageal endoscopic images. The system is designed to assist clinicians in the early detection of esophageal cancer, addressing challenges such as late diagnosis and the subjectivity of standard endoscopy.

Technical Architecture
The model utilizes a novel hybrid preprocessing pipeline to improve feature representation:


Input Layer: Standard RGB endoscopic images are processed into three distinct modalities.


Hybrid Preprocessing: The system applies Contrast Limited Adaptive Histogram Equalization (CLAHE) for local contrast enhancement and Fourier Transform for frequency domain feature extraction.


9-Channel Stacking: The raw RGB image, CLAHE-enhanced image, and Fourier magnitude image are stacked into a 9-channel input tensor.


Modified EfficientNet-B0: A custom 1x1 convolution layer reduces the 9 channels back to 3 to leverage the pre-trained EfficientNet-B0 backbone for feature extraction.


Classification Head: A fully connected layer with 512 units, ReLU activation, and a 0.5 dropout rate for improved generalization.

Performance Results
The model was evaluated on a dataset of 2,107 images across 7 classes:


Accuracy: Peak validation accuracy of 93.33%.


AUC Score: Multi-class Area Under the Curve of 0.99.


Optimization: Trained using the AdamW optimizer with a cosine annealing learning rate scheduler and early stopping.

Deployment
The system is deployed as a monolithic Django web application, allowing for a single-unit package that includes the user interface, preprocessing logic, and model inference coordination.

How to Use Inference
You can use the pre-trained MODEL.pth to classify new endoscopic images.

Option 1: Using the Web App (Django)
Navigate to the web app directory: cd web_app.

Start the server: python manage.py runserver.

Open your browser to http://127.0.0.1:8000/.

Log in and upload an image to see the classification result.

Option 2: Using the Jupyter Notebook
Open notebooks/inference_pipeline.ipynb.

This notebook loads the MODEL.pth file.

It applies the hybrid CLAHE + Fourier preprocessing.

It outputs the predicted class from the 7 categories.

How to Train
If you want to retrain the model using your own dataset:


Prepare your dataset: Organize your images into 7 folders named after the classes: dyed_lifted_polyps, dyed_resection_margins, esophagus, normal_pylorus, normal_z_line, polyps, and ulcerative_colitis.

Configure hyperparameters: In preprocess_and_model_pipeline.ipynb, you can adjust:


Learning Rate: 0.0001.


Batch Size: 32.


Epochs: 25 with Early Stopping.

Run the training script: Execute the cells in preprocess_and_model_pipeline.ipynb. This will:

Perform the 9-channel image stacking.

Train the modified EfficientNet-B0 backbone.

Save the best model as a new .pth file.
