# Tomato Leaf Image Classifier (using EfficientNetB4)

## 🚀 Project Overview

This project is a deep learning model for binary image classification. It is designed to determine whether an image of a tomato plant contains a leaf or not.

The model is built using TensorFlow and Keras, employing **Transfer Learning** with the powerful **EfficientNetB4** architecture.

**Note on Naming:** The project notebook file is named `MobileNetV2.ipynb`, but the code within *actually* implements the `EfficientNetB4` model. This `README` reflects the code that was written.

## 📊 Dataset

The dataset for this project was custom-built by combining two different sources to create the binary classes:

* **`leaves` (Class 0):** This data, containing images of tomato leaves (both healthy and diseased), was sourced from the **"Tomato-Village"** public GitHub repository.
    * *Source:* [github.com/mamta-joshi-gehlot/Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village/tree/main)

* **`no_leaves` (Class 1):** This data was manually collected by the project author. It consists of images downloaded from Kaggle, including images of tomatoes on the vine (fruit), soil, and other non-leaf parts of the plant.

The final dataset was organized into `train/leaves` and `train/no_leaves` folders. The `image_dataset_from_directory` utility from Keras was then used to load the data.

* **Data Split:** The dataset was split into 80% for training and 20% for validation using the `validation_split` argument.
* **Performance:** The dataset was prefetched using `tf.data.AUTOTUNE` for efficient loading during training.

---

## 🤖 Model Architecture (Transfer Learning)

The model is built by stacking a custom classifier on top of a pre-trained base model.

1.  **Base Model:** `EfficientNetB4` pre-trained on ImageNet. The input shape was set to `(380, 380, 3)` to match the model's requirements.
2.  **Freezing:** The entire `EfficientNetB4` base model was frozen (`base_model.trainable = False`) so that only the new, custom layers would be trained.
3.  **Custom Head:** A new classifier "head" was added to the base model:
    * `GlobalAveragePooling2D`: To reduce the feature maps from the base model into a single vector.
    * `Dropout(0.2)`: A dropout layer for regularization to help prevent overfitting.
    * `Dense(128, 'relu')`: A hidden dense layer with 128 neurons.
    * `Dense(1, 'sigmoid')`: The final output layer with a single neuron and 'sigmoid' activation, perfect for binary (0 or 1) classification.

---

## 📈 Training & Saving

* **Compiler:** The model was compiled with the `adam` optimizer and `binary_crossentropy` loss, which is standard for two-class (sigmoid) problems.
* **Training:** The model was trained for **10 epochs** on the `train_dataset` while validating against the `validation_dataset`.
* **Saving:** After training, the final model was saved to a `.h5` file (`leaf_classifier_model.h5`).

---

## 🔬 Inference (Testing the Model)

The notebook includes a section for loading the saved model and performing a prediction on a new image.

1.  The saved `leaf_classifier_model.h5` model is loaded.
2.  A test image is loaded and preprocessed to match the model's input (resized to `380x380`).
3.  `model.predict()` is called, which returns a single "score" between 0.0 and 1.0.
4.  A threshold of **0.5** is used to make a final decision:
    * If `score < 0.5`, it is classified as "Result: Leaf ✅ (Class: leaves)".
    * If `score >= 0.5`, it is classified as "Result: No leaf ❌ (Class: no_leaves)".

## 🛠️ Technologies Used

* Python 3
* TensorFlow & Keras
* `tf.keras.applications.EfficientNetB4`
* NumPy
* Matplotlib
* Google Colab & Google Drive
