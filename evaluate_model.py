import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score

def evaluate_model(model_path, test_data_dir):
    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully.")

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Adjust if your model has multiple classes
                  metrics=['accuracy'])

    # Prepare test data
    test_images = []
    test_labels = []

    # Load images and labels
    for label in os.listdir(test_data_dir):
        class_dir = os.path.join(test_data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image = load_img(img_path, target_size=(224, 224))  # Adjust to your model's input size
                    image = img_to_array(image) / 255.0  # Normalize the image
                    test_images.append(image)
                    test_labels.append(label)  # Use the directory name as label

    # Convert to numpy arrays
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Convert string labels to numeric labels
    label_mapping = {'NORMAL': 0, 'PNEUMONIA': 1}  # Update according to your classes
    try:
        numeric_labels = np.array([label_mapping[label.upper()] for label in test_labels])
    except KeyError as e:
        print(f"KeyError: Label '{e}' not found in label mapping. Please check your directory names and label mapping.")
        return

    # Perform predictions
    predictions = model.predict(test_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(numeric_labels, predicted_classes)
    print(f'Accuracy: {accuracy:.2f}')

    # Plotting true vs predicted labels
    plt.figure(figsize=(10, 6))
    plt.plot(numeric_labels, label='True Labels', marker='o', linestyle='-')
    plt.plot(predicted_classes, label='Predicted Labels', marker='x', linestyle='--')
    plt.title('True vs Predicted Labels for CT Classification')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.xticks(ticks=np.arange(len(numeric_labels)), labels=np.arange(len(numeric_labels)))
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Paths to your CT model and test data
    model_path = './models/xray_classification_model.h5'
    test_data_dir = './Xray-Dataset/chest_xray/val'  # Change to your test data directory

    # Evaluate the model
    if os.path.exists(model_path) and os.path.exists(test_data_dir):
        evaluate_model(model_path, test_data_dir)
    else:
        print("Model or test data directory does not exist. Please check the paths.")
