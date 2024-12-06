import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
import seaborn as sns

# Load and cache the MNIST dataset
def load_mnist_data():
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    dataset_path = 'mnist.npz'
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path=dataset_path)
    return x_train, y_train, x_test, y_test

# Train a neural network model on the MNIST dataset
def train_mnist_model(x_train, y_train, x_test, y_test):
    class EarlyStoppingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') > 0.99:
                print("\nReached 99% accuracy. Stopping training!")
                self.model.stop_training = True

    callbacks = EarlyStoppingCallback()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())

    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    print(history.epoch, history.history['accuracy'][-1])
    return model, history

# Predict a digit from an input image using the trained model
def predict_digit(model, img):
    img_array = np.array([img])
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    return str(predicted_digit)

# OpenCV: Handle mouse click event
start_inference = False
def toggle_inference(event, x, y, flags, params):
    global start_inference
    if event == cv2.EVENT_LBUTTONDOWN:
        start_inference = not start_inference

# OpenCV: Handle threshold slider
threshold = 100
def update_threshold(value):
    global threshold
    threshold = value

# Main OpenCV processing loop
def run_opencv_loop(model):
    global threshold
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow('MNIST Digit Recognition')
    cv2.setMouseCallback('MNIST Digit Recognition', toggle_inference)
    cv2.createTrackbar('threshold', 'MNIST Digit Recognition', threshold, 255, update_threshold)
    frame_count = 0
    processed_view_visible = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        if start_inference:
            processed_view_visible = True
            frame_count += 1

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
            _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY_INV)

            height, width = frame.shape[:2]
            box_size = 150
            center_x, center_y = width // 2, height // 2
            x1, y1 = center_x - box_size // 2, center_y - box_size // 2
            x2, y2 = center_x + box_size // 2, center_y + box_size // 2
            roi = binary_frame[y1:y2, x1:x2]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 3)
            cv2.putText(frame, "Show digit in box", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
            cv2.putText(frame, f"Threshold: {threshold}", (10, 130), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)

            height, width = roi.shape
            if width > height:
                new_width = 20
                new_height = int(height * (20 / width))
            else:
                new_height = 20
                new_width = int(width * (20 / height))

            resized_img = cv2.resize(roi, (new_width, new_height))
            padded_img = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_width) // 2
            y_offset = (28 - new_height) // 2
            padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

            prediction = predict_digit(model, padded_img)
            cv2.putText(frame, prediction, (center_x - 25, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)
            cv2.imshow('MNIST Digit Recognition', frame)
            cv2.imshow('Processed View', roi)
        else:
            cv2.putText(frame, "Click anywhere to start recognition", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
            cv2.imshow('MNIST Digit Recognition', frame)
            if processed_view_visible:
                if cv2.getWindowProperty('Processed View', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('Processed View')
                    processed_view_visible = False

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    model = None
    try:
        model = tf.keras.models.load_model('model.sav')
        print("Loaded saved model.")
        print(model.summary())
        
        # Generate predictions and evaluation data
        _, _, x_test, y_test = load_mnist_data()
        x_test = x_test / 255.0
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Create subplot figure
        plt.figure(figsize=(20, 10))
        
        # 1. Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(y_test, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. Per-class Accuracy
        plt.subplot(2, 3, 2)
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        plt.bar(range(10), class_accuracy)
        plt.title('Per-class Accuracy')
        plt.xlabel('Digit')
        plt.ylabel('Accuracy')
        
        # 3. ROC Curve
        plt.subplot(2, 3, 3)
        y_test_bin = label_binarize(y_test, classes=range(10))
        y_pred_proba = model.predict(x_test)
        
        # Calculate ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(10):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f'Digit {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves (One-vs-Rest)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 4. Precision-Recall Curve
        plt.subplot(2, 3, 4)
        for i in range(10):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            plt.plot(recall, precision, label=f'Digit {i}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # 5. Model Confidence Distribution
        plt.subplot(2, 3, 5)
        confidences = np.max(y_pred_proba, axis=1)
        plt.hist(confidences, bins=50)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Number of Samples')
        plt.title('Model Confidence Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes))
        
    except:
        print("Fetching MNIST data...")
        x_train, y_train, x_test, y_test = load_mnist_data()
        print("Training model...")
        model, history = train_mnist_model(x_train, y_train, x_test, y_test)
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        
        plt.tight_layout()
        plt.show()
        
        print("Saving model...")
        model.save('model.sav')

    print("Launching OpenCV...")
    run_opencv_loop(model)

if __name__ == '__main__':
    main()