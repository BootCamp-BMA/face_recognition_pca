import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        covariance_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
        self.components_ = eigenvectors
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / eigenvalues.sum()

    def transform(self, X):
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def load_images(subject_dir):
    images = []
    labels = []
    for folder in os.listdir(subject_dir):
        subject_path = os.path.join(subject_dir, folder)
        if os.path.isdir(subject_path):
            for img_file in os.listdir(subject_path):
                if img_file.endswith('.pgm') :
                    img_path = os.path.join(subject_path, img_file)
                    image = Image.open(img_path)
                    image = np.array(image)
                    images.append(image)
                    labels.append(int(folder[1:]))
    return np.array(images), np.array(labels)

def plot_pca_scatter(X, labels, pca, show_plot=True, save_image=False, image_path="pca_scatter.png"):
    X_pca = pca.transform(X)
    X_pca_2d = X_pca[:, :2]
    
    if show_plot:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Label (Subject)')
        plt.title('PCA Scatter Plot of Training Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
    
    if save_image:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter, label='Label (Subject)')
        plt.title('PCA Scatter Plot of Training Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.savefig(image_path)

def plot_sample_images(images, labels, num_samples=5, show_plot=True, save_image=False, image_path="sample_images.png"):
    image_height=112
    
    image_width=92
    if show_plot:
        plt.figure(figsize=(10, 10))
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].reshape(image_height, image_width), cmap='gray')
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.show()
    
    if save_image:
        plt.figure(figsize=(10, 10))
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i].reshape(image_height, image_width), cmap='gray')
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.savefig(image_path)

def plot_eigenface_with_test_image(test_image_path, eigenface_vector, predicted_label, true_label, nearest_distance, threshold, show_plot=True, save_image=False, image_path="eigenface.png"):
    image_height=112
    
    image_width=92
    test_image = Image.open(test_image_path)
    test_image = np.array(test_image).reshape(image_height, image_width)
    eigenface_image = eigenface_vector.reshape(image_height, image_width)
    
    if show_plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image, cmap='gray')
        plt.title(f"Test Image\nTrue: {true_label}", fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(eigenface_image, cmap='gray')
        info_text = f"Pred: {predicted_label}\nDist: {nearest_distance:.2f}, Threshold: {threshold:.2f}"
        plt.title(f"Reconstructed Eigenface\n{info_text}", fontsize=12)
        plt.axis('off')
        plt.show()
    
    if save_image:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(test_image, cmap='gray')
        plt.title(f"Test Image\nTrue: {true_label}", fontsize=12)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(eigenface_image, cmap='gray')
        info_text = f"Pred: {predicted_label}\nDist: {nearest_distance:.2f}, Threshold: {threshold:.2f}"
        plt.title(f"Reconstructed Eigenface\n{info_text}", fontsize=12)
        plt.axis('off')
        plt.savefig(image_path)

def reconstruct_eigenface(label, pca, knn, X_train_pca, labels):
    label_indices = np.where(labels == label)[0]
    if len(label_indices) == 0:
        return None

    sample_pca = X_train_pca[label_indices[0]]
    reconstructed_image = np.dot(sample_pca, pca.components_.T) + pca.mean_

    return reconstructed_image

def calculate_dynamic_threshold(X_train_pca, knn, percentile=95, min_threshold=0.1):
    distances, _ = knn.kneighbors(X_train_pca)
    nearest_distances = distances[:, 0]
    threshold = np.percentile(nearest_distances, percentile)
    threshold = max(threshold, min_threshold)
    return threshold

def predict_from_image_with_threshold(image_path, knn, pca, known_labels, true_label=None, threshold=None, show_plot=True, save_image=False):
    image_height=112
    
    image_width=92
    image = Image.open(image_path)
    image = np.array(image).reshape(1, image_height * image_width) / 255.0
    image_pca = pca.transform(image)
    
    distances, _ = knn.kneighbors(image_pca, n_neighbors=1)
    nearest_distance = distances[0][0]
    
    if threshold is not None:
        if nearest_distance > threshold:
            return "unknown"
    
    predicted_label = knn.predict(image_pca)[0]
    
    if predicted_label not in known_labels:
        return "unknown"
    
    reconstructed_eigenface = reconstruct_eigenface(predicted_label, pca, knn, X_train_pca, y_train)
    if reconstructed_eigenface is not None:
        if true_label is not None:
            plot_eigenface_with_test_image(image_path, reconstructed_eigenface, predicted_label, true_label, nearest_distance, threshold, show_plot, save_image)
        else:
            plot_eigenface_with_test_image(image_path, reconstructed_eigenface, predicted_label, "unknown", nearest_distance, threshold, show_plot, save_image)
    
    return predicted_label

def get_known_labels():
    return set(y_train)

def test_prediction(image_path, knn, pca, threshold=None, true_label=None, show_plot=True, save_image=False):
    known_labels = get_known_labels()
    predicted_label = predict_from_image_with_threshold(image_path, knn, pca, known_labels, true_label=true_label, threshold=threshold, show_plot=show_plot, save_image=save_image)
    print(f"Predicted label for the image: {predicted_label}")
    print('------------------------------------------------------------------')
