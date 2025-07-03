import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image, ImageDraw, ImageFilter
import os
import random

class SimpleFruitClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.class_names = ['apple', 'banana', 'orange']
        
    def create_sample_data(self, n_samples_per_class=100):
        """Create synthetic fruit data using color features"""
        print("Creating sample fruit data...")
        
        # Define color characteristics for each fruit
        fruit_colors = {
            'apple': {'r': (150, 255), 'g': (0, 100), 'b': (0, 100)},      # Red
            'banana': {'r': (200, 255), 'g': (200, 255), 'b': (0, 100)},   # Yellow
            'orange': {'r': (200, 255), 'g': (100, 200), 'b': (0, 50)}     # Orange
        }
        
        X = []
        y = []
        
        for class_idx, fruit in enumerate(self.class_names):
            colors = fruit_colors[fruit]
            
            for i in range(n_samples_per_class):
                # Generate random color values within fruit's range
                r = random.randint(*colors['r'])
                g = random.randint(*colors['g'])
                b = random.randint(*colors['b'])
                
                # Add some noise to make it more realistic
                r += random.randint(-20, 20)
                g += random.randint(-20, 20)
                b += random.randint(-20, 20)
                
                # Clamp values to 0-255
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                
                # Create feature vector (RGB values + some derived features)
                features = [
                    r, g, b,                           # RGB values
                    r + g + b,                         # Brightness
                    max(r, g, b) - min(r, g, b),      # Color contrast
                    r / (g + 1),                       # Red/Green ratio
                    (r + b) / (g + 1),                # Red+Blue/Green ratio
                    abs(r - g),                        # Red-Green difference
                    abs(g - b),                        # Green-Blue difference
                    abs(r - b)                         # Red-Blue difference
                ]
                
                X.append(features)
                y.append(class_idx)
        
        return np.array(X), np.array(y)
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        print("Training model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Predictions for detailed evaluation
        y_pred = self.model.predict(X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return X_train, X_test, y_train, y_test
    
    def feature_importance(self):
        """Show feature importance"""
        feature_names = [
            'Red', 'Green', 'Blue', 'Brightness', 'Contrast',
            'R/G Ratio', '(R+B)/G Ratio', 'R-G Diff', 'G-B Diff', 'R-B Diff'
        ]
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("\nFeature Importance Rankings:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    def create_synthetic_image(self, fruit_type, size=(100, 100)):
        """Create a synthetic fruit image for testing"""
        fruit_colors = {
            'apple': (220, 50, 50),      # Red
            'banana': (255, 255, 80),    # Yellow
            'orange': (255, 165, 0)      # Orange
        }
        
        # Create image with base color
        img = Image.new('RGB', size, fruit_colors[fruit_type])
        draw = ImageDraw.Draw(img)
        
        # Add some variation - draw a circle with slightly different color
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = min(size) // 3
        
        # Slightly darker color for the circle
        base_color = fruit_colors[fruit_type]
        circle_color = tuple(max(0, c - 30) for c in base_color)
        
        draw.ellipse([center_x - radius, center_y - radius, 
                     center_x + radius, center_y + radius], 
                    fill=circle_color)
        
        # Add some noise by applying a filter
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return img
    
    def extract_features_from_image(self, img):
        """Extract color features from an image"""
        # Convert to numpy array
        img_array = np.array(img)
        
        # Calculate average RGB values
        r = np.mean(img_array[:, :, 0])
        g = np.mean(img_array[:, :, 1])
        b = np.mean(img_array[:, :, 2])
        
        # Create feature vector (same as training)
        features = [
            r, g, b,                           # RGB values
            r + g + b,                         # Brightness
            max(r, g, b) - min(r, g, b),      # Color contrast
            r / (g + 1),                       # Red/Green ratio
            (r + b) / (g + 1),                # Red+Blue/Green ratio
            abs(r - g),                        # Red-Green difference
            abs(g - b),                        # Green-Blue difference
            abs(r - b)                         # Red-Blue difference
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_image(self, img):
        """Predict fruit type from image"""
        features = self.extract_features_from_image(img)
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        predicted_fruit = self.class_names[prediction]
        confidence = probabilities[prediction]
        
        return predicted_fruit, confidence, probabilities
    
    def test_predictions(self):
        """Test predictions on synthetic images"""
        print("\nTesting predictions on synthetic images...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, fruit in enumerate(self.class_names):
            # Create synthetic image
            img = self.create_synthetic_image(fruit)
            
            # Make prediction
            predicted_fruit, confidence, probabilities = self.predict_image(img)
            
            # Display results
            axes[i].imshow(img)
            axes[i].set_title(f'Actual: {fruit}\nPredicted: {predicted_fruit}\nConfidence: {confidence:.2f}')
            axes[i].axis('off')
            
            # Print detailed results
            print(f"\n{fruit.upper()} image:")
            print(f"Predicted: {predicted_fruit} (confidence: {confidence:.3f})")
            print("All probabilities:")
            for j, class_name in enumerate(self.class_names):
                print(f"  {class_name}: {probabilities[j]:.3f}")
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the simple fruit classifier"""
    print("Simple Fruit Classifier (No TensorFlow Required)")
    print("=" * 50)
    
    # Initialize classifier
    classifier = SimpleFruitClassifier()
    
    # Create sample data
    X, y = classifier.create_sample_data(n_samples_per_class=200)
    print(f"Created dataset with {len(X)} samples and {len(X[0])} features")
    
    # Train model
    X_train, X_test, y_train, y_test = classifier.train_model(X, y)
    
    # Show feature importance
    classifier.feature_importance()
    
    # Test on synthetic images
    classifier.test_predictions()
    
    print("\nClassifier training completed!")
    print("\nThis simple classifier uses:")
    print("- Random Forest algorithm")
    print("- Color-based features (RGB values and derived features)")
    print("- Works without TensorFlow or deep learning")
    print("\nTo use with real images:")
    print("1. Load images using PIL: img = Image.open('path/to/image.jpg')")
    print("2. Use classifier.predict_image(img) to get predictions")

if __name__ == "__main__":
    main()
