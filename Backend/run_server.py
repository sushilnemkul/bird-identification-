"""
Bird Identification System Using CNN (From Scratch)
====================================================
A complete implementation of bird species classification using a custom CNN
trained from scratch without any pretrained weights or transfer learning.

Dataset: Kaggle Bird Species Dataset
Framework: TensorFlow/Keras
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the bird classification system"""
    
    # Dataset paths (adjust based on your Kaggle dataset structure)
    DATASET_PATH = 'C:\Users\Legion\Desktop\project66\dataset\BirdIdentification'  # Update this path
    TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
    VALID_DIR = os.path.join(DATASET_PATH, 'valid')
    TEST_DIR = os.path.join(DATASET_PATH, 'test')
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Model saving
    MODEL_SAVE_PATH = 'models/bird_classifier_scratch.keras'
    CHECKPOINT_PATH = 'models/checkpoints/best_model.keras'
    
    # Results
    RESULTS_DIR = 'results'
    PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')


# ============================================================================
# 1. DATASET PREPARATION AND ANALYSIS
# ============================================================================

class DatasetAnalyzer:
    """Analyze and prepare the bird dataset"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.class_distribution = {}
        
    def analyze_dataset(self):
        """Perform comprehensive dataset analysis"""
        print("=" * 70)
        print("DATASET ANALYSIS")
        print("=" * 70)
        
        # Analyze each split
        for split in ['train', 'valid', 'test']:
            split_path = os.path.join(self.dataset_path, split)
            if os.path.exists(split_path):
                print(f"\n{split.upper()} SET:")
                self._analyze_split(split_path, split)
        
        return self.class_distribution
    
    def _analyze_split(self, split_path, split_name):
        """Analyze a single dataset split"""
        classes = sorted(os.listdir(split_path))
        classes = [c for c in classes if os.path.isdir(os.path.join(split_path, c))]
        
        class_counts = {}
        total_images = 0
        image_sizes = []
        
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            class_counts[cls] = count
            total_images += count
            
            # Sample image size analysis (check first 5 images)
            for img_file in images[:5]:
                img_path = os.path.join(cls_path, img_file)
                try:
                    img = tf.io.read_file(img_path)
                    img = tf.image.decode_jpeg(img, channels=3)
                    image_sizes.append(img.shape[:2])
                except:
                    continue
        
        self.class_distribution[split_name] = class_counts
        
        print(f"  Total classes: {len(classes)}")
        print(f"  Total images: {total_images}")
        print(f"  Average images per class: {total_images/len(classes):.1f}")
        print(f"  Min images per class: {min(class_counts.values())}")
        print(f"  Max images per class: {max(class_counts.values())}")
        
        if image_sizes:
            unique_sizes = set(image_sizes)
            print(f"  Unique image resolutions: {len(unique_sizes)}")
            print(f"  Sample resolutions: {list(unique_sizes)[:3]}")
        
        # Identify class imbalance
        mean_count = np.mean(list(class_counts.values()))
        std_count = np.std(list(class_counts.values()))
        imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
        
        print(f"  Class imbalance ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2.0:
            print("  ⚠️  WARNING: Significant class imbalance detected!")
    
    def plot_class_distribution(self, save_path=None):
        """Visualize class distribution"""
        if 'train' not in self.class_distribution:
            print("No training data to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, split in enumerate(['train', 'valid', 'test']):
            if split in self.class_distribution:
                data = self.class_distribution[split]
                classes = list(data.keys())[:20]  # Top 20 classes
                counts = [data[c] for c in classes]
                
                axes[idx].bar(range(len(classes)), counts)
                axes[idx].set_title(f'{split.upper()} Set Distribution')
                axes[idx].set_xlabel('Class Index')
                axes[idx].set_ylabel('Number of Images')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# 2. DATA GENERATORS WITH AUGMENTATION
# ============================================================================

def create_data_generators(config):
    """Create training, validation, and test data generators"""
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    # Validation and test data (only rescaling)
    valid_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    valid_generator = valid_test_datagen.flow_from_directory(
        config.VALID_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = valid_test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\n" + "=" * 70)
    print("DATA GENERATORS CREATED")
    print("=" * 70)
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {valid_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    print(f"Number of classes: {len(train_generator.class_indices)}")
    print(f"Class labels: {list(train_generator.class_indices.keys())[:10]}...")
    
    return train_generator, valid_generator, test_generator


# ============================================================================
# 3. CUSTOM CNN ARCHITECTURE (FROM SCRATCH)
# ============================================================================

def build_custom_cnn(input_shape, num_classes):
    """
    Build a custom CNN architecture from scratch
    
    Architecture:
    - 4 Convolutional blocks with increasing filters
    - Batch normalization for stability
    - MaxPooling for dimensionality reduction
    - Dropout for regularization
    - Dense layers for classification
    """
    
    model = models.Sequential(name='BirdCNN_FromScratch')
    
    # Block 1: Initial feature extraction
    model.add(layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2: Deeper features
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3: Complex patterns
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Block 4: High-level features
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))
    
    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

class BirdClassifierTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
        
    def compile_model(self, model, learning_rate=0.001):
        """Compile the model with optimizer and loss"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        self.model = model
        
        print("\n" + "=" * 70)
        print("MODEL COMPILED")
        print("=" * 70)
        print(f"Optimizer: Adam (lr={learning_rate})")
        print(f"Loss: Categorical Crossentropy")
        print(f"Metrics: Accuracy, Top-3 Accuracy")
        
        return model
    
    def create_callbacks(self):
        """Create training callbacks"""
        os.makedirs(os.path.dirname(self.config.CHECKPOINT_PATH), exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                self.config.CHECKPOINT_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_gen, valid_gen, epochs=50):
        """Train the model"""
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        callbacks = self.create_callbacks()
        
        self.history = self.model.fit(
            train_gen,
            validation_data=valid_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        return self.history
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation metrics"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model, test_generator):
        self.model = model
        self.test_generator = test_generator
        
    def evaluate(self):
        """Comprehensive model evaluation"""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        # Basic evaluation
        test_loss, test_acc, test_top3 = self.model.evaluate(
            self.test_generator, 
            verbose=1
        )
        
        print(f"\nTest Accuracy: {test_acc*100:.2f}%")
        print(f"Test Top-3 Accuracy: {test_top3*100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Predictions
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes
        
        # Classification report
        class_names = list(self.test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        return {
            'test_accuracy': test_acc,
            'test_top3_accuracy': test_top3,
            'test_loss': test_loss,
            'predictions': predictions,
            'y_true': y_true,
            'y_pred': y_pred
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        class_names = list(self.test_generator.class_indices.keys())
        
        # Limit to top 20 classes for readability
        if len(class_names) > 20:
            print("Note: Showing confusion matrix for first 20 classes")
            class_names = class_names[:20]
            mask = (y_true < 20) & (y_pred < 20)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ============================================================================
# 6. PREDICTION MODULE
# ============================================================================

class BirdPredictor:
    """Predict bird species from images"""
    
    def __init__(self, model_path, class_indices_path=None):
        self.model = keras.models.load_model(model_path)
        self.class_names = None
        
        if class_indices_path and os.path.exists(class_indices_path):
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
                self.class_names = {v: k for k, v in class_indices.items()}
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess a single image"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_size)
        img = img / 255.0
        img = tf.expand_dims(img, 0)
        return img
    
    def predict(self, image_path, top_k=5):
        """Predict bird species with confidence scores"""
        img = self.preprocess_image(image_path)
        predictions = self.model.predict(img, verbose=0)[0]
        
        # Get top K predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.class_names[idx] if self.class_names else f"Class_{idx}"
            confidence = predictions[idx]
            results.append({
                'species': class_name,
                'confidence': float(confidence),
                'confidence_pct': f"{confidence*100:.2f}%"
            })
        
        return results
    
    def predict_and_display(self, image_path, top_k=3):
        """Predict and display results"""
        results = self.predict(image_path, top_k)
        
        # Display image
        img = plt.imread(image_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Top Prediction: {results[0]['species']} ({results[0]['confidence_pct']})")
        
        # Print results
        print("\n" + "=" * 50)
        print("PREDICTION RESULTS")
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['species']}: {result['confidence_pct']}")
        
        plt.show()
        return results


# ============================================================================
# 7. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Configuration
    config = Config()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    print("=" * 70)
    print("BIRD IDENTIFICATION SYSTEM - CNN FROM SCRATCH")
    print("=" * 70)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Step 1: Analyze dataset
    analyzer = DatasetAnalyzer(config.DATASET_PATH)
    class_dist = analyzer.analyze_dataset()
    analyzer.plot_class_distribution(
        save_path=os.path.join(config.PLOTS_DIR, 'class_distribution.png')
    )
    
    # Step 2: Create data generators
    train_gen, valid_gen, test_gen = create_data_generators(config)
    
    # Save class indices for later use
    class_indices_path = os.path.join(config.RESULTS_DIR, 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    
    # Step 3: Build model
    input_shape = (config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS)
    num_classes = len(train_gen.class_indices)
    
    model = build_custom_cnn(input_shape, num_classes)
    model.summary()
    
    # Step 4: Train model
    trainer = BirdClassifierTrainer(config)
    trainer.compile_model(model, learning_rate=config.LEARNING_RATE)
    history = trainer.train(train_gen, valid_gen, epochs=config.EPOCHS)
    
    # Plot training history
    trainer.plot_training_history(
        save_path=os.path.join(config.PLOTS_DIR, 'training_history.png')
    )
    
    # Step 5: Evaluate model
    evaluator = ModelEvaluator(model, test_gen)
    results = evaluator.evaluate()
    
    evaluator.plot_confusion_matrix(
        results['y_true'], 
        results['y_pred'],
        save_path=os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
    )
    
    # Step 6: Save final model
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    model.save(config.MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to: {config.MODEL_SAVE_PATH}")
    
    # Step 7: Example prediction
    print("\n" + "=" * 70)
    print("PREDICTION MODULE READY")
    print("=" * 70)
    print("To make predictions on new images:")
    print(f"""
predictor = BirdPredictor(
    model_path='{config.MODEL_SAVE_PATH}',
    class_indices_path='{class_indices_path}'
)
results = predictor.predict_and_display('path/to/bird_image.jpg')
    """)
    
    return model, history, results


if __name__ == "__main__":
    # Execute the complete pipeline
    model, history, results = main()
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nProject Limitations (As Required):")
    print("- Accuracy lower than transfer learning approaches")
    print("- Requires substantial training time and computational resources")
    print("- Performance sensitive to image quality and class imbalance")
    print("- Dataset from Kaggle - ensure proper attribution and licensing")
    print("\nEthical Considerations:")
    print("- Model trained for educational/research purposes only")
    print("- Results should be validated by domain experts")
    print("- Be aware of potential biases in the training dataset")