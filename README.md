# Autonomous Vision Systems for Self-Driving Cars

A project focused on designing and optimizing object detection pipelines to enhance the capabilities of autonomous vehicles. The project evaluates and implements state-of-the-art machine learning techniques to achieve reliable object detection in real-world scenarios.

## Results (Using YOLO) 🌟
[![Demo](http://img.youtube.com/vi/Zljh-bciS-Q/0.jpg)](http://www.youtube.com/watch?v=Zljh-bciS-Q)

## Analysis 🧠
Here's a [presentation](https://docs.google.com/presentation/d/1UXpk47FWr--odaWwSold6jPDFvrQtz1Fob_nV4R4UMA/edit?usp=sharing) that I made to explore my thought process.

### **1. Sliding Windows Technique**
- **Method**: The image feed is divided into smaller boxes (sliding windows), and each box is classified independently.  
- **Drawbacks**: Computationally expensive and less efficient for real-time systems.

### **2. Perceptron Model**
- **Architecture**: A neural network classifier that consists of the following layers:
<div style="overflow-x:auto;">
<table style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr>
      <th style="border: 1px solid #ddd; padding: 8px;">Layer</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Input Shape</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Output Shape</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">1. Flatten</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Flatten</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(32, 32, 3)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(3072)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Converts 3D image data into a 1D array.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">2. Dense (Hidden)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fully Connected</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(3072)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(128)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Learns features using non-linearity.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">3. Dense (Output)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fully Connected</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(128)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(3)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Outputs probabilities for 3 classes.</td>
    </tr>
  </tbody>
</table>
</div>

- **Observation**: 50% accuracy demonstrates the limitations of basic models in handling complex object detection tasks.

### **3. Convolutional Neural Network (CNN) Model**
- **Architecture**: A custom CNN model with layers optimized for feature extraction and classification:
<div style="overflow-x:auto;">
<table style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr>
      <th style="border: 1px solid #ddd; padding: 8px;">Layer</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Input Shape</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Output Shape</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">1. Conv2D</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Convolutional</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(32, 32, 3)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(30, 30, 64)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Extracts spatial features using a 3x3 filter with 64 channels.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">2. Activation</td>
      <td style="border: 1px solid #ddd; padding: 8px;">ReLU</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(30, 30, 64)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(30, 30, 64)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Applies non-linearity to introduce activation.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">3. MaxPooling2D</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Pooling</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(30, 30, 64)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(15, 15, 64)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Downsamples feature maps using a 2x2 pooling window.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">4. Flatten</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Flatten</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(15, 15, 64)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(14400)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Flattens the 3D feature maps into a 1D array.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">5. Dense (Hidden)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fully Connected</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(14400)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(128)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Learns high-level features using non-linearity.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">6. Dense (Output)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fully Connected</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(128)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(3)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Outputs probabilities for 3 classes.</td>
    </tr>
  </tbody>
</table>
</div>

- **Observation**: Significant improvement (71% accuracy) over the perceptron model, showcasing the power of convolutional layers in spatial data analysis.

### **4. Transfer Learning with VGG16**
- **Architecture**: Transfer learning using the VGG16 model pre-trained on ImageNet:
<div style="overflow-x:auto;">
<table style="border-collapse: collapse; width: 100%; text-align: left;">
  <thead>
    <tr>
      <th style="border: 1px solid #ddd; padding: 8px;">Layer</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Type</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Input Shape</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Output Shape</th>
      <th style="border: 1px solid #ddd; padding: 8px;">Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">1. VGG Expert</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Pre-trained Feature Extractor</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(224, 224, 3)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Feature Maps</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Extracts rich features using pre-trained VGG model.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">2. GlobalAveragePooling2D</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Pooling</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Feature Maps</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(Channels)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Reduces spatial dimensions to a single value per channel.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">3. Dense (Hidden)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fully Connected</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(Channels)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(1024)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Learns high-level features using non-linearity.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">4. Dropout</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Regularization</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(1024)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(1024)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Reduces overfitting by randomly dropping neurons (30%).</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">5. Dense (Hidden)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fully Connected</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(1024)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(512)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Learns further refined features using non-linearity.</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">6. Dropout</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Regularization</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(512)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(512)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Reduces overfitting by randomly dropping neurons (30%).</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 8px;">7. Dense (Output)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Fully Connected</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(512)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">(3)</td>
      <td style="border: 1px solid #ddd; padding: 8px;">Outputs probabilities for 3 classes.</td>
    </tr>
  </tbody>
</table>
</div>

- **Observation**: Leveraging pre-trained models leads to higher accuracy (85%) and reduced training time for complex datasets.

### **5. YOLO (You Only Look Once)**
- **Architecture**: YOLO model processes the entire image using the DarkNet architecture:
![image](https://github.com/user-attachments/assets/ca8732b8-ed3e-4306-be52-009a6d0ad412)

- **Observation**: Combines speed and accuracy, ideal for real-time video feed processing by analyzing frames without relying on sliding windows.


## Technologies Used 🛠️

- **Programming Language**: Python  
- **Framework**: TensorFlow, Keras  
- **Models**: VGG16, YOLO, custom CNNs  
- **Libraries**: NumPy, Pandas, Matplotlib, Pillow for data processing and visualization  

## Installation 🖥️

1. Clone the repository:
   ```bash
   git clone https://github.com/Profilist/Autonomous-Driving.git
   ```
2. Launch Jupyter Notebook
   ```bash
   jupyter notebook
   ```

## License 📜
This project is licensed under the MIT License. See the LICENSE file for details.
