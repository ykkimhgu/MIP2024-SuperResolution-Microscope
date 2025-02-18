# Image Super-Resolution Model on Low-Resolution Microscopy Images for Enhancing White Blood Cell Classification

**Date**:  December 2024

**Class**: Mechatronics Integration Project

**Author**: School of Mechanical and Control Engineering, Handong Global University, Eunji Ko and Young-Keun Kim

**Github**: 

[https://github.com/eunjijuliako/Capstone.git](https://github.com/eunjijuliako/Capstone.git)

# Introduction

## 1. Objective

This research aims to improve classification performance by leveraging existing equipment and generating high-resolution images through deep learning-based super-resolution techniques. A quantitative analysis will be conducted to compare the classification accuracy of models with and without super-resolution application, evaluating the impact of high-quality image data on classification tasks. Additionally, the study seeks to compare and analyze optimal models that can achieve higher accuracy than existing research to improve performance. This approach explores the potential of distinguishing various types of white blood cells and contributing to deep learning-based immune system diagnostics, even in environments where acquiring high-resolution images is challenging.

## 2. Preparation

### Software

| **GPU Model** | NVIDIA A30 |
| --- | --- |
| **Server** | Handong University Mechanical Control Engineering Department Server |
| **Framework** | Pytorch |
| **Environment** | Python using anaconda |

### Hardware

| **Laptop Model** | ASUS Zenbook 14 OLED (UX3405) |
| --- | --- |
| **Operating System** | Windows 11 Home/Pro |
| **Processor** | Intel® Core™ Ultra 7-155H (1.4 GHz base, up to 4.8 GHz, 16 cores, 22 threads) or Ultra 9-185H (2.3 GHz base, up to 5.1 GHz, 16 cores, 22 threads) |
| **Graphics** | Intel® Arc™ Graphics |
| **Memory** | 32GB LPDDR5X (onboard) |

### Dataset

- Case Study 1: Four Types of White Blood Cell
    
    
    | **Class** | **Eosinophil** | **Lymphocyte** | **Monocyte** | **Neutrophil** |
    | --- | --- | --- | --- | --- |
    | **Explanation** | (3% in blood) First to respond to bacteria or a virus | (30% in blood) Known for their role in asthma | (6% in blood) Known for their role in allergy symptoms | (60% in blood) Fight infections by producing antibodies |
    | **Classification** | Train (2497), Test (623) | Train (2483), Test (620) | Train (2478), Test (620) | Train (2499), Test (624) |
    
    | **Super Resolution** | Train: 9,957 LR&HR pair dataset without class, Test: 2,487 with class (the test dataset of super-resolution will be a training dataset for classification) |
    | --- | --- |
    
    Link: [https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset/data](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data)
    
- Case Study 2: Five Types of White Blood Cell
    
    
    | **Class** | **Eosinophil** | **Lymphocyte** | **Monocyte** | **Neutrophil** | **Basophil** |
    | --- | --- | --- | --- | --- | --- |
    | **Explanation** | (3% in blood) First to respond to bacteria or a virus | (30% in blood) Known for their role in asthma | (6% in blood) Known for their role in allergy symptoms | (60% in blood) Fight infections by producing antibodies | (1% in blood) Clean up dead cells |
    | **Classification** | Train (787), Test (197) | Train (2128), Test (532) | Train (187), Test (47) | Train (76), Test (20) | Train (71), Test (18) |
    
    | **Super Resolution** | Train: 10,175 LR&HR pair dataset without class, Test: 4063 with class (the test dataset of super-resolution will be a training dataset for classification) |
    | --- | --- |
    
    Link: https://www.kaggle.com/datasets/masoudnickparvar/white-blood-cells-dataset/data
  
# Algorithm

## 1. Overview

### **Part 1: Super Resolution**

![**Figure 1. Part 1 Model Design**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%201.%20Part%201%20Model%20Design.png?raw=true)

**Figure 1. Part 1 Model Design**

### **Part 2: Classification**

![**Figure 2. Part 2 Model Design**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%202.%20Part%202%20Model%20Design.png?raw=true)

**Figure 2. Part 2 Model Design**

## 2. Procedure

### **Part 0. Pre-Processing**

1. **Generate  Raw Images (Low-Resolution Images)**
    - Since the dataset already contains high-resolution (HR) images, low-resolution (LR) images are generated for training and testing the deep learning-based super-resolution model.
    - The script `Part0.Pre_Processing/degradation_folder_for_super_resolution.py` is used to apply degradation to the HR images, producing degraded LR images.
2. **Split Dataset into Train and Test Sets**
    - The dataset is split, so 80% of the images are from the training and 20% from the testing sets.
    - The paired dataset is created, where LR and corresponding HR images are grouped. However, 20% of the HR images are excluded from training and reserved for testing.
    - This is done using the script `Part0.Pre_Processing/dataset_train_test_for_classification.py`.

---

### **Part 1&2: Super-Resolution and Classification**

The workflow varies depending on the dataset type: **Ground Truth**, **Bicubic**, **Raw Images**, **SRGAN**, or **Real-ESRGAN**. Below is the process for **Case Study 1** (similar steps apply to Case Study 2, except for resolution differences).

---

### **Case Study 1**

### **Dataset Preparation**

1. **Raw Images**
    - 20% of the raw LR images are reserved for classification.
    - The remaining 80% of the raw LR images are used for classification model training.
        
        ![**Figure 3. Raw-Images Prodecure**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%203.%20Raw-Images%20Prodecure.png?raw=true)
        
        **Figure 3. Raw-Images Prodecure**
        
2. **Method 1: Bicubic Interpolation**
    - Bicubic interpolation is applied to 20% of the raw LR images.
    - The output images are used for classification: 80% for training and 20% for testing.
    - Script: `Part1.Super_Resolution/Bicubic/Bicubic.py`.
        
        ![**Figure 4. Bicubic Prodecure**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%204.%20Bicubic%20Prodecure.png?raw=true)
        
        **Figure 4. Bicubic Prodecure**
        
3. **Method 2: SRGAN**
    - **Train the SRGAN Model:**
        - The paired dataset (80% training set) trains the SRGAN model.
        - Training script: `Part1.Super_Resolution/SRGAN/train.py`.
    - **Test the SRGAN Model:**
        - The trained model is tested using the 20% testing set.
        - The resulting restored HR images are saved in the SRGAN output folder.
        - Testing script: `Part1.Super_Resolution/SRGAN/test.py`.
4. **Method 3: Real-ESRGAN**
    - **Train the Real-ESRGAN Model:**
        - Custom modifications are made to the configuration (`finetune_realesrgan_x4plus_pairdata.yml`) and scripts to support the dataset.
        - Training script: `Part1.Super_Resolution/Real-ESRGAN/train.py`.
    - **Test the Real-ESRGAN Model:**
        - The model is tested using the 20% testing set.
        - The output HR images are saved in the Real-ESRGAN output folder.
        - Testing script: `Part1.Super_Resolution/Real-ESRGAN/test.py`.
        
        ![**Figure 5. SRGAN Real-ESRGAN Prodecure**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%205.%20SRGAN%20Real-ESRGAN%20Prodecure.png?raw=true)
        
        **Figure 5. SRGAN Real-ESRGAN Prodecure**
        
5. **Ground Truth**
    - 20% of the Ground Truth images are reserved for classification.
    - 80% of the reserved Ground Truth images are used for training, and 20% are reserved for testing.
        
        ![**Figure 6. Ground-Truth Images Prodecure**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%206.%20Ground-Truth%20Images%20Prodecure.png?raw=true)
        
        **Figure 6. Ground-Truth Images Prodecure**
        

---

### **Image Quality Evaluation**

For **Raw Images**, **Method 1 (Bicubic)**, **Method 2 (SRGAN)**, **Method 3 (Real-ESRGAN)**, and **Ground Truth**, the same image quality evaluation is conducted using PSNR and SSIM metrics.

- Script: `Part1.Super_Resolution/PSNR_SSIM.py`.

---

### **Classification**

For all datasets (Raw Images, Bicubic, SRGAN, Real-ESRGAN, Ground Truth), classification performance is evaluated by training the model on 80% of the dataset and testing it on the remaining 20%.

- Classification Script: `Part2.Classification/ResNet50nAdaptivePooling.py`.

# Result and Discussion

## **Part 1: Super Resolution**

### **Results**

- Case Study 1
    
    ![**Figure 7. Case Study 1 Super Resolution Results**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%207.%20Case%201%20Super%20Resolution%20Results.png?raw=true)
    
    **Figure 7. Case Study 1 Super Resolution Results**
    
    ![**Figure 8. Part 1 Case Study 1 Results**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%208.%20Part%201%20Case%20Study%201%20Results.png?raw=true)
    
    **Figure 8. Part 1 Case Study 1 Results**
    
- Case Study 2
    
    ![**Figure 9. Case 2 Super Resolution Results**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%209.%20Case%202%20Super%20Resolution%20Results.png?raw=true)
    
    **Figure 9. Case 2 Super Resolution Results**
    
    ![**Figure 10. Part 1 Case Study 2 Results**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%2010.%20Part%201%20Case%20Study%202%20Results.png?raw=true)
    
    **Figure 10. Part 1 Case Study 2 Results**
    

### Discussion

- Bicubic
    
    Using Bicubic interpolation to transform low-resolution images into high-resolution ones showed results that were visually hard to distinguish from the original low-resolution images. This is because interpolation does not create details from the original image but works by resizing it to minimize data loss. Thus, while the details were maintained when enlarging a low-resolution image with Bicubic interpolation, the results did not improve.
    
- SRGAN
    
    SRGAN is a deep learning model that introduces a new loss function to create fine textures and high-resolution images closer to the human visual experience. High-resolution images generated by SRGAN were cleaner than those produced by Bicubic but still contained noise. This noise often made the texture appear rough, unlike the smooth edges of the original image. Additionally, SRGAN images showed more precise details, but the noise also caused some features like cell boundaries and internal structures to appear faint.
    
- Real-ESRGAN
    
    Real-ESRGAN is an advanced model that improves on SRGAN's limitations, creating more detailed images using complex degradation techniques to generate high-resolution images similar to actual low-resolution images. Images produced by Real-ESRGAN were cleaner and had better detail restoration than other models. Compared to SRGAN, black spots and blurred white areas were more effectively restored, and the boundaries of cells were depicted as smoother and cleaner than in the original. It enabled more precise visualization of cell structures; however, distortions could occur when distinguishing cell size or type, so comparisons with the original image were necessary.
    
- **Overall**
    - **PSNR**
        
        In Case 1, SRGAN achieved the highest PSNR value at 28.1809 dB, while the low-resolution original had the lowest at 12.1323 dB. In Case 2, Real-ESRGAN reached the highest PSNR value at 30.5038 dB, with Bicubic being the lowest at 12.4899 dB. In Case 2, the PSNR value of Real-ESRGAN exceeded 30 dB, indicating a minor pixel difference from the high-resolution original image. On the other hand, the low-resolution original and Bicubic showed the most significant pixel differences from the high-resolution original.
        
    - **SSIM**
        
        Real-ESRGAN's SSIM values were significantly higher than those of other models, showing 0.9190 in Case 1 and 0.9490 in Case 2. The high-resolution images generated by Real-ESRGAN showed the most significant similarity to the high-resolution original regarding luminance, contrast, and structure. In visual comparisons of the five cases, images generated by Real-ESRGAN were evaluated as having the most similar appearance to the high-resolution original.
        

## **Part 2: Classification**

### **Results**

- Case Study 1
    
    ![**Figure 11. Part 2 Case Study 1 Results**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%2011.%20Part%202%20Case%20Study%201%20Results.png?raw=true)
    
    **Figure 11. Part 2 Case Study 1 Results**
    
- Case Study 2
    
    ![**Figure 12. Part 2 Case Study 2 Results**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%2012.%20Part%202%20Case%20Study%202%20Results.png?raw=true)
    
    **Figure 12. Part 2 Case Study 2 Results**
    

### Discussion

The Real-ESRGAN model demonstrated impressive performance, but further adjustments are required to address consistency and misclassification issues. In the first case study, the model achieved an accuracy of 97.19%, a recall of 97%, a precision of 97%, and an F1 score of 97. The second case study reported an accuracy of 95.33%, a recall of 90.19%, a precision of 88.20%, and an F1 score of 89.18, making it the highest-performing among the five cases.
However, recurring training of the model revealed a variation in performance, with an approximate 5% difference in accuracy observed. This fluctuation can be attributed to dataset diversity, random initialization, and hyperparameter configurations. Additionally, an analysis of the confusion matrix for Real-ESRGAN highlighted significant misclassifications: Eosinophils were misclassified as Neutrophils 15% of the time, and Monocytes were misclassified as Lymphocytes 13% of the time. This indicates that further data analysis is needed, and upon examining the microscopy images of the classes, it was inferred that the similarity in cell morphology contributed to these misclassifications.

![**Figure 13. Confusion Matrix and the Cell Similarity**](https://github.com/eunjijuliako/Capstone/blob/main/Images/Figure%2013.%20Confusion%20Matrix%20and%20the%20Cell%20Similarity.png.png?raw=true)

**Figure 13. Confusion Matrix and the Cell Similarity**

# Conclusion

**Achievement**

This study demonstrated the potential of using the deep learning-based super-resolution model, Real-ESRGAN, to enhance the quality of white blood cell images, combined with a ResNet50 and Adaptive Pooling classification model to improve classification accuracy. This approach shows that deep learning can enhance image quality in environments where obtaining high-resolution images is difficult, improving the performance of white blood cell type differentiation and suggesting a potential alternative to microscopy or automated diagnostic devices for immune system diagnostics.

**Comparison with Related Work**

In Part 1, an image quality analysis showed that Real-ESRGAN restored high-resolution images that were clearer and better preserved fine cellular features compared to those restored using the SRGAN model from previous studies. Additionally, image quality evaluation metrics such as PSNR and SSIM indicated superior performance for Real-ESRGAN.

In Part 2, regarding classification performance, prior studies reported an accuracy of 94.87%, whereas this study achieved a peak accuracy of 97.19%. When comparing cases using low-resolution original images without super-resolution techniques to those enhanced with Real-ESRGAN, a 35.21% improvement in classification accuracy was observed. These findings suggest that Real-ESRGAN can significantly improve image quality and that appropriate classification models enhance performance.

**Limitation and Future Work**

This study has three main limitations and areas for future research:

First, the study generated low-resolution images using the degradation methods presented in the Real-ESRGAN paper, starting from high-resolution originals. These methods included noise, blur, and compression treatments. However, further comparison with real-world low-resolution images from environments like developing countries, where obtaining high-resolution images is challenging, is necessary. It would help validate the utility of the applied degradation process.

Second, in Part 2's analysis of classification performance, performance variations of about 5% were observed due to model initialization and hyperparameter settings. Ensuring consistent classification performance is crucial for commercialization and is particularly important in medical imaging. Therefore, future work should optimize model initialization and fine-tuning hyperparameters to reduce variability.

Third, the acquisition of diverse datasets is needed. The datasets used in this study consisted of approximately 10,000 to 15,000 images per case for training and testing. However, as cells in the human body can have subtle differences, distinguishing between cell types can be difficult. Obtaining a greater variety and number of datasets is necessary to improve the model's ability to learn more variables and perform more accurately.

# References

[1] D. C. Lepcha, B. Goyal, A. Dogra, and V. Goyal, "Image super-resolution: A comprehensive review,
recent trends, challenges, and applications," Information Fusion, vol. 91, pp. 230–260, 2023.

[2] S. Han, S. I. Hwang, and H. J. Lee, "The classification of renal cancer in 3-phase CT images using
a deep learning method," J Digit Imaging, vol. 32, no. 4, pp. 638–643, 2019.

[3] J. Ferdousi, et al., "A deep learning approach for white blood cells image generation and
classification using SRGAN and VGG19," Telematics and Informatics Reports, vol. 16, p. 100163,
2024.

[4] Y. R. Seo and S. J. Kang, "Current status and recent trends in deep learning-based super resolution
technology," Broadcasting and Media, vol. 25, no. 2, pp. 7–16, 2020.

[5] M. S. Yoon, M. S. Chae, M. K. Lee, and S. Y. Hong, "Case study on the management of medical
devices in official development assistance (ODA): Consulting for the Tanzania biomedical engineering
education program," Journal of Health and Medical Industry, vol. 11, no. 3, pp. 129–144, 2017.

[6] C. Dong, C. C. Loy, K. He, and X. Tang, "Image super-resolution using deep convolutional
networks," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 38, no. 2, pp. 295–
307, 2016.

[7] B. Lim, S. Son, H. Kim, S. Nah, and K. M. Lee, "Enhanced deep residual networks for single image
super-resolution," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
Workshops, pp. 136–144, 2017.

[8] X. Wang, K. Yu, S. Wu, J. Gu, Y. Liu, C. Dong, et al., "Esrgan: Enhanced super-resolution generative
adversarial networks," in Proceedings of the European Conference on Computer Vision (ECCV)
Workshops, pp. 0–0, 2018.
