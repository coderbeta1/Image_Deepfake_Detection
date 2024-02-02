# Image_Deepfake_Detection
Implementation and improvement of a research paper with a new theory for image deepfake detection with high generalization to include General Adversarial Networks, Diffusion Models and any other new generative techniques 


## Approach 1: Learn the low level cues left behind by an Image Generator
It is believed the image generators leave behind some perceptual cues (encompassing elements such as colors, textures, brightness, and low-level visual cues). The primary concept of this approach revolves around identifying those unique low-level cues left by image generators, achieved through training over diverse images. The methodology employs pretrained ResNET 50 and ResNET 18 backbones, with the addition of a new binary classification dense layer for discriminating between real and fake images. 


The implementation is carried out using both Keras for ResNET 50 and PyTorch for ResNET 18. Image augmentation is incorporated in the ResNET 50 implementation. Both models are trained on the ImageNet Dataset. Key implementation details include a batch size of 128, a learning rate of 0.00001, Stochastic Gradient Descent (SGD) as the optimizer with momentum set to 0.9, Cross Entropy Loss as the chosen loss function, and a resized resolution of 256 x 256. The training process spans 80 epochs, ensuring a comprehensive learning experience for the models.


**Flaws with this approach:**
1. It was seen that while there decent accuracy over the training dataset, when an image generated using a different image generation technique was given the classifier failed to classify with good accuracy
2. Training the model takes too much time and is not entirely accurate
3. The was a sink in the parameters learned, such that, if it was a real image classification was good but if a fake image was given from a method not used in training it was considered as real. Any image that is not generated using the method used while training would be classified as real


## Approach 2: Use a generalized backbone for feature maps and then use Classifiers
Since it was noticed in approach 1 that training a neural network over a single generation technique would make it work only for that generation method, why not stop aiming to learn a neural network with the specific task of classifying real and fake over a given generation technique and just use a pretrained backbone to get feature maps and then use classifiers like linear probes, KNN, SVMs, etc

### Section 1: Implementaion details:
For this pupose I tried, <br>
<b>I. Datasets:</b>
1. Laion vs LDM100 (1000 images each = 2000 Total)
2. ImageNet vs LDM200 (1000 images each = 2000 Total)
3. bigGAN Real vs bigGAN Fake (2000 images each = 4000 Total)

<b>II. Transformations on image before training:</b>
1. Transform 0: No Change
2. Transform 1: Adding Gaussian Blur
3. Transform 2: Adding Jitter
4. Transform 3: Adding Guassian Blur and Jitter Both

<b>III. Backbone Models:</b>
1. DINO ViT-B/16
2. DINO ResNET50
3. CLIP ViT-B/16

<b>IV. As for the classifiers I used, </b>
1. Bagging Classifier
2. Decision Tree
3. Random Forest
4. Linear Discriminant Analysis
5. Quadratic Discriminant Analysis
6. KNN (1 neighbour)
7. KNN (3 neighbour)
8. KNN (5 neighbour)
9. Linear Probe
10. Support Vector Machine
11. Gradient Boosting
12. Naive Bayes.

<b>V. I also tried 3 different variations of dimensionality reductions</b>
1. No Reduction
2. Principal Component Analysis
3. Autoencoding 

### Section 2: Findings and Inference:

#### 1. Impact of using transforms before training and after training:
![Impact_of_transformations_on_training](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/fee63dbd-857d-4682-ac91-01b0ead60ed3)

**NOTE:** Goodness Factor was calculated by subtracting min(accuracy) from all data points for each line. Each line represents a different dataset-backbone model combination. The names have been omitted for clarity

**Inference 1:** Adding No transform while training is better. But adding transform 2 also shows good results. Dealing with Gaussian Blur might cause issues in accuracy (around 0.05 on average)

**Inference 2:** On adding no transform while training but adding transformation while testing it was found that the models were robust to those transforms. This means that if the user were to do some edits like adding jitter or gaussian blur (while compressing it) to the images the models would still be very accurate

#### 2. Impact of Dimensionality Reduction:
![Impact_of_dimensionality_reduction_vs_classifiers](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/5a5f5631-2c59-4816-b028-58ec1fb8a726)

**Inference:** In many cases no dimensionality reduction is the best choice. But in some cases autoencoding performs better than no reduction or PCA

#### 3. Impact of classifier used:
![image](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/f18a4814-476c-4a3d-8dba-dd9da4f97609)

**Inference:** SVM, Linear Probing and Linear Discriminant Analysis seem to be good classifiers

#### 4. Impact of backbone model used:
![image](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/301ed72d-6030-4ee8-bb5a-7744977480bb)

**Inference:** CLIP ViT-B/16 is the best backbone over all cases, followed by DINO ResNET50, followed by DINO ViT-b/16

### Section 3: Going Beyond ...

#### After this I tried to split the load for feature transforms over multiple backbones instead of just one backbone, I made the following combinations:
1. DINO ViT-B/16 and DINO ResNET50 and CLIP ViT-B/16
2. DINO ViT-B/16 and DINO ResNET50
3. CLIP ViT-B/16 and DINO ResNET50
4. CLIP ViT-B/16 and DINO ViT-B/16

I then trained each model over the entire dataset and took the best classifier for each. I used randomised jitter (p=0.5) and guassian blur (p=0.5).
![accuracy_vs_backbone_wrt_model](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/2dceccff-d11a-4439-8def-61afe9b1a714)

**Inference:** This showed that using a combination was still not able to beat the previous best model, i.e. CLIP ViT-B/16

#### Lastly I tried to test the models across datasets (Training on dataset A and testing on dataset B). This was to check for generalization of the models

![image](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/401d8a08-e6e7-4743-9e5f-f5cae53c4cca)

### Section 4: Results and Conclusion:

#### I. Results:
1. The best accuracy achieved was with no transformation, CLIP ViT-B/16 backbone model and Linear Discriminant Analysis with no reduction as the classifier. The test accuracy was **98.1875%** with train accuracy **98.75%** when trained and tested over the combined datasets
2. On testing across datasets, the model with CLIP ViT-B/16 backbone, Support Vector Machine classifier, autoencoder for dimensionality and randomised jitter and blur for transformation gave the best results.

> i. **Best Model for GANs:** It was generalized best when trained over a bigGAN and tested over other datasets like laion, ImageNet, ldm100, ldm200. When trained and tested   over bigGAN accuracy was **98.875%**. When tested over laion and ldm100, accuracy was **79.1%**. When tested over ImageNet and ldm200, accuracy was **81.1%**.

> ii. **Best Model for Diffusion Models:** It was generalized greatly when trained over a ImageNet vs ldm200 and tested over other datasets like laion, bigGAN, ldm100 also. When trained and tested over ImageNet vs ldm299 accuracy was **98.5%**. When tested over laion and ldm100, accuracy was **94.44%**. When tested over bigGAN Real vs bigGAN Fake, accuracy was **72.1%**.

#### II. Conclusion:
1. Best results are achieved when a generalized backbone is used with SVM or LDA as classifier. Using a well trained autoencoder can also be very vital in making the difference
2. While getting good accuracy within different diffusion models or within different GANs based generation techniques, it is still currently difficult to get good accuracy with unknown techniques of image generation

### Section 5: Future Scope:
1. Explore using text embeddings for classification of real and fake images (explore using the semantic information within an image)
2. Try to enhance the autoencoder hyperparameters
3. Use larger datasets for training the models
4. Try exploring other backbone models
5. Try building some neural network by using multiple backbones and then get feature maps
