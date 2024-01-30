# Image_Deepfake_Detection
Implementation and improvement of a research paper with a new theory for image deepfake detection with high generalization to include General Adversarial Networks, Diffusion Models and any other new generative techniques 
<br>
<h2>Approach 1: Learn the low level cues left behind by an Image Generator</h2>
It is believed the image generators leave behind some perceptual cues (encompassing elements such as colors, textures, brightness, and low-level visual cues). The primary concept of this approach revolves around identifying those unique low-level cues left by image generators, achieved through training over diverse images. The methodology employs pretrained ResNET 50 and ResNET 18 backbones, with the addition of a new binary classification dense layer for discriminating between real and fake images. 
<br><br>
The implementation is carried out using both Keras for ResNET 50 and PyTorch for ResNET 18. Image augmentation is incorporated in the ResNET 50 implementation. Both models are trained on the ImageNet Dataset. Key implementation details include a batch size of 128, a learning rate of 0.00001, Stochastic Gradient Descent (SGD) as the optimizer with momentum set to 0.9, Cross Entropy Loss as the chosen loss function, and a resized resolution of 256 x 256. The training process spans 80 epochs, ensuring a comprehensive learning experience for the models.
<br><br>

<b>Flaws with this approach:</b>
1. It was seen that while there decent accuracy over the training dataset, when an image generated using a different image generation technique was given the classifier failed to classify with good accuracy
2. Training the model takes too much time and is not entirely accurate
3. The was a sink in the parameters learned, such that, if it was a real image classification was good but if a fake image was given from a method not used in training it was considered as real. Any image that is not generated using the method used while training would be classified as real
   
<br>
<h2>Approach 2: Use a generalized backbone for feature maps and then use Classifiers</h2>
Since it was noticed in approach 1 that training a neural network over a single generation technique would make it work only for that generation method, why not stop aiming to learn a neural network with the specific task of classifying real and fake over a given generation technique and just use a pretrained backbone to get feature maps and then use classifiers like linear probes, KNN, SVMs, etc
<br><br>
<h3>Implementaion details:</h3>
For this pupose I tried, <br>
<b>Datasets:</b>
  1. Laion vs LDM100
  2. ImageNet vs LDM200
  3.bigGAN Real vs bigGAN Fake

<b>Transformations on image before training:</b>
1. Transform 0: No Change
2. Transform 1: Adding Gaussian Blur
3. Transform 2: Adding Jitter
4. Transform 3: Adding Guassian Blur and Jitter Both

<b>Backbone Models:</b>
1. DINO ViT-B/16
2. DINO ResNET50
3. CLIP ViT-B/16

<b>As for the classifiers I used, </b>
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

<b>I also tried 3 different variations of dimensionality reductions</b>
1. No Reduction
2. Principal Component Analysis
3. Autoencoding 

Here are the findings and the inference,

1. Impact of using transforms before training:
![Impact_of_transformations_on_training](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/fee63dbd-857d-4682-ac91-01b0ead60ed3)

NOTE: Goodness Factor was calculated by subtracting min(accuracy) from all data points for each line. Each line represents a different dataset-backbone model combination. The names have been omitted for clarity

Inference: Adding No transform while training is better. But adding transform 2 also shows good results. Dealing with Gaussian Blur might cause issues in accuracy (around 0.05 
 on average)

2. Impact of Dimensionality Reduction:
![Impact_of_dimensionality_reduction_vs_classifiers](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/5a5f5631-2c59-4816-b028-58ec1fb8a726)

Inference: In many cases no dimensionality reduction is the best choice. But in some cases autoencoding performs better than no reduction or PCA

3. Impact of classifier used:
![clip_vitb16_transform0_dim_red_comparison](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/52481e98-6b6a-4bdf-8004-1169050b8708)
Using CLIP ViT-B/16 backbone over all datasets combined
![dino_resnet50_transform0_dim_red_comparison](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/2871efec-56a4-45cd-bf71-e743c71ceb40)
Using DINO ResNET50 backbone over all datasets combined
![dino_vitb16_transform0_dim_red_comparison](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/d141a9a5-eeea-4533-a7e8-c7e6c2a00ec4)
Using DINO ViT-B/16 backbone over all datasets combined

Inference: SVM, Linear Probing and Linear Discriminant Analysis seem to be good classifiers

4. Impact of backbone model used:
![accuracy_vs_backbone_wrt_dataset](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/3faa299b-5518-4bb6-abcb-19565283c09c)
NOTE: Multiple lines of same color refer to different transforms used
![bigganReal_vs_bigganFake_accuracy_vs_transform](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/f9054b0d-de0b-4615-863b-424aa3641539)
Using bigGAN Real and bigGAN Fake Datasets
![laion_vs_ldm100_accuracy_vs_transform](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/ef6adac4-4c29-41d5-bd0c-132df2a69bce)
Using Laion and LDM100 Datasets
![imagenet_vs_ldm200_accuracy_vs_transform](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/f432071b-7e25-403d-aad8-3fc44107535f)
Using ImageNet and LDM200 Datasets

Inference: CLIP ViT-B/16 is the best backbone over all cases, followed by DINO ResNET50, followed by DINO ViT-b/16

After this I tried to split the load feature transforms over multiple backbones instead of just one backbone. I made the following combinations
1. DINO ViT-B/16 and DINO ResNET50 and CLIP ViT-B/16
2. DINO ViT-B/16 and DINO ResNET50
3. CLIP ViT-B/16 and DINO ResNET50
4. CLIP ViT-B/16 and DINO ViT-B/16

I then trained each model over the entire dataset and took the best classifier for each. I used randomised jitter (p=0.5) and guassian blur (p=0.5).
![accuracy_vs_backbone_wrt_model](https://github.com/coderbeta1/Image_Deepfake_Detection/assets/72234861/2dceccff-d11a-4439-8def-61afe9b1a714)

Lastly I tried to check the models across datasets. (Training on A and testing on B).

(UPDATE THE TABLES HERE)











