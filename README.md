# Image_Deepfake_Detection
Implementation and improvement of a research paper with a new theory for image deepfake detection with high generalization to include General Adversarial Networks, Diffusion Models and any other new generative techniques 

**Approach 1: Learn the low level cues left behind by an Image Generator**
It is believed the image generators leave behind some perceptual cues (encompassing elements such as colors, textures, brightness, and low-level visual cues). The primary concept of this approach revolves around identifying those unique low-level cues left by image generators, achieved through training over diverse images. The methodology employs pretrained ResNET 50 and ResNET 18 backbones, with the addition of a new binary classification dense layer for discriminating between real and fake images. 

The implementation is carried out using both Keras for ResNET 50 and PyTorch for ResNET 18. Image augmentation is incorporated in the ResNET 50 implementation. Both models are trained on the ImageNet Dataset. Key implementation details include a batch size of 128, a learning rate of 0.00001, Stochastic Gradient Descent (SGD) as the optimizer with momentum set to 0.9, Cross Entropy Loss as the chosen loss function, and a resized resolution of 256 x 256. The training process spans 80 epochs, ensuring a comprehensive learning experience for the models.
