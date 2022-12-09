# Seeing is not believing! 
## Privacy preservation using intelligent masking and inpainting using stable diffusion

### Description

Through this project, we wanted to preserve privacy whilst ensuring that non private information
is preserved. 

We try three methods, two of which are novel to the best of our knowledge for masking
private information. We use face recognition task as our privacy task and show performance on 
other downstream tasks as good enough by sharing qualitative results.

#### Fine-tuning FaceNet

[FaceNet](https://github.com/timesler/facenet-pytorch) models available publicly are trained on the VGGFace2 and CASIA-webface datasets.
However the diffusion model is trained on the 

#### Finding Regions to Mask

#### Diffusion Models for In-painting

### Structure

- **Finetuning**: `/pipeline/facenet_finetuning.ipynb` : This notebook does fine-tuning of facenet over the celeb-A dataset. The facenet model available publicly has been trained over the VGGFace2 model and the CASIA-webface model. Since our diffusion models have been trained on celeb-A dataset, we finetuned the model over celeb-A dataset. Since we ran this code as a script on GCP, there are no inline outputs.

- **Generation code**: The `pipeline` folder contains the scripts which loads a validation data loader and generates masks for the input images. For each of the input image and mask, we generate a privacy preserving face output using the stable diffusion pipeline.
  - `generate_with_random_masks.py`: Generates masks using random patching and generates images using stable diffusion
  - `generate_with_gradcam_masks.py`: Generates masks using gradcam and generates images using stable diffusion
  - `generate_with_saliency_gradients_masks.py`: Generates masks using saliency gradients and generates images using stable diffusion
  
- **Results**: The final output of the scripts will generate three folders `results` containing generated images, `masks` containing the mask applied and `original` containing the original images. Each image is identified by a unique id and the ground truth class label. The output images are named as follows:
  -  Generated image: `{image-id}-generated-label-{class-id}.jpg`
  -  Mask image: `{image-id}-masks-label-{class-id}.jpg`
  -  Original image: `{image-id}-img-label-{class-id}.jpg`

- **Evaluation**: 'evaluate.py' returns the accuracy of the finetuned facenet model on the validation sample and the accuracy with which privacy was preserved by creating the new images (number of times we successfully fooled the facenet model)

### Qualitative Results

![image](images/1.png)

![image](images/2.png)

![image](images/3.png)

![image](images/4.png)

![image](images/5.png)

![image](images/6.png)


