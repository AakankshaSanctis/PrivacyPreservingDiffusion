import glob
import os
import random
import zipfile

import cv2
import gdown
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from facenet_pytorch import fixed_image_standardization
from natsort import natsorted
from patchify import patchify, unpatchify
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

manual_seed = 999
random.seed(manual_seed)
torch.manual_seed(manual_seed)

MODEL_FILE = "data/celeba/facenet.pt"
img_folder = 'data/celeba/abc'
results_folder = "data/celeba/mae/results/"
original_folder = "data/celeba/mae/original/"
masks_folder = "data/celeba/mae/masks/"
grad_cam_embeddings = "data/celeba/mae/embeddings"
embedding_image_folder = f'{grad_cam_embeddings}/feature_vectors_image/'

ngpu = 1
device = torch.device('cuda:0' if (
        torch.cuda.is_available() and ngpu > 0) else 'cpu')


image_size = 160
default_transform  = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])
transform=transforms.Compose([
    fixed_image_standardization
])


prompt = ""
guidance_scale=7.5
num_samples = 1
generator = torch.Generator(device="cuda").manual_seed(0)



def setup_diffusion_pipeline():
    """
    Sets up a stable diffusion pipeline for image inpainting
    """
    device = "cuda"
    model_path = "runwayml/stable-diffusion-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        revision="fp16", 
        torch_dtype=torch.float16,
        use_auth_token=True
    ).to(device)
    pipe.safety_checker = dummy
    return pipe

def dummy(images, **kwargs):
    return images, False




def download_data():
    """
    Creates the folders and downloads the celebA dataset
    """
    ### Data Download CelebA
    data_root = 'data/celeba'
    download_path = f'{data_root}/CelebA.zip'
    url = 'https://drive.google.com/uc?id=1pUv-X_r738skkbJgoytwJ7bQLxVmGeye'

    model_root = 'data/celeba'
    model_download_path = f'{model_root}/facenet.pt'
    model_url = 'https://drive.google.com/uc?id=1WpB3vZ_L6ZctpfTAI4PAPJoiIvCk5CAP'

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    gdown.download(url, download_path, quiet=False)

    with zipfile.ZipFile(download_path, 'r') as ziphandler:
        ziphandler.extractall(img_folder)
    gdown.download(model_url, model_download_path, quiet=False)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if not os.path.exists(original_folder):
        os.makedirs(original_folder)

    if not os.path.exists(masks_folder):
        os.makedirs(masks_folder)



### Create Loaders
class CelebADataset(Dataset):


  def __init__(self, root_dir, transform=None):
    image_names = os.listdir(root_dir)
    self.root_dir = root_dir
    self.transform = transform 
    self.image_names = natsorted(glob.glob(root_dir + '/*/*.jpg'))


  def __len__(self): 
    return len(self.image_names)


  def __getitem__(self, idx):
    img_path = self.image_names[idx]
    img = Image.open(img_path).convert('RGB')
    if self.transform:
      img = self.transform(img)

    return (img, int(img_path.split('/')[-2]) - 1)


def get_data_loader():
    """
    Gets data loader for validation dataset
    """
    celeba_dataset_val = CelebADataset(f'{img_folder}/1', default_transform)

    batch_size = 1
    num_workers = 0 if device.type == 'cuda' else 2
    pin_memory = True if device.type == 'cuda' else False

    val_dataloader = torch.utils.data.DataLoader(celeba_dataset_val,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory,
                                                shuffle=True)
    return val_dataloader



def get_generated_image(img, mask, pipe):
  """
  Creates a pipeline to obtain the generated image using stable diffusion using a mask and a input image
  """
  return pipe(
    prompt=prompt,
    image=Image.fromarray(img),
    mask_image=Image.fromarray(mask),
    guidance_scale=guidance_scale,
    generator=generator,
    num_images_per_prompt=num_samples,
).images[0]  


def generate_mask(image, m, patch_size):
  """
  Masks random patches given the proportion and the patch size
  """
  mask_img = 255*np.zeros_like(image)
  mask_patches = patchify(mask_img, (patch_size,patch_size,3), step=patch_size)

  x = mask_patches.shape[0]
  y = mask_patches.shape[1]
  num_patches_sampled = m * (x-4) * (y-4)

  for iter in range(int(num_patches_sampled)):
    x_coord = random.randint(4, x-4)
    y_coord = random.randint(4, y-4)

    mask_patches[x_coord][y_coord] = 255*np.ones_like(mask_patches[x_coord][y_coord])
  
  reconstructed_mask = unpatchify(mask_patches, image.shape)
  return reconstructed_mask


def run_pp_gradient_image_generation(val_dataloader, pipe):
    """
    Runs the pipeline to generate masks and images through stable diffusion
    """
    ### Get Gradients and generate images

    for i_batch, (x, y) in enumerate(val_dataloader):
        if i_batch == 1000:
            break
        label = y.numpy()[0]

        x = np.array(x.squeeze().permute(1,2,0))
        x = cv2.resize(x, (512, 512))

        x= ((x - x.min()) * (1/(x.max() - x.min()) * 255)).astype('uint8')
        mask = generate_mask(x, 1, 32)


        x = x.astype('uint8')
        mask = mask.astype('uint8')
        generated_image  = get_generated_image(x, mask, pipe)

        generated_image.save("{}{}-generated-label-{}.jpg".format(results_folder, i_batch, label))
        img = Image.fromarray(x)
        img.save("{}{}-img-label-{}.jpg".format(original_folder, i_batch, label))
        mask = Image.fromarray(mask)
        mask.save("{}{}-masks-label-{}.jpg".format(masks_folder, i_batch, label))



if __name__=="__main__":
    download_data()
    val_data_loader = get_data_loader()
    pipe = setup_diffusion_pipeline()
    run_pp_gradient_image_generation(val_data_loader, pipe)
