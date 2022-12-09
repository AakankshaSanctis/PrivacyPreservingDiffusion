import torch
from PIL import Image
import gdown
import os
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization
from tqdm import tqdm
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_download_path = './facenet.pt'
model_url = 'https://drive.google.com/uc?id=1WpB3vZ_L6ZctpfTAI4PAPJoiIvCk5CAP'

gdown.download(model_url, model_download_path, quiet=False)
model = torch.load("./facenet.pt", device).eval()

image_size=160
transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    fixed_image_standardization
])


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("parent", type=str)
  args = parser.parse_args()

  PARENT_FOLDER = args.parent
  DIFFUSED_IMGS = "results"
  ORIGINAL_IMGS = "original"

  img_map = dict([(filename.split('-')[0], transform(Image.open(f"{PARENT_FOLDER}/{DIFFUSED_IMGS}/{filename}"))) for filename in os.listdir(f"{PARENT_FOLDER}/{DIFFUSED_IMGS}")])
  num_correct_preds = 0
  num_wrong_preds = 0
  num_fooled = 0
  num_not_fooled = 0

  count = 0
  for filename in tqdm(os.listdir(f"{PARENT_FOLDER}/{ORIGINAL_IMGS}")):
    img_id = filename.split('-')[0]
    class_id = filename.split('-')[-1].split('.')[0]

    orig_img = transform(Image.open(f"{PARENT_FOLDER}/{ORIGINAL_IMGS}/{filename}")).to(device)
    diffused_img = img_map[img_id].to(device)
    model_pred_orig = torch.argmax(model(orig_img.unsqueeze(0)).squeeze()).item()
    model_pred_diffused = torch.argmax(model(diffused_img.unsqueeze(0)).squeeze()).item()

    if(int(class_id) == model_pred_orig):
      num_correct_preds += 1
      if(model_pred_diffused != model_pred_orig):
        num_fooled += 1
      else:
        num_not_fooled +=1
    else:
      num_wrong_preds +=1

  print("Number of correct preds: {}, wrong preds:{}".format(num_correct_preds, num_wrong_preds))

  print(f"Number of times the model was fooled: {num_fooled}, not fooled: {num_not_fooled}")

  print(f"Accuracy: {num_fooled/num_correct_preds}")

