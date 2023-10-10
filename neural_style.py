# from typing import Any
import torch
import torchvision.models  as models
import torchvision.transforms as transforms
# from torchvision.io import read_image
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
import argparse
from PIL import Image
import warnings
# warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)
# pre_means = [0.485, 0.456, 0.406]
pre_means = [0.48501961, 0.45795686, 0.40760392]
pre_stds = [1, 1, 1]#[0.229, 0.224, 0.225]
img_size = 512

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=2000)
args = parser.parse_args()
content = "input/tubingen.jpg"
style = "input/vangogh_starry_night.jpg"

style_layer_nums = [1, 6, 11, 20, 29] # taken from https://www.mathworks.com/help/deeplearning/ref/vgg19.html
content_layer_num = 22

content_layers = {f"features.{content_layer_num}": "relu4_2"}
style_layers = {f"features.{i}": f"relu{j+1}_1" for j, i in enumerate(style_layer_nums)}

# _style_wt_list = [1e3/n**2 for n in [64,128,256,512,512]]
_style_wt_list = [1e2/5 for n in [64,128,256,512,512]]
style_weights = {f"relu{j+1}_1":_style_wt_list[j] for j, _ in enumerate(style_layer_nums)}
content_weights = {"relu4_2" : 1e0}

def get_model():
    weights = models.VGG19_Weights.DEFAULT
    model = models.vgg19(weights=weights, progress=True)
    # print second layer weights
    # print(model.features[5].weight)
    model = model.to(device).eval().requires_grad_(False)
    # print(model)
    return model

def get_images():
    # content_img = read_image(content).to(device)
    # style_img = read_image(style).to(device)
    content_img = Image.open(content).convert("RGB")
    style_img = Image.open(style).convert("RGB")
    return style_img, content_img

class processor:
    def __init__(self, img_size, means=pre_means, std=pre_stds) -> None:
        self.img_size = img_size
        self.means = means
        self.std = std
    def preprocess(self, img):
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means, std=self.std),
            transforms.Lambda(lambda x: x.mul_(255))
        ])(img)
    def postprocess(self, img):
        _ = transforms.Compose([
            transforms.Lambda(lambda x: x.mul_(1./255)),
            transforms.Normalize(mean=[-m/s for m, s in zip(pre_means, pre_stds)], std=[1/s for s in pre_stds]),
        ])(img)
        return transforms.ToPILImage()(_.clamp_(0, 1).cpu())

class extractor:
    def __init__(self, model, style_layers, content_layers) -> None:
        self.model = model
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.extractor = create_feature_extractor(self.model, {**self.style_layers, **self.content_layers})
    def extract(self, img):
        out = self.extractor(img)
        return {key: val for key, val in out.items() if key in self.style_layers.values()}, {key: val for key, val in out.items() if key in self.content_layers.values()}
    def __call__(self, img):
        style_out, content_out = self.extract(img.unsqueeze(0))
        # print(style_out)
        flat = {key: val.view(val.shape[1], -1) for key, val in style_out.items()}
        gram = {key: torch.matmul(val, val.t()) for key, val in flat.items()}
        assert all(val1.shape[2] * val1.shape[3] == val2.shape[1] for val1, val2 in zip(style_out.values(), flat.values()))
        return gram, content_out

model = get_model()
style_img, content_img = get_images()
proc = processor(img_size)
ext = extractor(model, style_layers, content_layers)
p_content, p_style = proc.preprocess(content_img).to(device), proc.preprocess(style_img).to(device)
actual_gram_matrices, _ = ext(p_style)
# print(actual_gram_matrices)
_, actual_content_outputs = ext(p_content)
# assert p_content.shape == p_style.shape

for i in style_layer_nums:
    assert model.features[i]._get_name() == "ReLU"

#getting the required image
# noise_img = proc.preprocess(content_img).to(device) # torch.rand([3, 224, 224], device = device)*0.5
noise_img = p_content.clone()
noise_img.requires_grad = True
num_steps = args.n 
iter_range = tqdm(range(num_steps))
lr = 1
optimizer = torch.optim.LBFGS([noise_img], max_iter=num_steps, lr=lr)

def closure():
    global iter_range, ext
    iter_range.update()
    style_outputs, content_outputs = ext(noise_img)
    loss = 0.
    for key, val in style_outputs.items():
        output_flat = val.view(val.shape[1], -1)
        gram_matrix = torch.matmul(output_flat, output_flat.t())
        loss += style_weights[key] * ((gram_matrix - actual_gram_matrices[key])**2).mean()
    for key, val in content_outputs.items():
        # print(f"============== iter : {num_iter[0]} ====================")
        # print(f"style: {loss}")
        # old_loss = loss.clone()
        # print(val.shape, actual_content_outputs[key].shape)
        loss += content_weights[key]*((val - actual_content_outputs[key])**2).mean()
        # print(f"content: {loss - old_loss}")
    optimizer.zero_grad()
    loss.backward()
    # print("diff : ", abs(prev_mean_losses-loss.item()))
    # diff = abs(prev_mean_losses-loss.item())
    return loss

for _ in range(1):
    optimizer.step(closure)

corr_img = noise_img.clone() # + torch.rand([3, 224, 224], device = device)*0.5
corr_img = proc.postprocess(corr_img)
corr_img.save(f"output/final.jpg")