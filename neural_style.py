import torch
import torchvision.models  as models
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
import argparse
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

class processor:
    def __init__(self, img_size, means, std) -> None:
        self.img_size = img_size
        self.means = means
        self.std = std
    def preprocess(self, img):
        return transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.means, std=self.std),
            transforms.Lambda(lambda x: x.mul_(255))
        ])(img).unsqueeze(0)
    def postprocess(self, img):
        temp = transforms.Compose([
            transforms.Lambda(lambda x: x.mul_(1./255)),
            transforms.Normalize(mean=[-m/s for m, s in zip(self.means, self.std)], std=[1/s for s in self.std]),
            transforms.Resize(self.img_size)
        ])(img[0].squeeze())
        return transforms.ToPILImage()(temp.clamp(0,1).cpu())

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
        style_out, content_out = self.extract(img)
        flat = {key: val.view(val.shape[1], -1) for key, val in style_out.items()}
        gram = {key: torch.matmul(val, val.t()).div_(val.shape[1]) for key, val in flat.items()}
        assert all(val1.shape[2] * val1.shape[3] == val2.shape[1] for val1, val2 in zip(style_out.values(), flat.values()))
        return gram, content_out

class ImageStyleTransfer:
    def __init__(self) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device being used:", self.device)
        # pre_means = [0.485, 0.456, 0.406]
        self.pre_means = [0.48501961, 0.45795686, 0.40760392]
        self.pre_stds = [1, 1, 1]#[0.229, 0.224, 0.225]
        self.img_size = 512

        style_layer_nums = [1, 6, 11, 20, 29] # taken from https://www.mathworks.com/help/deeplearning/ref/vgg19.html
        content_layer_num = 22

        self.content_layers = {f"features.{content_layer_num}": "relu4_2"}
        self.style_layers = {f"features.{i}": f"relu{j+1}_1" for j, i in enumerate(style_layer_nums)}

        _style_wt_list = [1e3/n**2 for n in [64,128,256,512,512]]
        self.style_weights = {f"relu{j+1}_1":_style_wt_list[j] for j, _ in enumerate(style_layer_nums)}
        self.content_weights = {"relu4_2" : 1e0}

        self.model = self.get_model()
        self.content_img = None
        self.style_img = None

        self.proc = processor(self.img_size, self.pre_means, self.pre_stds)
        self.ext = extractor(self.model, self.style_layers, self.content_layers)

    def get_model(self):
        weights = models.VGG19_Weights.DEFAULT
        model = models.vgg19(weights=weights, progress=True)
        model = model.to(self.device).eval().requires_grad_(False)
        return model
    
    def get_images(self, content, style):
        content_img = Image.open(content).convert("RGB")
        style_img = Image.open(style).convert("RGB")
        return style_img, content_img

    def __call__(self, content, style, save_path = None, num_steps = 500):
        self.style_img, self.content_img = self.get_images(content, style)
        p_content, p_style = self.proc.preprocess(self.content_img).to(self.device), self.proc.preprocess(self.style_img).to(self.device)
        actual_gram_matrices, _ = self.ext(p_style)
        _, actual_content_outputs = self.ext(p_content)
        noise_img = p_content.clone()
        noise_img.requires_grad = True
        num_iter = [0]
        iter_range = tqdm(range(num_steps))
        lr = 1
        optimizer = torch.optim.LBFGS([noise_img], max_iter=num_steps, lr=lr)
        def closure():
            iter_range.update()
            style_outputs, content_outputs = self.ext(noise_img)
            loss = 0.
            num_iter[0] += 1
            for key, val in style_outputs.items():
                # output_flat = val.view(val.shape[1], -1)
                # gram_matrix = torch.matmul(output_flat, output_flat.t())
                # loss += style_weights[key] * ((gram_matrix - actual_gram_matrices[key])**2).mean()
                loss += self.style_weights[key] * nn.functional.mse_loss(style_outputs[key], actual_gram_matrices[key])
            for key, val in content_outputs.items():
                # print(f"============== iter : {num_iter[0]} ====================")
                # print(f"style: {loss}")
                # old_loss = loss.clone()
                # print(val.shape, actual_content_outputs[key].shape)
                # loss += content_weights[key]*((val - actual_content_outputs[key])**2).mean()
                loss += self.content_weights[key]*nn.functional.mse_loss(val, actual_content_outputs[key])
                # print(f"content: {loss - old_loss}")
            optimizer.zero_grad()
            loss.backward()
            return loss
        for _ in range(1):
            optimizer.step(closure)

        corr_img = noise_img.clone()
        corr_img = self.proc.postprocess(corr_img)
        if save_path is not None:
            corr_img.save(save_path)
        return corr_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=500)
    args = parser.parse_args()
    content = "input/tubingen.jpg"
    # content = "input/ashwin.jpg"
    # content = "input/Aspark-Owl.jpg"
    # content = "input/jv_bokassa.png"
    # content = "output/frame_0.jpg"
    # style = "input/rain-princess-aframov.jpg"
    style = "input/vangogh_starry_night.jpg"
    # style = "input/escher_sphere.jpg"
    # style = "input/picasso_selfport1907.jpg"
    # style = "input/frida_kahlo.jpg"
    image_style_transfer = ImageStyleTransfer()
    image_style_transfer(content, style, save_path="output/final.jpg", num_steps=args.n)