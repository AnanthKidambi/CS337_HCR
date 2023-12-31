import torch
import torchvision.models  as models
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
from torchvision.models.feature_extraction import create_feature_extractor
import argparse
from PIL import Image
import utils
import inspect
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
        self.means = [0.48501961, 0.45795686, 0.40760392]
        self.std = [0.229, 0.224, 0.225]
        
    def extract(self, img):
        debug = utils.debug
        high_val = utils.high_val

        img2 = transforms.Normalize(mean=self.means, std=self.std)(img)
        if debug: 
            if img2.isnan().any() or (torch.max(torch.abs(img2)) >= high_val):
                print(f"Ipython from line {inspect.currentframe().f_lineno} of neural_style.py")
                import IPython
                IPython.embed()
                exit()   
        out = self.extractor(img2)
        if debug: 
            for outv in out.values():
                if outv.isnan().any() or (torch.max(torch.abs(outv)) >= high_val):
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of neural_style.py")
                    import IPython
                    IPython.embed()
                    exit()
        for key in out:
            out[key] = torch.where(out[key].isnan(), torch.zeros_like(out[key]), out[key])
        
        if debug: 
            for outv in out.values():
                if outv.isnan().any() or (torch.max(torch.abs(outv)) >= high_val):
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of neural_style.py")
                    import IPython
                    IPython.embed()
                    exit()
        return {key: val for key, val in out.items() if key in self.style_layers.values()}, {key: val for key, val in out.items() if key in self.content_layers.values()}
    def __call__(self, img, normalize = False):
        debug = utils.debug
        high_val = utils.high_val
        
        style_out, content_out = self.extract(img)
        flat = {key: val.view(val.shape[1], -1) for key, val in style_out.items()}
        if debug: 
            for key in flat:
                if flat[key].isnan().any() or (torch.max(torch.abs(flat[key])) >= high_val):
                    print(f"Ipython from line {inspect.currentframe().f_lineno} of neural_style.py")
                    import IPython
                    IPython.embed()
                    exit()
        if normalize:
            gram = {key: torch.matmul(val, val.t()).div_(4.0*val.numel()) for key, val in flat.items()}
        else:
            gram = {key: torch.matmul(val, val.t()).div_(val.shape[1]) for key, val in flat.items()}
        if debug: 
            for key, _ in gram.items():
                if gram[key].isnan().any():
                    print('nan:', [gram[key].isnan().nonzero() for key in gram])
        for key in gram:
            gram[key] = torch.where(gram[key].isnan(), torch.zeros_like(gram[key]), gram[key])
        if debug: 
            for key, _ in gram.items():
                if gram[key].isnan().any():
                    print('DUH:', [outv.isnan().nonzero() for outv in gram.values()])
        assert all(val1.shape[2] * val1.shape[3] == val2.shape[1] for val1, val2 in zip(style_out.values(), flat.values()))
        return gram, content_out

class ImageStyleTransfer:
    def __init__(self, img_size) -> None:

        self.device = torch.device(utils.device if torch.cuda.is_available() else "cpu")
        print("Device being used:", self.device)
        self.pre_means = [0.48501961, 0.45795686, 0.40760392]
        self.pre_stds = [1, 1, 1]
        self.img_size = img_size

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

    def __call__(self, content, style, save_path = None, num_steps = 500, init_img = None):
        self.style_img, self.content_img = self.get_images(content, style)
        p_content, p_style = self.proc.preprocess(self.content_img).to(self.device), self.proc.preprocess(self.style_img).to(self.device)
        actual_gram_matrices, _ = self.ext(p_style)
        _, actual_content_outputs = self.ext(p_content)
        if init_img is None:
            noise_img = p_content.clone()
        else:
            noise_img = Image.open(init_img).convert("RGB")
            noise_img = self.proc.preprocess(noise_img).to(self.device)
        noise_img.requires_grad = True
        num_iter = [0]
        iter_range = tqdm(range(num_steps))
        lr = 1
        optimizer = torch.optim.LBFGS([noise_img], max_iter=1, lr=lr)
        def closure():
            iter_range.update()
            style_outputs, content_outputs = self.ext(noise_img)
            loss = 0.
            num_iter[0] += 1
            for key, val in style_outputs.items():
                loss += self.style_weights[key] * nn.functional.mse_loss(style_outputs[key], actual_gram_matrices[key])
            for key, val in content_outputs.items():
                loss += self.content_weights[key]*nn.functional.mse_loss(val, actual_content_outputs[key])
            optimizer.zero_grad()
            loss.backward()
            return loss
        for _ in range(num_steps):
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
    content = "input/content/tubingen.jpg"
    # content = "input/content/mbs.jpg"
    # content = "input/content/leopard.jpg"
    # content = "input/content/iitb.png"
    # content = "input/content/taj_mahal.png"
    # content = "input/content/ashwin.jpg"
    # content = "input/content/Aspark-Owl.jpg"
    # content = "input/content/tom_and_jerry.jpg"
    # content = "input/content/jv_bokassa.png"
    content = "input/content/jv_sleeping.jpg"

    # style = "input/style/rain-princess-aframov.jpg"
    style = "input/style/kandinsky.jpg"
    # style = "input/style/eye_supernova.jpg"
    # style = "input/style/shipwreck.jpg"
    # style = "input/style/escher_sphere.jpg"
    # style = "input/style/picasso_selfport1907.jpg"
    # style = "input/style/frida_kahlo.jpg"

    size = Image.open(content).convert("RGB")._size
    img_size = tuple([i - i%8 for i in size[1:]])
    name = f'{content.split("/")[-1].split(".")[0]}_{style.split("/")[-1].split(".")[0]}'
    image_style_transfer = ImageStyleTransfer(img_size)
    image_style_transfer(content, style, save_path=f"output/{name}.jpg", num_steps=args.n)