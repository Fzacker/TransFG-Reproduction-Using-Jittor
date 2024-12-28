import gradio as gr
import jittor
import numpy as np
from PIL import Image
from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader
import argparse
import os

# 加载模型
def setup_model(args):
    # 配置和加载模型
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
    model.load_from(np.load(args.pretrained_dir))
    pretrained_model = jittor.load(args.pretrained_model)['model']
    model.load_state_dict(pretrained_model)
    model.eval()  # 设置为评估模式
    return model

def read_cub_file(file_path):
    result = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) > 1:
                result.append(parts[1])
    return result

def classify_image(image, model, test_loader):
    image = image.convert('RGB')
    image = jittor.array(test_loader.transform(image)).unsqueeze(0)

    logits = model(image)
    preds, _ = jittor.argmax(logits, dim=-1)

    predicted_class = preds.item()
    return predicted_class

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/opt/tiger/minist')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    args = parser.parse_args()

    args.data_root = '{}/{}'.format(args.data_root, args.dataset)

    device = "cuda"
    args.n_gpu = jittor.get_device_count()
    args.device = device

    model = setup_model(args)
    train_loader, test_loader = get_loader(args)

    cont = read_cub_file(os.path.join(args.data_root, "classes.txt"))

    def predict(image):
        correct_labels = classify_image(image, model, test_loader)
        
        return cont[correct_labels]

    # 创建Gradio界面
    gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Upload Image"),  # 图片输入
        ],
        outputs="text"
    ).launch()

if __name__ == "__main__":
    main()
