# 将fruit_vegetable_data的数据做一次清洗
#   - 统一文件后缀：{'PNG', 'JPG', 'png', 'jpeg', 'jpg'}五种
#   - 统一RGB通道
#   - (Optional) 裁剪大小

# resize_size=224 ->
# test calc complete: mean=0.5592794912530372, std=0.33495044731382206
# train calc complete: mean=0.5396750806068386, std=0.3302109593831783
# validation calc complete: mean=0.5586149059496127, std=0.3347352803119484

import os
import glob
import PIL.Image
import torch
import numpy as np
from PIL.Image import Image
from torchvision import transforms

input_folder = './fruit_vegetable_raw_data/'
output_folder = './fruit_vegetable_data_224'
calc_folder = './fruit_vegetable_data_224'
# folder_list = ['test']
folder_list = ['test', 'train', 'validation']
image_names = ['PNG', 'JPG', 'png', 'jpeg', 'jpg']
channel_name = 'RGB'
resize_size = 224


def preprocess():
    image_transform = None
    if resize_size is not None:
        image_transform = transforms.Compose([
            transforms.Resize(resize_size),
            # transforms.ToTensor()
        ])
    else:
        image_transform = transforms.Compose([
            # transforms.ToTensor()
        ])

    for folder in folder_list:
        input_folder_path = os.path.join(input_folder, folder)
        output_folder_path = os.path.join(output_folder, folder)
        print(f'working on {input_folder_path}..')
        categories = glob.glob(pathname=os.path.join(input_folder_path, '*'), recursive=False)
        print(f'{len(categories)} categories found')
        # 针对每个category
        for category in categories:
            category = category.replace(input_folder_path, '').lstrip('./\\')
            # Image扫描
            all_images = []
            for name in image_names:
                # 五种image应该全部进来了
                all_images += glob.glob(pathname=os.path.join(input_folder_path, category, f'*.{name}'))
            print(f'image scan({folder}/{category}): {len(all_images)} images')
            # Image处理
            for image in all_images:
                assert isinstance(image, str)
                # 处理
                img_pil = PIL.Image.open(image).convert('RGB')
                img_tensor = image_transform(img_pil)
                assert isinstance(img_tensor, PIL.Image.Image)
                # 保存，先提取带后缀的文件名
                image_name = image.replace(os.path.join(input_folder_path, category), '').lstrip('./\\')
                # 不带后缀的文件名
                image_name = os.path.splitext(image_name)[0]
                # 计算完整保存路径
                output_img_path = os.path.join(output_folder_path, category, f'{image_name}.png')
                # 生成文件夹
                os.makedirs(os.path.join(output_folder_path, category), exist_ok=True)
                img_tensor.save(output_img_path, 'PNG')


def mean_std():
    # 每个数据集分割分开来看
    for folder in folder_list:
        all_images = []
        num_pixels = 0.0
        sum_value = 0.0
        sum_squares = 0.0
        calc_folder_path = os.path.join(calc_folder, folder)
        categories = glob.glob(pathname=os.path.join(calc_folder_path, '*'), recursive=False)
        for category in categories:
            category = category.replace(calc_folder_path, '').lstrip('./\\')
            for name in image_names:
                all_images += glob.glob(pathname=os.path.join(calc_folder_path, category, f'*.{name}'))
        # 每个category的都合在一起
        for image in all_images:
            assert isinstance(image, str)
            img_pil = PIL.Image.open(image).convert('RGB')
            # 一定要是255.0，否则就会是int类型，基本上全都是0了
            img_np = np.array(img_pil).astype(np.uint8) / 255.0
            num_pixels += img_np.size
            sum_value += np.sum(img_np)
            sum_squares += np.sum(np.square(img_np))
        # print(np.shape(np_arrays))  # [DataSize, Height, Width, Channel]
        mean = sum_value / num_pixels
        std = np.sqrt(sum_squares / num_pixels - np.square(mean))
        print(f'{folder} calc complete: mean={mean}, std={std}')


if __name__ == '__main__':
    mean_std()
