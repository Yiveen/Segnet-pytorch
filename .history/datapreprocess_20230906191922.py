import os
import random
import numpy as np

def main():
    image_dir = ''
    image_names = os.listdir(image_dir)
    random.shuffle(image_names)
    
    total_images = len(image_names)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    train_split = int(total_images * train_ratio)
    val_split = int(total_images * (train_ratio + val_ratio))
    
    train_set = image_names[:train_split]
    val_set = image_names[train_split:val_split]
    test_set = image_names[val_split:]
    
    # 为每个类型选择一定数量的文件，并添加后缀"changed"
    num_files_per_type = 5  # 选择每个类型的文件数量
    random_numbers = np.random.randint(0, 1001, num_files_per_type)
    random_numbers_list = random_numbers.tolist()   

    # 初始化用于存储已处理文件名的列表
    changed_files = []

    # 遍历训练集、验证集和测试集
    for dataset, dataset_name in [(train_set, 'train'), (val_set, 'val'), (test_set, 'test')]:

        for filename in dataset:
            file_type = filename.split('.')[-1]  # 获取文件类型
            number = int(filename.split('.')[0])
            if number in random_numbers_list:
                new_filename = filename.split('.')[0] + '_changed.' + file_type
                os.rename(os.path.join(image_dir, filename), os.path.join(image_dir, new_filename))
                
    with open('train.txt', 'w') as f:
        f.write('\n'.join(train_set))

    with open('val.txt', 'w') as f:
        f.write('\n'.join(val_set))

    with open('test.txt', 'w') as f:
        f.write('\n'.join(test_set))


if __name__ == "__main__":

    main()