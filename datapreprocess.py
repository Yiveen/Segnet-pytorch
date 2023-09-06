import os
import random
import numpy as np

def main():
    image_dir = 'mini_dataset_vima'
    image_names = [filename for filename in os.listdir(image_dir) if not filename.endswith('.txt')]
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
    num_files_per_type = 1  # 选择每个类型的文件数量 

    # 遍历训练集、验证集和测试集
    for dataset in [train_set, val_set, test_set]:
        # random_numbers = np.random.randint(0, len(dataset), num_files_per_type)
        # random_numbers_list = random_numbers.tolist()  
        # for filename in dataset:
        #     number = int(filename)
        #     if number in random_numbers_list:
        sampled_indices = random.sample(range(len(dataset)), 1)
        for indice in sampled_indices:
                new_filename = str(dataset[indice]) + '_changed'
                dataset[indice] = new_filename 
                # os.rename(os.path.join(image_dir, filename), os.path.join(image_dir, new_filename))
                
    with open(str(image_dir)+ '/' + 'train.txt', 'w') as f:
        f.write('\n'.join(train_set))
        
        with open('background.txt', 'w') as f:

    with open(str(image_dir)+ '/' + 'val.txt', 'w') as f:
        f.write('\n'.join(val_set))

    with open(str(image_dir)+ '/' + 'test.txt', 'w') as f:
        f.write('\n'.join(test_set))


if __name__ == "__main__":

    main()