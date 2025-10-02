import utils.data_load_operate as data_load_operate

data_set_name_list = ['HanChuan', 'HongHu']
data_set_path = './data/'

if __name__ == '__main__':
    for name in data_set_name_list:
        data, gt = data_load_operate.load_data(name, data_set_path)
        print(f"Dataset: {name}, Data shape: {data.shape}, GT shape: {gt.shape}")