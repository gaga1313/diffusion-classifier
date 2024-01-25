import json
import os
import random
import torch

random.seed(43)
file_paths = ['0571_nse_dnn_0.00_bicycle_10_n03792782_1418.png', '0545_nse_dnn_0.00_boat_10_n04612504_30617.png', '0793_nse_dnn_0.10_elephant_10_n02504458_2054.png', '0151_nse_dnn_0.10_cat_10_n02123045_3098.png', '0860_nse_dnn_0.90_bear_10_n02132136_4636.png', '0908_nse_dnn_0.60_chair_10_n02791124_5089.png', '0649_nse_dnn_0.60_airplane_10_n02690373_6665.png', '0598_nse_dnn_0.35_clock_10_n04548280_156.png', '0695_nse_dnn_0.90_boat_10_n03344393_771.png', '1125_nse_dnn_0.20_bottle_10_n03937543_2105.png', '0894_nse_dnn_0.35_keyboard_10_n03085013_20784.png', '1028_nse_dnn_0.00_clock_10_n04548280_4181.png', '0581_nse_dnn_0.90_keyboard_10_n04505470_3499.png', '1199_nse_dnn_0.60_oven_10_n04111531_10216.png', '0180_nse_dnn_0.20_oven_10_n04111531_4467.png', '0617_nse_dnn_0.05_clock_10_n03196217_3987.png', '0919_nse_dnn_0.20_bird_10_n02033041_17119.png', '0819_nse_dnn_0.00_bicycle_10_n03792782_13608.png', '0740_nse_dnn_0.35_dog_10_n02097474_10860.png', '0614_nse_dnn_0.00_cat_10_n02123394_1602.png', '0713_nse_dnn_0.03_bicycle_10_n02835271_6167.png', '0748_nse_dnn_0.05_dog_10_n02086646_4881.png', '1029_nse_dnn_0.03_bear_10_n02132136_6967.png', '1277_nse_dnn_0.10_dog_10_n02089867_4717.png', '1008_nse_dnn_0.60_car_10_n03100240_9875.png', '1084_nse_dnn_0.10_oven_10_n04111531_7965.png', '0924_nse_dnn_0.60_cat_10_n02125311_22037.png', '0166_nse_dnn_0.00_elephant_10_n02504013_1647.png', '1260_nse_dnn_0.10_bird_10_n02051845_3479.png', '0797_nse_dnn_0.00_boat_10_n03662601_31508.png', '0671_nse_dnn_0.20_bottle_10_n04557648_11992.png', '0410_nse_dnn_0.05_bear_10_n02132136_3224.png', '0863_nse_dnn_0.05_keyboard_10_n04505470_1043.png', '0988_nse_dnn_0.05_chair_10_n02791124_2470.png', '0148_nse_dnn_0.35_dog_10_n02094258_1224.png', '0926_nse_dnn_0.60_chair_10_n03376595_8995.png', '0080_nse_dnn_0.03_airplane_10_n02690373_3860.png', '0772_nse_dnn_0.35_truck_10_n04467665_47602.png', '0734_nse_dnn_0.20_knife_10_n03041632_7448.png', '1205_nse_dnn_0.60_elephant_10_n02504013_5996.png', '0723_nse_dnn_0.35_knife_10_n03041632_2153.png', '0517_nse_dnn_0.03_bird_10_n01860187_24935.png', '0981_nse_dnn_0.60_keyboard_10_n03085013_25066.png', '0590_nse_dnn_0.35_cat_10_n02123597_7615.png', '0725_nse_dnn_0.60_airplane_10_n02690373_3377.png', '0857_nse_dnn_0.20_bicycle_10_n03792782_11417.png', '1216_nse_dnn_0.03_chair_10_n02791124_5795.png', '0047_nse_dnn_0.90_chair_10_n02791124_5400.png', '0293_nse_dnn_0.03_bicycle_10_n03792782_6613.png', '0334_nse_dnn_0.90_boat_10_n02951358_30335.png', '0953_nse_dnn_0.03_bicycle_10_n02835271_3135.png', '0983_nse_dnn_0.10_knife_10_n03041632_11887.png', '1195_nse_dnn_0.03_truck_10_n03417042_11704.png', '0418_nse_dnn_0.00_boat_10_n02951358_7315.png', '0750_nse_dnn_0.03_truck_10_n03796401_6444.png', '0585_nse_dnn_0.90_bicycle_10_n03792782_4067.png', '0627_nse_dnn_0.10_bottle_10_n03983396_10251.png', '0462_nse_dnn_0.90_car_10_n04285008_15003.png', '0588_nse_dnn_0.90_knife_10_n03041632_4879.png', '1018_nse_dnn_0.03_clock_10_n04548280_6068.png', '1228_nse_dnn_0.20_knife_10_n03041632_44269.png', '0851_nse_dnn_0.05_clock_10_n03196217_3868.png', '0432_nse_dnn_0.35_bottle_10_n04557648_2954.png', '0779_nse_dnn_0.10_truck_10_n03417042_5380.png', '0249_nse_dnn_0.35_truck_10_n03417042_6847.png', '0948_nse_dnn_0.90_knife_10_n03041632_41493.png', '1093_nse_dnn_0.90_boat_10_n04273569_18898.png', '1191_nse_dnn_0.00_elephant_10_n02504458_291.png', '0434_nse_dnn_0.10_keyboard_10_n04505470_6876.png', '0205_nse_dnn_0.03_bottle_10_n04557648_12671.png', '0818_nse_dnn_0.05_keyboard_10_n03085013_24577.png', '0059_nse_dnn_0.05_oven_10_n04111531_16583.png', '0187_nse_dnn_0.05_dog_10_n02101388_2968.png', '0610_nse_dnn_0.05_knife_10_n03041632_99912.png', '0800_nse_dnn_0.20_bottle_10_n04560804_9981.png', '0083_nse_dnn_0.60_bear_10_n02133161_4882.png', '1135_nse_dnn_0.20_bottle_10_n02823428_7570.png', '0501_nse_dnn_0.20_boat_10_n03344393_11590.png', '0330_nse_dnn_0.90_cat_10_n02124075_9291.png', '0847_nse_dnn_0.00_bear_10_n02132136_25897.png', '0437_nse_dnn_0.00_car_10_n03100240_11400.png', '0008_nse_dnn_0.00_chair_10_n03376595_16202.png', '0592_nse_dnn_0.10_clock_10_n02708093_527.png', '1181_nse_dnn_0.05_dog_10_n02112350_5035.png', '0170_nse_dnn_0.05_car_10_n02814533_35054.png', '0651_nse_dnn_0.10_knife_10_n03041632_90407.png', '0596_nse_dnn_0.35_boat_10_n04273569_12536.png', '0044_nse_dnn_0.20_elephant_10_n02504013_2464.png', '1186_nse_dnn_0.10_bottle_10_n03983396_11859.png', '0272_nse_dnn_0.20_elephant_10_n02504013_8522.png', '0859_nse_dnn_0.90_bicycle_10_n03792782_30942.png', '0381_nse_dnn_0.60_bicycle_10_n03792782_15679.png', '1098_nse_dnn_0.60_car_10_n02814533_8232.png', '0624_nse_dnn_0.35_car_10_n02814533_81455.png', '0211_nse_dnn_0.03_truck_10_n04467665_67908.png', '0090_nse_dnn_0.35_bear_10_n02132136_2760.png']

unique_filter = {}
class_results = {}
for i,fp in enumerate(file_paths):

    fp = fp.split('_')
    nl = fp[3]
    if nl in unique_filter:
        unique_filter[nl].append(i)
    else:
        unique_filter[nl] = [i]
        class_results[nl] = []
print(unique_filter)

data_folder = '/jobtmp/ggaonkar/diffusion-classifier/data/image16_uniform_noise'

results = os.listdir(data_folder)

for f in results:
    index = int(results.split('.')[0])
    x = torch.load(os.path.join(data_folder, results))
    for key,value in unique_filter.items():
        if index in value:
            if x['pred']==x['label']:
                class_results[key].append(1)
            else:
                class_results[key].append(0)

for key, value in class_results.items():
    print(f'level : {value}')
