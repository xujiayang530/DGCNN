import shutil
import os

extrac_path = '/NAS_REMOTE/wangzihan/SEED/ExtractedFeatures/'
label_path = '/NAS_REMOTE/wangzihan/SEED/ExtractedFeatures/label.mat'
save_path = './SEED/SEED_code/ExtractedFeatures/'

dir_list = os.listdir(extrac_path)

dic = {}

for f in dir_list:
    if '_' not in f:
        continue

#    print(extrac_path + f)
    sub = f.split('.')[0].split('_')[0]
    date = f.split('.')[0].split('_')[1]
    if sub not in dic.keys():
        dic[sub] = []
    dic[sub].append(date)


for i in range(3):
    folder = save_path + "session" + str(i+1) + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    for k, v in dic.items():
        v.sort()
        f = k + '_' + v[0] + '.mat'
        del v[0]
        shutil.copyfile(extrac_path + f, folder + f)

        print(extrac_path + f, '->' , folder + f)

    shutil.copyfile(label_path, folder + 'label.mat')
    print(label_path, '->' , folder + 'label.mat')

    print("")

#print(dic)
