%% 

clear,clc

% 读取数据就是SEED数据集下的ExtractFeatures文件夹，但是需要把文件分出session，分别放在session1 2 3下面
load_path = './SEED/ExtractedFeatures1/session3/';
save_path = './SEED/SEED_code/DE/session3/';
label_path = './SEED/ExtractedFeatures1/label';
load(label_path)
file_list = dir(load_path);
file_list(1:2)=[];

for sub_i = 1:length(file_list)
    disp(['subject>',file_list(sub_i).name])
    S = load([load_path,file_list(sub_i).name]);
    DE = [];
    labelAll = [];
    for ii = 1:15
        eval(['data=','S.de_LDS',num2str(ii),';']);
        DE = cat(2,DE,data);
        labelAll = [labelAll;zeros(size(data,2),1)+label(ii)];
    end
    
    save([save_path,file_list(sub_i).name],'DE','labelAll');
    
end
    
