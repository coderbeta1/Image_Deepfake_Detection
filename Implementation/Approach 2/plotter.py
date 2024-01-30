import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

datasets_path = os.getcwd()

datasets_folders = [
                    '_laion_vs_ldm100',                
                    '_bigganReal_vs_bigganFake',
                    '_imagenet_vs_ldm200',
                    ]
models_name = ['_dino_vitb16',
               '_m1_m2',
                '_dino_resnet50',
                
                '_m1_m3',
                '_m2_m3',
                '_m1_m2_m3',
                '_clip_vitb16',
                ]
best = []
next_best = []
next_next_best = []

for i in range(len(models_name)):
    plot_datasets_path = os.path.join(datasets_path, 'all_datasets_accuracy'+models_name[i]+'.csv')
    information = pd.read_csv(plot_datasets_path)
    best.append(information.iloc[0,:][-1])
    next_best.append(information.iloc[1,:][-1])
    next_next_best.append(information.iloc[2,:][-1])

plt.plot([a[1:] for a in models_name], best, color='green', label="Best Classifer")
plt.plot([a[1:] for a in models_name], next_best, color='blue', label="2nd Best Classifer")
plt.plot([a[1:] for a in models_name], next_next_best, color='red', label="3rd Best Classifer")
plt.legend()
plt.xlabel("Backbone Model Used")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Backbone Model")
plt.grid()
plt.show()
'''
New Code starts Here
'''
'''datasets_path = os.getcwd()

datasets_folders = [
                    '_laion_vs_ldm100',                
                    '_bigganReal_vs_bigganFake',
                    '_imagenet_vs_ldm200',
                    ]
models_name = ['_dino_vitb16',
                '_dino_resnet50',
                '_clip_vitb16',
                ]

model_name = '_m2_m3'

all_data = []
for i in datasets_folders:
    alpha = []
    for j in datasets_folders:
        plot_datasets_path = os.path.join(datasets_path, 'results_cross_dataset')
        plot_datasets_path = os.path.join(plot_datasets_path, 'cross_datasets_accuracy'+model_name+i+j+'.csv')
        information = pd.read_csv(plot_datasets_path, index_col=False)
        alpha.append(max(information.iloc[0,:][-1], 1-information.iloc[-1,:][-1]))
    all_data.append(alpha)

data_table = pd.DataFrame(all_data, index=[a[1:] for a in datasets_folders], columns=[a[1:] for a in datasets_folders])
data_table.to_csv('cross_datasets_accuracy'+model_name+'.csv')
print(data_table)
'''

'''
New Code starts Here
'''

# datasets_path = os.getcwd()

# datasets_folders = [
#                     '_laion_vs_ldm100',                
#                     '_bigganReal_vs_bigganFake',
#                     '_imagenet_vs_ldm200',
#                     ]
# models_name = ['_dino_vitb16',
#                 '_dino_resnet50',
#                 '_clip_vitb16',
#                 ]

# all_data = []
# for i in datasets_folders:
#     alpha = []
#     for j in datasets_folders:
#         plot_datasets_path = os.path.join(datasets_path, 'results_cross_dataset')
#         plot_datasets_path = os.path.join(plot_datasets_path, 'cross_datasets_accuracy'+models_name[2]+i+j+'.csv')
#         information = pd.read_csv(plot_datasets_path, index_col=False)
#         alpha.append(max(information.iloc[0,:][-1], 1-information.iloc[-1,:][-1]))
#     all_data.append(alpha)

# data_table = pd.DataFrame(all_data, index=[a[1:] for a in datasets_folders], columns=[a[1:] for a in datasets_folders])
# data_table.to_csv('cross_datasets_accuracy'+models_name[2]+'.csv')
# print(data_table)

'''
New Code starts Here
'''


# datasets_path = os.getcwd()

# datasets_folders = [
#                     '_laion_vs_ldm100',                
#                     '_bigganReal_vs_bigganFake',
#                     '_imagenet_vs_ldm200',
#                     ]
# models_name = ['_dino_vitb16',
#                 '_dino_resnet50',
#                 '_clip_vitb16',
#                 ]

# all_data = []
# for i in datasets_folders:
#     alpha = []
#     for j in datasets_folders:
#         plot_datasets_path = os.path.join(datasets_path, 'results_cross_dataset')
#         plot_datasets_path = os.path.join(plot_datasets_path, 'cross_datasets_accuracy'+models_name[2]+i+j+'.csv')
#         information = pd.read_csv(plot_datasets_path, index_col=False)
#         alpha.append(max(information.iloc[0,:][-1], 1-information.iloc[-1,:][-1]))
#     all_data.append(alpha)

# data_table = pd.DataFrame(all_data, index=[a[1:] for a in datasets_folders], columns=[a[1:] for a in datasets_folders])
# data_table.to_csv('cross_datasets_accuracy'+models_name[2]+'.csv')
# print(data_table)

'''
New Code starts Here
'''

'''
datasets_path = os.getcwd()

datasets_folders = [
                    'laion_vs_ldm100_results',                
                    'bigganReal_vs_bigganFake_results',
                    'imagenet_vs_ldm200_results',
                    ]
models_name = ['_dino_vitb16',
                '_dino_resnet50',
                '_clip_vitb16',
                ]
transforms_name = ['_transform0',
                    '_transform1',
                    '_transform2',
                    '_transform3',
                    ]

data1 = []
data2 = []
data3 = []
for soc in range(len(datasets_folders)):
    transform0 = []
    transform1 = []
    transform2 = []
    transform3 = []

    data_used = datasets_folders[soc]

    for boc in range(len(transforms_name)):
        model1_det = 0
        model2_det = 0
        model3_det = 0
        for koc in range(len(models_name)):
            plot_datasets_path = os.path.join(datasets_path, data_used)
            plot_datasets_path = os.path.join(plot_datasets_path, 'models_accuracy'+models_name[koc]+transforms_name[boc]+'.csv')
            information = pd.read_csv(plot_datasets_path, index_col=False)
            # file_name=models_name[1][1:]+transforms_name[0]+'_dim_red_comparison' 
            curr_row = information.iloc[0,:]
            if koc==0:
                model1_det = curr_row[-1]
            if koc==1:
                model2_det = curr_row[-1]
            if koc==2:
                model3_det = curr_row[-1]
        if boc==0:
            transform0.append([model1_det, model2_det, model3_det])
        if boc==1:
            transform1.append([model1_det, model2_det, model3_det])
        if boc==2:
            transform2.append([model1_det, model2_det, model3_det])
        if boc==3:
            transform3.append([model1_det, model2_det, model3_det])
    if soc == 0:
        data1.append(transform0[0])
        data1.append(transform1[0])
        data1.append(transform2[0])
        data1.append(transform3[0])
    if soc == 1:
        data2.append(transform0[0])
        data2.append(transform1[0])
        data2.append(transform2[0])
        data2.append(transform3[0])
    if soc == 2:
        data3.append(transform0[0])
        data3.append(transform1[0])
        data3.append(transform2[0])
        data3.append(transform3[0])

print(data1)
print(data2)
print(data3)
name_printer = [a[1:] for a in models_name]
plt.plot(name_printer, data1[0], label=datasets_folders[0][:-8], color='red')
for i in data1[1:]:
    plt.plot(name_printer, i, color='red')

plt.plot(name_printer, data2[0], label=datasets_folders[1][:-8], color='green')
for i in data2[1:]:
    plt.plot(name_printer, i, color='green')

plt.plot(name_printer, data3[0], label=datasets_folders[2][:-8], color='blue')
for i in data3[1:]:
    plt.plot(name_printer, i, color='blue')

plt.xticks(rotation=0)
plt.xlabel("Backbone Model")
plt.ylabel("Test Accuracy")
plt.title("Accuracy vs Backbone Model Used (w.r.t. Dataset)")
plt.legend()
plt.grid()  
plt.show()
'''

'''
New Code starts Here
'''

'''run_type = 'test'      # Choose "train" or "test"

if run_type == 'test':
    datasets_path = os.getcwd()
    
    datasets_folders = [
                        'imagenet_vs_ldm200_results',
                        'laion_vs_ldm100_results',
                        'bigganReal_vs_bigganFake_results'
                        ]
    models_name = ['_dino_vitb16',
                   '_dino_resnet50',
                   '_clip_vitb16',
                  ]
    transforms_name = ['_transform0',
                       '_transform1',
                       '_transform2',
                       '_transform3',
                      ]


no_all = []
pca_all = []
auto_all = []

for boc in datasets_folders:
    for koc in range(len(models_name)):
        plot_datasets_path = os.path.join(datasets_path, boc)
        plot_datasets_path = os.path.join(plot_datasets_path, 'models_accuracy'+models_name[koc]+transforms_name[0]+'.csv')
        information = pd.read_csv(plot_datasets_path, index_col=False)
        file_name=models_name[1][1:]+transforms_name[0]+'_dim_red_comparison' 

        no_reduction = []
        autoencoding = []
        with_pca = []
        def comp(a):
            return a[0]

        for i in range(len(information)):
            curr_row = information.iloc[i,:]
            if curr_row[0][-3:] == 'ion': 
                no_reduction.append([curr_row[0][:-18],curr_row[-1]])
            elif curr_row[0][-3:] == 'ing': 
                autoencoding.append([curr_row[0][:-18],curr_row[-1]])
            elif curr_row[0][-3:] == 'pca': 
                with_pca.append([curr_row[0][:-9],curr_row[-1]])

        no_reduction = sorted(no_reduction, key=comp)
        autoencoding = sorted(autoencoding, key=comp)
        with_pca = sorted(with_pca, key=comp)

        key_string = [x[0] for x in no_reduction]
        xvals = [x[1] for x in no_reduction]
        yvals = [x[1] for x in with_pca]
        zvals = [x[1] for x in autoencoding] 
        xvals -= min(xvals)
        yvals -= min(yvals)
        zvals -= min(zvals)
        no_all.append(xvals)
        pca_all.append(yvals)
        auto_all.append(zvals)

plt.plot(key_string, no_all[0], label='No Reduction', color='red')
for i in no_all[1:]:
    plt.plot(key_string, i, color='red')

plt.plot(key_string, pca_all[0], label='PCA', color='green')
for i in pca_all[1:]:
    plt.plot(key_string, i, color='green')

plt.plot(key_string, auto_all[0], label='Autoencoding', color='blue')
for i in auto_all[1:]:
    plt.plot(key_string, i, color='blue')

plt.xticks(rotation=90)
plt.xlabel("Classification Technique")
plt.ylabel("Goodness Factor")
plt.title("Impact of Dimensionality Reduction")
plt.legend()
plt.grid()  
plt.show()
'''

'''
New CODE starts here
'''

# datasets_folders = ['imagenet_vs_ldm200_results',
#                     'laion_vs_ldm100_results',
#                     'bigganReal_vs_bigganFake_results'
#                     ]
# classification_methods = [  'linear_probe', 
#                             'svm', 
#                             'random_forest', 
#                             'gradient_boosting', 
#                             'naive_bayes', 
#                             'decision_tree', 
#                             'knn1', 
#                             'knn3', 
#                             'knn5', 
#                             'bagging_classifier', 
#                             'qda', 
#                             'lda',
#                             ]
# dim_red_methods = [ 'no_reduction', 
#                     'pca', 
#                     'autoencoding',
#                     ]
# models_name = ['_dino_vitb16',
#                 '_dino_resnet50',
#                 '_clip_vitb16',
#                 ]
# transforms_name = ['_transform0',
#                     '_transform1',
#                     '_transform2',
#                     '_transform3',
#                     ]

# all_vect=[]
# for data_set in range(len(datasets_folders)):
#     for alp in range(len(models_name)):
#         to_comp = []
#         for i in range(4):
#             datasets_path = os.getcwd()
#             plot_datasets_path = os.path.join(datasets_path, datasets_folders[data_set])
#             plot_datasets_path = os.path.join(plot_datasets_path, 'models_accuracy'+models_name[alp]+transforms_name[i]+'.csv')
#             information = pd.read_csv(plot_datasets_path, index_col=False)
#             to_comp.append(information)
#         value_store = []
#         for i in range(len(to_comp[0])):
#             curr_row = to_comp[0].iloc[i,:]
#             curr_list = []
#             curr_list.append(curr_row[-1])
#             for j in range(1,4):
#                 curr_list.append(to_comp[j][to_comp[j][to_comp[j].columns[0]] == curr_row[0]].iloc[0,:][-1])
#             value_store.append(curr_list-min(curr_list))

#         final_ans = []
#         for i in range(len(value_store[0])):
#             summu = 0
#             for j in value_store:
#                 summu+=j[i]
#             final_ans.append(summu/len(value_store))

#         all_vect.append(final_ans)

# for i in all_vect:
#     plt.plot(['transform0', 'transform1', 'transform2','transform3'], i)
#     plt.grid()
#     plt.xlabel("Transformation Technique")
#     plt.ylabel("Goodness Factor")
#     plt.title("Impact of Transformations on Training Data")
# plt.show()


'''
New CODE starts here
'''

# run_type = 'test'      # Choose "train" or "test"

# # def load_saved_datasets(address=None, root_loc=None, model_name=None, transforms_name=None):
# #     if address:
# #         with open(address[0], 'rb') as file:
# #             feature_maps_train = pickle.load(file)
# #         with open(address[1], 'rb') as file:
# #             feature_labels_train = pickle.load(file)
# #         with open(address[2], 'rb') as file:
# #             feature_maps_test = pickle.load(file)
# #         with open(address[3], 'rb') as file:
# #             feature_labels_test = pickle.load(file)
# #     else:
# #         with open(os.path.join(root_loc, 'feature_maps_train'+model_name+transforms_name+'.pkl'), 'rb') as file:
# #             feature_maps_train = pickle.load(file)
# #         with open(os.path.join(root_loc, 'feature_labels_train'+model_name+transforms_name+'.pkl'), 'rb') as file:
# #             feature_labels_train = pickle.load(file)
# #         with open(os.path.join(root_loc, 'feature_maps_test'+model_name+transforms_name+'.pkl'), 'rb') as file:
# #             feature_maps_test = pickle.load(file)
# #         with open(os.path.join(root_loc, 'feature_labels_test'+model_name+transforms_name+'.pkl'), 'rb') as file:
# #             feature_labels_test = pickle.load(file)
# #     return feature_maps_train, feature_labels_train, feature_maps_test, feature_labels_test
# # # Create the dataset here
# if run_type == 'test':
#     datasets_path = os.getcwd()
    
#     datasets_folders = [
#                         'imagenet_vs_ldm200_results',
#                         'laion_vs_ldm100_results',
#                         'bigganReal_vs_bigganFake_results'
#                         ]
#     models_name = ['_dino_vitb16',
#                    '_dino_resnet50',
#                    '_clip_vitb16',
#                   ]
#     transforms_name = ['_transform0',
#                        '_transform1',
#                        '_transform2',
#                        '_transform3',
#                       ]

#     feature_maps_train = []
#     feature_labels_train = []
#     feature_maps_test = []
#     feature_labels_test = []

#     for folder in datasets_folders:
#         for curr_model in models_name:
#             for transformation in transforms_name:
#                 root_loc = os.path.join(datasets_path, folder)
#                 feature_train, labels_train, feature_test, labels_test = load_saved_datasets(root_loc=root_loc, model_name=curr_model, transforms_name=transformation)
#                 feature_maps_train += feature_train
#                 feature_labels_train += labels_train
#                 feature_maps_test += feature_test
#                 feature_labels_test += labels_test

# print("Train Data:", len(feature_labels_train))
# print("Test Data:", len(feature_labels_test))


# plot_datasets_path = os.path.join(datasets_path, "bigganReal_vs_bigganFake_results")
# plot_datasets_path = os.path.join(plot_datasets_path, 'models_accuracy'+models_name[2]+transforms_name[0]+'.csv')
# information = pd.read_csv('all_datasets_accuracy'+models_name[1]+'_transform0.csv', index_col=False)
# file_name=models_name[1][1:]+transforms_name[0]+'_dim_red_comparison' 

# no_reduction = []
# autoencoding = []
# with_pca = []
# def comp(a):
#     return a[0]

# for i in range(len(information)):
#     curr_row = information.iloc[i,:]
#     if curr_row[0][-3:] == 'ion': 
#         no_reduction.append([curr_row[0][:-18],curr_row[-1]])
#     elif curr_row[0][-3:] == 'ing': 
#         autoencoding.append([curr_row[0][:-18],curr_row[-1]])
#     elif curr_row[0][-3:] == 'pca': 
#         with_pca.append([curr_row[0][:-9],curr_row[-1]])

# no_reduction = sorted(no_reduction, key=comp)
# autoencoding = sorted(autoencoding, key=comp)
# with_pca = sorted(with_pca, key=comp)

# key_string = [x[0] for x in no_reduction]

# print(key_string)

# sizer = 3
# N = len(key_string)
# ind = np.arange(N)
# width = 0.2

# plt.figure(figsize=(8, 6))
# xvals = [x[1] for x in no_reduction]
# bar1 = plt.bar(ind, xvals, width, color = 'r')

# if sizer>1:
#     yvals = [x[1] for x in with_pca]
#     bar2 = plt.bar(ind+width, yvals, width, color='g') 
# if sizer>2:
#     zvals = [x[1] for x in autoencoding] 
#     bar3 = plt.bar(ind+width*2, zvals, width, color = 'b') 

# plt.xlabel("Classification Method") 
# plt.ylabel('Test Accuracy') 
# plt.title("Test Accuracy vs Classification Method") 
# if sizer == 1:
#     plt.xticks(ind, classification_methods, rotation = 90)
# if sizer == 2:
#     plt.xticks(ind+width/2, classification_methods, rotation = 90)
# elif sizer == 3:
#     plt.xticks(ind+width, key_string, rotation = 90)
# plt.yticks(np.arange(0,11)/10)
# if sizer==1: 
#     plt.legend((bar1), tuple(dim_red_methods))
# elif sizer==2: 
#     plt.legend((bar1, bar2), tuple(dim_red_methods))
# elif sizer==3: 
#     plt.legend((bar1, bar2, bar3), tuple(dim_red_methods)) 

# print(file_name)
# # plt.savefig(file_name+".png")
# plt.show()