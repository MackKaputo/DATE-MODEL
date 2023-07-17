import numpy as np 
import pandas as pd 
import pickle
import copy
import os 
import xgboost as xgb
from xgboost import XGBClassifier
from utils import find_best_threshold,process_leaf_idx,stratify_sample
from sklearn.metrics import f1_score,roc_auc_score
import torch
import torch.utils.data as Data
import warnings
warnings.filterwarnings("ignore")

# load preprocessed data
with open("./processed_data.pickle","rb") as f :
    processed_data = pickle.load(f)
# processed_data = pd.read_pickle("./processed_data.pickle")

print(processed_data.keys())
print("Finish loading data...")

# train/test data 
train = processed_data["raw"]["train"]
valid = processed_data["raw"]["valid"]
test = processed_data["raw"]["test"]

# Revenue data for regression target 
revenue_train, revenue_valid,revenue_test = processed_data["revenue"]["train"],\
                                            processed_data["revenue"]["valid"],\
                                            processed_data["revenue"]["test"]

# normalize revenue by f(x) = log(x+1)/max(xi)
norm_revenue_train, norm_revenue_test = np.log(revenue_train+1), np.log(revenue_test+1) 
global_max = max(norm_revenue_train) 
norm_revenue_train = norm_revenue_train/global_max

# Xgboost data 
xgb_trainx = processed_data["xgboost_data"]["train_x"]
xgb_trainy = processed_data["xgboost_data"]["train_y"]
xgb_validx = processed_data["xgboost_data"]["valid_x"]
xgb_validy = processed_data["xgboost_data"]["valid_y"]
xgb_testx = processed_data["xgboost_data"]["test_x"]
xgb_testy = processed_data["xgboost_data"]["test_y"]

# build xgboost model
print("Training xgboost model...")
columns = ['fob.value', 'cif.value', 'total.taxes', 'gross.weight', 'quantity', 'Unitprice', 'WUnitprice', 'TaxRatio', 'FOBCIFRatio', 'TaxUnitquantity', 'tariff.code', 'HS6', 'HS4', 'HS2', 'SGD.DayofYear', 'SGD.WeekofYear', 'SGD.MonthofYear'] + [col for col in train.columns if 'RiskH' in col] 
xgb_trainx = pd.DataFrame(xgb_trainx,columns=columns)
xgb_trainx['fob.value'] = xgb_trainx['fob.value'].astype('float')
xgb_trainx['cif.value'] = xgb_trainx['cif.value'].astype('float')
xgb_trainx['total.taxes'] = xgb_trainx['total.taxes'].astype('float')
xgb_trainx['gross.weight'] = xgb_trainx['gross.weight'].astype('float')
xgb_trainx['quantity'] = xgb_trainx['quantity'].astype('float')
xgb_trainx['Unitprice'] = xgb_trainx['Unitprice'].astype('float')
xgb_trainx['WUnitprice'] = xgb_trainx['WUnitprice'].astype('float')
xgb_trainx['TaxRatio'] = xgb_trainx['TaxRatio'].astype('float')
xgb_trainx['FOBCIFRatio'] = xgb_trainx['FOBCIFRatio'].astype('float')
xgb_trainx['TaxUnitquantity'] = xgb_trainx['TaxUnitquantity'].astype('float')

xgb_trainx['tariff.code'] = xgb_trainx['tariff.code'].astype('int')
xgb_trainx['HS6'] = xgb_trainx['HS6'].astype('int')
xgb_trainx['HS4'] = xgb_trainx['HS4'].astype('int')
xgb_trainx['HS2'] = xgb_trainx['HS2'].astype('int')
xgb_trainx['SGD.DayofYear'] = xgb_trainx['SGD.DayofYear'].astype('int')
xgb_trainx['SGD.WeekofYear'] = xgb_trainx['SGD.WeekofYear'].astype('int')
xgb_trainx['SGD.MonthofYear'] = xgb_trainx['SGD.MonthofYear'].astype('int')

for col in train.columns:
    if 'RiskH' in col:
        xgb_trainx[col] = xgb_trainx[col].astype('int')

xgb_testx = pd.DataFrame(xgb_testx,columns=columns)

xgb_testx['fob.value'] = xgb_testx['fob.value'].astype('float')
xgb_testx['cif.value'] = xgb_testx['cif.value'].astype('float')
xgb_testx['total.taxes'] = xgb_testx['total.taxes'].astype('float')
xgb_testx['gross.weight'] = xgb_testx['gross.weight'].astype('float')
xgb_testx['quantity'] = xgb_testx['quantity'].astype('float')
xgb_testx['Unitprice'] = xgb_testx['Unitprice'].astype('float')
xgb_testx['WUnitprice'] = xgb_testx['WUnitprice'].astype('float')
xgb_testx['TaxRatio'] = xgb_testx['TaxRatio'].astype('float')
xgb_testx['FOBCIFRatio'] = xgb_testx['FOBCIFRatio'].astype('float')
xgb_testx['TaxUnitquantity'] = xgb_testx['TaxUnitquantity'].astype('float')

xgb_testx['tariff.code'] = xgb_testx['tariff.code'].astype('int')
xgb_testx['HS6'] = xgb_testx['HS6'].astype('int')
xgb_testx['HS4'] = xgb_testx['HS4'].astype('int')
xgb_testx['HS2'] = xgb_testx['HS2'].astype('int')
xgb_testx['SGD.DayofYear'] = xgb_testx['SGD.DayofYear'].astype('int')
xgb_testx['SGD.WeekofYear'] = xgb_testx['SGD.WeekofYear'].astype('int')
xgb_testx['SGD.MonthofYear'] = xgb_testx['SGD.MonthofYear'].astype('int')

for col in train.columns:
    if 'RiskH' in col:
        xgb_testx[col] = xgb_testx[col].astype('int')

#print(xgb_testx)
xgb_clf = XGBClassifier(n_estimators=100, max_depth=4,n_jobs=-1)
xgb_clf.fit(xgb_trainx,xgb_trainy)
# evaluate xgboost model
print("------Evaluating xgboost model------")
test_pred = xgb_clf.predict_proba(xgb_testx)[:,1]

xgb_validx = pd.DataFrame(xgb_validx,columns=columns)
xgb_validx['fob.value'] = xgb_validx['fob.value'].astype('float')
xgb_validx['cif.value'] = xgb_validx['cif.value'].astype('float')
xgb_validx['total.taxes'] = xgb_validx['total.taxes'].astype('float')
xgb_validx['gross.weight'] = xgb_validx['gross.weight'].astype('float')
xgb_validx['quantity'] = xgb_validx['quantity'].astype('float')
xgb_validx['Unitprice'] = xgb_validx['Unitprice'].astype('float')
xgb_validx['WUnitprice'] = xgb_validx['WUnitprice'].astype('float')
xgb_validx['TaxRatio'] = xgb_validx['TaxRatio'].astype('float')
xgb_validx['FOBCIFRatio'] = xgb_validx['FOBCIFRatio'].astype('float')
xgb_validx['TaxUnitquantity'] = xgb_validx['TaxUnitquantity'].astype('float')

xgb_validx['tariff.code'] = xgb_validx['tariff.code'].astype('int')
xgb_validx['HS6'] = xgb_validx['HS6'].astype('int')
xgb_validx['HS4'] = xgb_validx['HS4'].astype('int')
xgb_validx['HS2'] = xgb_validx['HS2'].astype('int')
xgb_validx['SGD.DayofYear'] = xgb_validx['SGD.DayofYear'].astype('int')
xgb_validx['SGD.WeekofYear'] = xgb_validx['SGD.WeekofYear'].astype('int')
xgb_validx['SGD.MonthofYear'] = xgb_validx['SGD.MonthofYear'].astype('int')

for col in train.columns:
    if 'RiskH' in col:
        xgb_validx[col] = xgb_validx[col].astype('int')
        
xgb_auc = roc_auc_score(xgb_testy, test_pred)
xgb_threshold,_ = find_best_threshold(xgb_clf, xgb_validx, xgb_validy)
xgb_f1 = find_best_threshold(xgb_clf, xgb_testx, xgb_testy,best_thresh=xgb_threshold)
print("AUC = %.4f, F1-score = %.4f" % (xgb_auc, xgb_f1))

# Precision and Recall
y_prob = test_pred
for i in [99,98,95,90]:
    threshold = np.percentile(y_prob, i)
    print(f'Checking top {100-i}% suspicious transactions: {len(y_prob[y_prob > threshold])}')
    precision = np.mean(xgb_testy[y_prob > threshold])
    recall = sum(xgb_testy[y_prob > threshold])/sum(xgb_testy)
    revenue_recall = sum(revenue_test[y_prob > threshold]) /sum(revenue_test)
    print(f'Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, Seized Revenue (Recall): {round(revenue_recall, 4)}')

xgb_clf.get_booster().dump_model('xgb_model.txt', with_stats=False)

# Xgboost+LR model 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import  OneHotEncoder

# get leaf index from xgboost model 
X_train_leaves = xgb_clf.apply(xgb_trainx)
X_valid_leaves = xgb_clf.apply(xgb_validx)
X_test_leaves = xgb_clf.apply(xgb_testx)
train_rows = X_train_leaves.shape[0]

# one-hot encoding for leaf index
xgbenc = OneHotEncoder(categories="auto")
lr_trainx = xgbenc.fit_transform(X_train_leaves)
lr_validx = xgbenc.transform(X_valid_leaves)
lr_testx = xgbenc.transform(X_test_leaves)

# model 
print("Training Logistic regression model...")
lr = LogisticRegression(n_jobs=-1)
lr.fit(lr_trainx, xgb_trainy)
test_pred = lr.predict_proba(lr_testx)[:,1]
print("------Evaluating xgboost+LR model------")
xgb_auc = roc_auc_score(xgb_testy, test_pred)
xgb_threshold,_ = find_best_threshold(lr, lr_validx, xgb_validy) # threshold was select from validation set
xgb_f1 = find_best_threshold(lr, lr_testx, xgb_testy,best_thresh=xgb_threshold) # then applied on test set
print("AUC = %.4f, F1-score = %.4f" % (xgb_auc, xgb_f1))

# Precision and Recall
y_prob = test_pred
for i in [99,98,95,90]:
    threshold = np.percentile(y_prob, i)
    print(f'Checking top {100-i}% suspicious transactions: {len(y_prob[y_prob > threshold])}')
    precision = np.mean(xgb_testy[y_prob > threshold])
    recall = sum(xgb_testy[y_prob > threshold])/sum(xgb_testy)
    revenue_recall = sum(revenue_test[y_prob > threshold]) /sum(revenue_test)
    print(f'Precision: {round(precision, 4)}, Recall: {round(recall, 4)}, Seized Revenue (Recall): {round(revenue_recall, 4)}')

# user & item information 
train_raw_importers = train['importer.id'].values
train_raw_items = train['tariff.code'].values
valid_raw_importers = valid['importer.id'].values
valid_raw_items = valid['tariff.code'].values
test_raw_importers = test['importer.id']
test_raw_items = test['tariff.code']

# we need padding for unseen user or item 
importer_set = set(train_raw_importers)
item_set = set(train_raw_items)

# Remember to +1 for zero padding 
importer_mapping = {v:i+1 for i,v in enumerate(importer_set)} 
hs6_mapping = {v:i+1 for i,v in enumerate(item_set)}
importer_size = len(importer_mapping) + 1
item_size = len(hs6_mapping) + 1
train_importers = [importer_mapping[x] for x in train_raw_importers]
train_items = [hs6_mapping[x] for x in train_raw_items]

# for test data, we use padding_idx=0 for unseen data
valid_importers = [importer_mapping.get(x,0) for x in valid_raw_importers]
valid_items = [hs6_mapping.get(x,0) for x in valid_raw_items]
test_importers = [importer_mapping.get(x,0) for x in test_raw_importers] # use dic.get(key,deafault) to handle unseen
test_items = [hs6_mapping.get(x,0) for x in test_raw_items]

# Preprocess
train_rows = train.shape[0]
valid_rows = valid.shape[0] + train_rows
X_leaves = np.concatenate((X_train_leaves, X_valid_leaves, X_test_leaves), axis=0) # make sure the dimensionality
transformed_leaves, leaf_num, new_leaf_index = process_leaf_idx(X_leaves)
train_leaves, valid_leaves, test_leaves = transformed_leaves[:train_rows],\
                                          transformed_leaves[train_rows:valid_rows],\
                                          transformed_leaves[valid_rows:]

# Convert to torch type
train_leaves = torch.tensor(train_leaves).long()
train_user = torch.tensor(train_importers).long()
train_item = torch.tensor(train_items).long()

valid_leaves = torch.tensor(valid_leaves).long()
valid_user = torch.tensor(valid_importers).long()
valid_item = torch.tensor(valid_items).long()

test_leaves = torch.tensor(test_leaves).long()
test_user = torch.tensor(test_importers).long()
test_item = torch.tensor(test_items).long()

# cls data
train_label_cls = torch.tensor(xgb_trainy).float()
valid_label_cls = torch.tensor(xgb_validy).float()
test_label_cls = torch.tensor(xgb_testy).float()

# revenue data 
train_label_reg = torch.tensor(norm_revenue_train).float()
valid_label_reg = torch.tensor(revenue_valid).float()
test_label_reg = torch.tensor(revenue_test).float()

# create dataloader 

train_dataset = Data.TensorDataset(train_leaves,train_user,train_item,train_label_cls,train_label_reg)
valid_dataset = Data.TensorDataset(valid_leaves,valid_user,valid_item,valid_label_cls,valid_label_reg)
test_dataset = Data.TensorDataset(test_leaves,test_user,test_item,test_label_cls,test_label_reg)



data4embedding = {"train_dataset":train_dataset,"valid_dataset":valid_dataset,"test_dataset":test_dataset,\
                  "leaf_num":leaf_num,"importer_num":importer_size,"item_size":item_size}

# save data
with open("torch_data.pickle", 'wb') as f:
    pickle.dump(data4embedding, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("leaf_index.pickle", "wb") as f:
    pickle.dump(new_leaf_index, f, protocol=pickle.HIGHEST_PROTOCOL)
