
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import importlib

params = {
    't_span': 11,  
    't_start': 0.1,
    'L': 20,  
    'M': 20,  
    'N':128,
    't_gap':0.00001,
    'exp_k': 0.05,
    'hidden_layers': 128,
    'epochs': 500,  
    'learning_rate': 0.001,
    'lambda_diag':20,
    'drop_out':0.1,
    'batch_size':50,
    'formula_name':'Brusselator',
    'model_name':'AFD-STA',
    'noise':0.2
}


def data_extract(params):

    module = importlib.import_module(f"{params['formula_name']}")
    integration = getattr(module, 'integration')

    train_data,train_labels=integration(params['L'],params['M'],params['t_gap'],params['t_start'],params['N'],params['t_span'])

    return train_data,train_labels

def model_train(model_name,train_data,train_labels, parameters):

    module = __import__(model_name)  
    
    avg_loss, model_save_path= module.train_model(parameters, train_data, train_labels)
    
    return avg_loss, model_save_path

def model_test(model_path,test_data,test_labels,parameters):
    from test import model_test
    persist,predict,label=model_test(model_path,test_data,test_labels,parameters)
    
    return persist,predict,label


def result_process(label_all_save,predict_all_save,persist_all,predict_all,label_all,t_start,t_gap):
    from result_process import plot_process
    plot_process(label_all_save,predict_all_save,persist_all,predict_all,label_all,t_start,t_gap)
    return


data,labels=data_extract(params)

persist_all = []
predict_all = []
label_all = []
Rmse_all= []
Mape_all = []
Mae_all = []
Smape_all = []
Mad_all = []

persist_all_save = []
predict_all_save = []
label_all_save = []

for i in range(params['N']):

    current_data=data
    current_labels=np.array(labels[i])

    predict_one=[]
    persist_one=[]
    label_one=[]

    for j in range(params['t_span']-1):
        current_train_data = current_data[j:j+1, :, :] + params['noise'] * np.random.randn(*current_data[j:j+1, :, :].shape)
        current_train_label = current_labels[j,:] + params['noise'] * np.random.randn(*current_labels[j,:].shape)
        current_test_data = current_data[j+1:j+2, :, :] + params['noise']* np.random.randn(*current_data[j+1:j+2, :, :].shape)
        current_test_label = current_labels[j+1:j+2,:] + params['noise'] * np.random.randn(*current_labels[j+1:j+2,:].shape)

        current_train_label = np.array([current_train_label[k:k+params['M']] for k in range(params['L']+1)])
        current_train_label = np.expand_dims(current_train_label, axis=0)
        current_test_label=current_test_label[0:1,params['M']-1:]

        avg_loss,model_save_path=model_train(params['model_name'],current_train_data,current_train_label, params)

        print("The {:}th point's {:}th prediction average Loss is {:}".format(i+1,j+1,avg_loss))
        
        persist,predict,label=model_test(model_save_path,current_test_data,current_test_label,params)

        persist=np.array(persist)
        predict=np.array(predict)
        label=np.array(label)

        persist_flattened = persist.flatten()
        predict_flattened = predict.flatten()
        label_flattened = label.flatten()

        Rmse_all.append(np.sqrt(np.mean((predict_flattened - label_flattened)**2)))

        epsilon = 1e-10
        mae = np.mean(np.abs(label_flattened - predict_flattened))
        Mae_all.append(mae)

        denominator = (np.abs(label_flattened) + np.abs(predict_flattened) + epsilon)
        smape = np.mean(2 * np.abs(label_flattened - predict_flattened) / denominator) * 100
        Smape_all.append(smape)

        Mad_all.append(np.median(np.abs((label_flattened - predict_flattened) - np.median(label_flattened - predict_flattened))))
        
        persist_all.append(persist)
        predict_all.append(predict)
        label_all.append(label)

        persist_one.append(persist_flattened)
        predict_one.append(predict_flattened)
        label_one.append(label_flattened)

    predict_one=np.array(predict_one).flatten()
    persist_one=np.array(persist_one).flatten()   
    label_one=np.array(label_one).flatten()

    persist_all_save.append(persist_one)
    predict_all_save.append(predict_one)
    label_all_save.append(label_one)


persist_all_save=np.array(persist_all_save)
predict_all_save=np.array(predict_all_save)
label_all_save=np.array(label_all_save)

persist_all=np.array(persist_all)
predict_all=np.array(predict_all)
label_all=np.array(label_all)

current_dir = os.path.dirname(os.path.realpath(__file__))  
picture_data_dir = os.path.join(current_dir, 'picture_data')  


os.makedirs(picture_data_dir, exist_ok=True)

print("Overall average RMSE is{}".format(np.mean(Rmse_all)))
print("Overall average MAE is{}".format(np.mean(Mae_all)))
print("Overall average SAMPE is{}".format(np.mean(Smape_all)))
print("Overall average MAD is{}".format(np.mean(Mad_all)))


pd.DataFrame(persist_all_save).to_excel(os.path.join(picture_data_dir, 'persist_all_save.xlsx'), index=False, header=False)
pd.DataFrame(predict_all_save).to_excel(os.path.join(picture_data_dir, 'predict_all_save.xlsx'), index=False, header=False)
pd.DataFrame(label_all_save).to_excel(os.path.join(picture_data_dir, 'label_all_save.xlsx'), index=False, header=False)
pd.DataFrame(Rmse_all).to_excel(os.path.join(picture_data_dir, 'RMSE_All_save.xlsx'), index=False, header=False)



result_process(label_all_save,predict_all_save,persist_all,predict_all,label_all,params['t_start'],params['t_gap'])