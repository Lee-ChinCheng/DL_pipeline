import os, time, json, pickle, random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import importlib
from pathlib import Path
import numpy as np, tensorflow as tf
from training_func.callbacks import *
from training_func.data_processor import load_data
from training_func.data_processor import load_data_v2
from training_func.custom_model import load_model
from training_func.plotting import *
from training_func.utils import *
import training_func.environment as env
time_start=time.time()
st_train=datetime.now()
st_time=st_train.strftime("%Y-%m-%d %H:%M:%S")




W_path_str = 'Hemo'



# cd RDDL # conda activate py310
# python run_train.py -name AFP -m Baseline -nor 0 -v 2

# python run_train.py -name Hemo_predi -m CW -nor 0 -v 2

# python run_train.py -name Hemo -m OS1-1 -v 2





def train_log(W_path_str, content, write_mode):

    logtxt =f'{W_path_str}/log.txt'

    if not os.path.exists(logtxt):
        open(logtxt, "w").close()  # create empty file

    if write_mode == 0: #direct append ex: tabulate, date
        with open(logtxt, 'a') as file:
            file.write(content + "\n")  

    elif write_mode == 1: #dictionary k,v ex: args_p_d.items()
        with open(logtxt, 'a') as file:
            for a_key, a_va in content: #args_p_d.items()
                file.write(f"{a_key}: {a_va}\n")  
    
    elif write_mode == 2: #list k,v ex: op_li_2
        with open(logtxt, 'a') as file:
            for ele in content:
                file.write(str(ele)+'\n')
    else:
        print(' invalid write_mode')



def testset_prediction(KFOLD, csv_path, prob_output_folder, table_title):

    pf_li=[]
    combined_labels=[]
    
    for i in range(1, KFOLD+1):

        combined_labels.append(f'model{i}')
        md_path=f'{W_path_str}/outputs/models/{a.method}_model_{i}.keras'
        model = load_model(md_path) 
        input_num = len(model.inputs)
    
        pred_gen, pred_steps, pred_labels = load_data_v2(csv_path, 'test', None, input_num, z_mean=0, z_std=1)    
        pred_result = model.predict(pred_gen, steps=pred_steps, verbose=0)

        #print(pred_result.shape) 
        la_li, prb_li = pred_labels, pred_result[:, 1].tolist()

        mtx=metric_scores(la_li, prb_li)
        pf_li.append(mtx)
        t1_prob_txt = f'{prob_output_folder}/test1_model{i}.txt'
        write_prob_txt(la_li, prb_li, t1_prob_txt)

    str_table = show_table([_.values() for _ in pf_li],
                headers=pf_li[0].keys(),
                v_headers=combined_labels,
                title=table_title, float_fmt='%.3f')

    del model
    env.memory_recovery()

    return str_table




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-name', default='RDDL', type=case_name,
                        help='name of this deep learning project')
    parser.add_argument('-m', choices=balancing_methods(), required=True,
                        help='the balanced method', metavar='METHOD', dest='method')
    parser.add_argument('-nor', default=0, type=int, choices=[0, 1, 2],
                        help='the specified normalization option of this balancing scheme', dest='nor_mode')
    parser.add_argument('-g', default=0, type=int,
                        help='the specific GPU number', dest='gpu_number')
    parser.add_argument('-v', '--verbose', default=1, type=int, choices=[0, 1, 2],
                        help='verbosity mode')
    a = parser.parse_args()

    env.environment_setup(a.name, gpu_device='CUDA', gpu_number=a.gpu_number, quiet_mode=(a.verbose == 0))
    user_model = importlib.import_module('%s.USER_model' % a.name)

    #===  from environment.py
    data_dir = Path.cwd() / W_path_str 
   
    env.K_FOLD = len(list_files(data_dir, cond=lambda x: x.startswith('training')))
    #==============================================================================

    with open(env.WORKSPACE['hyperparameters'] / ('%s.json' % a.method), 'r') as f:
        p = argparse.Namespace(**json.load(f))

    # save the history in order to draw the learning curve
    z_stats, histories, predictions = list(), list(), list()

    # start training five folds
    for fold_num in range(1, env.K_FOLD + 1): #original code
    #for fold_num in range(1, 2):

        tf.random.set_seed(p.random_seed)
        np.random.seed(p.random_seed)
        random.seed(p.random_seed)


        model = user_model.self_defined_model(p.dropout_rate, model_name='%s_%s_%d' % (a.name, a.method, fold_num))
        model.summary()
        input_num = len(model.inputs)
        
        train_gen, train_steps, _, z_mean, z_std, alpha = load_data(data_dir,'training', fold_num, input_num,
                                                                    batch_size=p.batch_size, method=a.method,
                                                                    nor_mode=a.nor_mode)
        val_gen, val_steps, _ = load_data(data_dir,'validation', fold_num, input_num,
                                          batch_size=p.batch_size, z_mean=z_mean, z_std=z_std)
        #print( type(val_gen)) # <class 'generator'>
        #val_steps = 1

        val_evaluater = Evaluater('val', val_gen, val_steps, p.batch_size, verbose=a.verbose)

        scheduler = LearningRateScheduler(p.epochs, p.initial_lr, p.decay_rate, p.bottom_lr, fold_num)
        loss = SamplingLoss(a.method, alpha=alpha)

        if a.method in balancing_methods(resampled=True):
            samp_val_gen, samp_val_steps, _ = load_data(data_dir,'validation', fold_num, input_num,
                                                        batch_size=p.batch_size, method=a.method,
                                                        z_mean=z_mean, z_std=z_std)
            samp_val_evaluater = Evaluater('samp_val', samp_val_gen, samp_val_steps,
                                           p.batch_size, verbose=a.verbose)
            callbacks = [val_evaluater, samp_val_evaluater, scheduler.scheduler()]
        else:
            callbacks = [val_evaluater, scheduler.scheduler()]

        print('\n%s %s (Fold %d/%d)' % (a.name, a.method, fold_num, env.K_FOLD))
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss.sampling_loss())

        
        history = model.fit(train_gen, epochs=p.epochs, batch_size=p.batch_size,
                    steps_per_epoch=train_steps, verbose=a.verbose, callbacks=callbacks)
                         
      
        
        # save history & the weight of the model
        z_stats.append((z_mean, z_std))
        histories.append(refine_history(history.history))
        model.save(env.WORKSPACE['models'] / ('%s_model_%d.keras' % (a.method, fold_num)))
        #model.save(env.WORKSPACE['models'] / ('%s_model_%d.h5' % (a.method, fold_num))) #.h5 for running SHAP
      
        # start predicting five folds
        pred_gen, pred_steps, pred_labels = load_data(data_dir,'validation', fold_num, input_num,
                                                      z_mean=z_mean, z_std=z_std)
        pred_result = model.predict(pred_gen, steps=pred_steps, verbose=a.verbose)
        predictions.append({'y_true': pred_labels, 'y_pred': pred_result[:, 1].tolist()})
       
        del model
        env.memory_recovery()

    with open(env.WORKSPACE['tmp'] / ('%s_histories.pickle' % a.method), 'wb') as f:
        pickle.dump(histories, f)
    with open(env.WORKSPACE['tmp'] / ('%s_predictions.pickle' % a.method), 'wb') as f:
        pickle.dump(predictions, f)
    
    with open(env.WORKSPACE['models'] / 'ensemble_info.json', 'r+') as f:
        
        try:
            en_info = json.load(f)
        except:
            en_info = dict()

        f.truncate(0)
        f.seek(0)
        
        if a.method not in en_info:
            en_info[a.method] = dict()

        en_info[a.method]['data_time'] = get_nowtime()
       
        for method in en_info:
            en_info[method]['en_val_rank'] = None
            en_info[method]['selected'] = None

        en_info[a.method]['cv_folds'] = list()
        #val_ranks = performance_rank(predictions)
        
        #rank cross-fold models
        #for i in range(env.K_FOLD):
        #    en_info[a.method]['cv_folds'].append({'val_rank': val_ranks.index(i) + 1,
        #                                          'z_mean': z_stats[i][0],
        #                                          'z_std': z_stats[i][1]})
        #f.write(json.dumps(en_info, sort_keys=False, indent=4))
    
    draw_metric(env.WORKSPACE['plots'] / ('%s_learning_curve_f1-score.png' % a.method),
                histories, 'f1-score',
                title='%s %s %d-Fold Cross-Validation' % (a.name, a.method, env.K_FOLD))
    draw_curve(env.WORKSPACE['plots'] / ('%s_validation_ROC.png' % a.method),
               predictions, 'ROC',
               title='%s %s %d-Fold Cross-Validation ROC' % (a.name, a.method, env.K_FOLD))
    draw_curve(env.WORKSPACE['plots'] / ('%s_validation_PRC.png' % a.method),
               predictions, 'PRC',
               title='%s %s %d-Fold Cross-Validation PRC' % (a.name, a.method, env.K_FOLD))
    
    for metric in ['loss', 'accuracy', 'specificity', 'precision', 'recall', 'auROC']:
        draw_metric(env.WORKSPACE['plots'] / ('%s_%s.png' % (a.method, metric)),
                    histories, metric,
                    title='%s %s %d-Fold Cross-Validation' % (a.name, a.method, env.K_FOLD))
    #-----------------------------------------------
   
    train_log(W_path_str, f'\n\n==={st_time}===', 0)

    #print(p, type(p)) #<class 'argparse.Namespace'>
    args_p_d = vars(p) #dictionary
    print(a.method)
    
    train_log(W_path_str, f'balance method: {a.method}', 0)
    for a_key, a_va in args_p_d.items():
        print(f"{a_key}: {a_va}")
    train_log(W_path_str, args_p_d.items(), 1)
    #----------------------------------------



    performance = [metric_scores(**_) for _ in predictions]
    
    str_table = show_table([_.values() for _ in performance],
                headers=[to_formal(_) for _ in performance[0]],
                v_headers=['Fold %d' % (_ + 1) for _ in range(env.K_FOLD)],
                title='%d-FOLD CROSS-VALIDATION PERFORMANCE METRICS' % env.K_FOLD, float_fmt='%.3f')
   
    train_log(W_path_str, str_table, 0)

    time_end=time.time()
    print('duration:', time_end-time_start)
    #================================================

    ### run test set prediction
    prob_op = f'{W_path_str}/prob'
    if not os.path.exists(prob_op):  os.mkdir(prob_op)
    
    str_table = testset_prediction( env.K_FOLD, f'{W_path_str}/test1.csv', prob_op, 'testset 1' )
    train_log(W_path_str, str_table, 0)
    #================================================



