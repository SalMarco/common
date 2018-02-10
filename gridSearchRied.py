import os, traceback
import sys
import logging
from pprint import pprint as pp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import advanced_activations
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from keras.models import model_from_json
import random
import argparse
from datetime import datetime
sys.path.insert(0, '/home/marco/working-dir/GenderClass/')
from helper import Helper as h
from keras.models import model_from_json
from keras.models import load_model
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.cross_validation import PredefinedSplit
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC,LinearSVC,SVR
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from xgboost import XGBClassifier


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
SEP='|'
DEF_MODEL_NAME='best_model_grid'
CAT_LIST=['categ']
param_grid_RFC = dict(n_estimators=[20,50,100,200,],max_features=['auto',2,3,'sqrt'],max_depth=[5,10,15,20,None],min_samples_leaf=[1,10,25,50])
param_grid_SVC = dict(C=[0.5,1,2,3],kernel=['linear'],epsilon=[0,0.01,0.1],shrinking=[True,False]) #(kernel=['rbf','linear','sigmoid'],
batch_size = [10, 20]
epochs = [10, 100]
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
optimizer = ['SGD']#, 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid_NN = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer)
# param_grid_NN = dict(activation=['relu','tanh'])
PARAM_DICT={'RFC':param_grid_RFC,'SVC':param_grid_SVC,'NN':param_grid_NN}

TARGET_NAME='venduto_valore'

argparser = argparse.ArgumentParser(add_help=True)
argparser.add_argument('-i','--inFile', type=str, help=('csv with data in input'), required=True)
argparser.add_argument('-t','--testFile', type=str, help=('csv with data test set'))
argparser.add_argument('-m','--mod', type=str, help=('configuration file'))
argparser.add_argument('-n','--nameModel', type=str, help=('short name of the model to be used (NN, SVC, RFC, XGB)'),required=True)
argparser.add_argument('-o','--outFile', type=str, help=('csv with results in output [OPTIONAL]'))


script=os.path.basename(__file__)
logger=h.personalLogger(script=script,use_logfile=False)



#####################################################################


class TestClass:

    def __init__(self, **kargs):
        self.conf = kargs['conf']
        self.model_param = self.conf['model_param']
        self.nameModel = kargs['nm']
        self.numFeatures = kargs['nf']

    def selectModel(self):
        try:
            if self.nameModel =="XGB":
                return self.createModelXGB()
            if self.nameModel =="SVC":
                return self.createModelSVC()
            if self.nameModel =="RFC":
                return self.createModelRFC()
        except Exception as e:
            logger.error("msg:'%s', traceback:%s"%(str(e),traceback.format_exc()))


    def createModelNN(self,optimizer):
        leng = self.numFeatures
        model = Sequential()
        # activation='tanh'
        # if model_param['use_dropout']:
        #     model.add(Dropout(model_param['drop'], input_shape=(leng,)))
        model.add(Dense(leng, input_dim=leng))
        #model.add(Dropout(model_param['drop'], input_shape=(leng,)))
        model.add(Dense(30))
        # if model_param['use_second_hidden']:
        #     model.add(Dense(model_param['num_neur_hl2'], init='uniform', activation=activation_fun))
        model.add(Dense(1,activation='sigmoid'))
        # sgd = SGD(lr=model_param['lr'], momentum=model_param['momentum'], decay=0.0, nesterov=False)
        #sgd="SGD"
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #model.fit(X, Y, nb_epoch=numEpoch, batch_size=10)

        return model

    def createModelXGB(self):
        model = XGBClassifier(nthread=-1,seed=seed)
        return model

    def createModelRFC(self):
        logger.info("DEFINITION OF THE MODEL RFC")
        # model = ExtraTreesClassifier(n_estimators=self.model_param['n_estimators'],max_features=self.model_param['max_features'],n_jobs=-1)
        # model = RandomForestClassifier(n_jobs=-1),
        model = RandomForestRegressor(n_jobs=-1)
        logger.info("MODEL PARAMS: %s",model.get_params(deep=True))
        return model

###PROBLEMA utilizzare il bagging con ilgrid search e' un problema per il passaggio dei parametri dal grid a bagging e non al modello dentro
    def createModelSVC(self):
        logger.info("DEFINITION OF THE MODEL SVC")
        modelSVC =SVR() #probability=True
        # model = BaggingClassifier(base_estimator=modelSVC,verbose=4,n_jobs=-1,max_samples=1.0/self.model_param['n_estimators'] ,n_estimators=self.model_param['n_estimators'] )
        #model = SVC(kernel=self.model_param['kernel'],degree=self.model_param['degree'], C=self.model_param['C'],verbose=True,probability=True)
        model = modelSVC
        logger.info("MODEL PARAMS: %s",model.get_params(deep=True))
        return model


def gridSearch(model,param_grid,cv,X,Y,scoring):
    if cv:
        logger.info("USING CUSTOM CV")
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,cv=cv)
    else:
        logger.info("USIGN NORMAL CV")
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3,scoring=scoring,refit='RMS')
        # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=3,scoring=scoring,refit='RMS')
    logger.info("START GRID SEARCH")
    print("X",X)
    print("Y",Y)
    grid_result = grid.fit(X,Y)
    logger.info("END GRID SEARCH")
    return grid_result

def gridResults(grid_result,X,nameModel):
    logger.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(sorted(grid_result.cv_results_.keys()))
    if nameModel == "RFC":
        importances = grid_result.best_estimator_.feature_importances_
    if nameModel=='SVC':
        importances = grid_result.best_estimator_.coef_[0]
    if nameModel != 'NN':
        df_imp = pd.DataFrame({'features': X.columns.tolist(),'importances':importances})
        df_imp.sort_values(by="importances",ascending=False,inplace=True)
        print(df_imp)

def evalActivationFun(activation_fun):
    if activation_fun.find('advanced_activations')==-1:
        return activation_fun
    else:
        return eval(activation_fun)

def SaveModel(nameModel,grid_result):
    if nameModel=='NN':
        outFile='%s_%s.%s'%(DEF_MODEL_NAME,nameModel,"h5")
    else:
        outFile='%s_%s.%s'%(DEF_MODEL_NAME,nameModel,"pkl")
    joblib.dump(grid_result.best_estimator_,outFile)
    logger.info("SAVED BEST MODEL IN %s",outFile)

def readFile(inFile):
    df = pd.read_csv(inFile,delimiter=SEP)

    # df['targetBOOL'] = df.apply(lambda x: 1 if x['target']=='Best' else 0,axis=1)
    Y = df[TARGET_NAME]
    for cur_col in df.columns:
        t = df[cur_col].dtype
        print('cur_col, type',cur_col,t)
        if t != np.float64 and t != np.int64 and t!= np.bool:
            logger.info('making convertion for %s',cur_col)
            # le = LabelEncoder()
            # le.fit(df[cur_col])
            # df[cur_col] = le.transform(df[cur_col])
            df[cur_col] = df[cur_col].astype('category').cat.codes
            logger.info('end convertion for %s',cur_col)
    # for cur_col in CAT_LIST:
    #     le = LabelEncoder()
    #     le.fit(df[cur_col])
    #     df[cur_col] = le.transform(df[cur_col])
    # df.drop([TARGET_NAME,'venduto','classifica','venduto_1','classifica_1','delta','fotografia','se_no_perché','data','venduto_cp','new','fatte','n_pro_gr_dl','ispettore','agenzia','codice'],inplace=True,axis=1)
    df.drop([TARGET_NAME],inplace=True,axis=1)
    # df_out = df[['subtitles_insert_rel','extended_body_delete_rel','unique_subtitles_insert_rel','extended_body_insert_rel','unique_extended_body_delete_rel','body_delete_rel','body_equal_rel','extended_body_equal_rel']]
    X = df
    logger.info("SHAPE OF X:%s AND Y:%s",X.shape,Y.shape)
    len_train=X.shape[0]
    numFeatures = X.shape[1]
    X.fillna(value=0,inplace=True)
    print(X.dtypes)
    print(X)
    return X, Y, len_train, numFeatures


def main(argv):
    start_time = datetime.now()
    logger.info("START")
    args = argparser.parse_args()
    inFile = args.inFile
    testFile = args.testFile
    nameModel = args.nameModel
    conf_file = args.mod
    mod = __import__(conf_file, fromlist=['*'])
    model_conf=mod.gridSearch_Model_types[nameModel]
    conf = getattr(__import__(conf_file, fromlist=[model_conf]), model_conf)
    prefix_dict=conf['prefix_dict']
    out_dict=h.outfileName(fo=args.outFile,fi=inFile,prefix_dict=prefix_dict,add_date=True)
    logger.info("RUNNING WITH MOD: %s, INFILE: %s"%(conf_file,inFile))
    logger.info("LOADING THE DATA SET")
    param_grid = PARAM_DICT[nameModel]
    # scoring = {'Accuracy': make_scorer(accuracy_score),'RMS':make_scorer(mean_squared_error)}
    scoring = {'RMS':make_scorer(r2_score)}
    X, Y, len_train, numFeatures = readFile(inFile)
    cv = None
    if testFile:
        logger.info("USING TEST FILE %s AS TEST SET FOR THE CORSS VALIDATION"%testFile)
        X_test, Y_test,len_train_test, numFeatures_test = readFile(inFile)
        X = pd.concat([X,X_test],ignore_index=True)
        Y = pd.concat([Y,Y_test],ignore_index=True)
        cv_arr =[1]*len_train
        cv_arr.extend([0]*len_train_test)
        cv=PredefinedSplit(test_fold=cv_arr)
        print("Stampa di cv: ",cv)
        print("numero di fold",cv.get_n_splits())
        for train_index, test_index in cv.split():
            print("TRAIN:", train_index, "TEST:", test_index)
        logger.info("SHAPE OF X:%s AND Y:%s AFTER APPEND",X.shape,Y.shape)
    logger.info("CREATION OF THE MODEL")
    t=TestClass(conf=conf,nm=nameModel,nf=numFeatures)
    if nameModel == 'NN':
        model = KerasClassifier(build_fn=t.createModelNN)
        X = X.as_matrix()
        Y = Y.as_matrix()
    else:
        model = t.selectModel()
    logger.info("START GRID SEARCH")
    grid_result = gridSearch(model,param_grid,cv,X,Y,scoring)
    logger.info("END OF GRID SEARCH")
    logger.info("PRINTING RESULTS")
    gridResults(grid_result,X,nameModel)
    SaveModel(nameModel,grid_result)
    logger.info("EXECUTED IN %f SEC"%((datetime.now()-start_time)).total_seconds())
    logger.info("END")




if __name__=="__main__":
    main(sys.argv)
