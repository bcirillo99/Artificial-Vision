import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import *
from utils import *

def not_equal(y_true, y_pred):
    return np.abs(y_pred-y_true)>1

# Compute error rate, alpha and w
def compute_error(y_true, y_pred, w_i):
    '''
    Calculate the error rate of a weak classifier m. Arguments:
    y: actual target value
    y_pred: predicted value by weak classifier
    w_i: individual weights for each observation
    
    Note that all arrays should be the same length
    '''
    return sum(w_i * (np.squeeze(not_equal(y_true, y_pred))))/sum(w_i)

def compute_alpha(error):
    '''
    Calculate the weight of a weak classifier m in the majority vote of the final classifier. This is called
    alpha in chapter 10.1 of The Elements of Statistical Learning. Arguments:
    error: error rate from weak classifier m
    '''
    return np.log((1 - error) / error)+np.log(80)

def update_weights(w_i, alpha, y_true, y_pred):
    ''' 
    Update individual weights w_i after a boosting iteration. Arguments:
    w_i: individual weights for each observation
    y: actual target value
    y_pred: predicted value by weak classifier  
    alpha: weight of weak classifier used to estimate y_pred
    '''  
    
         
    return w_i * np.exp(alpha * (np.squeeze(not_equal(y_true, y_pred))))

def undersampling(df):
    d = dict([(key, 0) for key in range(1,82)])
    for elem in df["label"].tolist():
        d[int(elem)]+=1
    print(d)
    df_temp=df
    for key in d.keys():
        str_key = "\"%02d\"" % key
        q = "label == "+str_key
        df_q = df.query(q)
        if len(df_q) > n_sample_max_represented_class*0.3 and len(df_q) <= n_sample_max_represented_class*0.65:
            f = .7
            x = df.query(q).sample(frac = f)
        elif len(df_q) > n_sample_max_represented_class*0.65:
            f = .5
            x = df.query(q).sample(frac = f)
        else:
            x = df.query(q)
        
        df_temp = pd.merge(df_temp,x, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    df_train = pd.merge(df,df_temp, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    print(df_train)
    return df_train

# Define AdaBoost class
class AdaBoost:
    
    def __init__(self,train_datadir, dataframe, classes, label, epochs=50, patience=20,reduce_factor=0.1, M=3):
        self.alphas = []
        self.G_M = []
        self.M = M
        self.training_errors = []
        self.prediction_errors = []
        self.epochs = epochs
        self.dataframe = dataframe
        self.label = label
        self.patience = patience
        self.reduce_factor = reduce_factor
        self.classes = classes
        self.train_datadir = train_datadir
    
    def __create_model(self):

        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    
        # Add the preprocessing/augmentation layers.
        x = tf.keras.layers.Rescaling(1./255)(inputs)
        x = tf.keras.layers.RandomFlip(mode='horizontal_and_vertical')(x)
        x = tf.keras.layers.RandomRotation(0.1)(x)
        x = tf.keras.layers.RandomZoom(0.2)(x)

        base_model = tf.keras.applications.ResNet50(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(224, 224, 3),
            input_tensor=(x),
            include_top=False)
        X = base_model.output
        X = GlobalAveragePooling2D()(X)

        X = Dense(512)(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        output = Dense(81, activation='softmax', name='out')(X)
        model = Model(base_model.input, output)
        
        model.compile(
            loss=aar_class,      
            optimizer=SGD(0.005),
            metrics=[aar_metric_class,mae_class,sigma_class])
        
        return model
        
    def __update_generator(self,dataframe,predict):
        datagen = ImageDataGenerator()
        if predict:
            df_train=dataframe
        else:
            df_train = undersampling(dataframe)
        generator = datagen.flow_from_dataframe(
            dataframe=df_train,
            directory=self.train_datadir,
            x_col='filename',
            y_col='label',
            weight_col='w_col',
            target_size=(224, 224),
            batch_size=64,
            classes = self.classes,
            class_mode="categorical",
            validate_filenames=False,
            shuffle=False)
        return generator
            
    def fit(self, val_generator, M = 3, verbose=False):
        '''
        Fit model. Arguments:
        X: independent variables - array-like matrix
        y: target variable - array-like vector
        M: number of boosting rounds. Default is 100 - integer
        '''
        
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.M = M
        
        # Array containing all ages to be predicted
        positions = np.arange(1,82, dtype=np.float32)

        # Array containing all the true ages of the samples
        #### NOW CLASS 1 is 01 etc. (making the cast to float though does not cause problems)
        label= self.label
        predict_generator = self.__update_generator(self.dataframe, predict=True)
        
        # Iterate over M weak classifiers
        for m in range(0, M):
            
            
            if verbose:
                print("\nSet weights for classifier "+str(m))
            # Set weights for current boosting iteration
            if m == 0:
                # At the first iteresion the weights are all the same and equal to 1 / N
                w_i = np.array(self.dataframe["w_col"].tolist())
                sum_0 = sum(w_i)
            else:
                # Update w_i
                w_i = update_weights(w_i, alpha_m, label, pred)
                sum_i = np.sum(w_i)
                w_i = (w_i/sum_i)*sum_0
                self.dataframe["w_col"] = pd.Series(w_i)
                
            self.dataframe.to_csv("RUSboost/dataframe_"+ str(m)+".csv", index=False)
            #self.dataframe.sample(frac=1).reset_index()
            generator = self.__update_generator(self.dataframe,predict=False)
            generator.reset()
            if verbose:
                print("\nLoad calssifier "+str(m))
            # Instance of the classifier
            G_m = self.__create_model()
            
            log_path = "RUSboost/log_"+str(m)
            model_path = "RUSboost/model_"+str(m)
            tensorboard = TensorBoard(log_dir=log_path)

            #modelcheckpoint = ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True)
            modelcheckpoint = ModelCheckpoint(model_path, monitor="val_loss")
            early_stop = EarlyStopping(monitor='val_loss', patience=self.patience, verbose=1, 
                                    mode='auto', restore_best_weights=True)
            
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.reduce_factor, patience=10, 
                                        verbose=1, mode='auto')   
            if verbose:
                print("\n\n Training "+str(m)+" classifier:\n\n")
            G_m.fit(generator, epochs=self.epochs,
                            verbose=verbose,
                            validation_data=val_generator,
                            callbacks=[reduce_lr, early_stop, tensorboard, modelcheckpoint])
            if verbose:
                print("Training classifier "+str(m)+" done.\n\n")

            G_m = load_model(model_path,compile=False)
            G_m.compile(
                loss=aar_class,      
                optimizer=SGD(0.005),
                metrics=[aar_metric_class,mae_class,sigma_class])
            ####### SALVARE MODELLO E LOG #######  
            if verbose:
                print("\nPredict outputs from training set for classifier "+str(m))
            predict_generator.reset()
            pred=G_m.predict(predict_generator,verbose=verbose)
            prod = pred*positions
            pred = tf.reduce_sum(prod,axis=1,keepdims=True)
            pred = tf.round(pred)
            
            # Save to list of the classifiers
            self.G_M.append(G_m) 

            # 
            # We tested that if the parameter "shuffle" of the generator is set to False = 0 
            # then the order of the prediction is the order in which the data are collected into
            # the data frame. (To test just see that the first n results of the predict(generator),
            # are the results of the first x images passed individually to the predict(x))
            if verbose:
                print("\nCompute errors on the training set for classifier "+str(m))
            # Compute error
            
            error_m = compute_error(label, pred, w_i)
            if verbose:
                print("\nError rate classifier "+str(m)+": "+str(error_m))
            # 
            self.training_errors.append(error_m)

            # Compute alpha
            alpha_m = compute_alpha(error_m)
            if verbose:
                print("\nAlpha classifier "+str(m)+": "+str(alpha_m))
            # Save alpha in a list
            self.alphas.append(alpha_m)
            df_alpha = pd.DataFrame()
            df_alpha["alpha"] = self.alphas
            df_alpha.to_csv("RUSboost/alphas_adaboost.csv")
            # Save alpha(s) in a file
            with open("RUSboost/alphas_adaboost.txt", "w") as f:
                f.write("Iteration "+str(m) + ":\n")
                i=0
                for item in self.alphas:
                    f.write("Alpha "+str(i)+": "+str(item) + "\n")
                    i+=1

        # Assert that the number of alpha is equale to the number of classifiers
        assert len(self.G_M) == len(self.alphas)
    
    def predict(self, X, verbose=False):
        '''
        Predict using fitted model. Arguments:
        X: 
        '''
        positions = np.arange(1,82, dtype=np.float32)
        results = []
        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            X.reset()
            y_pred_m = self.G_M[m].predict(X,verbose=verbose)
            prod = y_pred_m*positions
            y_pred_m = tf.reduce_sum(prod,axis=1,keepdims=True)
            y_pred_m = tf.round(y_pred_m) * self.alphas[m]
            results.append(y_pred_m)
            print(self.alphas[m])

        # Calculate final predictions
        y_pred = np.round(np.sum(results,axis=0)/np.sum(self.alphas))
        y_pred = tf.convert_to_tensor(y_pred)
        return y_pred

    def evaluate(self, X, dataframe, verbose=False):
        label = tf.expand_dims(np.array(dataframe["label"].astype("float32").tolist(),dtype="float32"),-1)
        pred = self.predict(X,verbose)
        aar_m,mmae_m,sigma_m,mae_m, mae_j_list = aar(label,pred)
        return aar_m,mmae_m,sigma_m,mae_m,mae_j_list

    def load(self, path_model, path_alphas):
        df_alpha = pd.read_csv(path_alphas)
        self.alphas = df_alpha["alpha"].tolist()
        for m in range(self.M):
            print(m)
            self.G_M.append(load_model(path_model+"_"+str(m),compile=False))
            self.G_M[m].compile(
                loss=aar_class,      
                optimizer=SGD(0.005),
                metrics=[aar_metric_class,mae_class,sigma_class])
        #self.alphas = [4.130843614433012,3.761302463455208,3.42894608747818]
        #print(self.alphas)
    def get_single_experts(self):
        return self.G_M
                

        