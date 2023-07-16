######################################################################
# Load and batch data
# -------------------
#
''' 
Los datos de entrada como vectores ya se tienen, 
son los archivos genenerados por baks (o binning, incluso t2v pero este último 
es más difícil de entender para generar el vocabulario)

La tokenización será redondiar los valores dados por baks a un valor entero 
o a un decimal dado. Esto es realizado en el preprocesamiento.

Del tutorial la función "batchify" no necesito implementarla, mis datos con baks 
ya estan de la forma que requiero.
'''

# Cambiar nomenclatura a nombreFuncion, NombreClase, nombre_variable, VARIABLE_GLOBAL

######################################################################

import numpy as np
import os
import h5py
import torch

def datasetPreprocessing(filepath_dataset: str , filename_dataset: str , filepath_output: str, rounded_decimal: int = 1):
    """
    Lee el archivo dataset baks y redondea los elementos al decimal indicado como parámetro.
    Este preprocesamiento es para SUA y MUA.
    Guarda los datos preprocesados en un archivo .h5 con: X_sua, X_mua e y_task.
    ------------
    Parámetros: 
    filepath_dataset: String
        Dirección donde se enceuntra el dataset baks.
    filename_dataset: String
        Nombre del archivo dataset baks.  
    filepath_output: String
        Dirección donde se guardará el archivo redondeado.
    rounded_decimal: Int
        Decimal al que se quiere redondear los datos, por defecto 1.
    -------------
    """
    with h5py.File(filepath_dataset, 'r') as f:
        X_sua = f[f'X_sua'][()]
        X_mua = f[f'X_mua'][()]
        y_task = f['y_task'][()]   
        
    assert len(X_sua) == len(X_mua) and len(X_sua) == len(y_task),\
            "Largo de X_sua, X_mua e y_task no coinciden"
    
    # Para SUA
    i=0
    while i < len(X_sua):
        j=0
        while j < len(X_sua[0]):
            X_sua[i][j] = round(X_sua[i][j], rounded_decimal)
            j = j + 1
        i = i + 1
        
    # Para MUA
    i=0
    while i < len(X_mua):
        j=0
        while j < len(X_mua[0]):
            X_mua[i][j] = round(X_mua[i][j], rounded_decimal)
            j = j + 1
        i = i + 1
        
    # guardar en un archivo
    with h5py.File(f"{filepath_output}/{filename_dataset[:-3]}_rounded_{rounded_decimal}.h5", 'w') as f:      
        f['X_sua'] = X_sua
        f['X_mua'] = X_mua
        f['y_task'] = y_task


def generateVocabulary(data, decimal):
    '''
    Recibe un dataset, puede ser SUA o MUA, este fue obtenido luego de leer un archivo en rounded.
    Busca el valor máximo y mínimo de los datos del dataset.
    Entrega un array entre valor min y valor max, junto con el valor max y el valor min.
    Recordar que el vocabulario tiene números que representan la cantidad de spikes aproximados que 
    ocurrieron en un intervalo de tiempo.
    ------------
    Parámetros: 
    data: Tensor o array
        Contiene SUA o MUA de baks rounded.
    decimal: Int
        Decimal al que fue redondeado la data.
    -------------
    Retorna:
    vocabulary: Array
        Arreglo de un mínimo hasta un máximo.
    maxi: Int
        Valor máximo encontrado dentro de data.
    mini: Int
        Valor mínimo encontrado dentro de data.
    '''
    maxi = 0 # 0 spikes en ese intervalo de tiempo.
    mini = 99999999999 # valor muy alto.
    for i in data:
        if maxi < max(i):
            maxi = max(i)
        if mini > min(i):
            mini = min(i)
    # calcular salto según decimal
    # num = 1
    # for i in range(decimal):
    #     num = num/10
    
    vocabulary = np.arange(mini, maxi+1, 10**(-decimal))
    return vocabulary, maxi, mini

def generateBigVocabulary(dir_datasets: str, decimal: int):
    '''
    Genera un vocabulario que abarque a todos los archivos del directorio que se ingrese, 
    tanto para SUA como MUA, este vocabulario servirá para ambos datasets.
    Recordar que el vocabulario tiene números que representan la cantidad de spikes aproximados que 
    ocurrieron en un intervalo de tiempo.
    ------------
    Parámetros:
    dir_dataset: String
        Dirección de la carpeta donde se encuentran los archivos a leer para generar vocabulario genérico.
        ej: './datos/05_rounded'
    decimal: Int
        Cantidad de decimales con las que se desea generar el vocabulario genérico.
    ------------
    Retorna:
    vocabulary: Array
        Arreglo de un mínimo hasta un máximo.
    maxi: Int
        Valor máximo encontrado dentro de los datasets.
    mini: Int
        Valor mínimo encontrado dentro de los datasets.
    '''
    datasets = os.listdir(dir_datasets)
    # print(datasets)
    max_global = 0
    min_global = 99999999999

    for data in datasets:
        with h5py.File(f'{dir_datasets}/{data}', 'r') as f:
            X_sua = f[f'X_sua'][()]
            X_mua = f[f'X_mua'][()]

        _, max_sua, min_sua = generateVocabulary(X_sua, decimal=decimal)
        _, max_mua, min_mua = generateVocabulary(X_mua, decimal=decimal)
        if (max_sua > max_global or max_mua > max_global):
            max_global = max(max_sua, max_mua)
        if (min_sua < min_global or min_mua > min_global):
            min_global = min(min_sua, min_mua)
            
    # num = 1
    # for i in range(decimal):
    #     num = num/10  
    vocabulary = np.arange(min_global, max_global+1, 10**(-decimal)).round(decimal)
    return vocabulary, max_global, min_global


class DatasetTransformers(torch.utils.data.Dataset):
    def __init__(self, X, Y, decimal, vocabulary=None):       
        '''
        Recibe los datos X (SUA o MUA) e Y de un dataset leído, luego los datos de X los transforma 
        en un valor que representa al índice donde iría ubicado dicho valor en el vocabulario.
        ------------
        Parámetros: 
        X: Array
            Dataset que contiene SUA o MUA.
        Y: Array
            Arreglo con los valores reales que la salida del modelo desea alcanzar (comparar).
            ej: velocidades x e y del mono.
        decimal: Int
            Cantidad de decimales que contienen los números del vocabulario.
        -------------
        Retorna:
        X: Tensor
            Dataset (SUA o MUA) con los índices de los datos en el vocabulario.
        Y: Tensor
            Contiene las velocidades x e y del mono, o contiene posición, velocidad y aceleración x e y del mono, 
            depende de los datos ingresados en Y.
        '''
        self.decimal = decimal
        self.Y = Y 
           
        # Transformo el valor en el índice del vocabulario
        if vocabulary is not None: # vocabulario nuevo por archivo
            i=0
            res=[]
            while i < X.shape[0]:
                j=0
                l = []    
                while j < X.shape[1]:
                    aux = int(np.where(vocabulary==X[i][j])[0])
                    l.append(aux)
                    j=j+1
                res.append(l)
                i=i+1
            self.X = np.array(res)
        else:   # vocabulario original           
            self.X = np.multiply(X, 10**self.decimal).astype(int)
            

        assert len(self.X) == len(self.Y),\
            "Largo de X e Y no coinciden"

    def __getitem__(self, idx):           
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X)


def readDataset(filepath_dataset: str, feature: str, only_velocity: bool = True):
    with h5py.File(filepath_dataset, 'r') as f:
        X = f[f'X_{feature}'][()]
        Y = f['y_task'][()]   
    if only_velocity:
        # select the x-y velocity components
        Y = Y[:,2:4] # data shape: n x 6 (x-y position, x-y velocity, x-y acceleration)
    return X, Y


# Función de Ahmadi adaptada
def transform_data(X, y, timesteps):
    """
    Transform data into sequence data with timesteps

    Parameters
    ----------
    X : ndarray
        The input data 
    y : ndarray
        The output (target) data
    timesteps: int
        The umber of input steps to predict next step

    Returns
    ----------
    X_seq : ndarray
        The transformed input sequence data
    y_seq : ndarray
        The transformed ouput (target) sequence data
    """
    X_seq = []
    y_seq = []
    # check length X_in equals to y_in
    assert len(X) == len(y), "Both input data length must be equal"
    for i in range(len(X)):
        end_idx = i + timesteps
        if end_idx > len(X)-1:
            break # break if index exceeds the data length
        # get input and output sequence
        X_seq.append(np.concatenate(X[i:end_idx,:]).tolist())
        y_seq.append(y[end_idx-1,:].tolist())
    return np.array(X_seq), np.array(y_seq)


# Juntar varios archivos como uno solo
def appendFiles(list_filenames: list, filespath_baks: str, feature: str, only_velocity: bool = True, padding: bool = True,):
    '''
    Agrupa varios archivos en una lista para X e Y.
    Para que el número de columnas de todos los archivos coincidan se ocupa padding (rellenando con 0) o se trunca
    a la cantidad menor de columnas.
    ------------
    Parámetros:
    list_filenames: List
        Lista de strings con los nombres de los archivos a agrupar.
        ej: ['loco_01.h5', 'loco_03.h5', 'loco_04.h5']
    filespath_baks: String
        Dirección de la carpeta donde se encuentran los archivos a agrupar.
        ej: 'data/rounded/loco'
    feature: String
        Tipo de señal con la que se trabajará.
        ej: 'sua' o 'mua'
    only_velocity: Bool
        Si se trabaja solo con la salida velocidad o no.
    padding: Bool
        Si se desea usar padding o truncar.
    ------------
    Retorna:
    Xs: List
        Lista con SUA o MUA de todos los archivos.
    Ys: List
        Lista con las salidas de todos los archivos.
    '''
    assert len(list_filenames) > 1, "Lista nombre de archivos con 1 o menos archivos"
    Xs = []
    Ys = []
    max_cols = 0
    min_cols = 999999999999999999999
                
    for file in list_filenames:
        filepath = os.path.join(filespath_baks, file)
        X, Y = readDataset(filepath_dataset=filepath, feature=feature, only_velocity=only_velocity)
        Xs.append(X)
        Ys.append(Y)
        # calculando cantidad columnas máximo y mínimo de todos los archivos
        if max_cols < len(X[0]):
            max_cols = len(X[0]) 
        if min_cols > len(X[0]):
            min_cols = len(X[0])

    # Las columnas deben ser el mismo número para todos 
    if padding: # rellenando con 0 espacios faltantes
        i = 0
        while i < len(Xs):
            if len(Xs[i][0]) != max_cols:
                resto = max_cols - len(Xs[i][0])
                Xs[i] = np.pad(Xs[i], (0,resto), constant_values=(0))
                Xs[i] = Xs[i][:-resto]
            i=i+1
    else: # truncar
        i = 0
        while i < len(Xs):
            if len(Xs[i][0]) != min_cols:
                resto = len(Xs[i][0]) - min_cols
                temp = []
                for elem in Xs[i]:
                    elem = list(elem[:-resto])
                    temp.append(elem)
                Xs[i] = np.array(temp)
            i=i+1
    return Xs, Ys


def flattenFiles(X_train, Y_train, X_eval, Y_eval, X_test=None, Y_test=None):
    new_X_train = []
    new_Y_train = []
    new_X_eval = []
    new_Y_eval = []
    new_X_test = []
    new_Y_test = []
    i=0
    while i < len(X_train):
        for elem in X_train[i]:
            temp = elem.tolist()
            new_X_train.append(temp)
        for elem in Y_train[i]:
            temp = elem.tolist()
            new_Y_train.append(temp)
        i=i+1
    i=0
    while i < len(X_eval):
        for elem in X_eval[i]:
            temp = elem.tolist()
            new_X_eval.append(temp)
        for elem in Y_eval[i]:
            temp = elem.tolist()
            new_Y_eval.append(temp)
        i=i+1 
    if X_test is not None:
        i=0
        while i < len(X_test):
            for elem in X_test[i]:
                temp = elem.tolist()
                new_X_test.append(temp)
            for elem in Y_test[i]:
                temp = elem.tolist()
                new_Y_test.append(temp)
            i=i+1    

    return np.array(new_X_train), np.array(new_Y_train), np.array(new_X_eval), np.array(new_Y_eval), np.array(new_X_test), np.array(new_Y_test)