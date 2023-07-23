# Librerías
from scipy import stats
import h5py
import os


# Funciones
def readResults(dir_results: str, file_results: str):
    '''
    Función que lee un archivo con resultados y entrega el RMSE y CC.
    ------------
    Parámetros:
    dir_results: String
        Dirección de la carpeta donde se encuentran los archivos a leer.
        ej: my_transformers/eval/only_velocity/normalized/results/loco/sua
    file_results: String
        Nombre del archivo con resultados a leer.
    ------------
    Retorna:
    RMSE_folds: Numpy Array
        Arreglo con todos los RMSE obtenidos en cada ventana (fold).
    CC_folds: Numpy Array
        Arreglo con todos los CC obtenidos en cada ventana (fold).
    '''
    with h5py.File(f'{dir_results}/{file_results}', 'r') as f:
        RMSE_folds = f[f'rmse_test_folds'][()]
        CC_folds = f[f'cc_test_folds'][()]
    return RMSE_folds, CC_folds

def listFilenamesResults(dir_results: str, join: bool):
    '''
    Función que genera lista con nombre de los archivos con resultados.
    Dependiendo de la opción de join serán los resultados agrupados o sin agrupar.
    ------------
    Parámetros:
    dir_results: String
        Dirección de la carpeta donde se encuentran los archivos con resultados.
        ej: my_transformers/eval/only_velocity/normalized/results/loco/sua
    ------------
    Retorna:
    list_names: List
        Lista con nombre de los archivos encontrados y ordenados.
    '''
    list_names = []
    for result in os.listdir(dir_results):
        if ".json" in result:
            continue
        elif "join" in result and join:
            list_names.append(result)
        elif "join" not in result and join is False:
            list_names.append(result)
        else:
            continue
    list_names.sort()
    return list_names

def listResultsMetrics(dir_results:str, list_names:list):
    '''
    Función que genera la lista con RMSE y la lista con CC de todos los archvios con resultados.
    ------------
    Parámetros:
    dir_results: String
        Dirección de la carpeta donde se encuentran los archivos con resultados.
        ej: my_transformers/eval/only_velocity/normalized/results/loco/sua
    list_names: List
        Lista con nombre de los archivos con resultados.
    ------------
    Retorna:
    list_rmse: List
        Lista con RMSE de todos los archivos.
    list_cc: List
        Lista con CC de todos los archivos.
    '''
    list_rmse = []
    list_cc = []
    for file_result in list_names:
        temp_rmse, temp_cc = readResults(dir_results=dir_results, file_results=file_result)
        list_rmse.append(temp_rmse.tolist())
        list_cc.append(temp_cc.tolist())
    return list_rmse, list_cc



def main():
    # Genero lista con nombres de los archivos SIN agrupar, estos nombres son iguales para SUA, MUA y los datos normalizados o sin normalizar.
    list_names = listFilenamesResults(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/sua", join=False)
    # Genero lista con todos los RMSE de todos los archivos sin agrupar y hago lo mismo para CC.
    # 1) Loco SUA, sin normalizar
    list_rmse_sua_not_normalized, list_cc_sua_not_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/sua", list_names=list_names)
    # 2) Loco MUA, sin normalizar
    list_rmse_mua_not_normalized, list_cc_mua_not_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/mua", list_names=list_names)
    # 3) Loco SUA, normalizado
    list_rmse_sua_normalized, list_cc_sua_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/normalized/results/loco/sua", list_names=list_names)
    # 4) Loco MUA, normalizado
    list_rmse_mua_normalized, list_cc_mua_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/normalized/results/loco/mua", list_names=list_names)

    # Calculo el test Kruskal Wallis para RMSE SUA sin agrupar. Sin normalizar vs normalizar
    kruskal_rmse_sua_not_normalized = stats.kruskal (list_rmse_sua_not_normalized, list_rmse_sua_normalized)  
    print("kruskal_rmse_sua_not_normalized: ", kruskal_rmse_sua_not_normalized)
    # Calculo el test Kruskal Wallis para CC SUA sin agrupar. Sin normalizar vs normalizar
    kruskal_cc_sua_not_normalized = stats.kruskal (list_cc_sua_not_normalized, list_cc_sua_normalized)  
    print("kruskal_cc_sua_not_normalized: ", kruskal_cc_sua_not_normalized)
    
    # Calculo el test Kruskal Wallis para RMSE MUA sin agrupar. Sin normalizar vs normalizar
    kruskal_rmse_mua_not_normalized = stats.kruskal (list_rmse_mua_not_normalized, list_rmse_mua_normalized)  
    print("kruskal_rmse_mua_not_normalized: ", kruskal_rmse_mua_not_normalized)
    # Calculo el test Kruskal Wallis para CC MUA sin agrupar. Sin normalizar vs normalizar
    kruskal_cc_mua_not_normalized = stats.kruskal (list_cc_mua_not_normalized, list_cc_mua_normalized)  
    print("kruskal_cc_mua_not_normalized: ", kruskal_cc_mua_not_normalized)
    
    
    
    return

main()



# list_names_join = listFilenamesResults(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/sua", join=True)
# print(list_names)
# print("---"*10)
# print(list_names_join)
# print("---"*10)
# # 1) Loco SUA sin normalizar, total 10 archivos con 5 ventanas cada uno.


# # 1) Loco SUA normalizado, total 10 archivos con 5 ventanas cada uno.
# rmse_files_n = []
# rmse_file1, cc_file1 = readResults("my_transformers/eval/only_velocity/normalized/results/loco/sua", "loco_20170302_02_baks_rounded_1.h5")
# rmse_file2, cc_file2 = readResults("my_transformers/eval/only_velocity/normalized/results/loco/sua", "loco_20170301_05_baks_rounded_1.h5")

# rmse_files_n.append(rmse_file1.tolist())
# rmse_files_n.append(rmse_file2.tolist())
# print("Lista RMSE normalizados", rmse_files_n)


# rmse_files_sn = []
# rmse_file1_sn, cc_file1_sn = readResults("my_transformers/eval/only_velocity/not_normalized/results/loco/sua", "loco_20170302_02_baks_rounded_1.h5")
# rmse_file2_sn, cc_file2_sn = readResults("my_transformers/eval/only_velocity/not_normalized/results/loco/sua", "loco_20170301_05_baks_rounded_1.h5")

# rmse_files_sn.append(rmse_file1_sn.tolist())
# rmse_files_sn.append(rmse_file2_sn.tolist())
# print("Lista RMSE sin normalizar", rmse_files_sn)

# a = stats.kruskal (rmse_files_n, rmse_files_sn)
# print(a)


#########################################################################


# print(type(file1.tolist()))
# print(file1.tolist())



# grupo1 = [7, 14, 14, 13, 12, 9, 6, 14, 12, 8]
# grupo2 = [15, 17, 13, 15, 15, 13, 9, 12, 10, 8]
# grupo3 = [6, 8, 8, 9, 5, 14, 13, 8, 10, 9]

# # realizar la prueba de Kruskal-Wallis 
# a = stats.kruskal (grupo1, grupo2, grupo3)

# print(a)

