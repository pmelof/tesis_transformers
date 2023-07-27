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
    # Decidir qué ejecutar, opciones 
    # a) Sin normalizar vs. normalizados (archivos sin agrupar) 
    # b) Sin normalizar vs. normalizados (archivos agrupados) 
    # c) Agrupados vs. no agrupados (sin normalizar) 
    # d) Agrupados vs. no agrupados (normalizados)
    # e) 1 Agrupado vs. 1 no agrupado (sin normalizar) - test distinto?
    # f) 1 Agrupado vs. 1 no agrupado (normalizados) - test distinto?
    # g) QRNN vs. Transformers (sin normalizar) 
    # h) QRNN vs. Transformers (normalizados)
    # i) QRNN vs. Transformers (agrupados y sin normalizar) 
    # j) QRNN vs. Transformers (agrupados y normalizados)
    
    # all_options = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    real_option = ["a", "c", "b"]
    
    
    # SOLO LOCO
    # Genero lista con nombres de los archivos agrupados y otra sin agrupar, estos nombres son iguales para SUA, MUA y los datos normalizados o sin normalizar.
    list_names_join = listFilenamesResults(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/sua", join=True)
    list_names = listFilenamesResults(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/sua", join=False)
    print("TRANSFORMERS:")
    print("---\nLista de archivos sin agrupar:\n", list_names)
    print("---\nLista de archivos agrupados:\n", list_names_join)
    list_names_qrnn_sua = listFilenamesResults(dir_results="results_qrnn/loco/sua", join=False)
    list_names_qrnn_mua = listFilenamesResults(dir_results="results_qrnn/loco/mua", join=False)
    print("QRNN:")
    print("---\nLista de archivos SUA:\n", list_names_qrnn_sua)
    print("---\nLista de archivos MUA:\n", list_names_qrnn_mua)
    
    
    # TRANSFORMERS
    # Obtengo RMSE y CC archivos sin agrupar:
    # 1) Loco SUA, sin normalizar
    list_rmse_sua_not_normalized, list_cc_sua_not_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/sua", list_names=list_names)
    # 2) Loco MUA, sin normalizar
    list_rmse_mua_not_normalized, list_cc_mua_not_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/mua", list_names=list_names)
    # 3) Loco SUA, normalizado
    list_rmse_sua_normalized, list_cc_sua_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/normalized/results/loco/sua", list_names=list_names)
    # 4) Loco MUA, normalizado
    list_rmse_mua_normalized, list_cc_mua_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/normalized/results/loco/mua", list_names=list_names)
    
    # Obtengo RMSE y CC archivos agrupados:
    # Genero lista con todos los RMSE de todos los archivos agrupados y hago lo mismo para CC.
    # 1) Loco SUA, sin normalizar
    list_join_rmse_sua_not_normalized, list_join_cc_sua_not_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/sua", list_names=list_names_join)
    # 2) Loco MUA, sin normalizar
    list_join_rmse_mua_not_normalized, list_join_cc_mua_not_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/not_normalized/results/loco/mua", list_names=list_names_join)
    # 3) Loco SUA, normalizado
    list_join_rmse_sua_normalized, list_join_cc_sua_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/normalized/results/loco/sua", list_names=list_names_join)
    # 4) Loco MUA, normalizado
    list_join_rmse_mua_normalized, list_join_cc_mua_normalized = listResultsMetrics(dir_results="my_transformers/eval/only_velocity/normalized/results/loco/mua", list_names=list_names_join)
    
    if "a" in real_option:    
        print("\n", "=="*100)
        print("Opción a \nLOCO SUA Y MUA (SIN NORMALIZAR VS NORMALIZADOS) PARA ARCHIVOS SIN AGRUPAR \n")
    
        # Calculo el test Kruskal Wallis para RMSE SUA sin agrupar. Sin normalizar vs normalizar
        kruskal_rmse_sua = stats.kruskal (list_rmse_sua_not_normalized, list_rmse_sua_normalized)  
        print("\nkruskal_rmse_sua: \n", kruskal_rmse_sua)
        # Calculo el test Kruskal Wallis para CC SUA sin agrupar. Sin normalizar vs normalizar
        kruskal_cc_sua = stats.kruskal (list_cc_sua_not_normalized, list_cc_sua_normalized)  
        print("kruskal_cc_sua: \n", kruskal_cc_sua)
        
        # Calculo el test Kruskal Wallis para RMSE MUA sin agrupar. Sin normalizar vs normalizar
        kruskal_rmse_mua = stats.kruskal (list_rmse_mua_not_normalized, list_rmse_mua_normalized) 
        print("\nkruskal_rmse_mua: \n", kruskal_rmse_mua)
        # Calculo el test Kruskal Wallis para CC MUA sin agrupar. Sin normalizar vs normalizar
        kruskal_cc_mua = stats.kruskal (list_cc_mua_not_normalized, list_cc_mua_normalized)  
        print("kruskal_cc_mua: \n", kruskal_cc_mua)
    
    if "b" in real_option:    
        print("\n", "=="*100)
        print("Opción b \nLOCO SUA Y MUA (SIN NORMALIZAR VS NORMALIZADOS) PARA ARCHIVOS AGRUPADOS \n")        

        # Calculo el test Kruskal Wallis para RMSE SUA sin agrupar. Sin normalizar vs normalizar
        kruskal_join_rmse_sua = stats.kruskal (list_join_rmse_sua_not_normalized, list_join_rmse_sua_normalized)  
        print("\nkruskal_join_rmse_sua: \n", kruskal_join_rmse_sua)
        # Calculo el test Kruskal Wallis para CC SUA sin agrupar. Sin normalizar vs normalizar
        kruskal_join_cc_sua = stats.kruskal (list_join_cc_sua_not_normalized, list_join_cc_sua_normalized)  
        print("kruskal_join_cc_sua: \n", kruskal_join_cc_sua)
        
        # Calculo el test Kruskal Wallis para RMSE MUA sin agrupar. Sin normalizar vs normalizar
        kruskal_join_rmse_mua = stats.kruskal (list_join_rmse_mua_not_normalized, list_join_rmse_mua_normalized)  
        print("\nkruskal_join_rmse_mua: \n", kruskal_join_rmse_mua)
        # Calculo el test Kruskal Wallis para CC MUA sin agrupar. Sin normalizar vs normalizar
        kruskal_join_cc_mua = stats.kruskal (list_join_cc_mua_not_normalized, list_join_cc_mua_normalized)  
        print("kruskal_join_cc_mua: \n", kruskal_join_cc_mua)
    
    if "c" in real_option:    
        print("\n", "=="*100)
        print("Opción c \nLOCO SUA Y MUA (AGRUPADOS VS NO AGRUPADOS) SIN NORMALIZAR \n")
 
       # Calculo el test Kruskal Wallis para RMSE SUA. Agrupados vs. sin agrupar. Sin normalizar.
        kruskal_join_vs_not_join_rmse_sua_not_notmalized = stats.kruskal (list_join_rmse_sua_not_normalized, list_rmse_sua_not_normalized)  
        print("\nkruskal_join_vs_not_join_rmse_sua_not_notmalized: \n", kruskal_join_vs_not_join_rmse_sua_not_notmalized)
        # Calculo el test Kruskal Wallis para CC SUA. Agrupados vs. sin agrupar. Sin normalizar.
        kruskal_join_vs_not_join_cc_sua_not_normalized = stats.kruskal (list_join_cc_sua_not_normalized, list_cc_sua_not_normalized)  
        print("kruskal_join_vs_not_join_cc_sua_not_normalized: \n", kruskal_join_vs_not_join_cc_sua_not_normalized)
        
        # Calculo el test Kruskal Wallis para RMSE MUA. Agrupados vs. sin agrupar. Sin normalizar.
        kruskal_join_vs_not_join_rmse_mua_not_normalized = stats.kruskal (list_join_rmse_mua_not_normalized, list_rmse_mua_not_normalized)  
        print("\nkruskal_join_vs_not_join_rmse_mua_not_normalized: \n", kruskal_join_vs_not_join_rmse_mua_not_normalized)
        # Calculo el test Kruskal Wallis para CC MUA. Agrupados vs. sin agrupar. Sin normalizar.
        kruskal_join_vs_not_join_cc_mua_not_normalized = stats.kruskal (list_join_cc_mua_not_normalized, list_cc_mua_not_normalized)  
        print("kruskal_join_vs_not_join_cc_mua_not_normalized: \n", kruskal_join_vs_not_join_cc_mua_not_normalized)
    
    if "d" in real_option:    
        print("\n", "=="*100)
        print("Opción d \nLOCO SUA Y MUA (AGRUPADOS VS NO AGRUPADOS) NORMALIZADOS \n")
        # EN PROCESO... .
 
       # Calculo el test Kruskal Wallis para RMSE SUA. Agrupados vs. sin agrupar. Normalizado.
        kruskal_join_vs_not_join_rmse_sua_normalized = stats.kruskal (list_join_rmse_sua_normalized, list_rmse_sua_normalized)  
        print("\nkruskal_join_vs_not_join_rmse_sua_normalized: \n", kruskal_join_vs_not_join_rmse_sua_normalized)
        # Calculo el test Kruskal Wallis para CC SUA. Agrupados vs. sin agrupar. Normalizado.
        kruskal_join_vs_not_join_cc_sua_normalized = stats.kruskal (list_join_cc_sua_normalized, list_cc_sua_normalized)  
        print("kruskal_join_vs_not_join_cc_sua_normalized: \n", kruskal_join_vs_not_join_cc_sua_normalized)
        
        # Calculo el test Kruskal Wallis para RMSE MUA. Agrupados vs. sin agrupar. Normalizado.
        kruskal_join_vs_not_join_rmse_mua_normalized = stats.kruskal (list_join_rmse_mua_normalized, list_rmse_mua_normalized)  
        print("\nkruskal_join_vs_not_join_rmse_mua_normalized: \n", kruskal_join_vs_not_join_rmse_mua_normalized)
        # Calculo el test Kruskal Wallis para CC MUA. Agrupados vs. sin agrupar. Normalizado.
        kruskal_join_vs_not_join_cc_mua_normalized = stats.kruskal (list_join_cc_mua_normalized, list_cc_mua_normalized)  
        print("kruskal_join_vs_not_join_cc_mua_normalized: \n", kruskal_join_vs_not_join_cc_mua_normalized)
    
    
    return

main()

