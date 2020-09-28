import random
random.seed(0)
def al(tipo):
    if tipo==1:
        return 0.6+(0.4*random.random())
    else:
        return 0.5*random.random()

def generar_numeros(volumen):
#Devuelve una lista de tuplas que contienen una lista de datos y su clase
    datos=[]
    dato_1=()
    dato_2=()
    for n in range(volumen):
        dato_1=([al(1),al(2),al(1),al(2)],"Class 1")
        dato_2=([al(2),al(1),al(2),al(1)],"Class 2")
        datos.append(dato_1)
        datos.append(dato_2)
    return datos

def iniciar_perceptron():               #Pesos iniciados en valores entre 0 y 0.5
    percep=[None]*4
    for n in range(len(percep)):
        percep[n]=0.5*random.random()
    return percep
 
def entrenar_perceptron(datos,pesos):
    z = 0.0
    clase_perceptron = ""
    completo = True
    aciertos = 0
    errores = 0
    ciclo = 1
    while (completo):
        completo = False
        for n in datos:
            valores = n[0]
            clase = n[1]
            for n2 in range(len(valores)):  # Calcular z y clasificar
                z = z + valores[n2] * pesos[n2]
            if (z >= 0):
                clase_perceptron = "Class 1"
            if (z < 0):
                clase_perceptron = "Class 2"
            if (clase_perceptron == clase):  # Comprobar clasificacion y corregir
                aciertos += 1
            else:
                completo = True
                errores += 1
                for n2 in range(len(pesos)):  # Actualizacion de pesos segun la funcion de entrenamiento
                    pesos[n2] = pesos[n2] + (0.0 - z) * valores[n2]
            z = 0.0
        print("\nCICLO: ", ciclo)
        print("Aciertos: ", aciertos)
        print("Errores: ", errores)
        ciclo += 1
        aciertos = 0
        errores = 0

datos=generar_numeros(4)
print(datos)
pesos=iniciar_perceptron()
print(pesos)
entrenar_perceptron(datos,pesos)