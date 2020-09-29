

import xlrd
import pandas as pd
def function_activation(i,x,w,theta):
    n=0.0
    for j in range(len(w)):
        # calculamos z
        n += (x[j][i] * w[j])
        z = n - theta  # Restamos el bias
        # Si la neurona se activa o no
        activate=1 if z >= 0 else 0
    return z,n,activate
    
def trainer(theta, fac_ap, w, x, d,class_trainer):
    epochs=1
    count_errors = 0
    count_success = 0
    errores = True
    while errores:
        errores = False
        nsamples = len(d)
        for i in range(nsamples):
            z,n, active=function_activation(i,x,w,theta)
            print("(Epoca: {3}) Z=(n({1:.4}) - theta({2})) = {0:.2} Entonces z obtenido es {5} ({4}), Esperado es {6}".format(z,n,theta,epochs,active,class_trainer[active],class_trainer[d[i]]))
            if (active != d[i]):
                errores=True
                # Calculamos errores
                error=(d[i] - active)
                # Ajustar pesos
                theta=theta + (-(fac_ap * error))
                for xwi in range(len(w)):    
                    w[xwi]=w[xwi] + (x[xwi][i] * error * fac_ap)
                    print("w{0}={1}".format(xwi,w[xwi]))
                    #w2=w2 + (x2[i] * error * fac_ap)
                epochs += 1
                print("Epoca = " + str(epochs))
                print("Error\n")
                count_errors +=1
            else:
                print("Acierto\n")
                count_success +=1
                
        return w, epochs, theta,count_errors, count_success

if (__name__ == "__main__"):
    file_excel=pd.read_excel("./data_OR.xlsx")
    print("Datos")
    print(file_excel)
    print("\n")
    theta=0.4 #Umbral para restar el potencial post-sinaptico
    fac_ap=0.2
    w=[0.3,0.5,0.0] # Pesos de cada
    epochs=1
    x_trainer=[file_excel['x1'],file_excel['x2'],file_excel['x3']] # Set de entrenamiento
    y_trainer=file_excel['d'] # Llamado target o vector esperado
    class_trainer = ['Clase 1', 'Clase 2']
    w, epochs, theta, count_errors, count_success=trainer(theta, fac_ap, w, x_trainer, y_trainer,class_trainer)

    print("\nResultados: \nPesos (w) = {0} \nUmbral (Theta) = {1} \nEpocas recorridas= {2}, \nErrores = {3}, \nAciertos={4}\n".format(w, theta, epochs,count_errors, count_success))
