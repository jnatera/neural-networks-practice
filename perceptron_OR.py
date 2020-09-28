

import xlrd
import pandas as pd


def trainer(theta, fac_ap, w, epochs, x, d):
    errores = True
    while errores:
        errores = False
        nsamples = len(d)
        for i in range(nsamples):
            n = 0.0
            for xwi in range(len(w)):
                # calculamos z
                n += (x[xwi][i] * w[xwi])
            z = n - theta  # Restamos el bias
            print("(Epoca: {3}) Z=(n({1:.4}) - theta({2})) = {0:.2}".format(z,n,theta,epochs))
            # Si la neurona se activa o no
            z=1 if z >= 0 else 0
            print("Entonces: Z => {0}".format(z))

            if (z != d[i]):
                errores=True
                # Calculamos errores
                error=(d[i] - z)
                # Ajustar pesos
                theta=theta + (-(fac_ap * error))
                for xwi in range(len(w)):    
                    w[xwi]=w[xwi] + (x[xwi][i] * error * fac_ap)
                    print("w{0}={1}".format(xwi,w[xwi]))
                    #w2=w2 + (x2[i] * error * fac_ap)
                epochs += 1
                print("Epoca = " + str(epochs))
                print("Error\n")
            else:
                print("Acierto\n")
                
        return w, epochs, theta



if (__name__ == "__main__"):
    file_excel=pd.read_excel("./data_OR.xlsx")
    print("Datos")
    print(file_excel)
    print("\n")
    theta=0.4
    fac_ap=0.2
    w=[0.3,0.5,0.0]
    epochs=1
    x=[file_excel['x1'],file_excel['x2'],file_excel['x3']]
    d=file_excel['d']
    w, epochs, theta=trainer(theta, fac_ap, w, epochs, x, d)

    print("\nResultados: \nw = {0} \nTheta = {1} \nEpocas = {2}".format(w, theta, epochs))
