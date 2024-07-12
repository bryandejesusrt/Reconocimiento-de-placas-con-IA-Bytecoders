import csv

def agregar_placa(nueva_placa, archivo_csv="./placas_permitidas.csv"):
    with open(archivo_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([nueva_placa])
    print(f"Placa {nueva_placa} agregada con éxito al archivo {archivo_csv}.")

if __name__ == "__main__":
    nueva_placa = input("Introduce la nueva placa: ").upper()
    agregar_placa(nueva_placa)
