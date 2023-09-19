# ecg-estancia
Este repositorio contiene los entregables para la estancia de investigación del proyecto : Redes Neuronales Convolucionales para la Detección de Infarto de Miocardio en Electrocardiogramas.

--------------------------------------------------------
## *Notebooks*
Los *notebooks* ocupados durante la investigación son:

1. [Exploración y tratamiento de los datos](<ecg-eda-ptbxl-v3.ipynb>)
2. [Construcción y comparativa de modelos](<ecg-classification-ptbxl-v3.ipynb>)

En el primero se analizaron las distribuciones de las etiquetas, se visualizaron los *ECG* para comprender sus características, se desarrolló la limpieza por medio de filtros de mediana y se exploraron los criterios para filtrar los outliers.

En el segundo se incluye la generación de modelos predictivos; en la primera parte hay conjuntos de celdas similares en los que sucesivamente se fueron probando nuevas ideas y técnicas en búsqueda de mejorar el desempeño y la generalización, con lo que posteriormente se realizó el análisis de atribuciones, se optimizó la arquitectura tomando esto en cuenta y se obtuvieron los resultados del mejor modelo.

Ambos se incluyeron tras haber corrido las celdas, para que se pueda visualizar lo que se realizó en cada uno. Para correrlos se requiere que se almacenen los datos en la carpeta [data](<data>). Por limitaciones de almacenamiento en  repositorios, no es posible incluir los datos aquí. No obstante, en la sección de datos existe un [*README.md*](<data/README.md>) que detalla el cómo se pueden extraer de la ruta en que originalmente se obtuvieron.

--------------------------------------------------------

## *Scripts*
Los *scripts* ocupados durante la investigación son:

1. [ecg.py](<scripts/ecg.py>)
2. [antonior92.py](<scripts/antonior92.py>)

En el primero se encuentran todas las variables y funciones definidas para su uso en los *notebooks*. El motivo de ocupar este archivo fue para minimizar la carga visual en cuestión del volumen código que estaría presente en los *notebooks*. Contiene funciones de carga, tratamiento, visualización, análisis y predicción.

El segundo es un *script* perteneciente al usuario de *GitHub* [antonior92](<https://github.com/antonior92/automatic-ecg-diagnosis>). Se renombró al archivo *model.py* contenido en su repositorio con el fin de ocupar y comparar su arquitectura con las de los modelos desarrollados en esta investigación.

--------------------------------------------------------

## Datos
Los datos originales, como los generados, deben estar contenidos en el fichero [data](<data>) cuando se trabaje en el ambiente de elección. Como se mencionó previamente, su almacenamiento no es posible en este repositorio. Para más detalle, consultar el archivo [README.md](<data/README.md>) dentro de dicho fichero.

--------------------------------------------------------

## Modelos
El mejor modelo según las métricas de desempeño está contenido en el archivo [model.h5](<model.h5>). Es posible hacer uso de él para hacer predicciones si se le pasa una matriz de 5,000 (500 observaciones por segundo) por 12 (cada una de las señales) milivoltios correspondientes a un *ECG*, en conjunto con un vector que contenga la edad y el sexo del paciente. Para más detalle en cómo se debe acomodar el dato para pasarlo al modelo, se puede inspeccionar el [notebook](<ecg-classification-ptbxl-v3.ipynb>) en el que se realizaron los modelos.

--------------------------------------------------------

## Reporte
El detalle de todo lo que se hizo en la investigación está contenido en [reporte.pdf](<reporte.pdf>). Ahí se detalla la metodología y el por qué de las decisiones que se tomaron dentro de los *notebooks*, así como un análisis de los resultados del mejor modelo.