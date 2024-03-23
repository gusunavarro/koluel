import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import pandas as pd
import textract


# Función para analizar sentimientos
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    # Calculamos la objetividad como complemento de la subjetividad
    objectivity = 1 - subjectivity
    return polarity, subjectivity, objectivity


# Interfaz de usuario con Streamlit
st.title(" Koluel. Memorias de la Patagonia Austral. Análisis de Sentimientos.Por Gustavo Navarro  ")


# Caja de texto para ingresar texto
texto_input = st.text_area("Ingrese o pegue el texto que desea analizar:", height=200)

# Opción para subir múltiples archivos
archivos = st.file_uploader("O cargar archivos (.txt, .doc, .pdf, .csv)", type=["txt", "doc", "docx", "pdf", "csv"],
                            accept_multiple_files=True)

if texto_input:
    resultados_texto = []
    polarity, subjectivity, objectivity = analyze_sentiment(texto_input)
    resultados_texto.append(("Texto Ingresado", polarity, subjectivity, objectivity))

    # Visualización de resultados del texto ingresado en formato de tabla
    df_resultados_texto = pd.DataFrame(resultados_texto,
                                       columns=["Archivo", "Polaridad", "Subjetividad", "Objetividad"])
    st.write(df_resultados_texto)

    # Gráfico de barras para comparar los resultados
    fig, ax = plt.subplots(figsize=(10, 6))
    df_resultados_texto.plot(kind='bar', x='Archivo', ax=ax)
    ax.set_ylabel("Valor")
    ax.set_title("Análisis de Sentimientos del Texto Ingresado")
    st.pyplot(fig)

if archivos:
    resultados_archivos = []
    for archivo in archivos:
        contenido = ""
        if archivo.type == "text/plain":
            contenido = archivo.read().decode("utf-8")
        elif archivo.type == "application/pdf":
            contenido = textract.process(archivo)
            contenido = contenido.decode("utf-8")
        elif archivo.type == "text/csv":
            contenido = pd.read_csv(archivo)
            contenido = contenido.to_string()
        else:
            contenido = textract.process(archivo)
            contenido = contenido.decode("utf-8")

        polarity, subjectivity, objectivity = analyze_sentiment(contenido)
        resultados_archivos.append((archivo.name, polarity, subjectivity, objectivity))

    # Visualización de resultados de los archivos en formato de tabla
    df_resultados_archivos = pd.DataFrame(resultados_archivos,
                                          columns=["Archivo", "Polaridad", "Subjetividad", "Objetividad"])
    st.write(df_resultados_archivos)

    # Gráfico de barras para comparar los resultados
    fig, ax = plt.subplots(figsize=(10, 6))
    df_resultados_archivos.plot(kind='bar', x='Archivo', ax=ax)
    ax.set_ylabel("Valor")
    ax.set_title("Análisis de Sentimientos de los Archivos")
    st.pyplot(fig)