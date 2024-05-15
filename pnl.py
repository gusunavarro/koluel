import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Descargar los recursos necesarios para NLTK (esto solo necesita hacerse una vez)
nltk.download('vader_lexicon')

# Inicializar el analizador de sentimientos de NLTK
sia = SentimentIntensityAnalyzer()


# Función para analizar sentimientos con NLTK
def analyze_sentiment_nltk(text):
    # Calcular la polaridad de sentimiento con NLTK
    scores = sia.polarity_scores(text)
    polarity = scores['compound']  # Utilizamos el puntaje compuesto para la polaridad
    # Inferir la objetividad basada en el puntaje compuesto
    objectivity = 1 - abs(polarity)
    # Calculamos la subjetividad como complemento de la objetividad
    subjectivity = 1 - objectivity
    return polarity, subjectivity, objectivity


# Función para obtener texto de una URL
def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text
    except Exception as e:
        return str(e)


# Función para obtener el nombre del dominio de una URL
def get_domain_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        return domain.split('.')[1]  # Devuelve solo el primer segmento del dominio
    else:
        return domain  # Devuelve el dominio completo



# Interfaz de usuario con Streamlit
st.title("Análisis de Sentimientos")

# Opción para ingresar texto, cargar archivos o ingresar una lista de URL
option = st.radio("Seleccione una opción:", ("Texto", "Cargar Archivos", "URLs"))

if option == "Texto":
    # Caja de texto para ingresar texto
    texto_input = st.text_area("Ingrese o pegue el texto que desea analizar:", height=200)

    if st.button("Analizar"):
        if texto_input:
            polarity, subjectivity, objectivity = analyze_sentiment_nltk(texto_input)
            st.write("Análisis de Sentimientos del Texto Ingresado")
            st.write("Polaridad:", polarity)
            st.write("Subjetividad:", subjectivity)
            st.write("Objetividad:", objectivity)

            # Crear un DataFrame con los resultados
            df_resultados_texto = pd.DataFrame({
                "Archivo": ["Texto Ingresado"],
                "Polaridad": [polarity],
                "Subjetividad": [subjectivity],
                "Objetividad": [objectivity]
            })

            # Gráfico de barras para los resultados del texto ingresado
            fig, ax = plt.subplots(figsize=(6, 4))
            df_resultados_texto.plot(kind='bar', x='Archivo', ax=ax)
            ax.set_ylabel("Valor")
            ax.set_title("Análisis de Sentimientos del Texto Ingresado")
            st.pyplot(fig)

elif option == "Cargar Archivos":
    # Opción para subir múltiples archivos
    archivos = st.file_uploader("Cargar archivos (.txt, .csv)",
                                type=["txt", "csv"], accept_multiple_files=True)

    if st.button("Analizar Archivos"):
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

                polarity, subjectivity, objectivity = analyze_sentiment_nltk(contenido)
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

elif option == "URLs":
    # Campo de entrada para ingresar las URLs de la página web
    urls_input = st.text_area("Ingrese las URLs de la página web (una por línea):", height=200)

    if st.button("Obtener Texto y Analizar"):
        urls = urls_input.split('\n')
        resultados_urls = []
        for url in urls:
            url = url.strip()
            if url:
                text_from_url = get_text_from_url(url)
                if text_from_url:
                    polarity, subjectivity, objectivity = analyze_sentiment_nltk(text_from_url)
                    domain = get_domain_name(url)
                    resultados_urls.append((domain, polarity, subjectivity, objectivity))
                else:
                    st.write(f"No se pudo obtener texto de la URL: {url}")

        if resultados_urls:
            # Visualización de resultados de las URLs en formato de tabla
            df_resultados_urls = pd.DataFrame(resultados_urls,
                                              columns=["Dominio", "Polaridad", "Subjetividad", "Objetividad"])
            st.write(df_resultados_urls)

            # Gráfico de barras para comparar los resultados
            fig, ax = plt.subplots(figsize=(10, 6))
            df_resultados_urls.plot(kind='bar', x='Dominio', ax=ax)
            ax.set_ylabel("Valor")
            ax.set_title("Análisis de Sentimientos de las URLs")
            st.pyplot(fig)

