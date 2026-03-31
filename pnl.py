import streamlit as st
import nltk
import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import PyPDF2

# Descarga condicional de recursos NLTK (para VADER)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    with st.spinner("Descargando recursos de NLTK..."):
        nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def analyze_sentiment_nltk(text):
    scores = sia.polarity_scores(text)
    polarity = scores['compound']
    objectivity = 1 - abs(polarity)
    subjectivity = 1 - objectivity
    return polarity, subjectivity, objectivity

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")
        text = " ".join([p.get_text() for p in soup.find_all("p")])
        return text.strip()
    except Exception as e:
        st.error(f"Error al obtener {url}: {e}")
        return ""

def get_domain_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith("www."):
        return domain[4:]
    return domain

def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error al leer PDF: {e}")
        return ""

st.title("Análisis de Sentimientos")

option = st.radio("Seleccione una opción:", ("Texto", "Cargar Archivos", "URLs"))

if option == "Texto":
    texto_input = st.text_area("Ingrese o pegue el texto que desea analizar:", height=200)
    if st.button("Analizar"):
        if texto_input:
            with st.spinner("Analizando..."):
                polarity, subjectivity, objectivity = analyze_sentiment_nltk(texto_input)
                st.write("**Resultados**")
                st.write(f"Polaridad: {polarity:.4f}")
                st.write(f"Subjetividad: {subjectivity:.4f}")
                st.write(f"Objetividad: {objectivity:.4f}")

                df = pd.DataFrame({
                    "Archivo": ["Texto Ingresado"],
                    "Polaridad": [polarity],
                    "Subjetividad": [subjectivity],
                    "Objetividad": [objectivity]
                })
                fig, ax = plt.subplots(figsize=(6, 4))
                df.plot(kind='bar', x='Archivo', ax=ax)
                ax.set_ylabel("Valor")
                ax.set_title("Análisis de Sentimientos")
                st.pyplot(fig)
        else:
            st.warning("Por favor ingrese texto.")

elif option == "Cargar Archivos":
    archivos = st.file_uploader("Cargar archivos (.txt, .csv, .pdf)",
                                type=["txt", "csv", "pdf"], accept_multiple_files=True)
    if st.button("Analizar Archivos"):
        if archivos:
            resultados = []
            for archivo in archivos:
                with st.spinner(f"Procesando {archivo.name}..."):
                    contenido = ""
                    if archivo.type == "text/plain":
                        contenido = archivo.read().decode("utf-8")
                    elif archivo.type == "application/pdf":
                        contenido = extract_text_from_pdf(archivo)
                    elif archivo.type == "text/csv":
                        try:
                            df_csv = pd.read_csv(archivo)
                            contenido = df_csv.to_string()
                        except Exception as e:
                            st.error(f"Error al leer CSV {archivo.name}: {e}")
                            continue
                    else:
                        st.warning(f"Formato no soportado: {archivo.name}")
                        continue

                    if contenido.strip():
                        polarity, subjectivity, objectivity = analyze_sentiment_nltk(contenido)
                        resultados.append((archivo.name, polarity, subjectivity, objectivity))
                    else:
                        st.warning(f"El archivo {archivo.name} no contiene texto legible.")

            if resultados:
                df_resultados = pd.DataFrame(resultados, columns=["Archivo", "Polaridad", "Subjetividad", "Objetividad"])
                st.write(df_resultados)

                fig, ax = plt.subplots(figsize=(10, 6))
                df_resultados.plot(kind='bar', x='Archivo', ax=ax)
                ax.set_ylabel("Valor")
                ax.set_title("Comparativa de archivos")
                st.pyplot(fig)
        else:
            st.warning("No se seleccionaron archivos.")

elif option == "URLs":
    urls_input = st.text_area("Ingrese las URLs (una por línea):", height=200)
    if st.button("Obtener Texto y Analizar"):
        urls = [u.strip() for u in urls_input.split('\n') if u.strip()]
        if urls:
            resultados = []
            for url in urls:
                with st.spinner(f"Analizando {url}..."):
                    text = get_text_from_url(url)
                    if text:
                        polarity, subjectivity, objectivity = analyze_sentiment_nltk(text)
                        domain = get_domain_name(url)
                        resultados.append((domain, polarity, subjectivity, objectivity))
                    else:
                        st.warning(f"No se pudo extraer texto de {url}")

            if resultados:
                df_resultados = pd.DataFrame(resultados, columns=["Dominio", "Polaridad", "Subjetividad", "Objetividad"])
                st.write(df_resultados)

                fig, ax = plt.subplots(figsize=(10, 6))
                df_resultados.plot(kind='bar', x='Dominio', ax=ax)
                ax.set_ylabel("Valor")
                ax.set_title("Análisis de sentimientos por URL")
                st.pyplot(fig)
        else:
            st.warning("Ingrese al menos una URL.")
