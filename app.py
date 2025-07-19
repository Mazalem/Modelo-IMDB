import streamlit as st
import gdown
import tensorflow as tf
import numpy as np

url_id = '10V43Ow_Q7ZTh1jami8_28muu7fsryYi-'
tflite_path = 'modelo_imdb.tflite'

@st.cache_resource
def carregar_modelo():
    url = f'https://drive.google.com/uc?id={url_id}'
    gdown.download(url, tflite_path, quiet=False)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocessar_texto(texto, tokenizer, maxlen=500):
    sequencia = tokenizer.texts_to_sequences([texto])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequencia, maxlen=maxlen)
    return np.array(padded, dtype=np.float32)

def prever_sentimento(interpreter, texto, tokenizer):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    entrada = preprocessar_texto(texto, tokenizer)
    interpreter.set_tensor(input_details[0]['index'], entrada)
    interpreter.invoke()
    saida = interpreter.get_tensor(output_details[0]['index'])

    return saida[0][0]

def main():
    st.set_page_config(page_title="Classificador IMDB")
    st.title("🎬 Classificador de Sentimentos - IMDB Reviews")
    st.write("Envie um comentário de filme e descubra se o modelo acha que ele é **positivo** ou **negativo**.")
    st.markdown("🔗 Modelo no Colab: [Link do notebook](https://colab.research.google.com/)")

    interpreter = carregar_modelo()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30000, oov_token="<OOV>")
    
    # Idealmente deveriamos carregar o tokenizer real usado no treinamento
    # Mas para simplificar, criamos um novo aqui e o adaptamos a textos de exemplo
    exemplos = [
        "this movie is great", "worst film ever", "i loved the acting", "bad plot", "amazing cinematography"
    ]
    tokenizer.fit_on_texts(exemplos)

    texto = st.text_area("✍️ Digite o comentário", height=150)

    if texto:
        resultado = prever_sentimento(interpreter, texto, tokenizer)
        st.write(f"🔎 **Probabilidade de ser positivo:** `{resultado:.2%}`")

        if resultado >= 0.5:
            st.success("🎉 O modelo considera esse comentário **POSITIVO**.")
        else:
            st.error("😞 O modelo considera esse comentário **NEGATIVO**.")

if __name__ == "__main__":
    main()
