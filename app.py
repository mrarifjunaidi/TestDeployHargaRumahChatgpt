import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Prediksi Model Orange",
    page_icon="🤖",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "model_orange.pickle"

FEATURE_CONFIG = {
    "umur": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 100,
        "default": 30
    },
    "pendapatan": {
        "type": "numeric",
        "input": "number",
        "min": 0,
        "max": 100000000,
        "default": 5000000
    },
    "lama_bekerja": {
        "type": "numeric",
        "input": "slider",
        "min": 0,
        "max": 40,
        "default": 5
    },
    "jenis_kelamin": {
        "type": "categorical",
        "options": ["Laki-laki", "Perempuan"]
    },
    "status_pernikahan": {
        "type": "categorical",
        "options": ["Belum Menikah", "Menikah", "Cerai"]
    }
}


@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan di:\\n{MODEL_PATH}"
        )

    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        return model

    except Exception as e:
        raise Exception(f"Gagal load model: {str(e)}")


def create_input_form():

    input_data = {}

    with st.form("prediction_form"):

        st.subheader("Input Data")

        for feature_name, config in FEATURE_CONFIG.items():

            if config["type"] == "numeric":

                if config["input"] == "slider":

                    value = st.slider(
                        label=feature_name,
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"]
                    )

                else:

                    value = st.number_input(
                        label=feature_name,
                        min_value=float(config["min"]),
                        max_value=float(config["max"]),
                        value=float(config["default"])
                    )

                input_data[feature_name] = value

            elif config["type"] == "categorical":

                value = st.selectbox(
                    label=feature_name,
                    options=config["options"]
                )

                input_data[feature_name] = value

        submitted = st.form_submit_button("Prediksi")

    return submitted, input_data


def predict_with_model(model, input_df):

    prediction = model.predict(input_df)

    probability = None

    if hasattr(model, "predict_proba"):
        try:
            probability = model.predict_proba(input_df)
        except Exception:
            probability = None

    return prediction, probability


def predict_with_orange_fallback(model, input_data):

    try:
        import Orange
        from Orange.data import (
            Domain,
            ContinuousVariable,
            DiscreteVariable,
            Table
        )

    except ImportError:
        raise ImportError(
            "Library Orange3 belum terinstall."
        )

    try:

        variables = []
        data_row = []

        for feature_name, config in FEATURE_CONFIG.items():

            if config["type"] == "numeric":

                variables.append(
                    ContinuousVariable(feature_name)
                )

                data_row.append(
                    float(input_data[feature_name])
                )

            elif config["type"] == "categorical":

                variables.append(
                    DiscreteVariable(
                        feature_name,
                        values=config["options"]
                    )
                )

                category_index = config["options"].index(
                    input_data[feature_name]
                )

                data_row.append(category_index)

        domain = Domain(variables)

        X = np.array([data_row], dtype=float)

        orange_table = Table.from_numpy(domain, X)

        prediction = model(orange_table)

        return prediction

    except Exception as e:
        raise Exception(
            f"Gagal prediksi menggunakan fallback Orange: {str(e)}"
        )


def main():

    st.title("Aplikasi Prediksi Berbasis Model Orange")

    st.write(
        """
        Aplikasi ini menggunakan model machine learning hasil training
        dari Orange Data Mining dan dijalankan melalui Streamlit Cloud.
        """
    )

    st.sidebar.header("Petunjuk")

    st.sidebar.info(
        """
        1. Masukkan data pada form input.
        2. Klik tombol Prediksi.
        3. Sistem akan menjalankan model machine learning.
        4. Hasil prediksi akan ditampilkan otomatis.
        """
    )

    st.sidebar.success(
        """
        Model dimuat dari file pickle yang berada
        di GitHub repository yang sama dengan aplikasi.
        """
    )

    st.sidebar.warning(
        """
        Pastikan nama fitur pada FEATURE_CONFIG
        sama persis dengan fitur saat training model di Orange.
        """
    )

    try:

        model = load_model()

        st.success("Model berhasil dimuat.")

    except FileNotFoundError as e:

        st.error(str(e))

        st.stop()

    except Exception as e:

        st.error(f"Terjadi kesalahan saat load model:\\n{str(e)}")

        st.stop()

    submitted, input_data = create_input_form()

    if submitted:

        st.subheader("Data Input User")

        try:

            ordered_columns = list(FEATURE_CONFIG.keys())

            input_df = pd.DataFrame(
                [[input_data[col] for col in ordered_columns]],
                columns=ordered_columns
            )

            st.dataframe(input_df)

        except Exception as e:

            st.error(
                f"Gagal membuat DataFrame input:\\n{str(e)}"
            )

            st.stop()

        try:

            prediction = None
            probability = None

            try:

                prediction, probability = predict_with_model(
                    model,
                    input_df
                )

            except Exception as sklearn_error:

                st.warning(
                    f"""
                    Pendekatan scikit-learn gagal.
                    Mencoba fallback Orange...

                    Detail:
                    {str(sklearn_error)}
                    """
                )

                prediction = predict_with_orange_fallback(
                    model,
                    input_data
                )

            st.subheader("Hasil Prediksi")

            st.success(
                f"Hasil Prediksi: {prediction}"
            )

            if probability is not None:

                st.subheader("Probabilitas / Confidence")

                try:

                    prob_df = pd.DataFrame(probability)

                    st.dataframe(prob_df)

                except Exception:
                    st.write(probability)

        except Exception as e:

            st.error(
                f"""
                Prediksi gagal dilakukan.

                Kemungkinan penyebab:
                - Nama kolom tidak sesuai dengan training model
                - Struktur input berbeda
                - Format model Orange berbeda
                - Library Orange3 belum tersedia
                - Model pickle corrupt

                Detail Error:
                {str(e)}
                """
            )

    st.markdown("---")

    st.caption(
        """
        Catatan Deployment Streamlit Cloud:

        - Jangan gunakan path lokal komputer.
        - Semua file harus berada di GitHub repository.
        - File model_orange.pickle wajib ikut di-upload.
        - Jika ukuran model terlalu besar, gunakan Git LFS.
        - Main file path di Streamlit Cloud harus diarahkan ke app.py
        """
    )


if __name__ == "__main__":
    main()
