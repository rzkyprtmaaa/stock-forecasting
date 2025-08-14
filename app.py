import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Inisialisasi state halaman jika belum ada
if 'page' not in st.session_state:
    st.session_state.page = 'Home'  # default halaman awal

# Sidebar dengan 4 tombol navigasi dengan ukuran sama
st.sidebar.markdown(
    "<h2 style='text-align: center;'>Navigasi</h2>",
    unsafe_allow_html=True
)
button_style = """
    <style>
    div.stButton > button {
        width: 100%;
        height: 3em;
        margin-bottom: 0.5em;
    }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

col1 = st.sidebar.empty()
col2 = st.sidebar.empty()
col3 = st.sidebar.empty()
col4 = st.sidebar.empty()

if col1.button("Home"):
    st.session_state.page = 'Home'
if col2.button("LSTM"):
    st.session_state.page = 'LSTM'
if col3.button("GRU"):
    st.session_state.page = 'GRU'
if col4.button("Perbandingan"):
    st.session_state.page = 'Compare'

if st.session_state.page == 'Home':
    st.title('Prediksi Harga Penutupan Saham Bank BCA Menggunakan LSTM & GRU')

    # Baca langsung file CSV dari direktori lokal dengan penanganan error jika file tidak ditemukan
    try:
        df = pd.read_csv('bbcanew.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.date
        df = df.set_index('Date')

        # Delete 'Dividends' and 'Stock Splits' columns
        df = df.drop(['Dividends', 'Stock Splits'], axis=1)

        # Remove numbers after the decimal point in all columns
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]): # Check if column is numeric
                df[col] = df[col].astype(int) # Convert numeric columns to integers

        # Convert the 'Volume' column to string type
        df['Volume'] = df['Volume'].astype(str)

        # Apply a lambda function to format the 'Volume' column with commas
        df['Volume'] = df['Volume'].apply(lambda x: "{:,}".format(int(x)))

        # Tampilkan data
        st.header("Data Historis Saham Bank BCA:")

        # Buat plot
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='BCA Close Price',
            hovertemplate='%{x|%d-%m-%Y}<br>Harga: %{y}<extra></extra>'
        ))

        fig.update_layout(
            xaxis_title='Tahun',
            yaxis_title='Harga Saham (IDR)',
            width=1000,
            height=500,
        )

        # Tampilkan di Streamlit
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.error("Data Tidak Ditemukan")

elif st.session_state.page == 'LSTM':
    st.header("Model Algoritma LSTM")
    try:
        testing = pd.read_csv('lstm_testing.csv')
        testing['Date'] = pd.to_datetime(testing['Date'])  # Pastikan Date dikenali sebagai tanggal
        testing.set_index('Date', inplace=True)
        future = pd.read_csv('future_lstm2.csv')
        future['Date'] = pd.to_datetime(future['Date'])  # Pastikan Date dikenali sebagai tanggal
        future.set_index('Date', inplace=True)

        # Plot testing
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=testing.index,
            y=testing['Close'],
            mode='lines',
            name='Harga Asli',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=testing.index,
            y=testing['Predictions'],
            mode='lines',
            name='Prediksi LSTM',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Harga Asli vs Prediksi LSTM',
            xaxis_title='Waktu',
            yaxis_title='Harga Saham (IDR)',
            width=1000,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plot Future
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future.index,
            y=future['Predicted_Close'],
            mode='lines',
            name='Harga Asli',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title='Prediksi 1 Bulan ke Depan',
            xaxis_title='Waktu',
            yaxis_title='Harga Saham (IDR)',
            width=1000,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pilih rentang tanggal untuk download (letakkan di atas tombol download)
        min_date = future.index.min().date()
        max_date = future.index.max().date()
        st.subheader("Pilih Rentang Tanggal untuk Download Data Prediksi")
        date_range = st.date_input(
            "Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD"
        )

        # Tombol Download di bawah plot, hanya jika user memilih rentang tanggal
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            # Filter data sesuai rentang tanggal
            filtered = future.loc[(future.index.date >= start_date) & (future.index.date <= end_date)]
            csv = filtered.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediksi LSTM (CSV)",
                data=csv,
                file_name=f'Prediksi dari {start_date} sampai {end_date}.csv',
                mime='text/csv',
            )
        else:
            st.info("Silakan pilih rentang tanggal terlebih dahulu untuk mengunduh data.")
    except FileNotFoundError:
        st.error("Data Tidak Ditemukan")

elif st.session_state.page == 'GRU':
    
    st.header("Model Algoritma GRU")
    try:
        testing = pd.read_csv('gru_testing.csv')
        testing['Date'] = pd.to_datetime(testing['Date'])  # Pastikan Date dikenali sebagai tanggal
        testing.set_index('Date', inplace=True)
        future = pd.read_csv('future_gru4.csv')
        future['Date'] = pd.to_datetime(future['Date']) # Pastikan Date dikenali sebagai tanggal
        future.set_index('Date', inplace=True)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=testing.index,
            y=testing['Close'],
            mode='lines',
            name='Harga Asli',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=testing.index,
            y=testing['Predictions'],
            mode='lines',
            name='Prediksi GRU',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='Harga Asli vs Prediksi GRU',
            xaxis_title='Waktu',
            yaxis_title='Harga Saham (IDR)',
            width=1000,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plot Future
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future.index,
            y=future['Predicted_Close'],
            mode='lines',
            name='Harga Asli',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title='Prediksi 1 Bulan ke Depan',
            xaxis_title='Waktu',
            yaxis_title='Harga Saham (IDR)',
            width=1000,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pilih rentang tanggal untuk download (letakkan di atas tombol download)
        min_date = future.index.min().date()
        max_date = future.index.max().date()
        st.subheader("Pilih Rentang Tanggal untuk Download Data LSTM")
        date_range = st.date_input(
            "Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            format="YYYY-MM-DD"
        )

        # Tombol Download di bawah plot, hanya jika user memilih rentang tanggal
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            # Filter data sesuai rentang tanggal
            filtered = future.loc[(future.index.date >= start_date) & (future.index.date <= end_date)]
            csv = filtered.reset_index().to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Prediksi LSTM (CSV)",
                data=csv,
                file_name=f'Prediksi dari {start_date} sampai {end_date}.csv',
                mime='text/csv',
            )
        else:
            st.info("Silakan pilih rentang tanggal terlebih dahulu untuk mengunduh data.")
    except FileNotFoundError:
        st.error("Data Tidak Ditemukan")

elif st.session_state.page == 'Compare':
    st.header("Perbandingan Testing LSTM dan GRU")
    try:
        # Load data
        lstm_df = pd.read_csv("lstm_testing.csv")
        gru_df = pd.read_csv("gru_testing.csv")
        futurel = pd.read_csv('future_lstm2.csv')
        futureg = pd.read_csv('future_gru4.csv')

        lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])  # Pastikan Date dikenali sebagai tanggal
        lstm_df.set_index('Date', inplace=True)

        # Plot Prediksi Testing
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=lstm_df.index,
            y=lstm_df['Close'],
            mode='lines',
            name='Harga Asli',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=lstm_df.index,
            y=lstm_df['Predictions'],
            mode='lines',
            name='Prediksi LSTM (Testing)',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=lstm_df.index,
            y=gru_df['Predictions'],
            mode='lines',
            name='Prediksi GRU (Testing)',
            line=dict(color='red')
        ))
        fig.update_layout(
            xaxis_title='Waktu',
            yaxis_title='Harga Saham (IDR)',
            width=1000,
            height=500,
            title='Perbandingan Prediksi Testing LSTM vs GRU'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plot Prediksi Future
        futurel['Date'] = pd.to_datetime(futurel['Date'])
        futurel.set_index('Date', inplace=True)
        futureg['Date'] = pd.to_datetime(futureg['Date'])
        futureg.set_index('Date', inplace=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=futurel.index,
            y=futurel['Predicted_Close'],
            mode='lines',
            name='Prediksi LSTM (Future)',
            line=dict(color='blue')
        ))
        fig2.add_trace(go.Scatter(
            x=futureg.index,
            y=futureg['Predicted_Close'],
            mode='lines',
            name='Prediksi GRU (Future)',
            line=dict(color='red')
        ))
        fig2.update_layout(
            xaxis_title='Waktu',
            yaxis_title='Harga Saham (IDR)',
            width=1000,
            height=500,
            title='Perbandingan Prediksi 1 Bulan ke Depan LSTM vs GRU'
        )
        st.plotly_chart(fig2, use_container_width=True)
    except FileNotFoundError:
        st.error("Data Tidak Ditemukan")