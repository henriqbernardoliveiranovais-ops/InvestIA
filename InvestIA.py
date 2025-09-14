# ============ IMPORTS ============
import os
import warnings
import ssl

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- E-mail ---
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings("ignore")

# ============ STREAMLIT CONFIG (sempre primeiro comando) ============
st.set_page_config(page_title="InvestBot ML", page_icon="📈", layout="wide")

# ============ AMBIENTE/YFINANCE (opcional, ajuda no Windows) ============
# força yfinance a usar requests e, se existir, aponta o certificado sem acentos
os.environ.setdefault("YF_USE_CURL", "0")
_cert_path = r"C:\certifi\cacert.pem"
if os.path.exists(_cert_path):
    os.environ.setdefault("SSL_CERT_FILE", _cert_path)
    os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]
    os.environ["CURL_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]

# ============ .ENV ============
load_dotenv()

# ============ SESSION STATE (depois do set_page_config) ============
st.session_state.setdefault("dados_acao", None)
st.session_state.setdefault("dados_completo", None)
st.session_state.setdefault("ticker", None)
st.session_state.setdefault("model", None)
st.session_state.setdefault("scaler", None)
st.session_state.setdefault("X", None)
st.session_state.setdefault("y", None)
st.session_state.setdefault("scores", None)
st.session_state.setdefault("model_type", None)
st.session_state.setdefault("features", None)
st.session_state.setdefault("predictions", None)
st.session_state.setdefault("days_ahead", 5)

# ============ FUNÇÕES ============

# ---------------------------
# E-MAIL: provedores e envio
# ---------------------------

def _smtp_config(provider: str):
    """
    Retorna (host, port, mode) para o provedor.
    mode: 'SSL' ou 'STARTTLS'
    """
    p = (provider or "").strip().lower()
    if "gmail" in p or "google" in p:
        return "smtp.gmail.com", 465, "SSL"          # Gmail: SSL direto
    if "outlook" in p or "hotmail" in p or "live" in p or "office" in p:
        return "smtp.office365.com", 587, "STARTTLS" # Outlook/Hotmail: STARTTLS
    if "yahoo" in p:
        return "smtp.mail.yahoo.com", 465, "SSL"
    # padrão seguro
    return "smtp.gmail.com", 465, "SSL"

def _get_cred(st_module):
    """
    Busca credenciais na seguinte ordem de prioridade:
    1) st.secrets["GMAIL_USER"] / st.secrets["GMAIL_APP_PASSWORD"]
    2) st.secrets["sender_email"] / st.secrets["sender_password"]
    3) Variáveis de ambiente: GMAIL_USER / GMAIL_APP_PASSWORD
    4) Variáveis de ambiente: sender_email / sender_password
    """
    user = None
    pwd = None
    # 1 e 2 - st.secrets
    try:
        if "GMAIL_USER" in st_module.secrets and "GMAIL_APP_PASSWORD" in st_module.secrets:
            user = st_module.secrets["GMAIL_USER"]
            pwd = st_module.secrets["GMAIL_APP_PASSWORD"]
        elif "sender_email" in st_module.secrets and "sender_password" in st_module.secrets:
            user = st_module.secrets["sender_email"]
            pwd = st_module.secrets["sender_password"]
    except Exception:
        pass

    # 3 e 4 - env
    user = user or os.getenv("GMAIL_USER") or os.getenv("sender_email")
    pwd  = pwd  or os.getenv("GMAIL_APP_PASSWORD") or os.getenv("sender_password")

    # normaliza senha de app: remove espaços que o Google exibe
    if isinstance(pwd, str):
        pwd = pwd.replace(" ", "").strip()

    return user, pwd

def send_mail_alert(
    provider: str,
    sender_email: str,
    sender_password: str,
    recipient_email: str,
    subject: str,
    html_body: str,
) -> bool:
    """
    Envia e-mail usando o provedor escolhido.
    - Gmail/Yahoo: usar SENHA DE APLICATIVO (não a senha normal).
    - Outlook/Hotmail: geralmente requer STARTTLS e autenticação moderna da conta.
    """

    # Credenciais: args > st.secrets/env
    _user_fallback, _pwd_fallback = _get_cred(st)
    sender_email = (sender_email or _user_fallback or "").strip()
    sender_password = (sender_password or _pwd_fallback or "").replace(" ", "").strip()

    if not sender_email or not sender_password:
        st.error("Credenciais de e-mail não configuradas. Informe e-mail de remetente e a **SENHA DE APLICATIVO**.")
        return False

    host, port, mode = _smtp_config(provider)

    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        if mode == "SSL":
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=context, timeout=30) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, [recipient_email], msg.as_string())
        else:
            context = ssl.create_default_context()
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, [recipient_email], msg.as_string())

        return True

    except smtplib.SMTPAuthenticationError as e:
        code = getattr(e, "smtp_code", "535")
        msgb = getattr(e, "smtp_error", b"").decode(errors="ignore")
        st.error(f"Falha de autenticação SMTP ({code}).")
        st.info("⚠️ Use **senha de app** (com 2FA ativada). Confira se o e-mail do remetente corresponde à conta que gerou a senha de app.")
        st.caption(msgb)
        return False
    except Exception as e:
        st.error(f"Erro ao enviar e-mail: {e}")
        try:
            print("Erro SMTP:", repr(e))
        except:
            pass
        return False


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Médias Móveis
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12).mean()
    exp2 = df["Close"].ewm(span=26).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # Retornos/Vol
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()

    # Volume médio
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()

    return df


def prepare_features(df: pd.DataFrame, days_ahead: int = 5):
    df = calculate_technical_indicators(df)
    df["Target"] = df["Close"].shift(-days_ahead)
    df_clean = df.dropna()

    features = [
        "Close", "SMA_5", "SMA_20", "SMA_50", "RSI", "MACD",
        "MACD_Histogram", "BB_Upper", "BB_Lower", "Volatility",
        "Volume_MA", "Daily_Return",
    ]
    X = df_clean[features]
    y = df_clean["Target"]
    return X, y, df_clean


def train_ml_model(X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest"):
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_type == "linear_regression":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=1.0)
    elif model_type == "lasso":
        model = Lasso(alpha=0.1)
    else:
        model = RandomForestRegressor(random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    scores, predictions = [], []

    for tr, te in tscv.split(X):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        scores.append({
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2": r2_score(y_test, y_pred),
        })
        predictions.extend(y_pred)

    # Ajuste final com todos os dados
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)
    model.fit(X_scaled, y)

    return model, scaler_final, scores, np.asarray(predictions)


def make_predictions(model, scaler, last_data: pd.Series, features: list, days_ahead: int = 5) -> float:
    last_features = last_data[features].values.reshape(1, -1)
    last_features_scaled = scaler.transform(last_features)
    return float(model.predict(last_features_scaled)[0])


# ============ UI ============
st.title("🤖 InvestBot com Machine Learning")
st.markdown("**Análise avançada de ações com previsão de preços usando Machine Learning**")

# Sidebar
st.sidebar.header("⚙️ Configurações")
ticker_input = st.sidebar.text_input("Digite o ticker da ação (ex: PETR4, ITUB4, VALE3):", "PETR4").upper()
if not ticker_input.endswith(".SA"):
    ticker_input += ".SA"

period = st.sidebar.selectbox("Período dos dados:", ["1y", "2y", "5y", "max"], index=0)
model_type = st.sidebar.selectbox(
    "Modelo de Machine Learning:",
    ["random_forest", "gradient_boosting", "linear_regression", "ridge", "lasso"],
    index=0
)
days_ahead = st.sidebar.slider("Dias para previsão:", 1, 30, 5)
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Os modelos de ML são treinados com dados históricos e indicadores técnicos.")

# Botão: buscar e treinar
if st.sidebar.button("🚀 Buscar Dados e Treinar Modelo", type="primary"):
    st.write(f"🔎 Buscando dados para {ticker_input}...")
    try:
        dados_acao = yf.Ticker(ticker_input).history(period=period, auto_adjust=True)
        if dados_acao.empty:
            st.error(f"Não foi possível encontrar dados para {ticker_input}.")
        else:
            st.success(f"✅ Dados de {ticker_input} carregados com sucesso!")

            with st.spinner("🔄 Calculando indicadores técnicos..."):
                X, y, dados_completo = prepare_features(dados_acao, days_ahead)

            if len(X) > 0:
                with st.spinner("🤖 Treinando modelo de Machine Learning..."):
                    model, scaler, scores, predictions = train_ml_model(X, y, model_type)

                # Persistência
                st.session_state.update({
                    "dados_acao": dados_acao,
                    "dados_completo": dados_completo,
                    "ticker": ticker_input,
                    "model": model,
                    "scaler": scaler,
                    "X": X,
                    "y": y,
                    "scores": scores,
                    "model_type": model_type,
                    "features": X.columns.tolist(),
                    "predictions": predictions,
                    "days_ahead": days_ahead,   # <- salvar
                })

                st.success("✅ Modelo treinado com sucesso!")
            else:
                st.warning("Dados insuficientes para treinar o modelo.")
    except Exception as e:
        st.error(f"Ocorreu um erro ao buscar/treinar: {e}")

# Se temos dados, mostra as tabs
if st.session_state["dados_acao"] is not None and not st.session_state["dados_acao"].empty:
    dados_acao = st.session_state["dados_acao"]
    dados_completo = st.session_state["dados_completo"]
    ticker_atual = st.session_state["ticker"]
    model = st.session_state["model"]
    scaler = st.session_state["scaler"]
    X = st.session_state["X"]
    y = st.session_state["y"]
    scores = st.session_state["scores"]
    model_type = st.session_state["model_type"]
    features = st.session_state["features"]
    predictions_state = st.session_state["predictions"]
    days_ahead_state = st.session_state.get("days_ahead", days_ahead)

    avg_mae = float(np.mean([s["MAE"] for s in scores]))
    avg_rmse = float(np.mean([s["RMSE"] for s in scores]))
    avg_r2 = float(np.mean([s["R2"] for s in scores]))

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise", "🤖 Previsões", "📈 Gráficos", "📧 Alertas"])

    # -------- Tab 1: Análise --------
    with tab1:
        st.subheader(f"Análise de {ticker_atual}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Último Preço", f"R$ {dados_acao['Close'].iloc[-1]:.2f}")
        with c2:
            change = float(dados_acao["Close"].iloc[-1] - dados_acao["Close"].iloc[-2])
            st.metric("Variação Diária", f"R$ {change:.2f}")
        with c3:
            pct_change = (change / float(dados_acao["Close"].iloc[-2])) * 100
            st.metric("Variação %", f"{pct_change:.2f}%")

        st.subheader("📈 Performance do Modelo")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE Médio", f"R$ {avg_mae:.2f}")
        c2.metric("RMSE Médio", f"R$ {avg_rmse:.2f}")
        c3.metric("R² Médio", f"{avg_r2:.3f}")
        st.info(f"Modelo utilizado: {model_type.upper().replace('_', ' ')}")

    # -------- Tab 2: Previsões --------
    with tab2:
        st.subheader("🔮 Previsões de Preço")
        last_data = dados_completo.iloc[-1]
        prediction = make_predictions(model, scaler, last_data, features, days_ahead_state)
        current_price = float(dados_acao["Close"].iloc[-1])
        predicted_pct_change = ((prediction - current_price) / current_price) * 100

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Preço Atual", f"R$ {current_price:.2f}")
        with c2:
            st.metric(f"Previsão para {days_ahead_state} dias", f"R$ {prediction:.2f}", delta=f"{predicted_pct_change:.2f}%")

        if predicted_pct_change > 5:
            st.success("📈 Forte tendência de alta prevista!")
        elif predicted_pct_change > 2:
            st.info("📈 Tendência moderada de alta prevista")
        elif predicted_pct_change < -5:
            st.error("📉 Forte tendência de baixa prevista!")
        elif predicted_pct_change < -2:
            st.warning("📉 Tendência moderada de baixa prevista")
        else:
            st.info("➡️ Preço estável previsto")

        if hasattr(model, "feature_importances_"):
            st.subheader("🎯 Importância das Features")
            fi = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
            fig_importance = go.Figure()
            fig_importance.add_trace(go.Bar(x=fi["Importance"], y=fi["Feature"], orientation="h"))
            fig_importance.update_layout(title="Importância das Variáveis no Modelo", xaxis_title="Importância", yaxis_title="Variável", height=400)
            st.plotly_chart(fig_importance, use_container_width=True)

    # -------- Tab 3: Gráficos --------
    with tab3:
        st.subheader("📊 Visualizações")
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Preços e Médias Móveis", "Indicadores Técnicos"))

        fig.add_trace(go.Scatter(x=dados_completo.index, y=dados_completo["Close"], name="Preço", line=dict(color="blue")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dados_completo.index, y=dados_completo["SMA_20"], name="SMA 20", line=dict(color="orange", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=dados_completo.index, y=dados_completo["SMA_50"], name="SMA 50", line=dict(color="green", dash="dash")), row=1, col=1)

        fig.add_trace(go.Scatter(x=dados_completo.index, y=dados_completo["RSI"], name="RSI", line=dict(color="purple")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        if predictions_state is not None and len(predictions_state) > 0 and len(y) > 0:
            st.subheader("📋 Previsões vs Valores Reais")
            k = max(1, len(y) // 5)  # tamanho aproximado do último fold
            last_fold_predictions = np.asarray(predictions_state)[-k:]
            actual_values = y.iloc[-len(last_fold_predictions):]
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=actual_values.index, y=actual_values, name="Valores Reais", line=dict(color="blue")))
            fig_cmp.add_trace(go.Scatter(x=actual_values.index, y=last_fold_predictions, name="Previsões", line=dict(color="red", dash="dash")))
            fig_cmp.update_layout(title="Comparação: Previsões vs Valores Reais", xaxis_title="Data", yaxis_title="Preço (R$)")
            st.plotly_chart(fig_cmp, use_container_width=True)
        else:
            st.info("Treine o modelo para visualizar a comparação de previsões.")

    # -------- Tab 4: Alertas (E-mail) --------
    with tab4:
        st.subheader("📩 Alertas por E-mail")

        colA, colB = st.columns(2)
        with colA:
            email_provider = st.selectbox(
                "Provedor do remetente:",
                ["Gmail", "Outlook/Hotmail", "Yahoo"],
                index=0,
                help="Gmail/Yahoo: use **SENHA DE APLICATIVO** (2FA ativada)."
            )
            # tenta pré-preencher com secrets/env
            _user_prefill, _pwd_prefill = _get_cred(st)
            sender_email_input = st.text_input("E-mail remetente:", value=_user_prefill or os.getenv("sender_email", ""))

        with colB:
            sender_password_input = st.text_input(
                "Senha de aplicativo do remetente:",
                type="password",
                value=_pwd_prefill or os.getenv("sender_password", "")
            )

        recipient_email = st.text_input("Seu e-mail (destinatário):", "seu.email@exemplo.com", key="user_email")

        if st.button("📤 Enviar Relatório por E-mail"):
            if not recipient_email or "@" not in recipient_email:
                st.error("Por favor, insira um e-mail de destinatário válido.")
            else:
                try:
                    current_price = float(dados_acao["Close"].iloc[-1])
                    change_email = float(dados_acao["Close"].iloc[-1] - dados_acao["Close"].iloc[-2])
                    pct_change_email = (change_email / float(dados_acao["Close"].iloc[-2])) * 100
                except Exception:
                    current_price = None
                    pct_change_email = None

                prediction_email = make_predictions(model, scaler, dados_completo.iloc[-1], features, days_ahead_state)
                predicted_pct_change_email = None
                if current_price:
                    predicted_pct_change_email = ((prediction_email - current_price) / current_price) * 100

                subject = f"📊 Relatório InvestBot - {ticker_atual}"
                html_body = f"""
                <html><body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <h2 style="color:#2c3e50;">📈 Relatório de Análise - {ticker_atual}</h2>
                    <div style="background:#f8f9fa;padding:20px;border-radius:10px;margin:20px 0;">
                        <h3 style="color:#2c3e50;">💰 Informações Atuais</h3>
                        {"<p><strong>Preço Atual:</strong> R$ {:.2f}</p>".format(current_price) if current_price is not None else ""}
                        {"<p><strong>Variação Diária:</strong> {:.2f}%</p>".format(pct_change_email) if pct_change_email is not None else ""}
                        <p><strong>Volume Médio (20 dias):</strong> {dados_completo['Volume_MA'].iloc[-1]:.0f}</p>
                    </div>
                    <div style="background:#e8f5e8;padding:20px;border-radius:10px;margin:20px 0;">
                        <h3 style="color:#2c3e50;">🔮 Previsão para {days_ahead_state} dias</h3>
                        <p><strong>Preço Previsto:</strong> R$ {prediction_email:.2f}</p>
                        {"<p><strong>Variação Esperada:</strong> {:.2f}%</p>".format(predicted_pct_change_email) if predicted_pct_change_email is not None else ""}
                        <p><strong>Modelo Utilizado:</strong> {model_type.upper().replace('_', ' ')}</p>
                    </div>
                    <div style="background:#fff3cd;padding:20px;border-radius:10px;margin:20px 0;">
                        <h3 style="color:#2c3e50;">📊 Performance do Modelo</h3>
                        <p><strong>MAE:</strong> R$ {avg_mae:.2f}</p>
                        <p><strong>RMSE:</strong> R$ {avg_rmse:.2f}</p>
                        <p><strong>R²:</strong> {avg_r2:.3f}</p>
                    </div>
                    <p><em>⚠️ Esta análise é apenas educacional e não constitui recomendação de investimento.</em></p>
                </body></html>
                """

                with st.spinner("Enviando e-mail..."):
                    ok = send_mail_alert(
                        provider=email_provider,
                        sender_email=sender_email_input,
                        sender_password=sender_password_input,
                        recipient_email=recipient_email,
                        subject=subject,
                        html_body=html_body,
                    )

                if ok:
                    st.success(f"✅ E-mail enviado com sucesso para {recipient_email}!")
                else:
                    st.error("❌ Falha ao enviar e-mail. Verifique provedor, e-mail e senha de aplicativo.")

else:
    st.info("👆 Use a sidebar para buscar dados de uma ação e treinar o modelo de Machine Learning.")

# Footer
st.markdown("---")
st.caption("🤖 InvestBot com Machine Learning — Uso educacional. Não é recomendação de investimento.")
