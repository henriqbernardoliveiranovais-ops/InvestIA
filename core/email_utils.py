import os, ssl, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st

def get_email_credentials():
    user = os.getenv("GMAIL_USER") or os.getenv("sender_email")
    pwd = os.getenv("GMAIL_APP_PASSWORD") or os.getenv("sender_password")
    if pwd: pwd = pwd.replace(" ", "").strip()
    return user, pwd

def _smtp_config(provider):
    p = provider.lower()
    if "gmail" in p: return "smtp.gmail.com", 465, "SSL"
    if "outlook" in p or "hotmail" in p: return "smtp.office365.com", 587, "STARTTLS"
    if "yahoo" in p: return "smtp.mail.yahoo.com", 465, "SSL"
    return "smtp.gmail.com", 465, "SSL"

def send_mail_alert(provider, sender_email, sender_password, recipient_email, subject, html_body):
    host, port, mode = _smtp_config(provider)
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))
    try:
        if mode == "SSL":
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, [recipient_email], msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=30) as server:
                server.starttls(context=ssl.create_default_context())
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, [recipient_email], msg.as_string())
        return True
    except Exception as e:
        st.error(f"Erro ao enviar e-mail: {e}")
        return False
