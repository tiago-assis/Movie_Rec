import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from scipy.sparse import load_npz
import io
import time


def get_matrix_data():
    # Step 1: Load the credentials from Streamlit secrets
    CREDS_INFO = {
        "type": st.secrets['google_drive_api']['type'],
        "project_id": st.secrets['google_drive_api']['project_id'],
        "private_key_id": st.secrets['google_drive_api']['private_key_id'],
        "private_key": st.secrets['google_drive_api']['private_key'].replace("\\n", "\n"),
        "client_email": st.secrets['google_drive_api']['client_email'],
        "client_id": st.secrets['google_drive_api']['client_id'],
        "auth_uri": st.secrets['google_drive_api']['auth_uri'],
        "token_uri": st.secrets['google_drive_api']['token_uri'],
        "auth_provider_x509_cert_url": st.secrets['google_drive_api']['auth_provider_x509_cert_url'],
        "client_x509_cert_url": st.secrets['google_drive_api']['client_x509_cert_url'],
        "universe_domain": st.secrets['google_drive_api']['universe_domain']
    }
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    # Step 2: Create the Credentials object
    creds = service_account.Credentials.from_service_account_info(
        CREDS_INFO, scopes=SCOPES)
    # Step 3: Build the Drive API service
    service = build('drive', 'v3', credentials=creds)

    # Step 4: Specify the file ID and request the file
    file_id = "1SEYyy9csamwh2l0qdraTkkmubjIPTxhK"  # movie_recommendations.npz
    request = service.files().get_media(fileId=file_id)

    # Step 5: Download the file into a memory buffer
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    done = False

    t1 = time.time()
    t2 = t1
    while not done:
        if t2 - t1 > 60:
            print("File download timed out (60s).")
            return
        status, done = downloader.next_chunk()
        t2 = time.time()

    # Step 6: Load the pickle file from the memory buffer
    file_stream.seek(0)  # Reset the stream position to the beginning
    data = load_npz(file_stream)

    return data
