import socket
import time
import pandas as pd
import random

def normalize_data(df):
    df_norm = df.copy()

    for ind, mean in enumerate(df.mean()):
        df_norm[df.keys()[ind]] = df_norm[df.keys()[ind]] - mean

    for ind, std in enumerate(df.std()):
        df_norm[df.keys()[ind]] = df_norm[df.keys()[ind]] / std
        
    return df_norm

def get_df_every_2_seconds(csv_file_path, chunk_size):
    total_rows = pd.read_csv(csv_file_path).shape[0]
    num_chunks = total_rows // chunk_size  # Number of complete chunks
    remaining_rows = total_rows % chunk_size  # Remaining rows in the last chunk

    reader = pd.read_csv(csv_file_path, chunksize=chunk_size)
    
    for i in range(num_chunks):
        df_chunk = next(reader)
        df_chunk = normalize_data(df_chunk)
        df_chunk = df_chunk.sample(frac=1)  # Shuffle the rows
        yield df_chunk

    if remaining_rows > 0:
        remaining_chunk = pd.concat(list(reader)[-remaining_rows:])
        remaining_chunk = normalize_data(remaining_chunk)
        remaining_chunk = remaining_chunk.sample(frac=1)  # Shuffle the rows
        yield remaining_chunk


def send_data_by_socket(clientsocket, data):
    try:
        # Envío de datos
        clientsocket.sendall(data)
    except (ConnectionResetError, BrokenPipeError) as e:
        print('Error:', str(e))

csv_file_path = 'measures_v2.csv'
chunk_size = 42

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
HOST = ''
PORT = 5000

sock.bind((HOST, PORT))
sock.listen(5)
print("Server is running")

clientsocket, address = sock.accept()
print('Connection established:', address)

for df in get_df_every_2_seconds(csv_file_path, chunk_size):
    
    data = df.to_csv(index=False).encode()  # Convierte el DataFrame en bytes
    print(df.columns)
    print(df.shape[0])
    if df.shape[0] >= chunk_size:
        send_data_by_socket(clientsocket, data)
        
    # Wait for 2 minutes before the next iteration
    time.sleep(5)

clientsocket.close()
sock.close()