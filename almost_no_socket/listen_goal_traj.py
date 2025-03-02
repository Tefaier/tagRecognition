import socket

def start_server(host='192.168.56.1', port=30009):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print("Wait message...")
    client_socket, addr = server_socket.accept()
    data = client_socket.recv(1024).decode('utf-8').strip()
    client_socket.close()
    print("Get it")
    print("Put message in data.txt")
    with open("data.txt", "w") as file:
        file.write(data)

if __name__ == "__main__":
    start_server()
