import socket

# Connect to the server and send a message
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 8000))
message = 'Hello from the client!'
client_socket.send(message.encode())

# Wait for a response from the server
# while(1):
response = client_socket.recv(1024)
print(f'Received response: {response.decode()}')

# Clean up the socket
client_socket.close()
