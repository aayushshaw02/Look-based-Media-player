import socket

# Set up a socket and listen for incoming connections
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 8000))
server_socket.listen(1)

# Wait for a client to connect and send a message
client_socket, address = server_socket.accept()
print(f'Client connected: {address}')
message = client_socket.recv(1024)
print(f'Received message: {message.decode()}')

# Send a response back to the client
response = 'Hii Arush!'
# while(1):
client_socket.send(response.encode())

# Clean up the sockets
client_socket.close()
server_socket.close()
