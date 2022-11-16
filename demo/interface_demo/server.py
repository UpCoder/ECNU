import socket
import time
import cv2
import numpy as np

class SocketServer(object):
    def __init__(self):
        pass

    def bind(self, host="localhost", post=8888):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, post))  # 绑定端口
        self.server.listen(100)

    def receive_image(self):
        connection, address = self.server.accept()
        print(connection, address)

        head = connection.recv(1024)
        print(head.decode())
        length, send_time = head.decode().split('|')
        length = int(length)
        connection.send(b"Received head successfully")

        start_time = time.time()
        get_length = 0
        result_received = b''
        while get_length < length:
            received_data = connection.recv(1024*10)
            result_received += received_data
            get_length = get_length + len(received_data)
        print('应该接收{}, 实际接收{}'.format(length, len(result_received)))
        connection.send(b"Received image successfully")
        print('received image cost:', time.time()-start_time)

        img = cv2.imdecode(np.frombuffer(result_received, dtype='uint8'), cv2.IMREAD_COLOR)
        print(type(img), img.shape)
        cv2.imwrite('received.jpg', img)
        connection.close()

if __name__ == '__main__':
    server = SocketServer()
    server.bind()
    server.receive_image()