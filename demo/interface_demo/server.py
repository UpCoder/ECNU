import json
import socket
import time
import cv2
import base64
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

        received_data = connection.recv(1024*1024)
        received_data = json.loads(received_data.decode())

        send_time = received_data["time"]
        length = received_data["length"]
        image_data = base64.b64decode(received_data["data"].encode())
        print(send_time, length)
        # print(image_data)
        img = cv2.imdecode(np.frombuffer(image_data, dtype='uint8'), cv2.IMREAD_COLOR)
        print(type(img), img.shape)
        cv2.imwrite('received.jpg', img)
        connection.close()

if __name__ == '__main__':
    server = SocketServer()
    server.bind()
    server.receive_image()