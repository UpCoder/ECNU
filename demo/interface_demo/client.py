import socket
import time
import cv2
import datetime
import numpy as np

class SocketClient(object):
    def __int__(self):
        pass

    def connet(self, host="localhost", port=8888):
        try:
            self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # AF_INET（TCP/IP – IPv4）协议
            self.socket_client.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket_client.connect((host, port))
        except Exception as e:
            print(str(e))
        return

    def send_binary(self, binary_data):
        self.socket_client.send('{}|{}'.format(len(binary_data), datetime.datetime.now()).encode())
        reply = self.socket_client.recv(1024)

        if reply.decode() == "Received head successfully":
            self.socket_client.send(binary_data)
        reply = self.socket_client.recv(1024)
        if reply.decode() == "Received image successfully":
            print("Send succeeded.")
        return

    def send_image(self, image_path):
        img = cv2.imread(image_path)
        img_bytes = np.array(cv2.imencode('.jpg', img)[1]).tostring()
        print(len(img_bytes), type(img_bytes))
        self.send_binary(img_bytes)

if __name__ == '__main__':
    client = SocketClient()
    client.connet()
    client.send_image("test.jpg")


