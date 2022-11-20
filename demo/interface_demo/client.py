import base64
import socket
import time
import cv2
import datetime
import json
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

    def _pack_data(self, image_data):
        image_code = base64.b64encode(image_data).decode()
        packed_data = {
            "time": str(time.time()),
            "type": "jpg",
            "length": str(len(image_code)),
            "data": image_code
        }
        print(packed_data)
        return json.dumps(packed_data)

    def send_binary(self, image_data):
        send_data = self._pack_data(image_data)
        self.socket_client.send(send_data.encode())
        print("Send done.")
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

