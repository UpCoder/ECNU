import time
import cv2
import json
import base64
import socket
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

    def _pack_image_data(self, image_data, infos):
        image_code = base64.b64encode(image_data).decode('utf-8')
        packed_data = {
            "type": "",
            "time": "",
            "length": ""
        }
        # for k, v in infos.items():
        #     packed_data[k] = str(v)
        # print(packed_data)
        packed_data["data"] = image_code
        return json.dumps(packed_data)

    def send_image(self, image, infos):
        try:
            img_bytes = np.array(cv2.imencode('.jpg', image)[1]).tostring()
            packed_data = self._pack_image_data(img_bytes, infos)
            self.socket_client.send(packed_data.encode())
        except Exception as e:
            print(e)
        else:
            print("Send done.")

    def send_image_from_path(self, image_path):
        img = cv2.imread(image_path)
        self.send_image(img)

if __name__ == '__main__':
    client = SocketClient()
    client.connet()
    client.send_image("test.jpg")

