import json
import socket
import threading
import time

global send_msgs
send_msgs = []

def receive_message(ip, port):
    sock = socket.socket()
    print('start bind')
    sock.bind((ip, port))
    print('listen')
    sock.listen(1)
    while True:
        try:
            # 接收来自服务器的数据
            conn, addr = sock.accept()
            print(f'connect by: {(conn, addr)}')
            break
        except Exception as e:
            continue
    send_msgs.append('StartProgram')
    while True:
        messages = conn.recv(1024).decode('utf-8')
        print(f'messages:{messages}')
        content = json.loads(messages)
        print(f'receive: {content}')
        order_value = content.get('order', None)

        print(f'receive: {order_value}')
        if order_value is not None:
            time.sleep(3)
            send_msgs.append('AudioFinish')


def send_message(ip, port):
    send_client = socket.socket()
    while True:
        try:
            send_client.connect((ip, port))
            break
        except Exception as e:
            print(f'connect failed: {e}')
            time.sleep(1)
            continue
    while True:
        if len(send_msgs) > 0:
            print(f'\nsend: {send_msgs[-1]}\n')
            send_client.send(send_msgs[-1].encode('utf-8'))
            send_msgs.pop()


thread1 = threading.Thread(target=send_message, args=('localhost', 8881))
thread2 = threading.Thread(target=receive_message, args=('localhost', 8882))
thread1.start()
thread2.start()
time.sleep(100000)