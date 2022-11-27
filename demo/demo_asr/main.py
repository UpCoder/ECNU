
def audio_receive_message(conn, audio_processor_obj):
    global global_status
    while True:
        messages = conn.recv(1024).decode('utf-8')
        # messages = json.loads(messages)
        # content = messages.get('order', None)
        content = messages
        if content is None:
            print(f'ignore message: {messages}')
            continue
        if content == 'StartProgram':
            print('start interview')
            # audio_processor_obj.audio_stop_object.reset()
            # audio_processor_obj.audio_asr_object.reset()
            audio_processor_obj.send_socket_client.send_message(json.dumps({
                'order': 0
            }))
            global_status.current_question_id = 0
        elif content == 'AudioFinish':
            # 播放完成，开始监听说话是否停止，并且开始读取ASR的相关信息
            # 开启监听
            audio_processor_obj.audio_asr_object.in_listen = True
            audio_processor_obj.audio_stop_object.in_listening = True