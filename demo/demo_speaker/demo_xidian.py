import asrt_sdk

HOST = '127.0.0.1'
PORT = '20001'
PROTOCOL = 'http'
SUB_PATH = ''
speech_recognizer = asrt_sdk.get_speech_recognizer(HOST, PORT, PROTOCOL)
speech_recognizer.sub_path = SUB_PATH

FILENAME = 'A11_0.wav'
result = speech_recognizer.recognite_file(FILENAME)
print(result)
for index in range(0, len(result)):
    item = result[index]
    print("第", index, "段:", item.result)


wave_data = asrt_sdk.read_wav_datas(FILENAME)
result = speech_recognizer.recognite_speech(wave_data.str_data,
                                            wave_data.sample_rate,
                                            wave_data.channels,
                                            wave_data.byte_width)
print(result)
print(result.result)

result = speech_recognizer.recognite_language(result.result)
print(result)
print(result.result)