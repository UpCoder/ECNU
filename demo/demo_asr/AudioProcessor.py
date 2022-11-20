from realtime_demo_raw import AudioASRRecord
from realtime_if_stop import AudioStopRecord


class AudioProcessor(object):
    def __init__(self, sample_rate, stop_record_duration, asr_record_duration,
                 n_channels, stop_interval=1, stop_threshold=3000):
        audio_asr_object = AudioASRRecord(sample_rate, record_duration=asr_record_duration,
                                          n_channels=n_channels)
        audio_stop_object = AudioStopRecord(sample_rate, stop_record_duration, n_channels,
                                            stop_interval, stop_threshold)
