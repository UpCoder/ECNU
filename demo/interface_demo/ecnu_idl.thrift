namespace csharp ecnu.demo
namespace python ecnu.demo

struct imageReq {
    1: binary image_data,
    2: string image_url,
    254: int64 timestamp,
    255: optional map<string, string> extra
}

struct imageResp {
    1: binary image_data,
    2: map<string, string> infos,
    255: optional int64 resp_code,
}

struct audioReq {
    1: list<binary> audio_data,
    254: int64 timestamp,
    255: optional map<string, string> extra
}

struct audioResp {
    1: map<string, string> infos,
    255: optional int64 resp_code,
}

service imageService {
    imageResp imagePredict(1: imageReq),
    audioResp audioPredict(1: audioReq),
}