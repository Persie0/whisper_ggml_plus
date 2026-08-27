#include "whisper/whisper.h"

#define DR_WAV_IMPLEMENTATION
#include "whisper/examples/dr_wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include "json/json.hpp"

using json = nlohmann::json;

struct whisper_params {
    int32_t seed = -1;
    int32_t n_threads = std::min(4, (int32_t)std::thread::hardware_concurrency());
    int32_t n_processors = 1;
    int32_t offset_t_ms = 0;
    int32_t offset_n = 0;
    int32_t duration_ms = 0;
    int32_t max_context = -1;
    int32_t max_len = 0;
    int32_t best_of = 5;
    int32_t beam_size = -1;
    float word_thold = 0.01f;
    float entropy_thold = 2.40f;
    float logprob_thold = -1.00f;
    bool verbose = false;
    bool print_special_tokens = false;
    bool speed_up = false;
    bool translate = false;
    bool diarize = false;
    bool no_fallback = false;
    bool output_txt = false;
    bool output_vtt = false;
    bool output_srt = false;
    bool output_wts = false;
    bool output_csv = false;
    bool print_special = false;
    bool print_colors = false;
    bool print_progress = false;
    bool no_timestamps = false;
    bool split_on_word = false;
    std::string language = "id";
    std::string prompt;
    std::string model;
    std::string audio;
    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_outp = {};
};

static struct whisper_context * g_ctx = nullptr;
static std::string g_model_path;
static std::mutex g_mutex;

static void releaseContextLocked() {
    if (g_ctx != nullptr) {
        whisper_free(g_ctx);
        g_ctx = nullptr;
    }
    g_model_path.clear();
}

static char *jsonToChar(const json &jsonData) {
    std::string result = jsonData.dump();
    char *ch = new char[result.size() + 1];
    std::memcpy(ch, result.c_str(), result.size() + 1);
    return ch;
}

static bool loadMonoPcmF32(const std::string &path, std::vector<float> &pcmf32) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) return false;
    const drwav_uint64 totalFrames = wav.totalPCMFrameCount;
    const unsigned channels = wav.channels;
    if (channels == 0 || totalFrames == 0 ||
        totalFrames > static_cast<drwav_uint64>(std::numeric_limits<std::size_t>::max())) {
        drwav_uninit(&wav);
        return false;
    }

    pcmf32.resize(static_cast<std::size_t>(totalFrames));
    if (channels == 1) {
        const drwav_uint64 read = drwav_read_pcm_frames_f32(&wav, totalFrames, pcmf32.data());
        pcmf32.resize(static_cast<std::size_t>(read));
        drwav_uninit(&wav);
        return read > 0;
    }

    constexpr drwav_uint64 kChunkFrames = 4096;
    std::vector<float> interleaved(kChunkFrames * channels);
    drwav_uint64 written = 0;
    while (written < totalFrames) {
        const drwav_uint64 wanted = std::min(kChunkFrames, totalFrames - written);
        const drwav_uint64 read = drwav_read_pcm_frames_f32(&wav, wanted, interleaved.data());
        if (read == 0) break;
        for (drwav_uint64 frame = 0; frame < read; ++frame) {
            double sum = 0.0;
            for (unsigned channel = 0; channel < channels; ++channel) {
                sum += interleaved[static_cast<std::size_t>(frame) * channels + channel];
            }
            pcmf32[static_cast<std::size_t>(written + frame)] =
                static_cast<float>(sum / channels);
        }
        written += read;
    }
    drwav_uninit(&wav);
    pcmf32.resize(static_cast<std::size_t>(written));
    return written > 0;
}

static json transcribe(const json &jsonBody) {
    std::lock_guard<std::mutex> lock(g_mutex);
    whisper_params params;
    params.n_threads = jsonBody["threads"];
    params.verbose = jsonBody["is_verbose"];
    params.translate = jsonBody["is_translate"];
    params.language = jsonBody["language"];
    params.print_special_tokens = jsonBody["is_special_tokens"];
    params.no_timestamps = jsonBody["is_no_timestamps"];
    params.model = jsonBody["model"];
    params.audio = jsonBody["audio"];
    params.split_on_word = jsonBody["split_on_word"];
    params.diarize = jsonBody["diarize"];

    json jsonResult;
    jsonResult["@type"] = "transcribe";

    if (g_ctx == nullptr || g_model_path != params.model) {
        releaseContextLocked();
        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = true;
        cparams.flash_attn = true;
        g_ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
        if (g_ctx != nullptr) g_model_path = params.model;
    }

    if (g_ctx == nullptr) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "failed to initialize whisper context (possibly OOM)";
        return jsonResult;
    }

    std::vector<float> pcmf32;
    if (!loadMonoPcmF32(params.audio, pcmf32)) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "failed to open or decode WAV file";
        return jsonResult;
    }

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_realtime = false;
    wparams.print_progress = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate = params.translate;
    wparams.language = params.language.c_str();
    wparams.n_threads = params.n_threads;
    wparams.split_on_word = params.split_on_word;
    wparams.vad = true;
    if (params.split_on_word) {
        wparams.max_len = 1;
        wparams.token_timestamps = true;
    }

    if (whisper_full(g_ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        jsonResult["@type"] = "error";
        jsonResult["message"] = "failed to process audio";
        return jsonResult;
    }

    const int n_segments = whisper_full_n_segments(g_ctx);
    std::vector<json> segmentsJson;
    std::string text_result;
    for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(g_ctx, i);
        text_result += std::string(text);
        if (!params.no_timestamps) {
            json segment;
            segment["from_ts"] = whisper_full_get_segment_t0(g_ctx, i);
            segment["to_ts"] = whisper_full_get_segment_t1(g_ctx, i);
            segment["text"] = text;
            segmentsJson.push_back(segment);
        }
    }
    if (!params.no_timestamps) jsonResult["segments"] = segmentsJson;
    jsonResult["text"] = text_result;
    return jsonResult;
}

extern "C" {
    char *request(char *body) {
        try {
            json jsonBody = json::parse(body);
            if (jsonBody["@type"] == "getTextFromWavFile") return jsonToChar(transcribe(jsonBody));
            if (jsonBody["@type"] == "getVersion") {
                return jsonToChar({{"@type", "version"}, {"message", "lib v1.8.3-accel"}});
            }
            return jsonToChar({{"@type", "error"}, {"message", "method not found"}});
        } catch (const std::exception &e) {
            return jsonToChar({{"@type", "error"}, {"message", e.what()}});
        }
    }

    void free_result(char *result) { delete[] result; }

    void release_model() {
        std::lock_guard<std::mutex> lock(g_mutex);
        releaseContextLocked();
    }
}
