import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';
import 'package:flutter/foundation.dart';
import 'package:universal_io/io.dart';
import 'package:whisper_ggml_plus/src/models/whisper_model.dart';
import 'package:whisper_ggml_plus/src/whisper_audio_convert.dart';

import 'models/requests/transcribe_request.dart';
import 'models/requests/transcribe_request_dto.dart';
import 'models/requests/version_request.dart';
import 'models/responses/whisper_transcribe_response.dart';
import 'models/responses/whisper_version_response.dart';
import 'models/whisper_dto.dart';

export 'models/_models.dart';
export 'whisper_audio_convert.dart';

typedef WReqNative = Pointer<Utf8> Function(Pointer<Utf8> body);
typedef WFreeResultNative = Void Function(Pointer<Utf8> result);
typedef WFreeResultDart = void Function(Pointer<Utf8> result);
typedef WReleaseModelNative = Void Function();
typedef WReleaseModelDart = void Function();

class Whisper {
  const Whisper({required this.model, this.modelDir});

  final WhisperModel model;
  final String? modelDir;

  DynamicLibrary _openLib() {
    if (Platform.isAndroid) {
      return DynamicLibrary.open('libwhisper.so');
    }
    return DynamicLibrary.process();
  }

  Future<Map<String, dynamic>> _request({
    required WhisperRequestDto whisperRequest,
  }) async {
    return Isolate.run(() {
      final lib = _openLib();
      final request = lib.lookupFunction<WReqNative, WReqNative>('request');
      final freeResult = lib.lookupFunction<WFreeResultNative, WFreeResultDart>(
        'free_result',
      );
      final data = whisperRequest.toRequestString().toNativeUtf8();
      Pointer<Utf8> resultPointer = nullptr;
      try {
        resultPointer = request(data);
        if (resultPointer == nullptr) {
          throw StateError('Native Whisper request returned a null result');
        }
        final decoded = json.decode(resultPointer.toDartString());
        return decoded as Map<String, dynamic>;
      } finally {
        malloc.free(data);
        if (resultPointer != nullptr) freeResult(resultPointer);
      }
    });
  }

  /// Explicitly releases the native model/context. The native implementation is
  /// process-global, so this releases the model loaded by any Whisper instance.
  void releaseModel() {
    final lib = _openLib();
    lib.lookupFunction<WReleaseModelNative, WReleaseModelDart>(
      'release_model',
    )();
  }

  Future<WhisperTranscribeResponse> transcribe({
    required TranscribeRequest transcribeRequest,
    required String modelPath,
  }) async {
    File? convertedFile;
    try {
      final input = File(transcribeRequest.audio);
      final alreadyWav = transcribeRequest.audio.toLowerCase().endsWith('.wav');
      if (!alreadyWav) {
        final converter = WhisperAudioConvert(
          audioInput: input,
          audioOutput: File('${transcribeRequest.audio}.whisper.wav'),
        );
        convertedFile = await converter.convert();
      }

      final req = transcribeRequest.copyWith(
        audio: convertedFile?.path ?? transcribeRequest.audio,
      );
      final result = await _request(
        whisperRequest: TranscribeRequestDto.fromTranscribeRequest(req, modelPath),
      );

      if (result['text'] == null) {
        throw Exception(result['message']);
      }
      return WhisperTranscribeResponse.fromJson(result);
    } catch (e) {
      debugPrint(e.toString());
      rethrow;
    } finally {
      final temporary = convertedFile;
      if (temporary != null) {
        try {
          if (await temporary.exists()) await temporary.delete();
        } catch (_) {}
      }
    }
  }

  Future<String?> getVersion() async {
    final Map<String, dynamic> result = await _request(
      whisperRequest: const VersionRequest(),
    );
    final WhisperVersionResponse response = WhisperVersionResponse.fromJson(
      result,
    );
    return response.message;
  }
}
