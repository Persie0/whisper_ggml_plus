import 'package:path_provider/path_provider.dart';
import 'package:universal_io/io.dart';
import 'package:whisper_ggml_plus/src/models/whisper_model.dart';

import 'models/whisper_result.dart';
import 'whisper.dart';

class WhisperController {
  String _modelPath = '';
  String? _dir;

  Future<void> initModel(WhisperModel model) async {
    _dir ??= await getModelDir();
    _modelPath = '$_dir/ggml-${model.modelName}.bin';
  }

  Future<TranscribeResult?> transcribe({
    required WhisperModel model,
    required String audioPath,
    String lang = 'en',
    bool diarize = false,
  }) async {
    await initModel(model);

    final Whisper whisper = Whisper(model: model);
    final DateTime start = DateTime.now();
    const bool translate = false;
    const bool withSegments = false;
    const bool splitWords = false;

    try {
      final transcription = await whisper.transcribe(
        transcribeRequest: TranscribeRequest(
          audio: audioPath,
          language: lang,
          isTranslate: translate,
          isNoTimestamps: !withSegments,
          splitOnWord: splitWords,
          isRealtime: true,
          diarize: diarize,
        ),
        modelPath: _modelPath,
      );

      final Duration transcriptionDuration = DateTime.now().difference(start);
      return TranscribeResult(
        time: transcriptionDuration,
        transcription: transcription,
      );
    } catch (_) {
      return null;
    } finally {
      // The model can be hundreds of MB. Do not retain the native global context
      // after a transcription; the Android 17 memory metric counts that RSS even
      // while the UI is idle/backgrounded.
      whisper.releaseModel();
    }
  }

  static Future<String> getModelDir() async {
    final Directory libraryDirectory = Platform.isAndroid
        ? await getApplicationSupportDirectory()
        : await getLibraryDirectory();
    return libraryDirectory.path;
  }

  Future<String> getPath(WhisperModel model) async {
    _dir ??= await getModelDir();
    return '$_dir/ggml-${model.modelName}.bin';
  }

  /// Downloads a model directly to disk. The old implementation consolidated
  /// the complete response into a Uint8List first, temporarily duplicating a
  /// 100MB+ model in Dart heap.
  Future<String> downloadModel(WhisperModel model) async {
    final path = await getPath(model);
    final file = File(path);
    if (await file.exists()) return path;

    final client = HttpClient();
    final partial = File('$path.part');
    try {
      final request = await client.getUrl(model.modelUri);
      final response = await request.close();
      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw HttpException(
          'Model download failed with HTTP ${response.statusCode}',
          uri: model.modelUri,
        );
      }
      final sink = partial.openWrite();
      try {
        await response.pipe(sink);
      } catch (_) {
        await sink.close();
        rethrow;
      }
      if (await file.exists()) await file.delete();
      await partial.rename(path);
      return path;
    } finally {
      client.close(force: true);
      if (await partial.exists() && !await file.exists()) {
        try {
          await partial.delete();
        } catch (_) {}
      }
    }
  }
}
