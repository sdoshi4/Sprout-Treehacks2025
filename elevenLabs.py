# Automatic FlutterFlow imports
import '/flutter_flow/flutter_flow_theme.dart';
import '/flutter_flow/flutter_flow_util.dart';
import '/custom_code/actions/index.dart'; // Imports other custom actions
import '/flutter_flow/custom_functions.dart'; // Imports custom functions
import 'package:flutter/material.dart';
# // Begin custom action code
# // DO NOT REMOVE OR MODIFY THE CODE ABOVE!

# // Set your action name, define your arguments and return parameter,
# // and then add the boilerplate code using the green button on the right!

# // import '/backend/schema/structs/index.dart';

import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:just_audio/just_audio.dart';

Future<bool> playAudioFromElevenLabs(
  String text,
  String voiceId,
  String apiKey,
  double stability,
  double similarityBoost,
) async {
  final player = AudioPlayer();

  try {
    if (text.isEmpty || voiceId.isEmpty || apiKey.isEmpty) {
      return false;
    }

    final url =
        Uri.parse('https://api.elevenlabs.io/v1/text-to-speech/$voiceId');

    final response = await http.post(
      url,
      headers: {
        'xi-api-key': apiKey,
        'Content-Type': 'application/json',
        'Accept': 'audio/mpeg',
      },
      body: jsonEncode({
        'text': text,
        'voice_settings': {
          'stability': stability,
          'similarity_boost': similarityBoost
        }
      }),
    );

    if (response.statusCode == 200) {
      final audioSource = AudioSource.uri(
        Uri.dataFromBytes(
          response.bodyBytes,
          mimeType: 'audio/mpeg',
        ),
      );
      await player.setAudioSource(audioSource);
      await player.play();
      return true;
    } else {
      print('Error: ${response.body}');
      return false;
    }
  } catch (e) {
    print('Error: $e');
    return false;
  }
}