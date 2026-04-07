import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({
    super.key,
    required this.processedJson,
    required this.originalBytes,
  });

  final Map<String, dynamic> processedJson;
  final Uint8List originalBytes;

  @override
  Widget build(BuildContext context) {
    final processedBase64 = processedJson['image_base64'] as String;
    final labels = (processedJson['labels'] as List<dynamic>?) ?? [];
    final text = (processedJson['recognized_text'] as String?) ?? '';
    final processedBytes = base64Decode(processedBase64);

    return Scaffold(
      appBar: AppBar(title: const Text('Processed Result')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          const Text('Preview', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          ClipRRect(
            borderRadius: BorderRadius.circular(16),
            child: Image.memory(processedBytes, fit: BoxFit.cover),
          ),
          const SizedBox(height: 16),
          Text('Detected objects: ${labels.length}'),
          const SizedBox(height: 8),
          if (text.trim().isNotEmpty) ...[
            const Text('Recognized text', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 6),
            Text(text),
          ],
          const SizedBox(height: 16),
          const Text('You can add sliders here for live preview edits.'),
        ],
      ),
    );
  }
}