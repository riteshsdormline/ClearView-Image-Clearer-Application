class ProcessedImage {
  final String imageBase64;
  final List<dynamic> labels;
  final String recognizedText;
  final int width;
  final int height;

  ProcessedImage({
    required this.imageBase64,
    required this.labels,
    required this.recognizedText,
    required this.width,
    required this.height,
  });

  factory ProcessedImage.fromJson(Map<String, dynamic> json) {
    final size = json['size'] as Map<String, dynamic>;
    return ProcessedImage(
      imageBase64: json['image_base64'] as String,
      labels: (json['labels'] as List<dynamic>?) ?? const [],
      recognizedText: (json['recognized_text'] as String?) ?? '',
      width: (size['width'] as num).toInt(),
      height: (size['height'] as num).toInt(),
    );
  }
}