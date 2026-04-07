import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;

class ApiClient {
  ApiClient({required this.baseUrl});
  final String baseUrl;

  Future<Map<String, dynamic>> processImage(
    Uint8List bytes, {
    required Map<String, dynamic> params,
  }) async {
    final uri = Uri.parse('$baseUrl/process');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      http.MultipartFile.fromBytes('file', bytes, filename: 'image.jpg'),
    );
    request.fields['params'] = jsonEncode(params);

    final response = await request.send();
    final body = await response.stream.bytesToString();

    if (response.statusCode != 200) {
      throw Exception('Server error: ${response.statusCode} $body');
    }

    return jsonDecode(body) as Map<String, dynamic>;
  }
}