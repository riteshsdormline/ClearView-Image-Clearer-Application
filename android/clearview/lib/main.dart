import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'firebase_options.dart';

/// ClearView Flutter App
///
/// Flow:
/// Splash -> Google Login -> Home -> Gallery/Camera -> Backend processing ->
/// Editing screen with sliders -> Report screen -> Download options
///
/// Replace [backendBaseUrl] with your Python backend URL.
/// Example local dev on same Wi-Fi: http://192.168.1.10:8000

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
);
  await AppSession.init();
  runApp(const ClearViewApp());
}

const String backendBaseUrl = 'http://10.0.2.2:8000';

class ClearViewApp extends StatelessWidget {
  const ClearViewApp({super.key});

  @override
  Widget build(BuildContext context) {
    const purple = Color(0xFF7E57C2);
    const deepPurple = Color(0xFF5E35B1);
    const bg = Color(0xFFFDFBFF);

    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'ClearView',
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: bg,
        colorScheme: ColorScheme.fromSeed(
          seedColor: purple,
          primary: purple,
          secondary: deepPurple,
          surface: Colors.white,
        ),
        appBarTheme: const AppBarTheme(
          centerTitle: true,
          backgroundColor: Colors.transparent,
          foregroundColor: Colors.black87,
          elevation: 0,
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(18),
            borderSide: BorderSide.none,
          ),
        ),
      ),
      home: const SplashScreen(),
    );
  }
}

class AppSession {
  static final GoogleSignIn googleSignIn = GoogleSignIn.instance;
  static GoogleSignInAccount? user;

  static Future<void> init() async {
    await googleSignIn.initialize();
  }
}

class ProcessedImageData {
  final Uint8List enhancedBytes;
  final String? enhancedUrl;
  final String? editedUrl;
  final String description;
  final String location;
  final String locationHint;
  final List<dynamic> objects;

  ProcessedImageData({
    required this.enhancedBytes,
    this.enhancedUrl,
    this.editedUrl,
    required this.description,
    required this.location,
    required this.locationHint,
    required this.objects,
  });
}

class EditValues {
  double brightness = 1.0;
  double saturation = 1.0;
  double warmth = 1.0;
  double gamma = 1.0;
  double sharpness = 1.0;
}

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _goNext();
  }

  Future<void> _goNext() async {
    await Future.delayed(const Duration(seconds: 2));
    final GoogleSignInAccount? current =
    await AppSession.googleSignIn.attemptLightweightAuthentication();
    if (current != null) {
      AppSession.user = current;
      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const HomeScreen()),
      );
      return;
    }

    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(builder: (_) => const LoginScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFFFDFBFF), Color(0xFFF1E8FF), Color(0xFFEADCFD)],
          ),
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(22),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.75),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.auto_fix_high_rounded, size: 64, color: Color(0xFF7E57C2)),
              ),
              const SizedBox(height: 20),
              const Text(
                'ClearView',
                style: TextStyle(fontSize: 30, fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              const Text('Image clear, edit, and report in one flow'),
              const SizedBox(height: 26),
              const SizedBox(
                width: 180,
                child: LinearProgressIndicator(minHeight: 6),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  bool _loading = false;

  Future<void> _signIn() async {
    setState(() => _loading = true);
    try {
      final account = await AppSession.googleSignIn.authenticate();

      AppSession.user = account;

      if (!mounted) return;
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (_) => const HomeScreen()),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Login failed: $e')),
      );
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFFFDFBFF), Color(0xFFF7F0FF)],
          ),
        ),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Card(
              elevation: 10,
              shadowColor: Colors.purple.withOpacity(0.12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(28)),
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.person_rounded, size: 62, color: Color(0xFF7E57C2)),
                    const SizedBox(height: 16),
                    const Text(
                      'Welcome to ClearView',
                      style: TextStyle(fontSize: 22, fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'Sign in to continue',
                      style: TextStyle(color: Colors.black54),
                    ),
                    const SizedBox(height: 20),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: _loading ? null : _signIn,
                        icon: _loading
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(strokeWidth: 2),
                              )
                            : const Icon(Icons.g_mobiledata_rounded),
                        label: Text(_loading ? 'Signing in...' : 'Continue with Google'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF7E57C2),
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(vertical: 14),
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  Future<void> _chooseGallery(BuildContext context) async {
    final picker = ImagePicker();
    final xfile = await picker.pickImage(source: ImageSource.gallery);
    if (xfile == null) return;
    if (!context.mounted) return;
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ProcessingScreen(imageFile: File(xfile.path), sourceName: 'Gallery'),
      ),
    );
  }

  Future<void> _openCamera(BuildContext context) async {
    final picker = ImagePicker();
    final xfile = await picker.pickImage(source: ImageSource.camera);
    if (xfile == null) return;
    if (!context.mounted) return;
    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ProcessingScreen(imageFile: File(xfile.path), sourceName: 'Camera'),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final user = AppSession.user;
    return Scaffold(
      appBar: AppBar(
        title: const Text('ClearView'),
        actions: [
          if (user != null)
            Padding(
              padding: const EdgeInsets.only(right: 14),
              child: Center(
                child: Text(
                  user.displayName ?? 'User',
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              ),
            ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const SizedBox(height: 8),
            const Text(
              'Choose image source',
              style: TextStyle(fontSize: 26, fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 8),
            const Text(
              'Select from gallery or capture with camera, then process in your Python backend.',
              style: TextStyle(color: Colors.black54),
            ),
            const SizedBox(height: 24),
            _HomeOptionCard(
              icon: Icons.photo_library_rounded,
              title: 'From Gallery',
              subtitle: 'Pick an existing image',
              onTap: () => _chooseGallery(context),
            ),
            const SizedBox(height: 16),
            _HomeOptionCard(
              icon: Icons.camera_alt_rounded,
              title: 'Camera',
              subtitle: 'Capture a new image',
              onTap: () => _openCamera(context),
            ),
          ],
        ),
      ),
    );
  }
}

class _HomeOptionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final VoidCallback onTap;

  const _HomeOptionCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(24),
      child: Card(
        elevation: 6,
        shadowColor: Colors.purple.withOpacity(0.08),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        child: Padding(
          padding: const EdgeInsets.all(18),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFFF1E8FF),
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Icon(icon, color: const Color(0xFF7E57C2), size: 30),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700)),
                    const SizedBox(height: 4),
                    Text(subtitle, style: const TextStyle(color: Colors.black54)),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right_rounded),
            ],
          ),
        ),
      ),
    );
  }
}

class ProcessingScreen extends StatefulWidget {
  final File imageFile;
  final String sourceName;

  const ProcessingScreen({super.key, required this.imageFile, required this.sourceName});

  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  bool _loading = true;
  String _status = 'Processing image...';
  ProcessedImageData? _data;
  File? _enhancedFile;

  @override
  void initState() {
    super.initState();
    _process();
  }

  Future<void> _process() async {
    try {
      final uri = Uri.parse('$backendBaseUrl/pipeline');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', widget.imageFile.path));

      final response = await request.send();
      final body = await response.stream.bytesToString();

      if (response.statusCode != 200) {
        throw Exception(body);
      }

      final decoded = jsonDecode(body) as Map<String, dynamic>;
      final enhancedUrl = decoded['enhanced_download_url'] as String?;
      final editedUrl = decoded['edited_download_url'] as String?;
      final report = decoded['report'] as Map<String, dynamic>? ?? {};
      final objects = (decoded['objects'] as List?) ?? [];

      Uint8List enhancedBytes = await _downloadBytes('$backendBaseUrl$enhancedUrl');
      final tempDir = await getTemporaryDirectory();
      final file = File('${tempDir.path}/enhanced_${DateTime.now().millisecondsSinceEpoch}.jpg');
      await file.writeAsBytes(enhancedBytes);

      if (!mounted) return;
      setState(() {
        _data = ProcessedImageData(
          enhancedBytes: enhancedBytes,
          enhancedUrl: enhancedUrl,
          editedUrl: editedUrl,
          description: (report['description'] ?? '') as String,
          location: (report['location'] ?? 'unknown location') as String,
          locationHint: (report['location_hint'] ?? '') as String,
          objects: objects,
        );
        _enhancedFile = file;
        _loading = false;
        _status = 'Ready';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _status = 'Failed: $e';
      });
    }
  }

  Future<Uint8List> _downloadBytes(String url) async {
    final res = await http.get(Uri.parse(url));
    if (res.statusCode != 200) throw Exception('Failed to fetch image');
    return res.bodyBytes;
  }

  @override
  Widget build(BuildContext context) {
    final data = _data;

    return Scaffold(
      appBar: AppBar(title: const Text('Processing')),
      body: _loading
          ? Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const CircularProgressIndicator(),
                  const SizedBox(height: 18),
                  Text(_status, style: const TextStyle(color: Colors.black54)),
                ],
              ),
            )
          : data == null
              ? Center(child: Text(_status))
              : EditScreen(
                  originalFile: widget.imageFile,
                  enhancedBytes: data.enhancedBytes,
                  processedData: data,
                ),
    );
  }
}

class EditScreen extends StatefulWidget {
  final File originalFile;
  final Uint8List enhancedBytes;
  final ProcessedImageData processedData;

  const EditScreen({
    super.key,
    required this.originalFile,
    required this.enhancedBytes,
    required this.processedData,
  });

  @override
  State<EditScreen> createState() => _EditScreenState();
}

class _EditScreenState extends State<EditScreen> {
  final EditValues values = EditValues();
  bool _updating = false;
  Uint8List? _editedBytes;
  String? _editedRemoteUrl;

  @override
  void initState() {
    super.initState();
    _editedBytes = widget.enhancedBytes;
  }

  Future<void> _applyEdits() async {
    setState(() => _updating = true);
    try {
      final uri = Uri.parse('$backendBaseUrl/edit');
      final request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', widget.originalFile.path));
      request.fields['brightness'] = values.brightness.toStringAsFixed(2);
      request.fields['saturation'] = values.saturation.toStringAsFixed(2);
      request.fields['warmth'] = values.warmth.toStringAsFixed(2);
      request.fields['gamma'] = values.gamma.toStringAsFixed(2);
      request.fields['sharpness'] = values.sharpness.toStringAsFixed(2);
      request.fields['denoise'] = 'true';
      request.fields['blur'] = 'false';
      request.fields['detail'] = 'true';
      request.fields['upscale'] = 'false';

      final response = await request.send();
      final body = await response.stream.bytesToString();
      if (response.statusCode != 200) {
        throw Exception(body);
      }

      final decoded = jsonDecode(body) as Map<String, dynamic>;
      final remoteUrl = decoded['download_url'] as String?;
      final bytes = remoteUrl == null ? widget.enhancedBytes : await _downloadBytes('$backendBaseUrl$remoteUrl');

      if (!mounted) return;
      setState(() {
        _editedBytes = bytes;
        _editedRemoteUrl = remoteUrl;
        _updating = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _updating = false);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Edit failed: $e')));
    }
  }

  Future<Uint8List> _downloadBytes(String url) async {
    final res = await http.get(Uri.parse(url));
    if (res.statusCode != 200) throw Exception('Failed to fetch edited image');
    return res.bodyBytes;
  }

  Future<void> _downloadImage() async {
    final choice = await showDialog<String>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Choose download quality'),
          content: const Text('Pick standard or high quality.'),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context, 'standard'),
              child: const Text('Standard'),
            ),
            ElevatedButton(
              onPressed: () => Navigator.pop(context, 'high'),
              child: const Text('High Quality'),
            ),
          ],
        );
      },
    );

    if (choice == null) return;

    final bytes = _editedBytes ?? widget.enhancedBytes;
    final fileName = choice == 'high' ? 'clearview_high_${DateTime.now().millisecondsSinceEpoch}.jpg' : 'clearview_std_${DateTime.now().millisecondsSinceEpoch}.jpg';

    // Android 10+ uses scoped storage. This saves into app-specific external storage.
    // You can later move to a public Downloads implementation using SAF or a plugin.
    final directory = await getExternalStorageDirectory();
    if (directory == null) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Storage not available')));
      return;
    }

    final saveDir = Directory('${directory.path}/ClearViewDownloads');
    if (!await saveDir.exists()) {
      await saveDir.create(recursive: true);
    }

    final file = File('${saveDir.path}/$fileName');
    await file.writeAsBytes(bytes);

    if (!mounted) return;
    showDialog(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Saved'),
        content: Text('Image saved to:\n${file.path}'),
        actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final data = widget.processedData;
    final objects = data.objects;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Editing'),
        actions: [
          IconButton(
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => ReportScreen(processedData: data),
                ),
              );
            },
            icon: const Icon(Icons.description_rounded),
            tooltip: 'Report',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(22),
              child: Image.memory(
                _editedBytes ?? widget.enhancedBytes,
                fit: BoxFit.cover,
              ),
            ),
            const SizedBox(height: 14),
            Card(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Generated description', style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 8),
                    Text(data.description.isEmpty ? 'No description yet' : data.description),
                    const SizedBox(height: 10),
                    Text('Location: ${data.location}'),
                    const SizedBox(height: 6),
                    Text(
                      data.locationHint,
                      style: const TextStyle(color: Colors.black54),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 14),
            Card(
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Detected objects', style: Theme.of(context).textTheme.titleMedium),
                    const SizedBox(height: 10),
                    if (objects.isEmpty)
                      const Text('No objects detected yet')
                    else
                      ...objects.take(8).map((o) {
                        return Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Container(
                            width: double.infinity,
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF8F2FF),
                              borderRadius: BorderRadius.circular(16),
                            ),
                            child: Text(
                              '${o['label']}  •  confidence: ${(o['confidence'] ?? 0).toString()}  •  priority: ${(o['priority'] ?? 0).toString()}',
                            ),
                          ),
                        );
                      }),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 14),
            _SliderBlock(
              title: 'Brightness',
              value: values.brightness,
              min: 0.5,
              max: 1.8,
              onChanged: (v) => setState(() => values.brightness = v),
            ),
            _SliderBlock(
              title: 'Saturation',
              value: values.saturation,
              min: 0.5,
              max: 1.8,
              onChanged: (v) => setState(() => values.saturation = v),
            ),
            _SliderBlock(
              title: 'Warmth',
              value: values.warmth,
              min: 0.7,
              max: 1.6,
              onChanged: (v) => setState(() => values.warmth = v),
            ),
            _SliderBlock(
              title: 'Gamma',
              value: values.gamma,
              min: 0.6,
              max: 1.8,
              onChanged: (v) => setState(() => values.gamma = v),
            ),
            _SliderBlock(
              title: 'Detail',
              value: values.sharpness,
              min: 0.6,
              max: 2.0,
              onChanged: (v) => setState(() => values.sharpness = v),
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _updating ? null : _applyEdits,
                    icon: _updating
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.tune_rounded),
                    label: Text(_updating ? 'Applying...' : 'Apply edits'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF7E57C2),
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _downloadImage,
                    icon: const Icon(Icons.download_rounded),
                    label: const Text('Download'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _SliderBlock extends StatelessWidget {
  final String title;
  final double value;
  final double min;
  final double max;
  final ValueChanged<double> onChanged;

  const _SliderBlock({
    required this.title,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('$title: ${value.toStringAsFixed(2)}', style: const TextStyle(fontWeight: FontWeight.w600)),
            Slider(value: value, min: min, max: max, onChanged: onChanged),
          ],
        ),
      ),
    );
  }
}

class ReportScreen extends StatelessWidget {
  final ProcessedImageData processedData;

  const ReportScreen({super.key, required this.processedData});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Report')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Card(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('Image report', style: Theme.of(context).textTheme.headlineSmall),
                const SizedBox(height: 14),
                Text(
                  processedData.description,
                  style: const TextStyle(fontSize: 16, height: 1.4),
                ),
                const SizedBox(height: 18),
                Text('Predicted location: ${processedData.location}', style: const TextStyle(fontWeight: FontWeight.w700)),
                const SizedBox(height: 8),
                Text(processedData.locationHint, style: const TextStyle(color: Colors.black54)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
