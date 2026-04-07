// TODO Implement this library.import 'package:flutter/material.dart';
import 'package:flutter/material.dart';

import 'features/camera/camera_screen.dart';

class ClearviewApp extends StatelessWidget {
  const ClearviewApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Clearview',
      debugShowCheckedModeBanner: false,

      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.light,
        ),
      ),

      darkTheme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.blue,
          brightness: Brightness.dark,
        ),
      ),

      themeMode: ThemeMode.system,

      home: const CameraScreen(),
    );
  }
}