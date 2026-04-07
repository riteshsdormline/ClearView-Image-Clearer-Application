import 'package:flutter/material.dart';
import 'app.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Future: initialize services here
  // e.g. camera, shared prefs, firebase, etc.

  runApp(const ClearviewApp());
}