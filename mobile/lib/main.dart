import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Stock Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'NSE 20 Stock Predictor'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final symbolController = TextEditingController();
  String prediction = "";

  @override
  void dispose() {
    symbolController.dispose();
    super.dispose();
  }

  Future<void> _predictStock() async {
    final symbol = symbolController.text.trim().toUpperCase();

    if (symbol.isEmpty) {
      setState(() {
        prediction = "Please enter a stock symbol.";
      });
      return;
    }

    final result = await getPrediction(symbol);

    setState(() {
      prediction = result;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            TextField(
              controller: symbolController,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Stock Symbol (e.g., EGAD)',
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _predictStock,
              child: const Text("Predict Buy/Sell/Hold"),
            ),
            const SizedBox(height: 20),
            Text(
              "Prediction: $prediction",
              style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}

// ----------------- API CALL -----------------
Future<String> getPrediction(String symbol) async {
  final url = Uri.parse("http://192.168.1.181:8001/predict");
  final response = await http.post(
    url,
    headers: {"Content-Type": "application/json"},
    body: jsonEncode({"symbol": symbol}),
  );

  if (response.statusCode == 200) {
    final data = jsonDecode(response.body);
    return data["prediction"];
  } else {
    return "Error: ${response.statusCode}";
  }
}
