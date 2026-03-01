# ⚡ ESP32 Power Monitor — Web Dashboard

Real-time power monitoring dashboard with **ONNX ML Runtime** intelligence, built for Vercel deployment.

## Features

- **Live Gauges** — Voltage (Vrms), Current (A), Power (W) with radial gauges
- **Waveform Charts** — Real-time voltage, current, and power waveforms via Chart.js
- **Energy Auditor** — Cumulative kWh tracking with cost estimation
- **ONNX Runtime Monitor** — Tracks ML inference latency (P50/P95/P99), throughput, and per-model breakdown
- **AI Insights** — System health scoring, anomaly detection, fault classification, and smart recommendations
- **Reports** — Download CSV reports with peak analysis
- **Dark Theme** — Premium glassmorphism UI with AMD Ryzen™ branding

## Deploy to Vercel

### Option 1: Vercel CLI
```bash
cd web-dashboard
npx vercel --prod
```

### Option 2: Git Integration
1. Push this folder to a GitHub repo
2. Import in [vercel.com/new](https://vercel.com/new)
3. Set **Root Directory** to `web-dashboard`
4. Deploy!

### Option 3: Drag & Drop
1. Go to [vercel.com](https://vercel.com)
2. Drag the `public/` folder into the deploy area

## Local Development

```bash
cd web-dashboard
npx serve public -l 3000
```

Then open [http://localhost:3000](http://localhost:3000)

## Project Structure

```
web-dashboard/
├── public/
│   ├── index.html    # Main dashboard page
│   ├── style.css     # Complete dark theme styles
│   └── app.js        # Data engine + charts + ONNX monitor + AI
├── vercel.json       # Vercel deployment config
├── package.json      # Project metadata
└── README.md         # This file
```

## Architecture

The dashboard runs entirely client-side with a built-in **SensorDataEngine** that generates high-fidelity AC power line data matching real ESP32 PZEM-004T sensor readings. The ONNX inference simulation mirrors the Python backend's performance characteristics.

When connected to a real ESP32, the `updateFromESP32(v, c, p)` function can be called directly from a WebSocket, HTTP polling, or Web Serial API integration.

---

**Processed on AMD Ryzen™** · **ONNX Runtime v1.23** · **Chart.js v4.4**
