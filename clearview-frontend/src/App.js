import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Camera,
  Upload,
  FileImage,
  Download,
  RefreshCw,
  Sliders,
  Sparkles,
  Droplets,
  Settings,
  Zap,
  Palette
} from 'lucide-react';
import './App.css';

const API_BASE_URL = 'http://localhost:5000';

export default function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('upload');
  const [detectedObjects, setDetectedObjects] = useState([]);

  // Algorithm / controls
  const [histogramMethod, setHistogramMethod] = useState('clahe');
  const [dehazeMethod, setDehazeMethod] = useState('fast');
  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
  const [saturation, setSaturation] = useState(0);
  const [dehazeStrength, setDehazeStrength] = useState(50);
  const [denoise, setDenoise] = useState(false);
  const [denoiseH, setDenoiseH] = useState(10);
  const [sharpness, setSharpness] = useState(0);
  const [gamma, setGamma] = useState(1.0);
  const [claheClip, setClaheClip] = useState(2.0);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const histogramMethods = [
    { value: 'none', label: 'None' },
    { value: 'standard', label: 'Standard HE' },
    { value: 'clahe', label: 'CLAHE' },
    { value: 'ahe', label: 'AHE' },
    { value: 'bbhe', label: 'BBHE' },
    { value: 'dsihe', label: 'DSIHE' },
    { value: 'rmshe', label: 'RMSHE' },
    { value: 'multi_he', label: 'Multi-HE' },
    { value: 'contrast_stretch', label: 'Contrast Stretch' },
    { value: 'ycbcr', label: 'YCbCr HE' },
  ];

  const dehazeMethods = [
    { value: 'none', label: 'None' },
    { value: 'fast', label: 'Fast DCP' },
    { value: 'dcp_full', label: 'DCP + Guided' },
    { value: 'cap', label: 'CAP' },
  ];

  // --- file / camera handlers ---
  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) return alert('Choose an image file');
    const reader = new FileReader();
    reader.onload = (ev) => {
      setOriginalImage(ev.target.result);
      setProcessedImage(null);
      setActiveTab('process');
    };
    reader.readAsDataURL(file);
  };

  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: 1280, height: 720 },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        streamRef.current = s;
        setActiveTab('camera');
      }
    } catch (err) {
      alert('Camera access denied or not available');
    }
  };

  const capturePhoto = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
    setOriginalImage(dataUrl);
    setProcessedImage(null);
    setActiveTab('process');

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  // --- processing (calls backend) ---
  const processImage = useCallback(async () => {
    if (!originalImage) return;
    setIsProcessing(true);
    setProcessedImage(null);
    setDetectedObjects([]);

    const params = {
      histogram_method: histogramMethod,
      dehaze_method: dehazeMethod,
      brightness,
      contrast,
      saturation,
      dehazeStrength: Math.max(0.3, Math.min(1.2, dehazeStrength / 100)),
      denoise,
      denoise_h: denoiseH,
      sharpness,
      gamma,
      clahe_clip: claheClip,
      detect: true,
    };

    try {
      const res = await fetch(`${API_BASE_URL}/api/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: originalImage, params }),
      });

      const data = await res.json();
      if (data.success) {
        // backend returns data.processed_image as data URL
        setProcessedImage(data.processed_image || (data.url ? `${API_BASE_URL}${data.url}` : null));

        // support different backend keys: detections / objects
        const detections = data.detections || data.objects || data.detected || [];
        if (Array.isArray(detections) && detections.length) {
          const mapped = detections.map((d) => ({
            name: d.name || d.label || d.class || 'Unknown',
            confidence: Number(d.confidence ?? d.score ?? d.conf ?? 0),
          }));
          setDetectedObjects(mapped);
        } else {
          setDetectedObjects([]);
        }
      } else {
        alert('Processing failed: ' + (data.error || 'Unknown'));
      }
    } catch (err) {
      console.error(err);
      alert('Error contacting backend. Is it running on port 5000?');
    } finally {
      setIsProcessing(false);
    }
  }, [
    originalImage,
    histogramMethod,
    dehazeMethod,
    brightness,
    contrast,
    saturation,
    dehazeStrength,
    denoise,
    denoiseH,
    sharpness,
    gamma,
    claheClip,
  ]);

  // auto-run when tab is process and image available
  useEffect(() => {
    if (activeTab === 'process' && originalImage) {
      const t = setTimeout(() => processImage(), 450);
      return () Timeout(t);
    }
  }, [activeTab, originalImage, processImage]);

  const downloadImage = () => {
    if (!processedImage) return;
    const a = document.createElement('a');
    a.href = processedImage;
    a.download = 'clearview-enhanced.jpg';
    a.click();
  };

  const resetAll = () => {
    setBrightness(0);
    setContrast(0);
    setSaturation(0);
    setDehazeStrength(50);
    setDenoise(false);
    setDenoiseH(10);
    setSharpness(0);
    setGamma(1.0);
    setClaheClip(2.0);
    setHistogramMethod('clahe');
    setDehazeMethod('fast');
    setProcessedImage(null);
    setDetectedObjects([]);
  };

  return (
    <div className="app-root">
      <div className="app-card">
        <header className="app-header">
          <div className="brand">
            <div className="logo-gradient"><Sparkles size={22} color="#fff" /></div>
            <div>
              <h1 className="title">ClearView Pro</h1>
              <div className="subtitle">Advanced image dehaze & enhancement</div>
            </div>
          </div>

          <div className="tab-buttons">
            <button className={activeTab === 'upload' ? 'tab active' : 'tab'} onClick={() => setActiveTab('upload')}>Upload</button>
            <button className={activeTab === 'camera' ? 'tab active' : 'tab'} onClick={() => setActiveTab('camera')}>Camera</button>
            <button className={activeTab === 'process' ? 'tab active' : 'tab'} onClick={() => setActiveTab('process')}>Process</button>
          </div>
        </header>

        <main className="app-main">
          {/* LEFT: upload/camera/process preview */}
          <section className="left-col">
            {/* Upload */}
            {activeTab === 'upload' && (
              <div className="card">
                <div className="center">
                  <div className="big-icon"><Upload size={44} color="#fff" /></div>
                  <h2>Upload a hazy image</h2>
                  <p className="muted">We will analyze and enhance using multiple methods.</p>
                  <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
                  <button className="primary" onClick={() => fileInputRef.current?.click()}><Upload /> Choose Image</button>
                </div>
              </div>
            )}

            {/* Camera */}
            {activeTab === 'camera' && (
              <div className="card">
                <h2>Capture Photo</h2>
                <div className="camera-box">
                  <video ref={videoRef} autoPlay playsInline className="camera-video" />
                  <canvas ref={canvasRef} className="hidden" />
                </div>
                <div className="row">
                  <button onClick={startCamera} className="ghost"><Camera /> Start</button>
                  <button onClick={capturePhoto} className="primary"><FileImage /> Capture</button>
                </div>
              </div>
            )}

            {/* Process */}
            {activeTab === 'process' && originalImage && (
              <div className="card">
                <div className="two-col">
                  <div>
                    <h3>Original</h3>
                    <div className="preview-box"><img src={originalImage} alt="original" className="preview-img" /></div>
                  </div>
                  <div>
                    <h3>Enhanced</h3>
                    <div className="preview-box">
                      {isProcessing ? (
                        <div className="processing">
                          <RefreshCw className="spin" size={44} />
                          <div>Processing…</div>
                        </div>
                      ) : processedImage ? (
                        <img src={processedImage} alt="processed" className="preview-img" />
                      ) : (
                        <div className="muted">Press Reprocess to call backend.</div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="controls-row">
                  <button onClick={processImage} className="primary" disabled={isProcessing}><RefreshCw /> Reprocess</button>
                  <button onClick={downloadImage} className="secondary" disabled={!processedImage}><Download /> Download</button>
                  <button onClick={resetAll} className="ghost">Reset</button>
                </div>
              </div>
            )}
          </section>

          {/* RIGHT: algorithm selectors & sliders */}
          <aside className="right-col">
            <div className="card">
              <h4><Settings /> Algorithm selection</h4>
              <label className="label">Histogram</label>
              <select value={histogramMethod} onChange={(e) => setHistogramMethod(e.target.value)} className="select">
                {histogramMethods.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>

              <label className="label">Dehaze</label>
              <select value={dehazeMethod} onChange={(e) => setDehazeMethod(e.target.value)} className="select">
                {dehazeMethods.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </select>
            </div>

            <div className="card">
              <h4><Sliders /> Advanced controls</h4>
              <label className="label">Dehaze strength: {dehazeStrength}%</label>
              <input type="range" min="0" max="100" value={dehazeStrength} onChange={(e) => setDehazeStrength(Number(e.target.value))} />

              <label className="label">Brightness: {brightness}</label>
              <input type="range" min="-50" max="50" value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} />

              <label className="label">Contrast: {contrast}</label>
              <input type="range" min="-50" max="50" value={contrast} onChange={(e) => setContrast(Number(e.target.value))} />

              <label className="label">Saturation: {saturation}</label>
              <input type="range" min="-50" max="50" value={saturation} onChange={(e) => setSaturation(Number(e.target.value))} />

              <label className="label">Sharpness: {sharpness.toFixed(1)}</label>
              <input type="range" min="0" max="2" step="0.1" value={sharpness} onChange={(e) => setSharpness(Number(e.target.value))} />

              <label className="label">Gamma: {gamma.toFixed(1)}</label>
              <input type="range" min="0.5" max="3" step="0.1" value={gamma} onChange={(e) => setGamma(Number(e.target.value))} />

              <label className="label">CLAHE clip: {claheClip.toFixed(1)}</label>
              <input type="range" min="0.5" max="4" step="0.1" value={claheClip} onChange={(e) => setClaheClip(Number(e.target.value))} />

              <label className="label">
                <input type="checkbox" checked={denoise} onChange={(e) => setDenoise(e.target.checked)} /> Enable denoise
              </label>
              {denoise && (
                <>
                  <label className="label">Denoise H: {denoiseH}</label>
                  <input type="range" min="0" max="30" value={denoiseH} onChange={(e) => setDenoiseH(Number(e.target.value))} />
                </>
              )}
            </div>

            {/* detections */}
            <div className="card">
              <h4><Sparkles /> AI detections</h4>
              {detectedObjects.length === 0 ? (
                <div className="muted">Run processing to see detected objects.</div>
              ) : (
                <div className="detections">
                  {detectedObjects.map((d, i) => (
                    <div key={i} className="detection">
                      <div>{d.name}</div>
                      <div className="bar"><div style={{ width: `${d.confidence * 100}%` }} /></div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>
        </main>
      </div>
    </div>
  );
}
