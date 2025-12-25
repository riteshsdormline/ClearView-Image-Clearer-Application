// src/App.js
// src/App.js (top portion)
import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  Camera,
  FileImage,
  Download,
  RefreshCw,
  Sliders,
  Sparkles,
} from "lucide-react";

/* helper convert file->dataURL */
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}


function dataURLToFile(dataurl, filename = "capture.jpg") {
  const arr = dataurl.split(",");
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);
  while (n--) u8arr[n] = bstr.charCodeAt(n);
  return new File([u8arr], filename, { type: mime });
}

export default function App() {
  const [originalImage, setOriginalImage] = useState(null); // dataURL
  const [processedImage, setProcessedImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeTab, setActiveTab] = useState("upload");
  const [detectedObjects, setDetectedObjects] = useState([]);

  const [brightness, setBrightness] = useState(0);
  const [contrast, setContrast] = useState(0);
  const [saturation, setSaturation] = useState(0);
  const [dehazeStrength, setDehazeStrength] = useState(100);
  const [sharpness, setSharpness] = useState(0);

  const fileRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // handle file pick
  const onChoose = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const dataurl = await fileToDataURL(f);
    setOriginalImage(dataurl);
    setProcessedImage(null);
    setDetectedObjects([]);
    setActiveTab("process");
  };

  // start camera
  const startCamera = async () => {
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 1280, height: 720 } });
      videoRef.current.srcObject = s;
      streamRef.current = s;
      setActiveTab("camera");
    } catch (err) {
      alert("Camera access denied or not available");
    }
  };

  const capture = () => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c) return;
    c.width = v.videoWidth || 1280;
    c.height = v.videoHeight || 720;
    const ctx = c.getContext("2d");
    ctx.drawImage(v, 0, 0, c.width, c.height);
    const dataurl = c.toDataURL("image/jpeg", 0.95);
    setOriginalImage(dataurl);
    setProcessedImage(null);
    setDetectedObjects([]);
    setActiveTab("process");
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
  };

  // Core: call backend /api/enhance
  const process = async () => {
    if (!originalImage) {
      alert("Choose or capture an image first.");
      return;
    }
    setIsProcessing(true);
    setProgress(0);
    setProcessedImage(null);
    setDetectedObjects([]);

    try {
      // Build request body (we will send JSON with base64)
      const payload = {
        image: originalImage,
        params: {
          dehaze: true,
          dehazeStrength: Math.max(0.3, Math.min(1.5, dehazeStrength / 100)),
          histogram: true,
          brightness: brightness,
          contrast: contrast,
          saturation: saturation,
          sharpness: sharpness,
          detect: true
        }
      };

      // Send
      const resp = await fetch("http://localhost:5000/api/enhance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error("Backend error: " + txt);
      }

      const data = await resp.json();
      if (data.success) {
        setProcessedImage(data.processed_image);
        setDetectedObjects(data.detections || []);
      } else {
        alert("Processing failed: " + (data.error || "unknown"));
      }
    } catch (err) {
      console.error(err);
      alert("Processing error: " + err.message);
    } finally {
      setIsProcessing(false);
      setProgress(100);
      setTimeout(() => setProgress(0), 400);
    }
  };

  useEffect(() => {
    // Auto-process when switching to process tab and there is an image
    if (activeTab === "process" && originalImage && !processedImage) {
      const id = setTimeout(() => process(), 350);
      return () => clearTimeout(id);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTab, originalImage]);

  const download = () => {
    if (!processedImage) return;
    const a = document.createElement("a");
    a.href = processedImage;
    a.download = "clearview-enhanced.jpg";
    a.click();
  };

  return (
    <div style={{ minHeight: "100vh", background: "#0f172a", color: "#e6eef8", padding: 20, fontFamily: "Inter, Arial, sans-serif" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", flexDirection: "column", gap: 18 }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <div style={{ width: 44, height: 44, borderRadius: 10, background: "linear-gradient(90deg,#06b6d4,#8b5cf6)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Sparkles color="#fff" size={20} />
            </div>
            <div>
              <h1 style={{ margin: 0 }}>ClearView</h1>
              <div style={{ fontSize: 12, opacity: 0.8 }}>AI-powered haze removal & enhancement</div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            <button onClick={() => { setActiveTab("upload"); }} style={{ background: activeTab === "upload" ? "#0369a1" : "#0b1220", color: "#fff", padding: "8px 12px", borderRadius: 999, border: "none" }}>
              <Upload size={14} /> Upload
            </button>
            <button onClick={() => { setActiveTab("camera"); }} style={{ background: activeTab === "camera" ? "#0369a1" : "#0b1220", color: "#fff", padding: "8px 12px", borderRadius: 999, border: "none" }}>
              <Camera size={14} /> Camera
            </button>
            <button onClick={() => { setActiveTab("process"); }} style={{ background: activeTab === "process" ? "#0369a1" : "#0b1220", color: "#fff", padding: "8px 12px", borderRadius: 999, border: "none" }}>
              <RefreshCw size={14} /> Process
            </button>
          </div>
        </header>

        <main style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 18 }}>
          {/* LEFT: images */}
          <section style={{ background: "#071025", borderRadius: 12, padding: 14 }}>
            {/* uploader / camera */}
            {activeTab === "upload" && (
              <div>
                <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={onChoose} />
                <div style={{ marginBottom: 12 }}>
                  <h3 style={{ margin: 0, fontSize: 14 }}>Select a hazy image</h3>
                  <div style={{ fontSize: 12, opacity: 0.8 }}>We will send it to the server for dehazing + enhancement.</div>
                </div>
                <button onClick={() => fileRef.current?.click()} style={{ background: "#0369a1", color: "#fff", padding: "10px 14px", borderRadius: 8, border: "none" }}>
                  <Upload size={14} /> Choose Image
                </button>
              </div>
            )}

            {activeTab === "camera" && (
              <div>
                <div style={{ marginBottom: 8, fontSize: 12, opacity: 0.85 }}>Start your device camera and capture a frame.</div>
                <div style={{ background: "#000", height: 360, marginBottom: 8, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <video ref={videoRef} autoPlay playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={startCamera} style={{ padding: "8px 12px", borderRadius: 8 }}>Start Camera</button>
                  <button onClick={capture} style={{ padding: "8px 12px", borderRadius: 8, background: "#0369a1", color: "#fff" }}><FileImage size={14}/> Capture</button>
                </div>
                <canvas ref={canvasRef} style={{ display: "none" }} />
              </div>
            )}

            {activeTab === "process" && (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <div>
                    <div style={{ fontSize: 12, opacity: 0.8 }}>Original</div>
                    <div style={{ marginTop: 8, background: "#020617", borderRadius: 8, minHeight: 260, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
                      {originalImage ? <img src={originalImage} alt="original" style={{ width: "100%", height: "100%", objectFit: "contain" }} /> : <div style={{ opacity: 0.6 }}>No image</div>}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, opacity: 0.8 }}>Enhanced</div>
                    <div style={{ marginTop: 8, background: "#020617", borderRadius: 8, minHeight: 260, display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
                      {isProcessing ? (<div style={{ textAlign: "center" }}><RefreshCw className="spin" /> Processing... {progress>0 && `${progress}%`}</div>) : (processedImage ? <img src={processedImage} alt="processed" style={{ width: "100%", height: "100%", objectFit: "contain" }} /> : <div style={{ opacity: 0.6 }}>No processed image yet</div>)}
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
                  <button onClick={process} disabled={isProcessing} style={{ padding: "8px 12px", borderRadius: 8, background: "#0369a1", color: "#fff" }}><RefreshCw/> Reprocess</button>
                  <button onClick={download} disabled={!processedImage} style={{ padding: "8px 12px", borderRadius: 8, background: "#059669", color: "#fff" }}><Download/> Download</button>
                </div>
              </div>
            )}
          </section>

          {/* RIGHT: controls */}
          <aside style={{ background: "#071025", borderRadius: 12, padding: 12 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
              <Sliders/> <strong>Enhancement</strong>
            </div>

            <div style={{ display: "grid", gap: 10 }}>
              <label style={{ fontSize: 13, opacity: 0.85 }}>Dehaze {Math.round(dehazeStrength)}%</label>
              <input type="range" min="30" max="150" value={dehazeStrength} onChange={(e) => setDehazeStrength(Number(e.target.value))} />

              <label style={{ fontSize: 13, opacity: 0.85 }}>Brightness {brightness}</label>
              <input type="range" min="-80" max="80" value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} />

              <label style={{ fontSize: 13, opacity: 0.85 }}>Contrast {contrast}</label>
              <input type="range" min="-80" max="80" value={contrast} onChange={(e) => setContrast(Number(e.target.value))} />

              <label style={{ fontSize: 13, opacity: 0.85 }}>Saturation {saturation}</label>
              <input type="range" min="-80" max="80" value={saturation} onChange={(e) => setSaturation(Number(e.target.value))} />

              <label style={{ fontSize: 13, opacity: 0.85 }}>Sharpness {sharpness}</label>
              <input type="range" min="0" max="3" step="0.1" value={sharpness} onChange={(e) => setSharpness(Number(e.target.value))} />
            </div>

            <div style={{ marginTop: 16 }}>
              <strong>Detected objects</strong>
              <div style={{ marginTop: 8 }}>
                {detectedObjects.length === 0 ? <div style={{ fontSize: 13, opacity: 0.7 }}>No objects yet — reprocess to detect.</div> :
                  <div style={{ display: "grid", gap: 8 }}>
                    {detectedObjects.map((d, idx) => (
                      <div key={idx} style={{ display: "flex", justifyContent: "space-between", background: "#041022", padding: 8, borderRadius: 8 }}>
                        <div>{d.label || d.name || "object"}</div>
                        <div style={{ opacity: 0.85 }}>{Math.round((d.score || d.confidence || 0) * 100)}%</div>
                      </div>
                    ))}
                  </div>
                }
              </div>
            </div>
          </aside>
        </main>

        <footer style={{ textAlign: "center", opacity: 0.7, marginTop: 8 }}>ClearView — Local demo</footer>
      </div>
    </div>
  );
}
