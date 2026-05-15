'use client';

import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import Webcam from 'react-webcam';
import { Upload, Fish, AlertCircle, CheckCircle2, Camera, RefreshCw, X } from 'lucide-react';

export default function FishDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<'upload' | 'scan'>('upload');
  const [isCameraReady, setIsCameraReady] = useState(false);
  
  const webcamRef = useRef<Webcam>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResult(null);
      setError(null);
    }
  };

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setPreview(imageSrc);
      
      // Convert base64 to File object
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const capturedFile = new File([blob], "capture.jpg", { type: "image/jpeg" });
          setFile(capturedFile);
          setResult(null);
          setError(null);
        });
    }
  }, [webcamRef]);

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to connect to the backend server.');
    } finally {
      setLoading(false);
    }
  };

  const resetSelection = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans">
      <div className="max-w-4xl mx-auto">
        <header className="flex items-center gap-3 mb-10">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Fish className="text-white w-8 h-8" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-slate-900">Fish Species Classification</h1>
            <p className="text-slate-500">AI-powered detection and identification system</p>
          </div>
        </header>

        <div className="flex gap-2 mb-6 bg-white p-1 rounded-xl w-fit border border-slate-200 shadow-sm">
          <button 
            onClick={() => { setMode('upload'); resetSelection(); }}
            className={`px-6 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2
              ${mode === 'upload' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-500 hover:bg-slate-50'}`}
          >
            <Upload className="w-4 h-4" /> Upload File
          </button>
          <button 
            onClick={() => { setMode('scan'); resetSelection(); }}
            className={`px-6 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2
              ${mode === 'scan' ? 'bg-blue-600 text-white shadow-md' : 'text-slate-500 hover:bg-slate-50'}`}
          >
            <Camera className="w-4 h-4" /> Live Scan
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4 flex items-center justify-between">
              <span className="flex items-center gap-2">
                {mode === 'upload' ? <Upload className="w-5 h-5" /> : <Camera className="w-5 h-5" />}
                {mode === 'upload' ? 'Choose Image' : 'Camera Scan'}
              </span>
              {preview && (
                <button onClick={resetSelection} className="text-slate-400 hover:text-red-500 transition-colors">
                  <X className="w-5 h-5" />
                </button>
              )}
            </h2>
            
            <div className="relative border-2 border-dashed border-slate-300 rounded-xl overflow-hidden bg-slate-50 min-h-[300px] flex items-center justify-center">
              {preview ? (
                <img src={preview} alt="Preview" className="max-h-[400px] w-full object-contain shadow-sm" />
              ) : mode === 'upload' ? (
                <div 
                  className="w-full h-full py-16 text-center cursor-pointer hover:bg-slate-100 transition-colors"
                  onClick={() => document.getElementById('fileInput')?.click()}
                >
                  <div className="bg-white w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm">
                    <Upload className="text-blue-600 w-8 h-8" />
                  </div>
                  <p className="text-slate-600 font-medium">Click to upload or drag and drop</p>
                  <p className="text-slate-400 text-sm mt-1">PNG, JPG or JPEG (Max 10MB)</p>
                  <input id="fileInput" type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
                </div>
              ) : (
                <div className="relative w-full h-full">
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    className="w-full h-full object-cover"
                    onUserMedia={() => setIsCameraReady(true)}
                  />
                  {isCameraReady && (
                    <div className="absolute bottom-4 left-1/2 -translate-x-1/2">
                      <button 
                        onClick={capture}
                        className="bg-white p-4 rounded-full shadow-xl border-4 border-blue-100 hover:scale-110 transition-transform group"
                      >
                        <div className="w-6 h-6 bg-blue-600 rounded-full group-active:scale-90 transition-transform"></div>
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            <div className="flex gap-3 mt-6">
              <button
                onClick={handleUpload}
                disabled={!file || loading}
                className={`flex-1 py-3 rounded-xl font-semibold transition-all flex items-center justify-center gap-2
                  ${!file || loading 
                    ? 'bg-slate-200 text-slate-400 cursor-not-allowed' 
                    : 'bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg'}`}
              >
                {loading ? <RefreshCw className="w-5 h-5 animate-spin" /> : <Fish className="w-5 h-5" />}
                {loading ? 'Analyzing...' : 'Identify Species'}
              </button>
            </div>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-100 rounded-lg flex gap-3 text-red-700 text-sm">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <p>{error}</p>
              </div>
            )}
          </div>

          {/* Results Section */}
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <CheckCircle2 className="w-5 h-5" /> Analysis Result
            </h2>

            {result ? (
              <div className="space-y-6">
                <div className="bg-blue-50 p-4 rounded-xl border border-blue-100 text-center">
                  <p className="text-blue-600 text-sm font-semibold uppercase tracking-wider mb-1">Top Prediction</p>
                  <p className="text-2xl font-bold text-slate-900">{result.prediction}</p>
                  <div className="mt-2 flex items-center justify-center gap-2">
                    <div className="w-32 bg-slate-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${result.confidence * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-sm font-medium text-slate-600">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">Top 5 Candidates</h3>
                  <div className="space-y-2">
                    {result.top_5.map((item: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-slate-50 rounded-lg border border-slate-100">
                        <span className="text-slate-700 font-medium truncate pr-4">{item.species}</span>
                        <span className="text-slate-400 text-sm">{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="p-3 bg-amber-50 border border-amber-100 rounded-lg flex gap-2 text-amber-800 text-xs italic">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <p>{result.disclaimer}</p>
                </div>
              </div>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-slate-400 py-10">
                <Fish className="w-16 h-16 opacity-20 mb-4" />
                <p>Upload an image to see analysis</p>
              </div>
            )}
          </div>
        </div>

        {/* Footer info */}
        <footer className="mt-12 pt-8 border-t border-slate-200 text-center text-slate-400 text-sm">
          <p>Built with QUT Fish Dataset & PyTorch</p>
          <p className="mt-1">Focus: Species Classification Only</p>
        </footer>
      </div>
    </div>
  );
}
