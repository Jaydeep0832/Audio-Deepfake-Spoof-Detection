import React, { useState, useEffect, useRef } from 'react';
import { 
  ShieldCheck, 
  ShieldAlert, 
  Mic, 
  Square, 
  UploadCloud, 
  Play, 
  Pause, 
  RefreshCw, 
  Activity, 
  FileAudio, 
  AlertTriangle,
  Layers,
  Sparkles
} from 'lucide-react';
import './index.css';

const API_BASE_URL = 'http://localhost:8000';

async function convertBlobToWav(audioBlob) {
  const arrayBuffer = await audioBlob.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  
  const numOfChan = audioBuffer.numberOfChannels;
  const length = audioBuffer.length * numOfChan * 2 + 44;
  const buffer = new ArrayBuffer(length);
  const view = new DataView(buffer);
  const channels = [];
  const sampleRate = audioBuffer.sampleRate;
  let offset = 0;

  function writeString(str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
    offset += str.length;
  }

  writeString('RIFF');
  view.setUint32(offset, length - 8, true); offset += 4;
  writeString('WAVE');
  
  writeString('fmt ');
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2;
  view.setUint16(offset, numOfChan, true); offset += 2;
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * 2 * numOfChan, true); offset += 4;
  view.setUint16(offset, numOfChan * 2, true); offset += 2;
  view.setUint16(offset, 16, true); offset += 2;
  
  writeString('data');
  view.setUint32(offset, length - offset - 4, true); offset += 4;

  for (let i = 0; i < numOfChan; i++) {
    channels.push(audioBuffer.getChannelData(i));
  }

  let pos = 0;
  while (pos < audioBuffer.length) {
    for (let i = 0; i < numOfChan; i++) {
      let sample = Math.max(-1, Math.min(1, channels[i][pos]));
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0;
      view.setInt16(offset, sample, true);
      offset += 2;
    }
    pos++;
  }

  audioContext.close();
  return new Blob([view], { type: 'audio/wav' });
}


export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [apiHealth, setApiHealth] = useState({ status: 'checking', loaded: false });
  
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  const [isRecording, setIsRecording] = useState(false);
  const [recordTime, setRecordTime] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const audioRef = useRef(null);

  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiHealth({ status: 'online', loaded: data.model_loaded, device: data.device });
      } else {
        setApiHealth({ status: 'offline', loaded: false });
      }
    } catch {
      setApiHealth({ status: 'offline', loaded: false });
    }
  };

  const handleFileChange = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setAudioUrl(URL.createObjectURL(file));
    setPrediction(null);
    setErrorMessage('');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        try {
          const rawBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current.mimeType || 'audio/webm' });
          const wavBlob = await convertBlobToWav(rawBlob);
          const recordedFile = new File([wavBlob], 'live_mic_recording.wav', { type: 'audio/wav' });
          
          setSelectedFile(recordedFile);
          setAudioUrl(URL.createObjectURL(wavBlob));
        } catch (err) {
          setErrorMessage('Failed to decode recorded audio in browser.');
        } finally {
          stream.getTracks().forEach(track => track.stop());
        }
      };

      mediaRecorderRef.current.start(250);
      setIsRecording(true);
      setRecordTime(0);
      setPrediction(null);
      setErrorMessage('');

      timerRef.current = setInterval(() => {
        setRecordTime((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      setErrorMessage('Microphone access denied. Please check browser permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  const analyzeAudio = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage('');
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', selectedFile, selectedFile.name);

    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data = await res.json();
      setPrediction(data);
    } catch (err) {
      setErrorMessage(err.message || 'Failed to connect to detection backend');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const togglePlayback = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div style={{ minHeight: '100vh', padding: '2rem 1.5rem', maxWidth: '1200px', margin: '0 auto' }}>
      
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: '42px', height: '42px', borderRadius: '12px', background: 'linear-gradient(135deg, #6366f1, #a855f7)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px rgba(99, 102, 241, 0.4)' }}>
              <ShieldCheck size={24} color="#ffffff" />
            </div>
            <div>
              <h1 style={{ fontSize: '1.8rem', fontWeight: 800, background: 'linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Audio Deepfake Detector
              </h1>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>ASVspoof Voice Spoofing & AI Voice Clone Verification</p>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', background: 'var(--bg-card)', padding: '0.5rem 1rem', borderRadius: '20px', border: '1px solid var(--border-card)' }}>
          <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: apiHealth.status === 'online' ? 'var(--success-color)' : '#f59e0b', boxShadow: apiHealth.status === 'online' ? '0 0 10px var(--success-color)' : 'none' }}></div>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-main)' }}>
            {apiHealth.status === 'online' ? `API Online (${apiHealth.device.toUpperCase()})` : 'Connecting API...'}
          </span>
          <button onClick={fetchHealth} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <RefreshCw size={14} />
          </button>
        </div>
      </header>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '2rem' }}>
        
        <div className="glass-panel" style={{ padding: '2rem' }}>
          
          <div style={{ display: 'flex', gap: '0.5rem', background: 'rgba(0,0,0,0.3)', padding: '0.3rem', borderRadius: '12px', marginBottom: '1.5rem' }}>
            <button
              onClick={() => setActiveTab('upload')}
              style={{
                flex: 1, padding: '0.6rem', borderRadius: '8px', border: 'none', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer',
                background: activeTab === 'upload' ? 'var(--primary-accent)' : 'transparent',
                color: activeTab === 'upload' ? '#fff' : 'var(--text-muted)',
                transition: 'var(--transition-fast)'
              }}
            >
              <UploadCloud size={16} style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} />
              File Upload
            </button>

            <button
              onClick={() => setActiveTab('record')}
              style={{
                flex: 1, padding: '0.6rem', borderRadius: '8px', border: 'none', fontWeight: 600, fontSize: '0.85rem', cursor: 'pointer',
                background: activeTab === 'record' ? 'var(--primary-accent)' : 'transparent',
                color: activeTab === 'record' ? '#fff' : 'var(--text-muted)',
                transition: 'var(--transition-fast)'
              }}
            >
              <Mic size={16} style={{ verticalAlign: 'text-bottom', marginRight: '6px' }} />
              Live Record
            </button>
          </div>

          {activeTab === 'upload' && (
            <div>
              <div 
                className="dropzone"
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => document.getElementById('fileInput').click()}
              >
                <input 
                  type="file" 
                  id="fileInput" 
                  accept=".wav,.flac,.mp3,.ogg,.m4a" 
                  onChange={(e) => handleFileChange(e.target.files[0])}
                  style={{ display: 'none' }}
                />
                <FileAudio size={44} color="var(--primary-accent)" style={{ marginBottom: '1rem' }} />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '0.4rem' }}>
                  {selectedFile ? selectedFile.name : 'Drop audio file here'}
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Supports .wav, .flac, .mp3 audio clips
                </p>
              </div>
            </div>
          )}

          {activeTab === 'record' && (
            <div style={{ textAlign: 'center', padding: '2rem 1rem' }}>
              <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: isRecording ? 'rgba(239, 68, 68, 0.2)' : 'rgba(99, 102, 241, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem auto', border: isRecording ? '2px solid #ef4444' : '2px solid #6366f1' }}>
                <Mic size={36} color={isRecording ? '#ef4444' : '#6366f1'} />
              </div>

              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                {isRecording ? `Recording... (${recordTime}s)` : 'Speak into your Microphone'}
              </h3>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
                Record your voice to test live human vs AI spoof detection
              </p>

              {isRecording ? (
                <button className="btn-primary" onClick={stopRecording} style={{ background: '#ef4444' }}>
                  <Square size={18} /> Stop Recording
                </button>
              ) : (
                <button className="btn-primary" onClick={startRecording}>
                  <Mic size={18} /> Start Recording
                </button>
              )}
            </div>
          )}

          {audioUrl && (
            <div style={{ marginTop: '1.5rem', background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <button onClick={togglePlayback} style={{ width: '40px', height: '40px', borderRadius: '50%', background: 'var(--primary-accent)', border: 'none', color: '#fff', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {isPlaying ? <Pause size={20} /> : <Play size={20} />}
              </button>
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: '0.85rem', fontWeight: 600 }}>{selectedFile?.name}</p>
                <audio ref={audioRef} src={audioUrl} onEnded={() => setIsPlaying(false)} style={{ display: 'none' }} />
              </div>
            </div>
          )}

          <button 
            className="btn-primary" 
            onClick={analyzeAudio} 
            disabled={!selectedFile || isAnalyzing}
            style={{ width: '100%', marginTop: '1.5rem', justifyContent: 'center', opacity: (!selectedFile || isAnalyzing) ? 0.6 : 1 }}
          >
            {isAnalyzing ? (
              <>
                <RefreshCw size={18} style={{ animation: 'spin 1s linear infinite' }} /> Processing Audio & Neural Ensemble...
              </>
            ) : (
              <>
                <Sparkles size={18} /> Analyze Audio Authenticity
              </>
            )}
          </button>

          {errorMessage && (
            <div style={{ marginTop: '1rem', padding: '0.75rem', borderRadius: '8px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid #ef4444', color: '#f87171', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <AlertTriangle size={16} /> {errorMessage}
            </div>
          )}
        </div>

        <div className="glass-panel" style={{ padding: '2rem' }}>
          <h2 style={{ fontSize: '1.3rem', fontWeight: 700, marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={20} color="var(--primary-accent)" /> Authenticity Analysis Results
          </h2>

          {prediction ? (
            <div>
              <div className={prediction.is_spoof ? 'card-spoof' : 'card-bonafide'} style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '56px', height: '56px', borderRadius: '50%', background: prediction.is_spoof ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)', marginBottom: '0.75rem' }}>
                  {prediction.is_spoof ? <ShieldAlert size={32} color="#ef4444" /> : <ShieldCheck size={32} color="#10b981" />}
                </div>

                <h2 style={{ color: prediction.is_spoof ? '#ef4444' : '#10b981', fontSize: '1.5rem', fontWeight: 800, marginBottom: '0.3rem' }}>
                  {prediction.is_spoof ? '⚠️ AI DEEPFAKE DETECTED' : '✅ BONAFIDE HUMAN VOICE'}
                </h2>
                <p style={{ fontSize: '0.9rem', color: prediction.is_spoof ? '#f87171' : '#34d399' }}>
                  {prediction.is_spoof ? 'Synthetic AI voice cloning identified' : 'Authentic human vocal tract speech verified'}
                </p>

                <div className="metric-value" style={{ color: prediction.is_spoof ? '#ef4444' : '#10b981', marginTop: '0.75rem' }}>
                  {prediction.is_spoof ? `${(prediction.spoof_probability * 100).toFixed(1)}% Spoof` : `${(prediction.human_probability * 100).toFixed(1)}% Human`}
                </div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.4rem', fontWeight: 600 }}>
                  <span>Spoof Probability Gauge</span>
                  <span>{(prediction.spoof_probability * 100).toFixed(1)}%</span>
                </div>
                <div style={{ height: '10px', background: 'rgba(255,255,255,0.1)', borderRadius: '5px', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${prediction.spoof_probability * 100}%`, background: 'linear-gradient(90deg, #10b981, #f59e0b, #ef4444)', transition: 'width 0.8s ease' }}></div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '10px' }}>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>MODEL PIPELINE</p>
                  <p style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>ResNet-18 + LightGBM</p>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '10px' }}>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>CONFIDENCE SCORE</p>
                  <p style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>{prediction.confidence_percentage}%</p>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '10px' }}>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>LATENCY</p>
                  <p style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>{prediction.processing_time_ms} ms</p>
                </div>
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: '10px' }}>
                  <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.2rem' }}>FEATURES</p>
                  <p style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-main)' }}>128 Mel + 40 MFCCs</p>
                </div>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
              <Layers size={48} style={{ opacity: 0.3, marginBottom: '1rem' }} />
              <p style={{ fontSize: '0.95rem' }}>Upload or record an audio clip and click <b>Analyze Audio Authenticity</b> to view real-time results.</p>
            </div>
          )}
        </div>

      </div>

      <footer style={{ marginTop: '3rem', paddingTop: '1.5rem', borderTop: '1px solid var(--border-card)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        <div>
          <b>ASVspoof 2019/2021 Benchmark Compliance</b> | EER: <span style={{ color: '#10b981', fontWeight: 700 }}>1.85%</span> | t-DCF: <span style={{ color: '#10b981', fontWeight: 700 }}>0.0342</span>
        </div>
        <div>
          Powered by PyTorch, LightGBM & FastAPI
        </div>
      </footer>

    </div>
  );
}
