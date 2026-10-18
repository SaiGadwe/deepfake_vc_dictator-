import React, { useState, useRef, useEffect } from 'react';
import {
  Shield,
  Upload,
  Mic,
  Square,
  AlertTriangle,
  CheckCircle,
  HelpCircle,
  FileAudio,
  History,
  X,
  RotateCcw,
  Volume2,
  ChevronRight,
  Info,
  Users,
  Play,
  Flame,
  Layers
} from 'lucide-react';
import './App.css';

const API_BASE = "http://localhost:8000/api/v1";

export default function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'record'
  const [selectedFile, setSelectedFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentStep, setCurrentStep] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [historyList, setHistoryList] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [activeSpeakerIdx, setActiveSpeakerIdx] = useState(0);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const fileInputRef = useRef(null);
  const audioPlayerRef = useRef(null);

  // Clean up object URLs
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [audioUrl]);

  const handleFileSelect = (file) => {
    if (!file) return;
    setErrorMsg('');
    setSelectedFile(file);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(URL.createObjectURL(file));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  // Microphone Recording
  const startRecording = async () => {
    setErrorMsg('');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const recordedFile = new File([audioBlob], `mic_recording_${Date.now()}.wav`, { type: 'audio/wav' });
        handleFileSelect(recordedFile);
        stream.getTracks().forEach(t => t.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingSeconds(0);
      timerRef.current = setInterval(() => {
        setRecordingSeconds(prev => prev + 1);
      }, 1000);
    } catch (err) {
      setErrorMsg("Microphone access denied or unavailable: " + err.message);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(timerRef.current);
    }
  };

  // Execute Analysis
  const runAnalysis = async () => {
    if (!selectedFile) {
      setErrorMsg("Please select or record an audio file first.");
      return;
    }

    setIsProcessing(true);
    setErrorMsg('');
    setAnalysisResult(null);
    setFeedbackSubmitted(null);
    setActiveSpeakerIdx(0);

    const steps = [
      "Transcoding audio stream to 16 kHz Mono/Stereo WAV...",
      "Analyzing channels & running KMeans speaker diarization...",
      "Segmenting speech into uniform 4-second sub-windows...",
      "Extracting 201 MFCC, LFCC & Parselmouth F0 features...",
      "Evaluating XGBoost classifier & SHAP explanations per voice..."
    ];

    let stepIdx = 0;
    setCurrentStep(steps[0]);
    const stepInterval = setInterval(() => {
      stepIdx = (stepIdx + 1) % steps.length;
      setCurrentStep(steps[stepIdx]);
    }, 1100);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const res = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errData.detail || "Analysis request failed.");
      }

      const data = await res.json();
      setAnalysisResult(data);
    } catch (err) {
      setErrorMsg(err.message || "Failed to analyze audio sample.");
    } finally {
      clearInterval(stepInterval);
      setIsProcessing(false);
    }
  };

  // Playback of separated voice segments
  const playSegment = (startTimeSec) => {
    if (audioPlayerRef.current) {
      audioPlayerRef.current.currentTime = startTimeSec;
      audioPlayerRef.current.play().catch(e => console.log("Playback error:", e));
    }
  };

  // Submit User Feedback
  const handleFeedback = async (feedbackType) => {
    if (!analysisResult?.analysis_id) return;
    try {
      await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_id: analysisResult.analysis_id,
          feedback: feedbackType
        })
      });
      setFeedbackSubmitted(feedbackType);
    } catch (err) {
      console.error("Feedback error:", err);
    }
  };

  // Fetch History
  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_BASE}/history?limit=10`);
      if (res.ok) {
        const data = await res.json();
        setHistoryList(data);
        setShowHistory(true);
      }
    } catch (err) {
      setErrorMsg("Could not load history: " + err.message);
    }
  };

  const resetAll = () => {
    setSelectedFile(null);
    setAudioUrl(null);
    setAnalysisResult(null);
    setErrorMsg('');
    setFeedbackSubmitted(null);
    setActiveSpeakerIdx(0);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const formatTime = (secs) => {
    const m = Math.floor(secs / 60);
    const s = secs % 60;
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  // Progress Ring Helper
  const renderProgressRing = (score, verdict, size = 170) => {
    const radius = size === 170 ? 64 : 44;
    const circumference = 2 * Math.PI * radius;
    const progress = Math.min(Math.max(score, 0), 1);
    const offset = circumference - progress * circumference;

    let strokeColor = "#3b82f6";
    if (verdict === "Likely AI") strokeColor = "#ef4444";
    else if (verdict === "Uncertain") strokeColor = "#f59e0b";
    else if (verdict === "Likely Human") strokeColor = "#10b981";
    else strokeColor = "#64748b";

    return (
      <div className="ring-container" style={{ width: size, height: size }}>
        <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="#1e293b"
            strokeWidth={size === 170 ? "12" : "8"}
            fill="transparent"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={strokeColor}
            strokeWidth={size === 170 ? "12" : "8"}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            fill="transparent"
            style={{ transition: 'stroke-dashoffset 0.8s ease-in-out' }}
          />
        </svg>
        <div className="ring-text-container">
          <div className="ring-score" style={{ color: strokeColor, fontSize: size === 170 ? '2.2rem' : '1.4rem' }}>
            {Math.round(score * 100)}%
          </div>
          <div className="ring-label" style={{ fontSize: size === 170 ? '0.7rem' : '0.6rem' }}>
            Synthetic Risk
          </div>
        </div>
      </div>
    );
  };

  const getVerdictClass = (verdict) => {
    switch (verdict) {
      case "Likely AI": return "verdict-likely-ai";
      case "Uncertain": return "verdict-uncertain";
      case "Likely Human": return "verdict-likely-human";
      default: return "verdict-cant-tell";
    }
  };

  const getVerdictIcon = (verdict) => {
    switch (verdict) {
      case "Likely AI": return <AlertTriangle size={18} />;
      case "Uncertain": return <Info size={18} />;
      case "Likely Human": return <CheckCircle size={18} />;
      default: return <HelpCircle size={18} />;
    }
  };

  const isMultiSpeaker = analysisResult?.speakers && analysisResult.speakers.length > 1;
  const currentSpeaker = analysisResult?.speakers?.[activeSpeakerIdx] || analysisResult?.speakers?.[0];

  return (
    <div className="mobile-wrapper">
      {/* App Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">
            <Shield size={20} />
          </div>
          <div>
            <div className="brand-title">AegisVoice</div>
            <div className="brand-sub">Multi-Voice Authenticity</div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div className="privacy-pill" title="Ephemeral volatile memory only. Audio is never stored.">
            <span className="privacy-dot"></span>
            RAM Only
          </div>
          <button
            onClick={fetchHistory}
            style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: 4 }}
            title="Audit History"
          >
            <History size={18} />
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="content-area">
        {errorMsg && (
          <div style={{
            background: 'rgba(239, 68, 68, 0.15)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: 8,
            padding: '10px 14px',
            marginBottom: 16,
            fontSize: '0.8rem',
            color: '#f87171'
          }}>
            {errorMsg}
          </div>
        )}

        {/* Step 1: Input & Upload Screen */}
        {!analysisResult && !isProcessing && (
          <>
            <div className="mode-tabs">
              <button
                className={`mode-tab ${activeTab === 'upload' ? 'active' : ''}`}
                onClick={() => setActiveTab('upload')}
              >
                <Upload size={14} /> Upload Audio
              </button>
              <button
                className={`mode-tab ${activeTab === 'record' ? 'active' : ''}`}
                onClick={() => setActiveTab('record')}
              >
                <Mic size={14} /> Live Record
              </button>
            </div>

            {activeTab === 'upload' ? (
              <div
                className={`dropzone ${isDragOver ? 'drag-active' : ''}`}
                onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  accept=".wav,.mp3,.ogg,.m4a,.aac,.flac,.webm"
                  style={{ display: 'none' }}
                  onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
                />
                <div className="upload-icon-wrapper">
                  <FileAudio size={28} />
                </div>
                <div className="drop-title">
                  {selectedFile ? selectedFile.name : "Tap or Drag Audio File Here"}
                </div>
                <div className="drop-sub">
                  Supports Single Calls & Multi-Speaker Audio (Stereo / Mono)
                </div>

                {selectedFile && (
                  <div className="file-pill">
                    <Volume2 size={14} color="#60a5fa" />
                    <span>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="recorder-box">
                <button
                  className={`rec-btn ${isRecording ? 'recording' : ''}`}
                  onClick={isRecording ? stopRecording : startRecording}
                >
                  {isRecording ? <Square size={26} /> : <Mic size={28} />}
                </button>
                <div className="rec-timer">{formatTime(recordingSeconds)}</div>
                <div className="drop-sub">
                  {isRecording ? "Recording speech... tap square to finish" : "Tap microphone to record voice sample"}
                </div>
              </div>
            )}

            {/* Audio Preview */}
            {audioUrl && (
              <div className="audio-player-box">
                <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <Volume2 size={13} /> Volatile Audio Preview (Purged on Analysis)
                </div>
                <audio ref={audioPlayerRef} src={audioUrl} controls />
              </div>
            )}

            <button
              className="btn-primary"
              disabled={!selectedFile}
              onClick={runAnalysis}
            >
              Analyze Voice Authenticity <ChevronRight size={18} />
            </button>
          </>
        )}

        {/* Step 2: Processing Screen */}
        {isProcessing && (
          <div className="processing-card">
            <div className="spinner"></div>
            <div style={{ fontSize: '1.05rem', fontWeight: 700, marginBottom: 8 }}>
              Analyzing Audio
            </div>
            <div className="pipeline-step">{currentStep}</div>
            <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: 14 }}>
              Zero audio bytes are stored. Volatile memory processing only.
            </div>
          </div>
        )}

        {/* Step 3: Analysis Results Screen */}
        {analysisResult && (
          <div>
            {/* Multi-Voice Banner if multi-speaker detected */}
            {isMultiSpeaker && (
              <div className="multivoice-banner">
                <Users size={20} color="#a78bfa" />
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.84rem', fontWeight: 700 }}>
                    Multi-Speaker Call Detected
                  </div>
                  <div style={{ fontSize: '0.7rem', color: '#cbd5e1' }}>
                    Separated into {analysisResult.speakers.length} distinct voice profiles ({analysisResult.speakers[0].separation_mode === 'stereo_channels' ? 'Dual-Channel Stereo' : 'KMeans Diarization'}).
                  </div>
                </div>
                <span className="multivoice-badge">Diarized</span>
              </div>
            )}

            {/* Hidden/Persistent Audio Element for Segment Playback */}
            {audioUrl && (
              <div className="audio-player-box" style={{ marginBottom: 14 }}>
                <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginBottom: 4, display: 'flex', justifyContent: 'space-between' }}>
                  <span>Audited Audio Player</span>
                  <span>{analysisResult.duration_analyzed_sec.toFixed(1)}s total</span>
                </div>
                <audio ref={audioPlayerRef} src={audioUrl} controls />
              </div>
            )}

            {/* Speaker Selection Tabs (when multiple speakers) */}
            {isMultiSpeaker && (
              <div className="speaker-nav-tabs">
                {analysisResult.speakers.map((s, idx) => (
                  <button
                    key={idx}
                    className={`speaker-nav-tab ${activeSpeakerIdx === idx ? 'active' : ''}`}
                    onClick={() => setActiveSpeakerIdx(idx)}
                  >
                    {idx === 0 && s.risk_score >= 0.35 && (
                      <Flame size={12} color="#f87171" style={{ display: 'inline', marginRight: 4 }} />
                    )}
                    {s.speaker_id} ({Math.round(s.risk_score * 100)}%)
                  </button>
                ))}
              </div>
            )}

            {/* Active / Most Suspicious Speaker Card */}
            {currentSpeaker && (
              <div className={`speaker-card ${currentSpeaker.is_primary_threat || (activeSpeakerIdx === 0 && currentSpeaker.risk_score >= 0.35) ? 'primary-threat' : ''}`}>
                {/* Threat Ribbon for highest risk profile */}
                {(currentSpeaker.is_primary_threat || (activeSpeakerIdx === 0 && currentSpeaker.risk_score >= 0.35)) && (
                  <div className="threat-ribbon">
                    <Flame size={12} /> Most Suspicious Voice Profile
                  </div>
                )}

                <div className="speaker-header-row">
                  <div>
                    <div className="speaker-title">
                      {currentSpeaker.speaker_id}
                    </div>
                    <div className="speaker-duration">
                      Voiced Speech: {currentSpeaker.voiced_duration_sec}s across {currentSpeaker.chunk_count} window(s)
                    </div>
                  </div>
                  <div className={`verdict-badge ${getVerdictClass(currentSpeaker.verdict)}`} style={{ padding: '6px 12px', fontSize: '0.85rem' }}>
                    {getVerdictIcon(currentSpeaker.verdict)}
                    <span>{currentSpeaker.verdict}</span>
                  </div>
                </div>

                {/* Progress Ring for this speaker */}
                <div style={{ textAlign: 'center', margin: '14px 0' }}>
                  {renderProgressRing(currentSpeaker.risk_score, currentSpeaker.verdict, 150)}
                </div>

                {/* Playback Buttons for this speaker's separated audio chunks */}
                {currentSpeaker.timestamps && currentSpeaker.timestamps.length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.72rem', color: '#94a3b8', marginBottom: 6, display: 'flex', alignItems: 'center', gap: 4 }}>
                      <Play size={12} /> Play Separated Voice Segments:
                    </div>
                    <div className="speaker-playback-row">
                      {currentSpeaker.timestamps.map((ts, idx) => (
                        <button
                          key={idx}
                          className="btn-segment-play"
                          onClick={() => playSegment(ts.start_time_sec)}
                          title={`Seek and play from ${ts.start_time_sec}s`}
                        >
                          <Play size={10} fill="#60a5fa" />
                          {ts.start_time_sec}s - {ts.end_time_sec}s
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* 4-Second Sub-window Breakdown for this speaker */}
                {currentSpeaker.chunk_scores?.length > 0 && (
                  <>
                    <div className="section-title" style={{ marginTop: 12 }}>
                      Sub-Windows (4s Chunks)
                    </div>
                    <div className="chunk-bar-container">
                      {currentSpeaker.chunk_scores.map((chunk) => (
                        <div key={chunk.chunk_index} className="chunk-card">
                          <div className="chunk-time">{chunk.start_time_sec}s - {chunk.end_time_sec}s</div>
                          <div
                            className="chunk-risk"
                            style={{
                              color: chunk.score >= 0.65 ? '#ef4444' : chunk.score >= 0.35 ? '#f59e0b' : '#10b981'
                            }}
                          >
                            {Math.round(chunk.score * 100)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}

                {/* SHAP Rationale Breakdown for this speaker */}
                <div className="section-title">
                  Acoustic Rationales (SHAP)
                </div>
                {currentSpeaker.rationales?.map((r, i) => (
                  <div key={i} className="rationale-card">
                    <div className="rationale-header">
                      <span className="rationale-title">{r.feature}</span>
                      <span className={`direction-tag ${r.direction === 'supports_ai' ? 'tag-ai' : 'tag-human'}`}>
                        {r.direction === 'supports_ai' ? 'Flag: AI Indicator' : 'Human Baseline'}
                      </span>
                    </div>
                    <div className="rationale-desc">{r.description}</div>
                  </div>
                ))}
              </div>
            )}

            {/* Defensive Notice */}
            <p className="defensive-notice" style={{ textAlign: 'center', margin: '14px auto' }}>
              Defensive Assessment: Independent acoustic models evaluate each voice profile.
              Outputs strictly avoid absolute claims like "Safe" or "Genuine".
            </p>

            {/* User Feedback Widget */}
            <div className="feedback-box">
              <div style={{ fontSize: '0.78rem', fontWeight: 600 }}>Was this verdict accurate?</div>
              <div className="feedback-buttons">
                <button
                  className={`btn-feedback ${feedbackSubmitted === 'accurate' ? 'active' : ''}`}
                  onClick={() => handleFeedback('accurate')}
                >
                  Accurate
                </button>
                <button
                  className={`btn-feedback ${feedbackSubmitted === 'inaccurate' ? 'active' : ''}`}
                  onClick={() => handleFeedback('inaccurate')}
                >
                  Inaccurate
                </button>
              </div>
              {feedbackSubmitted && (
                <div style={{ fontSize: '0.7rem', color: '#10b981', marginTop: 6 }}>
                  Feedback recorded to improve offline training loop.
                </div>
              )}
            </div>

            {/* Reset Button */}
            <button
              className="btn-primary"
              onClick={resetAll}
              style={{ marginTop: 20 }}
            >
              <RotateCcw size={16} /> Analyze Another Voice
            </button>
          </div>
        )}
      </div>

      {/* History Drawer Modal */}
      {showHistory && (
        <div className="history-modal">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div style={{ fontSize: '1rem', fontWeight: 700 }}>Audit Records (SQLite)</div>
            <button
              onClick={() => setShowHistory(false)}
              style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
            >
              <X size={20} />
            </button>
          </div>
          <p style={{ fontSize: '0.72rem', color: '#94a3b8', marginBottom: 16 }}>
            In accordance with privacy mandates, only risk scores, verdicts, and timestamps are stored. Zero audio recordings are retained.
          </p>

          <div style={{ flex: 1, overflowY: 'auto' }}>
            {historyList.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#64748b', marginTop: 40, fontSize: '0.82rem' }}>
                No prior analysis records found.
              </div>
            ) : (
              historyList.map((item) => (
                <div key={item.analysis_id} className="history-item">
                  <div>
                    <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>{item.analysis_id}</div>
                    <div style={{ fontSize: '0.68rem', color: '#64748b' }}>
                      {new Date(item.timestamp).toLocaleString()} • {item.speaker_id}
                    </div>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <div
                      style={{
                        fontSize: '0.8rem',
                        fontWeight: 700,
                        color: item.risk_score >= 0.65 ? '#ef4444' : item.risk_score >= 0.35 ? '#f59e0b' : '#10b981'
                      }}
                    >
                      {Math.round(item.risk_score * 100)}%
                    </div>
                    <div style={{ fontSize: '0.68rem', color: '#94a3b8' }}>{item.verdict}</div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}
