import streamlit as st
from streamlit.components.v1 import html
import requests
import json

st.set_page_config(page_title="SMS Phishing Detection", layout="wide", initial_sidebar_state="collapsed")

# Load secrets or environment variables
bert_token = st.secrets.get("BERT_TOKEN", "")
bert_model = st.secrets.get("BERT_MODEL", "")
distilbert_token = st.secrets.get("DISTILBERT_TOKEN", "")
distilbert_model = st.secrets.get("DISTILBERT_MODEL", "")
default_model = st.secrets.get("DEFAULT_MODEL", "bert")

# Initialize session state
if 'api_result' not in st.session_state:
    st.session_state.api_result = None

def call_hf_api(text, model_name):
    """Call HF API from Python instead of JavaScript"""
    token = bert_token if model_name == "bert" else distilbert_token
    model_id = bert_model if model_name == "bert" else distilbert_model

    if not token or not model_id:
        return None, "Model not configured"

    headers = {"Authorization": f"Bearer {token}"}
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"

    try:
        response = requests.post(
            api_url,
            headers=headers,
            json={"inputs": text},
            timeout=30
        )

        if response.status_code == 200:
            hf_data = response.json()

            if isinstance(hf_data, list) and len(hf_data) > 0:
                scores = hf_data[0] if isinstance(hf_data[0], list) else hf_data
                if isinstance(scores, list) and len(scores) > 0:
                    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
                    top = sorted_scores[0]

                    confidence = round(top['score'] * 100)
                    label = top['label'].upper()
                    is_phishing = label in ['PHISHING', 'SMISHING', '2']

                    return {
                        'prediction': label,
                        'confidence': confidence,
                        'is_phishing': is_phishing
                    }, None

        return None, f"API error: {response.status_code}"
    except Exception as e:
        return None, str(e)

# Hidden input for JavaScript to send messages
col1, col2 = st.columns([1, 20])
with col1:
    pass
with col2:
    message_input = st.text_input("", key="phishing_message", label_visibility="collapsed", placeholder="Message from JS")

# Process API call if new message
if message_input and message_input != st.session_state.get('last_processed', ''):
    st.session_state.last_processed = message_input
    model = st.session_state.get('selected_model', default_model)
    result, error = call_hf_api(message_input, model)
    st.session_state.api_result = {'result': result, 'error': error}

# Create JSON data for JavaScript
api_data = st.session_state.api_result if st.session_state.api_result else {'result': None, 'error': None}
api_json = json.dumps(api_data)

html_content = """
<!DOCTYPE html>
<html>
<head>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

.wrapper { background: #0d1117; min-height: 120vh; padding: 60px 40px; position: relative; }

.model-selector {
  background: #1e1e1e;
  border: 1px solid #333;
  border-radius: 12px;
  padding: 20px;
  min-width: 260px;
  width: 280px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  align-self: flex-start;
  margin-top: 30px;
}

.model-selector label {
  display: block;
  font-size: 13px;
  font-weight: 700;
  color: #0084ff;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 10px;
}

.model-selector select {
  width: 100%;
  background: #2a2a2a;
  border: 1px solid #444;
  border-radius: 8px;
  color: white;
  font-size: 15px;
  padding: 10px 14px;
  outline: none;
  cursor: pointer;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='%230084ff'%3E%3Cpath d='M7 10l5 5 5-5z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 32px;
}

.model-selector select:focus {
  border-color: #0084ff;
}

.model-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #00c853;
  margin-right: 6px;
  vertical-align: middle;
}


.model-info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.info-label {
  font-size: 14px;
  color: #888;
}

.info-value {
  font-size: 15px;
  font-weight: 700;
  color: #0084ff;
}

.model-status {
  margin-top: 10px;
  font-size: 14px;
  color: #00c853;
  font-weight: 700;
}

.theme-toggle-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid #333;
}

.theme-toggle-label {
  font-size: 13px;
  font-weight: 700;
  color: #0084ff;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin: 0;
}

.toggle-switch {
  position: relative;
  display: inline-flex;
  align-items: center;
  width: 70px;
  height: 28px;
}

.toggle-switch input {
  opacity: 0;
  width: 0;
  height: 0;
  cursor: pointer;
}

.toggle-slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #444;
  border-radius: 14px;
  transition: 0.3s;
  border: 1px solid #555;
  user-select: none;
}

.toggle-switch input:checked + .toggle-slider {
  background-color: #0084ff;
  border-color: #0073e6;
}

.toggle-slider::after {
  content: '';
  position: absolute;
  height: 20px;
  width: 20px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  border-radius: 50%;
  transition: 0.3s;
}

.toggle-switch input:checked + .toggle-slider::after {
  transform: translateX(42px);
}

.toggle-label-light,
.toggle-label-dark {
  position: absolute;
  font-size: 10px;
  font-weight: 600;
  color: white;
  pointer-events: none;
}

.toggle-label-light {
  left: 8px;
}

.toggle-label-dark {
  right: 8px;
}

.model-info {
  margin-top: 14px;
  border-top: 1px solid #333;
  padding-top: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.header-box {
  background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
  border-radius: 12px;
  padding: 40px 50px;
  margin-bottom: 40px;
  box-shadow: 0 0 2px rgba(255,255,255,0.8), 0 0 8px rgba(255,255,255,0.6), 0 0 20px rgba(255,255,255,0.4), 0 0 40px rgba(255,255,255,0.2), 0 4px 15px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255,255,255,0.5);
  width: 95%;
  margin-left: auto;
  margin-right: auto;
}

.page-title {
  text-align: center;
  font-size: 42px;
  font-weight: 700;
  color: white;
  margin: 0 0 12px 0;
  letter-spacing: -0.5px;
}

.page-subtitle {
  text-align: center;
  font-size: 18px;
  color: #ccc;
  margin: 0;
}

.container { display: flex; gap: 280px; justify-content: center; background: transparent; }

.phone-wrap {
  position: relative;
  width: 520px;
}

/* Power button - right side */
.btn-power {
  position: absolute;
  right: -6px;
  top: 250px;
  width: 6px;
  height: 80px;
  background: linear-gradient(90deg, #2a2a2a, #444);
  border-radius: 0 4px 4px 0;
  box-shadow: 3px 0 6px rgba(0,0,0,0.6);
  z-index: 10;
}

/* Volume buttons - left side */
.btn-vol-up {
  position: absolute;
  left: -6px;
  top: 180px;
  width: 6px;
  height: 65px;
  background: linear-gradient(270deg, #2a2a2a, #444);
  border-radius: 4px 0 0 4px;
  box-shadow: -3px 0 6px rgba(0,0,0,0.6);
  z-index: 10;
}

.btn-vol-down {
  position: absolute;
  left: -6px;
  top: 260px;
  width: 6px;
  height: 65px;
  background: linear-gradient(270deg, #2a2a2a, #444);
  border-radius: 4px 0 0 4px;
  box-shadow: -3px 0 6px rgba(0,0,0,0.6);
  z-index: 10;
}

.phone-frame {
  width: 520px;
  height: 1100px;
  background: #ffffff;
  border-radius: 35px;
  border: 1px solid #2a2a2a;
  box-shadow:
    inset 0 0 30px rgba(255, 255, 255, 0.08),
    0 40px 80px rgba(0, 0, 0, 0.8),
    0 0 0 1px rgba(255,255,255,0.08),
    0 0 60px rgba(0, 132, 255, 0.15);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  position: relative;
}

/* Notch with camera */
.phone-frame::before {
  content: '';
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 160px;
  height: 32px;
  background: #000;
  border-radius: 0 0 20px 20px;
  z-index: 20;
}

/* Front camera dot */
.camera-dot {
  position: absolute;
  top: 18px;
  left: 50%;
  transform: translateX(-50%);
  width: 12px;
  height: 12px;
  background: radial-gradient(circle at 35% 35%, #1a1a2e, #000);
  border-radius: 50%;
  border: 1.5px solid #1a1a1a;
  z-index: 25;
  box-shadow: 0 0 4px rgba(0,150,255,0.2), inset 0 0 3px rgba(0,100,200,0.3);
}

.phone-frame::before {
  content: '';
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 150px;
  height: 28px;
  background: #000;
  border-radius: 0 0 18px 18px;
  z-index: 20;
}

.phone-frame::after {
  content: none;
}

.label {
  text-align: center;
  font-size: 18px;
  font-weight: 700;
  color: #0084ff;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 12px;
  padding: 0 10px;
}

.screen {
  flex: 1;
  background: #ffffff;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  margin: 3px;
  border-radius: 32px;
  box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.9);
  position: relative;
}

.header {
  background: linear-gradient(180deg, #f5f5f5 0%, #f0f0f0 100%);
  padding: 40px 16px 16px 16px;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.header-left { display: flex; align-items: center; gap: 12px; }
.header-avatar { width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #0084ff 0%, #0073e6 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 16px; }
.header-title { font-weight: 600; color: #000000; font-size: 17px; }

.header-title-new { font-weight: 600; color: #333; font-size: 18px; flex: 1; text-align: center; }
.header-close { background: none; border: 1px solid rgba(255,255,255,0.5); color: white; font-size: 20px; width: 28px; height: 28px; border-radius: 50%; cursor: pointer; display: flex; align-items: center; justify-content: center; }

.compose-field { background: linear-gradient(180deg, #f5f5f5 0%, #f0f0f0 100%); padding: 12px 16px; border-bottom: 1px solid #e0e0e0; display: flex; align-items: center; gap: 12px; }
.compose-label { font-weight: 600; color: #333; font-size: 16px; min-width: 35px; }
.compose-input { flex: 1; background: transparent; border: none; color: #000; font-size: 16px; outline: none; padding: 0; }
.header-status { font-size: 13px; color: #666666; }
.header-icons { display: flex; gap: 16px; }
.header-icons button { background: none; border: none; color: #0084ff; cursor: pointer; font-size: 18px; }
.call-btn { background: none; border: none; cursor: pointer; display: flex; align-items: center; }

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: #ffffff;
}

.date-separator {
  text-align: center;
  color: #666;
  font-size: 13px;
  margin: 12px 0;
}

.message {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  animation: slideIn 0.3s ease;
}

@keyframes slideIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.message.sent { justify-content: flex-end; }
.message.sent .avatar { display: none; }

.avatar { width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, #0084ff 0%, #0073e6 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: 600; flex-shrink: 0; }

.message-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.bubble {
  max-width: 280px;
  padding: 12px 16px;
  border-radius: 18px;
  font-size: 15px;
  line-height: 1.4;
  word-wrap: break-word;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
}

.message.received .bubble {
  background: #e8e8e8;
  color: #000000;
  border-radius: 20px 20px 4px 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

.message.sent .bubble {
  background: linear-gradient(135deg, #0084ff 0%, #0073e6 100%);
  color: white;
  border-radius: 20px 4px 20px 20px;
  box-shadow: 0 4px 12px rgba(0, 132, 255, 0.25);
}

.timestamp {
  font-size: 13px;
  color: #666;
  margin-top: 4px;
  padding: 0 4px;
}

.message.sent .timestamp { text-align: right; }
.message.received .timestamp { text-align: left; }

.badge {
  background: #ff2222;
  color: #ffffff;
  padding: 6px 12px;
  border-radius: 10px;
  font-size: 12px;
  font-weight: 700;
  display: block;
  margin-top: 10px;
  width: fit-content;
  box-shadow: 0 3px 8px rgba(255, 34, 34, 0.4);
  letter-spacing: 0.3px;
}

.badge.safe {
  background: #00c853;
  box-shadow: 0 3px 8px rgba(0, 200, 83, 0.4);
}

.input-area {
  background: linear-gradient(180deg, #f5f5f5 0%, #f0f0f0 100%);
  padding: 10px 16px;
  border-top: 1px solid #2a2a2a;
  display: flex;
  gap: 10px;
  align-items: flex-end;
  flex-shrink: 0;
}

.input-icons { display: flex; gap: 8px; }
.input-icons button {
  background: none;
  border: none;
  color: #0084ff;
  cursor: pointer;
  font-size: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.input-field {
  flex: 1;
  background: #f5f5f5;
  border: 1px solid #d0d0d0;
  border-radius: 20px;
  padding: 10px 16px;
  color: #000000;
  font-size: 13px;
  outline: none;
  resize: none;
  max-height: 100px;
}

.input-field::placeholder { color: #999999; }
.input-field:disabled { opacity: 0.4; cursor: not-allowed; }

.send-btn {
  background: linear-gradient(135deg, #0084ff 0%, #0073e6 100%);
  border: none;
  color: white;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  box-shadow: 0 4px 10px rgba(0, 132, 255, 0.3);
  transition: all 0.1s;
}

.send-btn:active {
  transform: scale(0.95);
  box-shadow: 0 2px 5px rgba(255, 149, 0, 0.2);
}

.send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.phone-keyboard {
  display: none;
  flex-direction: column;
  background: #f0f0f0;
  padding: 6px 3px 10px;
  border-top: none;
  flex-shrink: 0;
  gap: 4px;
  width: 100%;
  box-sizing: border-box;
  pointer-events: none;
  opacity: 0;
}

.phone-keyboard.active {
  display: flex;
  max-height: 500px;
  opacity: 1;
  pointer-events: auto;
  border-top: none;
  animation: slideInKeyboard 0.4s ease forwards;
}

@keyframes slideInKeyboard {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.kb-row {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 5px;
  width: 100%;
}

.kb-row.row-9  { grid-template-columns: repeat(9, 1fr); padding: 0 4%; }
.kb-row.row-7  { grid-template-columns: 1.5fr repeat(7, 1fr) 1.5fr; }
.kb-row.row-bot { grid-template-columns: 1.3fr 1.3fr 4fr 1.3fr 1.3fr; }

.kb-key {
  background: #ffffff;
  border: 1px solid #d0d0d0;
  border-radius: 5px;
  height: 40px;
  width: 100%;
  font-size: 14px;
  font-weight: 400;
  color: #000000;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 2px 0 #111;
  transition: background 0.05s;
  user-select: none;
  font-family: 'Roboto', sans-serif;
}

.kb-key:active {
  background: #0084ff;
  box-shadow: 0 1px 0 #004499;
  transform: translateY(1px);
}

.finger-tap {
  position: fixed;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  background: rgba(100, 200, 255, 0.3);
  border: 2px solid rgba(100, 200, 255, 0.8);
  pointer-events: none;
  transform: translate(-50%, -50%) scale(0);
  animation: finger-anim 0.6s ease-out forwards;
  z-index: 9999;
}

@keyframes finger-anim {
  0%   { transform: translate(-50%, -50%) scale(0.2); opacity: 1; }
  50%  { transform: translate(-50%, -50%) scale(1);   opacity: 0.5; }
  100% { transform: translate(-50%, -50%) scale(1.2); opacity: 0; }
}

.kb-func {
  background: #ffffff;
  border: 1px solid #d0d0d0;
  flex: 1.5;
  font-size: 12px;
  color: #000000;
}

.kb-space {
  flex: 5;
  font-size: 13px;
  color: #999999;
}

.kb-enter {
  background: #ffffff;
  border: 1px solid #d0d0d0;
  flex: 2;
  font-size: 12px;
  color: #0084ff;
}

.kb-emoji {
  background: #ffffff;
  border: 1px solid #d0d0d0;
  flex: 1.5;
  font-size: 18px;
  color: #ffa500;
}

.kb-mic {
  background: #ffffff;
  border: 1px solid #d0d0d0;
  flex: 1.5;
  font-size: 16px;
  color: #0084ff;
}

.kb-func {
  background: #ffffff;
  border: 1px solid #d0d0d0;
  flex: 1.3;
  font-size: 11px;
  color: #000000;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }

/* Dark Mode Styles */
.phone-frame.dark-mode {
  background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%);
}

.phone-frame.dark-mode .screen {
  background: #1e1e1e;
}

.phone-frame.dark-mode .messages {
  background: #1e1e1e;
}

.phone-frame.dark-mode .header {
  background: linear-gradient(180deg, #1a1a1a 0%, #121212 100%);
  border-bottom-color: #2a2a2a;
}

.phone-frame.dark-mode .header-title {
  color: white;
}

.phone-frame.dark-mode .header-title-new {
  color: white;
}

.phone-frame.dark-mode .header-status {
  color: #888;
}

.phone-frame.dark-mode .compose-field {
  background: linear-gradient(180deg, #1a1a1a 0%, #121212 100%);
  border-bottom-color: #2a2a2a;
}

.phone-frame.dark-mode .compose-label {
  color: #ccc;
}

.phone-frame.dark-mode .compose-input {
  color: #fff;
}

.phone-frame.dark-mode .input-area {
  background: linear-gradient(180deg, #1a1a1a 0%, #121212 100%);
}

.phone-frame.dark-mode .input-field {
  background: #2a2a2a;
  border-color: #3a3a3a;
  color: #e0e0e0;
}

.phone-frame.dark-mode .input-field::placeholder {
  color: #666;
}

.phone-frame.dark-mode .message.received .bubble {
  background: #2d2d2d;
  color: #f0f0f0;
}

.phone-frame.dark-mode .phone-keyboard {
  background: #282828;
}

.phone-frame.dark-mode .phone-keyboard.active {
  border-top-color: #444;
}

.phone-frame.dark-mode .kb-key {
  background: #404040;
  border-color: #555;
  color: white;
}

.phone-frame.dark-mode .kb-key:active {
  background: #0084ff;
}

.phone-frame.dark-mode .kb-func,
.phone-frame.dark-mode .kb-enter,
.phone-frame.dark-mode .kb-emoji,
.phone-frame.dark-mode .kb-mic {
  background: #2a2a2a;
}

</style>
</head>
<body>
<div class="wrapper">
  <div class="header-box">
    <h1 class="page-title">📧 SMS Phishing Detection ⚠️</h1>
    <p class="page-subtitle">Real-time threat detection & analysis</p>
  </div>

  <div class="container">
  <!-- SENDER SCREEN -->
  <div>
    <div class="label">Sender</div>
    <div class="phone-wrap">
      <div class="btn-vol-up"></div>
      <div class="btn-vol-down"></div>
      <div class="btn-power"></div>
    <div class="phone-frame">
      <div class="camera-dot"></div>
      <div class="screen">
        <div class="header" style="flex-direction: column; justify-content: center;">
          <div class="header-title-new">New Message</div>
          <button class="header-close" onclick="alert('Close Message')" style="position: absolute; right: 16px; top: 16px;">✕</button>
        </div>

        <div class="compose-field">
          <div class="compose-label">To:</div>
          <input type="text" id="recipientField" class="compose-input" placeholder="Enter recipient">
        </div>

        <div class="messages" id="senderMessages">
        </div>

        <div class="input-area">
          <div class="input-icons">
            <button>➕</button>
          </div>
          <input type="text" id="senderInput" class="input-field" placeholder="Chat message" onclick="showKeyboard(); event.stopPropagation();">
          <button class="send-btn" onclick="sendMessage()">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>

        <div class="phone-keyboard" id="phoneKeyboard">
          <div id="letterpad">
          <div class="kb-row">
            <button class="kb-key" onclick="addChar('q')">q</button>
            <button class="kb-key" onclick="addChar('w')">w</button>
            <button class="kb-key" onclick="addChar('e')">e</button>
            <button class="kb-key" onclick="addChar('r')">r</button>
            <button class="kb-key" onclick="addChar('t')">t</button>
            <button class="kb-key" onclick="addChar('y')">y</button>
            <button class="kb-key" onclick="addChar('u')">u</button>
            <button class="kb-key" onclick="addChar('i')">i</button>
            <button class="kb-key" onclick="addChar('o')">o</button>
            <button class="kb-key" onclick="addChar('p')">p</button>
          </div>
          <div class="kb-row row-9">
            <button class="kb-key" onclick="addChar('a')">a</button>
            <button class="kb-key" onclick="addChar('s')">s</button>
            <button class="kb-key" onclick="addChar('d')">d</button>
            <button class="kb-key" onclick="addChar('f')">f</button>
            <button class="kb-key" onclick="addChar('g')">g</button>
            <button class="kb-key" onclick="addChar('h')">h</button>
            <button class="kb-key" onclick="addChar('j')">j</button>
            <button class="kb-key" onclick="addChar('k')">k</button>
            <button class="kb-key" onclick="addChar('l')">l</button>
          </div>
          <div class="kb-row row-7">
            <button class="kb-key kb-func">⇧</button>
            <button class="kb-key" onclick="addChar('z')">z</button>
            <button class="kb-key" onclick="addChar('x')">x</button>
            <button class="kb-key" onclick="addChar('c')">c</button>
            <button class="kb-key" onclick="addChar('v')">v</button>
            <button class="kb-key" onclick="addChar('b')">b</button>
            <button class="kb-key" onclick="addChar('n')">n</button>
            <button class="kb-key" onclick="addChar('m')">m</button>
            <button class="kb-key kb-func" onclick="delChar()">⌫</button>
          </div>
          <div class="kb-row row-bot">
            <button class="kb-key kb-emoji" onclick="toggleEmojiPicker()">😊</button>
            <button class="kb-key kb-func" onclick="toggleNumpad()">?123</button>
            <button class="kb-key kb-space" onclick="addChar(' ')">space</button>
            <button class="kb-key kb-mic">🎤</button>
            <button class="kb-key kb-enter" onclick="sendMessage()">↵</button>
          </div>

          </div><!-- end letterpad -->

          <!-- NUMBER PAD -->
          <div id="numpad" style="display:none; flex-direction:column; gap:4px;">
            <div class="kb-row">
              <button class="kb-key" onclick="addChar('1')">1</button>
              <button class="kb-key" onclick="addChar('2')">2</button>
              <button class="kb-key" onclick="addChar('3')">3</button>
              <button class="kb-key" onclick="addChar('4')">4</button>
              <button class="kb-key" onclick="addChar('5')">5</button>
              <button class="kb-key" onclick="addChar('6')">6</button>
              <button class="kb-key" onclick="addChar('7')">7</button>
              <button class="kb-key" onclick="addChar('8')">8</button>
              <button class="kb-key" onclick="addChar('9')">9</button>
              <button class="kb-key" onclick="addChar('0')">0</button>
            </div>
            <div class="kb-row">
              <button class="kb-key" onclick="addChar('@')">@</button>
              <button class="kb-key" onclick="addChar('#')">#</button>
              <button class="kb-key" onclick="addChar('$')">$</button>
              <button class="kb-key" onclick="addChar('%')">%</button>
              <button class="kb-key" onclick="addChar('&')">&</button>
              <button class="kb-key" onclick="addChar('-')">-</button>
              <button class="kb-key" onclick="addChar('+')">+</button>
              <button class="kb-key" onclick="addChar('(')">&#40;</button>
              <button class="kb-key" onclick="addChar(')')">&#41;</button>
              <button class="kb-key kb-back" onclick="delChar()">⌫</button>
            </div>
            <div class="kb-row">
              <button class="kb-key kb-func" onclick="toggleNumpad()">ABC</button>
              <button class="kb-key" onclick="addChar('.')">.</button>
              <button class="kb-key" onclick="addChar(',')">','</button>
              <button class="kb-key" onclick="addChar('?')">?</button>
              <button class="kb-key" onclick="addChar('!')">!</button>
              <button class="kb-key" onclick="addChar('\"')">"</button>
              <button class="kb-key kb-enter" onclick="sendMessage()">↵</button>
            </div>
          </div>

          <!-- EMOJI PICKER -->
          <div id="emojipicker" style="display:none; flex-direction:column; gap:4px;">
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addChar('😀')">😀</button>
              <button class="kb-key" onclick="addChar('😃')">😃</button>
              <button class="kb-key" onclick="addChar('😄')">😄</button>
              <button class="kb-key" onclick="addChar('😂')">😂</button>
              <button class="kb-key" onclick="addChar('😍')">😍</button>
              <button class="kb-key" onclick="addChar('🥰')">🥰</button>
              <button class="kb-key" onclick="addChar('😍')">😍</button>
              <button class="kb-key" onclick="addChar('😘')">😘</button>
            </div>
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addChar('😡')">😡</button>
              <button class="kb-key" onclick="addChar('😠')">😠</button>
              <button class="kb-key" onclick="addChar('😔')">😔</button>
              <button class="kb-key" onclick="addChar('😢')">😢</button>
              <button class="kb-key" onclick="addChar('😭')">😭</button>
              <button class="kb-key" onclick="addChar('😱')">😱</button>
              <button class="kb-key" onclick="addChar('😲')">😲</button>
              <button class="kb-key" onclick="addChar('😳')">😳</button>
            </div>
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addChar('🎉')">🎉</button>
              <button class="kb-key" onclick="addChar('🎊')">🎊</button>
              <button class="kb-key" onclick="addChar('🔥')">🔥</button>
              <button class="kb-key" onclick="addChar('✨')">✨</button>
              <button class="kb-key" onclick="addChar('💯')">💯</button>
              <button class="kb-key" onclick="addChar('👍')">👍</button>
              <button class="kb-key" onclick="addChar('❤️')">❤️</button>
              <button class="kb-key" onclick="addChar('💪')">💪</button>
            </div>
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addChar('🚀')">🚀</button>
              <button class="kb-key" onclick="addChar('⚡')">⚡</button>
              <button class="kb-key" onclick="addChar('🎯')">🎯</button>
              <button class="kb-key" onclick="addChar('🤔')">🤔</button>
              <button class="kb-key" onclick="addChar('😎')">😎</button>
              <button class="kb-key" onclick="addChar('🤓')">🤓</button>
              <button class="kb-key" onclick="addChar('👀')">👀</button>
              <button class="kb-key" onclick="addChar('🙏')">🙏</button>
            </div>
            <div class="kb-row">
              <button class="kb-key kb-func" style="flex:2;" onclick="toggleEmojiPicker()">ABC</button>
              <button class="kb-key kb-back" style="flex:1;" onclick="delChar()">⌫</button>
              <button class="kb-key kb-enter" style="flex:2;" onclick="sendMessage()">↵</button>
            </div>
          </div>
        </div>

      </div>
    </div>
    </div><!-- end phone-wrap sender -->
  </div>

  <!-- RECEIVER SCREEN (DETECTION) -->
  <div>
    <div class="label">Receiver</div>
    <div class="phone-wrap">
      <div class="btn-vol-up"></div>
      <div class="btn-vol-down"></div>
      <div class="btn-power"></div>
    <div class="phone-frame">
      <div class="camera-dot"></div>
      <div class="screen">
        <div class="header">
          <div class="header-left">
            <div class="header-avatar">S</div>
            <div>
              <div class="header-title">Spam Detector</div>
            </div>
          </div>
          <div class="header-icons">
            <button class="call-btn">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="#00c853"><path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1-9.4 0-17-7.6-17-17 0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.3 0 .7-.2 1L6.6 10.8z"/></svg>
            </button>
            <button>🔍</button>
            <button>⋮</button>
          </div>
        </div>

        <div class="messages" id="receiverMessages">
        </div>

        <div class="input-area">
          <div class="input-icons">
            <button>➕</button>
          </div>
          <input type="text" id="receiverInput" class="input-field" placeholder="Chat message" onclick="showReceiverKeyboard(); event.stopPropagation();">
          <button class="send-btn" onclick="sendReceiverMessage()">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>

        <div class="phone-keyboard" id="receiverKeyboard">
          <div id="receiverLetterpad">
          <div class="kb-row">
            <button class="kb-key" onclick="addReceiverChar('q')">q</button>
            <button class="kb-key" onclick="addReceiverChar('w')">w</button>
            <button class="kb-key" onclick="addReceiverChar('e')">e</button>
            <button class="kb-key" onclick="addReceiverChar('r')">r</button>
            <button class="kb-key" onclick="addReceiverChar('t')">t</button>
            <button class="kb-key" onclick="addReceiverChar('y')">y</button>
            <button class="kb-key" onclick="addReceiverChar('u')">u</button>
            <button class="kb-key" onclick="addReceiverChar('i')">i</button>
            <button class="kb-key" onclick="addReceiverChar('o')">o</button>
            <button class="kb-key" onclick="addReceiverChar('p')">p</button>
          </div>
          <div class="kb-row row-9">
            <button class="kb-key" onclick="addReceiverChar('a')">a</button>
            <button class="kb-key" onclick="addReceiverChar('s')">s</button>
            <button class="kb-key" onclick="addReceiverChar('d')">d</button>
            <button class="kb-key" onclick="addReceiverChar('f')">f</button>
            <button class="kb-key" onclick="addReceiverChar('g')">g</button>
            <button class="kb-key" onclick="addReceiverChar('h')">h</button>
            <button class="kb-key" onclick="addReceiverChar('j')">j</button>
            <button class="kb-key" onclick="addReceiverChar('k')">k</button>
            <button class="kb-key" onclick="addReceiverChar('l')">l</button>
          </div>
          <div class="kb-row row-7">
            <button class="kb-key kb-func">⇧</button>
            <button class="kb-key" onclick="addReceiverChar('z')">z</button>
            <button class="kb-key" onclick="addReceiverChar('x')">x</button>
            <button class="kb-key" onclick="addReceiverChar('c')">c</button>
            <button class="kb-key" onclick="addReceiverChar('v')">v</button>
            <button class="kb-key" onclick="addReceiverChar('b')">b</button>
            <button class="kb-key" onclick="addReceiverChar('n')">n</button>
            <button class="kb-key" onclick="addReceiverChar('m')">m</button>
            <button class="kb-key kb-func" onclick="delReceiverChar()">⌫</button>
          </div>
          <div class="kb-row row-bot">
            <button class="kb-key kb-emoji" onclick="toggleReceiverEmojiPicker()">😊</button>
            <button class="kb-key kb-func" onclick="toggleReceiverNumpad()">?123</button>
            <button class="kb-key kb-space" onclick="addReceiverChar(' ')">space</button>
            <button class="kb-key kb-mic">🎤</button>
            <button class="kb-key kb-enter" onclick="sendReceiverMessage()">↵</button>
          </div>

          </div><!-- end letterpad -->

          <!-- NUMBER PAD -->
          <div id="receiverNumpad" style="display:none; flex-direction:column; gap:4px;">
            <div class="kb-row">
              <button class="kb-key" onclick="addReceiverChar('1')">1</button>
              <button class="kb-key" onclick="addReceiverChar('2')">2</button>
              <button class="kb-key" onclick="addReceiverChar('3')">3</button>
              <button class="kb-key" onclick="addReceiverChar('4')">4</button>
              <button class="kb-key" onclick="addReceiverChar('5')">5</button>
              <button class="kb-key" onclick="addReceiverChar('6')">6</button>
              <button class="kb-key" onclick="addReceiverChar('7')">7</button>
              <button class="kb-key" onclick="addReceiverChar('8')">8</button>
              <button class="kb-key" onclick="addReceiverChar('9')">9</button>
              <button class="kb-key" onclick="addReceiverChar('0')">0</button>
            </div>
            <div class="kb-row">
              <button class="kb-key" onclick="addReceiverChar('@')">@</button>
              <button class="kb-key" onclick="addReceiverChar('#')">#</button>
              <button class="kb-key" onclick="addReceiverChar('$')">$</button>
              <button class="kb-key" onclick="addReceiverChar('%')">%</button>
              <button class="kb-key" onclick="addReceiverChar('&')">&</button>
              <button class="kb-key" onclick="addReceiverChar('-')">-</button>
              <button class="kb-key" onclick="addReceiverChar('+')">+</button>
              <button class="kb-key" onclick="addReceiverChar('(')">&#40;</button>
              <button class="kb-key" onclick="addReceiverChar(')')">&#41;</button>
              <button class="kb-key kb-back" onclick="delReceiverChar()">⌫</button>
            </div>
            <div class="kb-row">
              <button class="kb-key kb-func" onclick="toggleReceiverNumpad()">ABC</button>
              <button class="kb-key" onclick="addReceiverChar('.')">.</button>
              <button class="kb-key" onclick="addReceiverChar(',')">','</button>
              <button class="kb-key" onclick="addReceiverChar('?')">?</button>
              <button class="kb-key" onclick="addReceiverChar('!')">!</button>
              <button class="kb-key" onclick="addReceiverChar('\"')">"</button>
              <button class="kb-key kb-enter" onclick="sendReceiverMessage()">↵</button>
            </div>
          </div>

          <!-- EMOJI PICKER -->
          <div id="receiverEmojipicker" style="display:none; flex-direction:column; gap:4px;">
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addReceiverChar('😀')">😀</button>
              <button class="kb-key" onclick="addReceiverChar('😃')">😃</button>
              <button class="kb-key" onclick="addReceiverChar('😄')">😄</button>
              <button class="kb-key" onclick="addReceiverChar('😂')">😂</button>
              <button class="kb-key" onclick="addReceiverChar('😍')">😍</button>
              <button class="kb-key" onclick="addReceiverChar('🥰')">🥰</button>
              <button class="kb-key" onclick="addReceiverChar('😍')">😍</button>
              <button class="kb-key" onclick="addReceiverChar('😘')">😘</button>
            </div>
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addReceiverChar('😡')">😡</button>
              <button class="kb-key" onclick="addReceiverChar('😠')">😠</button>
              <button class="kb-key" onclick="addReceiverChar('😔')">😔</button>
              <button class="kb-key" onclick="addReceiverChar('😢')">😢</button>
              <button class="kb-key" onclick="addReceiverChar('😭')">😭</button>
              <button class="kb-key" onclick="addReceiverChar('😱')">😱</button>
              <button class="kb-key" onclick="addReceiverChar('😲')">😲</button>
              <button class="kb-key" onclick="addReceiverChar('😳')">😳</button>
            </div>
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addReceiverChar('🎉')">🎉</button>
              <button class="kb-key" onclick="addReceiverChar('🎊')">🎊</button>
              <button class="kb-key" onclick="addReceiverChar('🔥')">🔥</button>
              <button class="kb-key" onclick="addReceiverChar('✨')">✨</button>
              <button class="kb-key" onclick="addReceiverChar('💯')">💯</button>
              <button class="kb-key" onclick="addReceiverChar('👍')">👍</button>
              <button class="kb-key" onclick="addReceiverChar('❤️')">❤️</button>
              <button class="kb-key" onclick="addReceiverChar('💪')">💪</button>
            </div>
            <div class="kb-row" style="grid-template-columns: repeat(8, 1fr);">
              <button class="kb-key" onclick="addReceiverChar('🚀')">🚀</button>
              <button class="kb-key" onclick="addReceiverChar('⚡')">⚡</button>
              <button class="kb-key" onclick="addReceiverChar('🎯')">🎯</button>
              <button class="kb-key" onclick="addReceiverChar('🤔')">🤔</button>
              <button class="kb-key" onclick="addReceiverChar('😎')">😎</button>
              <button class="kb-key" onclick="addReceiverChar('🤓')">🤓</button>
              <button class="kb-key" onclick="addReceiverChar('👀')">👀</button>
              <button class="kb-key" onclick="addReceiverChar('🙏')">🙏</button>
            </div>
            <div class="kb-row">
              <button class="kb-key kb-func" style="flex:2;" onclick="toggleReceiverEmojiPicker()">ABC</button>
              <button class="kb-key kb-back" style="flex:1;" onclick="delReceiverChar()">⌫</button>
              <button class="kb-key kb-enter" style="flex:2;" onclick="sendReceiverMessage()">↵</button>
            </div>
          </div>
        </div>
      </div>
    </div>
    </div><!-- end phone-wrap receiver -->
  </div>

  <!-- MODEL SELECTOR PANEL -->
  <div class="model-selector">
    <div class="theme-toggle-container">
      <label class="theme-toggle-label">Theme</label>
      <div class="toggle-switch" onclick="toggleDarkModeUI()">
        <input type="checkbox" id="darkModeToggle">
        <span class="toggle-slider"></span>
        <span class="toggle-label-light">Light</span>
        <span class="toggle-label-dark">Dark</span>
      </div>
    </div>
    <label><span class="model-dot"></span>Detection Model</label>
    <select id="modelSelect" onchange="updateModel(this.value)">
      <option value="bert">💡 BERT</option>
      <option value="distilbert">⚡ DistilBERT</option>
    </select>
    <div class="model-info">
      <div class="model-info-row">
        <span class="info-label">Accuracy</span>
        <span class="info-value" id="infoAccuracy">98.2%</span>
      </div>
      <div class="model-info-row">
        <span class="info-label">Speed</span>
        <span class="info-value" id="infoSpeed">~120ms</span>
      </div>
      <div class="model-info-row">
        <span class="info-label">Parameters</span>
        <span class="info-value" id="infoParams">110M</span>
      </div>
      <div class="model-info-row">
        <span class="info-label">Type</span>
        <span class="info-value" id="infoType">Transformer</span>
      </div>
      <div class="model-status">
        <span class="model-dot"></span> Model Ready
      </div>
    </div>
    <div style="border-top: 1px solid #333; margin-top: 16px; padding-top: 16px;">
      <label style="display: block; font-size: 11px; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Prediction Status</label>
      <div style="background: linear-gradient(135deg, #1e3a1f 0%, #2d5a32 100%); border-radius: 12px; padding: 16px; text-align: center; margin-bottom: 16px; border: 1px solid #22c55e;">
        <div style="font-size: 28px; font-weight: 700; color: #22c55e; margin-bottom: 8px;" id="predictionStatus">Safe</div>
      </div>
      <label style="display: block; font-size: 11px; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Confidence Score</label>
      <div style="background: linear-gradient(135deg, #1e3a1f 0%, #2d5a32 100%); border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #22c55e;">
        <div style="font-size: 48px; font-weight: 700; color: #22c55e; margin-bottom: 4px;" id="confidenceScore">0%</div>
      </div>
    </div>
  </div>

  </div>
</div>

<script>
// CONFIG_PLACEHOLDER
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

function playSendSound() {
  const t = audioCtx.currentTime;
  // WhatsApp: soft "doo-dum" two quick notes
  [[0, 783.99], [0.1, 659.25]].forEach(([delay, freq]) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.type = 'sine';
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.001, t + delay);
    gain.gain.linearRampToValueAtTime(0.4, t + delay + 0.01);
    gain.gain.setValueAtTime(0.4, t + delay + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.001, t + delay + 0.15);
    osc.start(t + delay);
    osc.stop(t + delay + 0.16);
  });
}

function playSafeSound() {
  [523, 659, 784].forEach((freq, i) => {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.type = 'sine';
    const t = audioCtx.currentTime + i * 0.12;
    osc.frequency.setValueAtTime(freq, t);
    gain.gain.setValueAtTime(0.25, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    osc.start(t);
    osc.stop(t + 0.22);
  });
}

function playSmishingAlert() {
  for (let i = 0; i < 3; i++) {
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.type = 'square';
    const t = audioCtx.currentTime + i * 0.25;
    osc.frequency.setValueAtTime(880, t);
    osc.frequency.setValueAtTime(440, t + 0.1);
    gain.gain.setValueAtTime(0.3, t);
    gain.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    osc.start(t);
    osc.stop(t + 0.22);
  }
}

function getTime() {
  const now = new Date();
  return now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

function isPhishing(text) {
  // Model detection happens server-side
  return false;
}

function toggleDarkMode(enabled) {
  const phoneFrames = document.querySelectorAll('.phone-frame');
  phoneFrames.forEach(frame => {
    if (enabled) {
      frame.classList.add('dark-mode');
    } else {
      frame.classList.remove('dark-mode');
    }
  });
  localStorage.setItem('darkMode', enabled);
}

function toggleDarkModeUI() {
  const toggle = document.getElementById('darkModeToggle');
  if (toggle) {
    toggle.checked = !toggle.checked;
    toggleDarkMode(toggle.checked);
  }
}

// Load dark mode preference from localStorage (default to true/dark mode)
document.addEventListener('DOMContentLoaded', function() {
  setTimeout(function() {
    const darkModeSaved = localStorage.getItem('darkMode');
    const darkModeEnabled = darkModeSaved === null ? true : darkModeSaved === 'true';
    const toggle = document.getElementById('darkModeToggle');
    if (toggle) {
      toggle.checked = darkModeEnabled;
      toggleDarkMode(darkModeEnabled);
    }
  }, 300);
});

async function sendMessage() {
  const inp = document.getElementById('senderInput');
  const text = inp.value.trim();
  if (!text) return;

  const time = getTime();

  // Add to sender
  const senderDiv = document.getElementById('senderMessages');
  const sentMsg = `<div class="message sent">
    <div class="message-group">
      <div class="bubble">${text}</div>
      <div class="timestamp">${time} ✓✓</div>
    </div>
  </div>`;
  senderDiv.insertAdjacentHTML('beforeend', sentMsg);
  senderDiv.scrollTop = senderDiv.scrollHeight;
  playSendSound();

  // Submit message to Python for detection
  const modelSelect = document.getElementById('modelSelect');
  const selectedModel = modelSelect.value || appConfig.default_model;

  // Send to Python backend
  submitMessageToPython(text, selectedModel);

  // Wait briefly for Python to process, then check result
  setTimeout(() => {
    // Get latest result from page data
    const resultDiv = document.getElementById('apiResultData');
    let phish = false;
    let confidence = 0;

    if (resultDiv && resultDiv.textContent) {
      try {
        const data = JSON.parse(resultDiv.textContent);
        if (data.result) {
          phish = data.result.is_phishing || false;
          confidence = data.result.confidence || 0;

          if (phish) playSmishingAlert();

          document.getElementById('confidenceScore').textContent = confidence + '%';
          document.getElementById('predictionStatus').textContent = phish ? 'Smishing' : 'Safe';
          document.getElementById('predictionStatus').style.color = phish ? '#ef4444' : '#22c55e';
        }
      } catch (e) {
        console.error('Parse error:', e);
      }
    }

    // Add to receiver
    const receiverDiv = document.getElementById('receiverMessages');
    const receivedMsg = `<div class="message received">
      <div class="avatar">?</div>
      <div class="message-group">
        <div style="display:flex; align-items:center; gap:8px;">
          <div class="bubble">${text}</div>
          ${phish ? `<span style="font-size:28px; line-height:1;" title="SMISHING DETECTED">🚩</span>` : ''}
        </div>
        <div class="timestamp">${time}</div>
      </div>
    </div>`;
    receiverDiv.insertAdjacentHTML('beforeend', receivedMsg);
    receiverDiv.scrollTop = receiverDiv.scrollHeight;
  }, 500);

  inp.value = '';
  document.getElementById('phoneKeyboard').classList.remove('active');
}

async function sendReceiverMessage() {
  const inp = document.getElementById('receiverInput');
  const text = inp.value.trim();
  if (!text) return;

  const time = getTime();

  // Add to receiver
  const receiverDiv = document.getElementById('receiverMessages');
  const sentMsg = `<div class="message sent">
    <div class="message-group">
      <div class="bubble">${text}</div>
      <div class="timestamp">${time} ✓✓</div>
    </div>
  </div>`;
  receiverDiv.insertAdjacentHTML('beforeend', sentMsg);
  receiverDiv.scrollTop = receiverDiv.scrollHeight;
  playSendSound();

  // Submit message to Python for detection
  const modelSelect = document.getElementById('modelSelect');
  const selectedModel = modelSelect.value || appConfig.default_model;

  // Send to Python backend
  submitMessageToPython(text, selectedModel);

  // Wait briefly for Python to process
  setTimeout(() => {
    const resultDiv = document.getElementById('apiResultData');
    let confidence = 0;

    if (resultDiv && resultDiv.textContent) {
      try {
        const data = JSON.parse(resultDiv.textContent);
        if (data.result) {
          confidence = data.result.confidence || 0;
        }
      } catch (e) {
        console.error('Parse error:', e);
      }
    }

    document.getElementById('confidenceScore').textContent = confidence + '%';
    document.getElementById('predictionStatus').textContent = 'Safe';
    document.getElementById('predictionStatus').style.color = '#22c55e';

    // Add to sender
    const senderDiv = document.getElementById('senderMessages');
    const receivedMsg = `<div class="message received">
      <div class="avatar">?</div>
      <div class="message-group">
        <div class="bubble">${text}</div>
        <div class="timestamp">${time}</div>
      </div>
    </div>`;
    senderDiv.insertAdjacentHTML('beforeend', receivedMsg);
    senderDiv.scrollTop = senderDiv.scrollHeight;
  }, 500);

  inp.value = '';
  document.getElementById('receiverKeyboard').classList.remove('active');
}

function addReceiverChar(c) {
  const inp = document.getElementById('receiverInput');
  inp.value += c;
  inp.focus();
}

function delReceiverChar() {
  const inp = document.getElementById('receiverInput');
  inp.value = inp.value.slice(0, -1);
}

function showTap(x, y) {
  const tap = document.createElement('div');
  tap.className = 'finger-tap';
  tap.style.left = x + 'px';
  tap.style.top = y + 'px';
  document.body.appendChild(tap);
  setTimeout(() => tap.remove(), 650);
}

document.addEventListener('click', function(e) {
  if (e.target.closest('.kb-key')) {
    showTap(e.clientX, e.clientY);
  }
});

document.addEventListener('keydown', function(e) {
  const senderInp = document.getElementById('senderInput');
  const receiverInp = document.getElementById('receiverInput');
  const inp = document.activeElement === senderInp ? senderInp : (document.activeElement === receiverInp ? receiverInp : null);
  if (!inp) return;

  const isReceiver = inp === receiverInp;

  const pressedKey = e.key.toLowerCase();

  // Find matching key button in phone keyboard
  const allKeys = document.querySelectorAll('.kb-key');
  let matchedKey = null;

  allKeys.forEach(btn => {
    if (btn.textContent.trim().toLowerCase() === pressedKey) {
      matchedKey = btn;
    }
  });

  // Backspace
  if (e.key === 'Backspace') {
    allKeys.forEach(btn => {
      if (btn.textContent.trim() === '⌫') matchedKey = btn;
    });
  }

  if (matchedKey) {
    if (isReceiver) {
      document.getElementById('receiverKeyboard').classList.add('active');
    } else {
      showKeyboard();
    }
    const rect = matchedKey.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    showTap(x, y);

    // Flash the key
    matchedKey.style.background = '#0084ff';
    setTimeout(() => matchedKey.style.background = '', 150);
  }
});

function toggleNumpad() {
  const numpad = document.getElementById('numpad');
  const letterpad = document.getElementById('letterpad');
  const emojipicker = document.getElementById('emojipicker');
  const showingNum = numpad.style.display === 'flex';
  if (showingNum) {
    numpad.style.display = 'none';
    emojipicker.style.display = 'none';
    letterpad.style.display = 'block';
  } else {
    numpad.style.display = 'flex';
    numpad.style.flexDirection = 'column';
    emojipicker.style.display = 'none';
    letterpad.style.display = 'none';
  }
}

function toggleEmojiPicker() {
  const letterpad = document.getElementById('letterpad');
  const numpad = document.getElementById('numpad');
  const emojipicker = document.getElementById('emojipicker');

  const emojiShowing = emojipicker.style.display === 'flex';

  if (emojiShowing) {
    // Go back to keyboard (letterpad)
    emojipicker.style.display = 'none';
    numpad.style.display = 'none';
    letterpad.style.display = 'block';
  } else {
    // Show emoji picker
    letterpad.style.display = 'none';
    numpad.style.display = 'none';
    emojipicker.style.display = 'flex';
  }
}

function addChar(c) {
  const inp = document.getElementById('senderInput');
  inp.value += c;
  inp.focus();
}

function delChar() {
  const inp = document.getElementById('senderInput');
  inp.value = inp.value.slice(0, -1);
  inp.focus();
}

function toggleReceiverNumpad() {
  const numpad = document.getElementById('receiverNumpad');
  const letterpad = document.getElementById('receiverLetterpad');
  const emojipicker = document.getElementById('receiverEmojipicker');
  const showingNum = numpad.style.display === 'flex';
  if (showingNum) {
    numpad.style.display = 'none';
    emojipicker.style.display = 'none';
    letterpad.style.display = 'block';
  } else {
    numpad.style.display = 'flex';
    numpad.style.flexDirection = 'column';
    emojipicker.style.display = 'none';
    letterpad.style.display = 'none';
  }
}

function toggleReceiverEmojiPicker() {
  const letterpad = document.getElementById('receiverLetterpad');
  const numpad = document.getElementById('receiverNumpad');
  const emojipicker = document.getElementById('receiverEmojipicker');

  const emojiShowing = emojipicker.style.display === 'flex';

  if (emojiShowing) {
    // Go back to keyboard (letterpad)
    emojipicker.style.display = 'none';
    numpad.style.display = 'none';
    letterpad.style.display = 'block';
  } else {
    // Show emoji picker
    letterpad.style.display = 'none';
    numpad.style.display = 'none';
    emojipicker.style.display = 'flex';
  }
}

function showKeyboard() {
  document.getElementById('phoneKeyboard').classList.add('active');
}

function showReceiverKeyboard() {
  document.getElementById('receiverKeyboard').classList.add('active');
}

const modelData = {
  bert: { accuracy: '98.2%', speed: '~120ms', params: '110M', type: 'Transformer' },
  distilbert: { accuracy: '96.8%', speed: '~60ms', params: '66M', type: 'Distilled' }
};

function updateModel(val) {
  const d = modelData[val];
  document.getElementById('infoAccuracy').textContent = d.accuracy;
  document.getElementById('infoSpeed').textContent = d.speed;
  document.getElementById('infoParams').textContent = d.params;
  document.getElementById('infoType').textContent = d.type;
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('senderInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendMessage();
    }
  });

  document.getElementById('receiverInput').addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendReceiverMessage();
    }
  });

  let initialMessageShown = false;
  document.getElementById('recipientField').addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && e.target.value.trim() !== '' && !initialMessageShown) {
      e.preventDefault();
      initialMessageShown = true;
      const senderMessages = document.getElementById('senderMessages');
      const initialMsg = `<div class="message sent">
        <div class="message-group">
          <div class="bubble">Hi, check this out!</div>
          <div class="timestamp">10:30 AM ✓✓</div>
        </div>
      </div>`;
      senderMessages.insertAdjacentHTML('beforeend', initialMsg);

      const receiverMessages = document.getElementById('receiverMessages');
      const receiverMsg = `<div class="message received">
        <div class="avatar">?</div>
        <div class="message-group">
          <div class="bubble">Hi, check this out!</div>
          <div class="timestamp">10:30 AM</div>
        </div>
      </div>`;
      receiverMessages.insertAdjacentHTML('beforeend', receiverMsg);
    }
  });
});

function hideKeyboard(e) {
  if (!e.target.closest('#phoneKeyboard') && !e.target.closest('.input-field')) {
    document.getElementById('phoneKeyboard').classList.remove('active');
  }
}

document.addEventListener('click', hideKeyboard);
</script>
</body>
</html>
"""

# Inject config and API data into HTML
config_js = f"""
const appConfig = {{
  bert_token: '{bert_token}',
  bert_model: '{bert_model}',
  distilbert_token: '{distilbert_token}',
  distilbert_model: '{distilbert_model}',
  default_model: '{default_model}'
}};
const apiResult = {api_json};

function submitMessageToPython(text, model) {{
  const input = document.querySelector('input[data-testid="TextInput"]');
  if (input) {{
    input.value = text;
    input.dispatchEvent(new Event('input', {{ bubbles: true }}));
    input.dispatchEvent(new Event('change', {{ bubbles: true }}));
  }}
}}
"""
html_content = html_content.replace('// CONFIG_PLACEHOLDER', config_js)

# Add hidden data div before closing body tag
hidden_div = f'<div id="apiResultData" style="display:none;">{api_json}</div>'
html_content = html_content.replace('</body>', f'{hidden_div}\n</body>')

html(html_content, height=1600)
