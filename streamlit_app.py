import streamlit as st
from streamlit.components.v1 import html
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="SMS Phishing Detection", layout="wide", initial_sidebar_state="collapsed")

# Load secrets - supports two models (bert, distilbert)
default_model = st.secrets.get("default_model", "bert")

def get_model_config(name):
    cfg = st.secrets.get(name, {})
    return cfg.get("repo_id", ""), cfg.get("token", "")

# Initialize session state
if 'last_result' not in st.session_state or st.session_state.last_result is None:
    st.session_state.last_result = {'prediction': 'Waiting', 'confidence': 0, 'is_phishing': False}
if 'last_message' not in st.session_state:
    st.session_state.last_message = ""
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = default_model

@st.cache_resource
def load_model(model_name):
    repo_id, token = get_model_config(model_name)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=token)
    model.eval()
    return tokenizer, model

@st.cache_data
def get_model_info(model_name):
    """Get real model info from loaded model"""
    try:
        _, model = load_model(model_name)
        params = model.num_parameters()
        params_str = f"{params/1e6:.0f}M" if params >= 1e6 else f"{params/1e3:.0f}K"
        model_type = model.config.model_type.upper() if hasattr(model.config, 'model_type') else "Unknown"
        arch = model.config.architectures[0] if hasattr(model.config, 'architectures') and model.config.architectures else model_type
        return {
            'params': params_str,
            'type': arch,
            'model_type': model_type
        }
    except Exception:
        return {'params': 'N/A', 'type': 'N/A', 'model_type': 'N/A'}

label_map = {0: "Legitimate", 1: "Spam", 2: "Smishing"}

def detect_phishing(text, model_name):
    """Run inference on text"""
    try:
        import time
        tokenizer, model = load_model(model_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        elapsed_ms = round((time.time() - start) * 1000)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        prediction = label_map.get(pred, "Unknown")
        confidence = round(probs[0][pred].item() * 100)
        is_phishing = prediction in ["Smishing", "Spam"]
        st.session_state.last_speed = elapsed_ms
        return {
            'prediction': prediction,
            'confidence': confidence,
            'is_phishing': is_phishing
        }, None
    except Exception as e:
        return None, str(e)

# Get result to inject into HTML
result_data = st.session_state.last_result
messages_data = st.session_state.messages

# Get real model info
try:
    model_info = get_model_info(st.session_state.current_model)
    real_params = model_info['params']
    real_type = model_info['type']
    real_speed = f"~{st.session_state.get('last_speed', 0)}ms" if st.session_state.get('last_speed', 0) > 0 else "N/A"
except Exception:
    real_params = "N/A"
    real_type = "N/A"
    real_speed = "N/A"

html_content = """
<!DOCTYPE html>
<html>
<head>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { height: 100%; overflow: hidden; }
body { background: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

.wrapper { background: #0d1117; height: 100vh; padding: 8px 16px; position: relative; display: flex; flex-direction: column; box-sizing: border-box; overflow: hidden; }

.model-selector {
  background: #1e1e1e;
  border: 1px solid #333;
  border-radius: 10px;
  padding: 14px;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  align-self: stretch;
  margin-top: 18px;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  overflow-y: auto;
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
  border-radius: 10px;
  padding: 26px 32px;
  margin-bottom: 12px;
  box-shadow: 0 0 2px rgba(255,255,255,0.8), 0 0 8px rgba(255,255,255,0.6), 0 0 20px rgba(255,255,255,0.4), 0 0 40px rgba(255,255,255,0.2), 0 4px 15px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255,255,255,0.5);
  width: 100%;
  flex-shrink: 0;
  box-sizing: border-box;
}

.page-title {
  text-align: center;
  font-size: 30px;
  font-weight: 700;
  color: white;
  margin: 0 0 8px 0;
  letter-spacing: -0.5px;
}

.page-subtitle {
  text-align: center;
  font-size: 14px;
  color: #ccc;
  margin: 0;
}

.container { display: flex; gap: 60px; justify-content: center; align-items: stretch; background: transparent; flex-wrap: nowrap; flex: 1; min-height: 0; width: 100%; }
.container > div { flex: 1; min-width: 0; max-width: 380px; display: flex; flex-direction: column; min-height: 0; }
.container > .model-selector { flex: 0 0 340px; max-width: 340px; margin-left: 80px; }

.phone-wrap {
  position: relative;
  flex: 1;
  min-height: 0;
  width: 100%;
  display: flex;
  flex-direction: column;
}

/* Power button - right side */
.btn-power {
  position: absolute;
  right: -4px;
  top: 140px;
  width: 4px;
  height: 50px;
  background: linear-gradient(90deg, #2a2a2a, #444);
  border-radius: 0 4px 4px 0;
  box-shadow: 3px 0 6px rgba(0,0,0,0.6);
  z-index: 10;
}

/* Volume buttons - left side */
.btn-vol-up {
  position: absolute;
  left: -4px;
  top: 100px;
  width: 4px;
  height: 40px;
  background: linear-gradient(270deg, #2a2a2a, #444);
  border-radius: 4px 0 0 4px;
  box-shadow: -3px 0 6px rgba(0,0,0,0.6);
  z-index: 10;
}

.btn-vol-down {
  position: absolute;
  left: -4px;
  top: 150px;
  width: 4px;
  height: 40px;
  background: linear-gradient(270deg, #2a2a2a, #444);
  border-radius: 4px 0 0 4px;
  box-shadow: -3px 0 6px rgba(0,0,0,0.6);
  z-index: 10;
}

.phone-frame {
  width: 100%;
  flex: 1;
  min-height: 0;
  background: #ffffff;
  border-radius: 28px;
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
  top: 6px;
  left: 50%;
  transform: translateX(-50%);
  width: 90px;
  height: 18px;
  background: #000;
  border-radius: 0 0 12px 12px;
  z-index: 20;
}

/* Front camera dot */
.camera-dot {
  position: absolute;
  top: 10px;
  left: 50%;
  transform: translateX(-50%);
  width: 8px;
  height: 8px;
  background: radial-gradient(circle at 35% 35%, #1a1a2e, #000);
  border-radius: 50%;
  border: 1.5px solid #1a1a1a;
  z-index: 25;
  box-shadow: 0 0 4px rgba(0,150,255,0.2), inset 0 0 3px rgba(0,100,200,0.3);
}

.phone-frame::before {
  content: '';
  position: absolute;
  top: 6px;
  left: 50%;
  transform: translateX(-50%);
  width: 88px;
  height: 16px;
  background: #000;
  border-radius: 0 0 10px 10px;
  z-index: 20;
}

.phone-frame::after {
  content: none;
}

.label {
  text-align: center;
  font-size: 16px;
  font-weight: 700;
  color: #0084ff;
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 6px;
  padding: 0 10px;
}

.screen {
  flex: 1;
  background: #ffffff;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  margin: 2px;
  border-radius: 26px;
  box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.9);
  position: relative;
}

.header {
  background: linear-gradient(180deg, #f5f5f5 0%, #f0f0f0 100%);
  padding: 26px 12px 10px 12px;
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
.header-icons { display: flex; gap: 16px; align-items: center; }
.header-icons button { background: none; border: none; color: #0084ff; cursor: pointer; font-size: 20px; line-height: 1; padding: 0; display: flex; align-items: center; justify-content: center; width: 22px; height: 22px; }
.header-icons button svg { width: 20px; height: 20px; }
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
  max-width: 260px;
  padding: 12px 16px;
  border-radius: 18px;
  font-size: 16px;
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
  font-size: 14px;
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
  align-items: center;
  flex-shrink: 0;
}

.input-icons { display: flex; gap: 8px; align-items: center; }
.input-icons button {
  background: none;
  border: none;
  color: #0084ff;
  cursor: pointer;
  font-size: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0;
  line-height: 1;
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
    <h1 class="page-title">📩 SMS Phishing Detection</h1>
    <p class="page-subtitle">Real-time threat detection & analysis | Classifying SMS messages as Legitimate, Spam or Smishing</p>
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
          <input type="text" id="recipientField" class="compose-input" placeholder="Enter recipient" value="Dr Lubani">
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
            <div class="header-avatar">U</div>
            <div>
              <div class="header-title">Unknown</div>
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
      <div class="model-status">
        <span class="model-dot"></span> Model Ready
      </div>
    </div>
    <div style="border-top: 1px solid #333; margin-top: 12px; padding-top: 12px; flex: 1; display: flex; flex-direction: column;">
      <label style="display: block; font-size: 12px; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Prediction Status</label>
      <div style="background: linear-gradient(135deg, #3a2818 0%, #5a3c22 100%); border-radius: 12px; padding: 16px; text-align: center; margin-bottom: 14px; border: 1px solid #f59e0b;">
        <div style="font-size: 26px; font-weight: 700; color: #f59e0b;" id="predictionStatus">Waiting</div>
      </div>
      <label style="display: block; font-size: 12px; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Confidence Score</label>
      <div style="background: linear-gradient(135deg, #1e2a3a 0%, #1e3a5a 100%); border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #3b82f6; margin-bottom: 14px;">
        <div style="font-size: 40px; font-weight: 700; color: #3b82f6;" id="confidenceScore">0%</div>
      </div>
      <label style="display: block; font-size: 12px; font-weight: 700; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;">Risk Level</label>
      <div style="background: linear-gradient(135deg, #3a2818 0%, #5a3c22 100%); border-radius: 12px; padding: 16px; text-align: center; border: 1px solid #f59e0b; margin-bottom: 14px;">
        <div style="font-size: 22px; font-weight: 700; color: #f59e0b;" id="riskLevel">Waiting</div>
      </div>
      <div id="userAdvice" style="display:none; background: linear-gradient(135deg, #3a1818 0%, #5a2222 100%); border-radius: 12px; padding: 18px; border: 1px solid #ef4444;">
        <div style="font-size: 14px; font-weight: 700; color: #ef4444; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">⚠️ Advice</div>
        <div style="font-size: 15px; color: #fca5a5; line-height: 1.5;" id="adviceText">Do not click suspicious links or share personal information!</div>
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
    const darkModeEnabled = false;
    localStorage.setItem('darkMode', 'false');
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
  submitMessageToPython(text, document.getElementById('modelSelect').value, 'sender');

  // Add to receiver (flag will update after API processes)
  setTimeout(() => {
    const phish = apiResult && apiResult.is_phishing ? true : false;
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
  }, 50);

  inp.value = '';
  inp.focus();
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
  submitMessageToPython(text, document.getElementById('modelSelect').value, 'receiver');

  // Add to sender (receiver messages don't get flagged, always safe)
  setTimeout(() => {
    document.getElementById('confidenceScore').textContent = '0%';
    document.getElementById('predictionStatus').textContent = 'Safe';
    document.getElementById('predictionStatus').style.color = '#22c55e';

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
  }, 50);

  inp.value = '';
  inp.focus();
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

# Hide form offscreen but keep functional (visibility hidden breaks interactions)
st.markdown("""
<style>
div[data-testid="stForm"] {
    position: fixed !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 1px !important;
    height: 1px !important;
    overflow: hidden !important;
    opacity: 0 !important;
}
.block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { height: 0 !important; display: none !important; }
#MainMenu, footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# Hidden form - JS will trigger this
@st.fragment
def phishing_form_fragment():
    with st.form("phishing_form", clear_on_submit=True):
        message = st.text_input("hidden_msg", label_visibility="collapsed", placeholder="PHISH_INPUT_TARGET")
        model_input = st.text_input("hidden_model", label_visibility="collapsed", placeholder="MODEL_INPUT_TARGET")
        direction_input = st.text_input("hidden_direction", label_visibility="collapsed", placeholder="DIRECTION_INPUT_TARGET")
        submit = st.form_submit_button("PHISH_SUBMIT_BTN")

        if submit and message:
            model_to_use = model_input.strip() if model_input and model_input.strip() in ("bert", "distilbert") else st.session_state.current_model
            st.session_state.current_model = model_to_use
            direction = direction_input.strip() if direction_input and direction_input.strip() in ("sender", "receiver") else "sender"
            with st.spinner("Analyzing..."):
                result, _ = detect_phishing(message, model_to_use)

            if result:
                import random
                hour = random.randint(1, 12)
                minute = random.randint(0, 59)
                ampm = random.choice(['AM', 'PM'])
                ts = f"{hour}:{minute:02d} {ampm}"
                st.session_state.last_result = result
                st.session_state.messages.append({'text': message, 'result': result, 'time': ts, 'direction': direction})
                st.rerun()

phishing_form_fragment()

# Build messages HTML for sender/receiver (start with default greeting)
sender_msgs_html = '<div class="message sent"><div class="message-group"><div class="bubble">Hi, check this out!</div><div class="timestamp">10:30 AM ✓✓</div></div></div>'
receiver_msgs_html = '<div class="message received"><div class="avatar">?</div><div class="message-group"><div class="bubble">Hi, check this out!</div><div class="timestamp">10:30 AM</div></div></div>'
RED_FLAG = '<svg width="28" height="28" viewBox="0 0 24 24" fill="#ef4444" style="flex-shrink:0;"><path d="M14.4 6L14 4H5v17h2v-7h5.6l.4 2h7V6z"/></svg>'
YELLOW_FLAG = '<svg width="28" height="28" viewBox="0 0 24 24" fill="#f59e0b" style="flex-shrink:0;"><path d="M14.4 6L14 4H5v17h2v-7h5.6l.4 2h7V6z"/></svg>'
for msg in messages_data:
    pred = msg['result']['prediction']
    if pred == 'Smishing':
        flag = RED_FLAG
    elif pred == 'Spam':
        flag = YELLOW_FLAG
    else:
        flag = ''
    ts = msg.get('time', '')
    direction = msg.get('direction', 'sender')
    if direction == 'sender':
        # Sender typed → sent on sender phone (right), received on receiver phone (left, with flag)
        sender_msgs_html += f'''<div class="message sent"><div class="message-group"><div class="bubble">{msg['text']}</div><div class="timestamp">{ts} ✓✓</div></div></div>'''
        receiver_msgs_html += f'''<div class="message received"><div class="avatar">?</div><div class="message-group"><div style="display:flex;align-items:center;gap:8px;"><div class="bubble">{msg['text']}</div>{flag}</div><div class="timestamp">{ts}</div></div></div>'''
    else:
        # Receiver typed → sent on receiver phone (right), received on sender phone (left)
        receiver_msgs_html += f'''<div class="message sent"><div class="message-group"><div class="bubble">{msg['text']}</div><div class="timestamp">{ts} ✓✓</div></div></div>'''
        sender_msgs_html += f'''<div class="message received"><div class="avatar">?</div><div class="message-group"><div class="bubble">{msg['text']}</div><div class="timestamp">{ts}</div></div></div>'''

# Risk level
risk_level = "High Risk" if result_data['prediction'] == 'Smishing' else ("Medium Risk" if result_data['prediction'] == 'Spam' else ("Low Risk" if result_data['prediction'] == 'Legitimate' else "Waiting"))

# Remove placeholder
html_content = html_content.replace('// CONFIG_PLACEHOLDER', '')

# Inject prediction result into HTML
prediction_text = result_data['prediction']
confidence_val = result_data['confidence']
is_phishing = result_data['is_phishing']
status_color = '#ef4444' if is_phishing else '#22c55e'

import json as json_lib
sender_json = json_lib.dumps(sender_msgs_html)
receiver_json = json_lib.dumps(receiver_msgs_html)

if prediction_text == "Waiting":
    ps_color = "#f59e0b"
    ps_bg = "linear-gradient(135deg, #3a2818 0%, #5a3c22 100%)"
    ps_border = "#f59e0b"
elif prediction_text in ["Spam", "Smishing"]:
    ps_color = "#ef4444"
    ps_bg = "linear-gradient(135deg, #3a1818 0%, #5a2222 100%)"
    ps_border = "#ef4444"
else:
    ps_color = "#22c55e"
    ps_bg = "linear-gradient(135deg, #1e3a1f 0%, #2d5a32 100%)"
    ps_border = "#22c55e"

current_model_js = st.session_state.current_model
inject_script = f'''
<script>
window.addEventListener('load', function() {{
  const sel = document.getElementById('modelSelect');
  if (sel) sel.value = '{current_model_js}';
  const ps = document.getElementById('predictionStatus');
  const psBox = ps ? ps.parentElement : null;
  const cs = document.getElementById('confidenceScore');
  const rl = document.getElementById('riskLevel');
  const ua = document.getElementById('userAdvice');
  if (ps) {{
    ps.textContent = '{prediction_text}';
    ps.style.color = '{ps_color}';
  }}
  if (psBox) {{
    psBox.style.background = '{ps_bg}';
    psBox.style.border = '1px solid {ps_border}';
  }}
  if (cs) cs.textContent = '{confidence_val}%';
  if (rl) {{
    rl.textContent = '{risk_level}';
    rl.style.color = '{("#ef4444" if risk_level == "High Risk" else ("#f59e0b" if risk_level == "Medium Risk" else ("#22c55e" if risk_level == "Low Risk" else "#f59e0b")))}';
    const rlBox = rl.parentElement;
    if (rlBox) {{
      rlBox.style.background = '{("linear-gradient(135deg, #3a1818 0%, #5a2222 100%)" if risk_level == "High Risk" else ("linear-gradient(135deg, #3a2818 0%, #5a3c22 100%)" if risk_level == "Medium Risk" else ("linear-gradient(135deg, #1e3a1f 0%, #2d5a32 100%)" if risk_level == "Low Risk" else "linear-gradient(135deg, #3a2818 0%, #5a3c22 100%)")))}';
      rlBox.style.border = '1px solid {("#ef4444" if risk_level == "High Risk" else ("#f59e0b" if risk_level == "Medium Risk" else ("#22c55e" if risk_level == "Low Risk" else "#f59e0b")))}';
    }}
  }}
  if (ua) ua.style.display = '{("block" if prediction_text == "Smishing" else "none")}';
  const sm = document.getElementById('senderMessages');
  if (sm) sm.innerHTML = {sender_json};
  const rm = document.getElementById('receiverMessages');
  if (rm) rm.innerHTML = {receiver_json};
  // Play alert if phishing detected
  if ({("true" if is_phishing else "false")}) {{
    try {{ playSmishingAlert(); }} catch(e) {{}}
  }}
  // Auto-focus sender input after first send so user can keep chatting
  if ({("true" if len(messages_data) > 0 else "false")}) {{
    const senderInp = document.getElementById('senderInput');
    if (senderInp) {{
      senderInp.focus();
      showKeyboard();
    }}
  }}
}});

// Bridge: send phone chat to Streamlit hidden input
function submitMessageToPython(text, model, direction) {{
  try {{
    const parentDoc = window.parent.document;
    const nativeSetter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, 'value').set;
    const msgInput = parentDoc.querySelector('input[placeholder="PHISH_INPUT_TARGET"]');
    const modelInput = parentDoc.querySelector('input[placeholder="MODEL_INPUT_TARGET"]');
    const dirInput = parentDoc.querySelector('input[placeholder="DIRECTION_INPUT_TARGET"]');
    if (msgInput) {{
      nativeSetter.call(msgInput, text);
      msgInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }}
    if (modelInput) {{
      nativeSetter.call(modelInput, model || 'bert');
      modelInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }}
    if (dirInput) {{
      nativeSetter.call(dirInput, direction || 'sender');
      dirInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
    }}
    setTimeout(() => {{
      const buttons = parentDoc.querySelectorAll('button');
      for (let btn of buttons) {{
        if (btn.textContent.includes('PHISH_SUBMIT_BTN')) {{
          btn.click();
          break;
        }}
      }}
    }}, 100);
  }} catch(e) {{
    console.log('Bridge error:', e);
  }}
}}
</script>
'''
html_content = html_content.replace('</body>', f'{inject_script}</body>')

# Render UI
html(html_content, height=900)
