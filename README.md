# Aria Voice AI: Self-Hosted Low-Latency Voice Agent

A high-performance, cost-optimized, production-ready AI voice agent powered by **Pipecat**, **Plivo** (bidirectional Media Streams & SIP Trunking), **Groq LLM (Llama 3.3 70B)**, local **Kokoro ONNX TTS**, and local **Faster-Whisper STT**.

Packaged with Docker for **Oracle Cloud Always Free (ARM64 Ampere A1)** and connected securely via **Cloudflare Tunnel**.

---

## 🌟 Architectural Overview

```
 [ Inbound PSTN / Mobile Call ]              [ Outbound Dial Trigger ]
               │                                         │
               ▼                                         ▼
   ┌──────────────────────────────────────────────────────────┐
   │                     Plivo Telephony                      │
   │  - Rented DID / Number: +1...                            │
   │  - Audio Format: 8kHz μ-law (PCMU) Bidirectional Stream  │
   └─────────────────────────────┬────────────────────────────┘
                                 │
                     WSS Stream via Cloudflare Tunnel
                                 │
                                 ▼
   ┌──────────────────────────────────────────────────────────┐
   │               FastAPI Server & Media Gateway             │
   │  - Inbound Answer URL: returns <Stream> XML              │
   │  - Outbound Trigger API: POST /api/call                  │
   │  - Web UI Control Dashboard: http://host:7860/dashboard  │
   └─────────────────────────────┬────────────────────────────┘
                                 │
                     Pipecat Pipeline Orchestrator
                                 │
       ┌─────────────────────────┼────────────────────────┐
       ▼                         ▼                        ▼
[ Faster-Whisper ]      [ Silero VAD ]           [ Groq LLM ]
 Local STT (small)    Speech Turn Detection    Llama 3.3 70B Versatile
 int8 Quantization    350ms Silence Timeout    <200ms TTFT Latency
       │                                                  │
       └─────────────────────────┬────────────────────────┘
                                 │
                                 ▼
                        [ Kokoro ONNX TTS ]
                      Local Speech Synthesis
                       Voice: af_heart (8kHz)
```

---

## 💰 Cost & Free Tier Breakdown

| Component | Provider / Engine | Resource / Specs | Monthly Cost |
| :--- | :--- | :--- | :--- |
| **Server Host** | **Oracle Cloud Always Free** | Up to 4 ARM OCPUs, 24GB RAM (Ampere A1) | **$0.00** |
| **LLM Inference** | **Groq Free Tier** | Llama 3.3 70B Versatile (fastest TTFT) | **$0.00** |
| **STT (Speech-to-Text)** | **Faster-Whisper (small)** | Runs locally on Oracle ARM CPU | **$0.00** |
| **TTS (Text-to-Speech)** | **Kokoro ONNX** | Runs locally on Oracle ARM CPU | **$0.00** |
| **Tunnel / Ingress** | **Cloudflare Tunnel** | Unlimited encrypted HTTPS/WSS proxy | **$0.00** |
| **Telephony Minutes** | **Plivo** | SIP / Media Stream (~$0.005/min) | Pay as you go (~$1-3) |
| **Phone Number (DID)** | **Plivo** | US local phone number | ~$0.50 – $1.00/mo |

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- `ffmpeg` and `espeak-ng` installed on your host system:
  - **macOS**: `brew install ffmpeg espeak-ng`
  - **Ubuntu/Debian**: `sudo apt update && sudo apt install -y ffmpeg espeak-ng libsndfile1`

### 2. Clone and Configure
```bash
git clone <your-repo>
cd hopeful-mendel

# Create python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 3. Set Required API Keys in `.env`
Open `.env` and fill in:
- `GROQ_API_KEY`: Get free from [console.groq.com](https://console.groq.com/keys)
- `PLIVO_AUTH_ID` & `PLIVO_AUTH_TOKEN`: From [console.plivo.com](https://console.plivo.com/)
- `PLIVO_PHONE_NUMBER`: Your rented Plivo number in E.164 format (e.g. `+14155552671`)

### 4. Pre-Download Model Weights (Zero Runtime Lag)
```bash
python scripts/download_models.py
```

### 5. Start the Server
```bash
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload
```
Open your browser at:
- **Control Dashboard**: [http://localhost:7860/dashboard](http://localhost:7860/dashboard)
- **Health Diagnostics**: [http://localhost:7860/health](http://localhost:7860/health)

---

## ☁️ Deployment on Oracle Cloud Always Free

Oracle Cloud offers permanent free ARM instances with plenty of compute for local Whisper and Kokoro.

### Step 1: Provision Oracle Cloud VM
1. Log into your Oracle Cloud Console.
2. Go to **Compute > Instances > Create Instance**.
3. Choose **Image**: Ubuntu 22.04 or 24.04 LTS.
4. Choose **Shape**: `VM.Standard.A1.Flex` (Ampere ARM). Select **2 to 4 OCPUs** and **12 to 24 GB RAM** (all 100% free under Always Free tier limits).
5. Attach your SSH key and launch the instance.

### Step 2: Install Docker on the Oracle Instance
SSH into your instance:
```bash
ssh ubuntu@<your-instance-ip>

# Update and install Docker
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-v2 curl git

# Enable docker permissions
sudo usermod -aG docker $USER
newgrp docker
```

### Step 3: Clone Code and Launch Stack
```bash
git clone <your-repo> voice-agent
cd voice-agent
cp .env.example .env
nano .env # Add your GROQ_API_KEY, PLIVO keys, and PUBLIC_URL

# Launch with Docker Compose
docker compose up -d --build
```
Check health:
```bash
curl http://localhost:7860/health
```

---

## 🔒 Cloudflare Tunnel Setup (Exposing without Open Ports)

Using Cloudflare Tunnel provides:
- Free SSL/TLS certificates (`https://` and `wss://`)
- Zero port forwarding needed in Oracle Cloud Security Lists
- Protection against DDoS and network sniffing

### Option A: Using Docker Compose `cloudflared` (Recommended)
1. Go to [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/) > **Networks > Tunnels**.
2. Click **Create a tunnel**, choose **Cloudflared**.
3. Name your tunnel (e.g. `aria-voice`).
4. Copy the `TUNNEL_TOKEN` from the command displayed.
5. In your `.env` file, set:
   ```env
   CLOUDFLARE_TUNNEL_TOKEN=eyJh...your_token_here
   PUBLIC_URL=https://agent.yourdomain.com
   ```
6. Route the hostname in Cloudflare to: `http://voice-agent:7860`.
7. Start the tunnel container:
   ```bash
   docker compose --profile with-tunnel up -d
   ```

### Option B: Quick Cloudflare Tunnel (CLI)
```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64 -o cloudflared
chmod +x cloudflared
sudo mv cloudflared /usr/local/bin/

# Start quick ad-hoc tunnel
cloudflared tunnel --url http://localhost:7860
```
Copy the generated `https://xxxx.trycloudflare.com` URL into your `.env` as `PUBLIC_URL`.

---

## 📞 Plivo Telephony Configuration

### Inbound Calls (Receive calls to Aria)
1. In the [Plivo Console](https://console.plivo.com/):
2. Navigate to **Voice > XML Applications > Add New Application**.
3. Configure the application:
   - **Application Name**: `Aria Voice AI`
   - **Primary Answer URL**: `https://your-domain.com/` (or `https://xxxx.trycloudflare.com/`)
   - **Answer URL Method**: `POST` (or `GET`)
   - **Fallback URL**: leave empty or set backup
4. Click **Create Application**.
5. Navigate to **Phone Numbers > Your Numbers** and click on your rented phone number.
6. Under **Application Type**, choose `XML Application` and select `Aria Voice AI`.
7. Click **Update Number**.
8. Dial your Plivo number from your phone! Aria will answer and converse with you.

---

### Outbound Calls (Have Aria dial a phone number)

#### Via the Web Dashboard
1. Open `https://your-domain.com/dashboard`.
2. Enter the recipient's phone number in E.164 format (e.g., `+14155552671`).
3. Click **Dial Number Now**.

#### Via REST API (cURL / Backend Integration)
```bash
curl -X POST https://your-domain.com/api/call \
  -H "Content-Type: application/json" \
  -d '{
    "to": "+14155552671",
    "from": "+19876543210",
    "extra_context": {
      "customer_name": "Jordan Smith",
      "purpose": "Appointment confirmation"
    }
  }'
```

When the recipient answers:
1. Plivo invokes `https://your-domain.com/outbound/answer`.
2. Plivo receives the `<Stream>` XML and establishes a WebSocket to `wss://your-domain.com/ws`.
3. Aria greets the recipient and conducts the call according to your persona and tools.

---

## 🛠️ Tool Calling & Capabilities

Aria is equipped with dynamic tool calling defined in `app/tools.py`:
- `get_current_time`: Provides current date, day, and time.
- `lookup_customer_account`: Retrieves mock CRM data (easily swapped with Postgres/Supabase/Hubspot).
- `schedule_appointment`: Books appointments and returns confirmation codes.
- `transfer_to_human_agent`: Hands off complex requests to human agents.

To add new custom tools:
1. Define an `async def your_function(params: FunctionCallParams, ...)` in `app/tools.py`.
2. Add comprehensive docstrings and type hints (Groq automatically parses this into JSON schema).
3. Add the function to `REGISTERED_TOOLS`.

---

## ⚡ Latency Tuning Checklist

1. **Groq Model Selection**:
   - `llama-3.3-70b-versatile` offers top-tier intelligence with ~150–250ms TTFT.
   - `llama-3.1-8b-instant` offers ultra-snappy ~80–120ms responses if you need maximum speed.
2. **Turn Detection**:
   - Set `VAD_STOP_SECS=0.30` or `0.35` in `.env` for snappy back-and-forth conversation without cutting off natural pauses.
3. **Model Quantization**:
   - `WHISPER_COMPUTE_TYPE=int8` provides the lowest CPU execution latency on ARM Ampere.
4. **Telephony Sample Rate**:
   - Audio is streamed natively in 8kHz $\mu$-law, eliminating unnecessary resampling overhead.

---

## 🧪 Diagnostics & Verification

Run the automated integration tester to verify all local components:
```bash
python scripts/test_agent.py
```
This tests:
1. ✅ Groq API connectivity and Llama 3.3 70B response
2. ✅ Local Kokoro ONNX voice generation
3. ✅ Local Faster-Whisper initialization
