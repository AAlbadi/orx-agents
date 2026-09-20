# 🚀 Production Deployment Guide: `agents.orxlabs.com` on Oracle Cloud (Arizona)

Complete production setup guide to host your ORX Agents Voice Receptionist platform on **Oracle Cloud Always Free (Phoenix / Arizona Region - ARM64 Ampere A1)** and connect the subdomain **`agents.orxlabs.com`** (registered on Spaceship.com).

---

## 🏗️ Architecture Overview

```
                          ┌────────────────────────┐
                          │     Spaceship.com      │
                          │   DNS: orxlabs.com     │
                          └───────────┬────────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │                                                   │
            ▼                                                   ▼
     orxlabs.com                                        agents.orxlabs.com
  (Vercel Front-End)                                 (Oracle Cloud Arizona VM)
                                                                │
                                                    ┌───────────┴───────────┐
                                                    │   Cloudflare Tunnel   │
                                                    │   (Zero-Port Ingress) │
                                                    └───────────┬───────────┘
                                                                │
                                                    ┌───────────▼───────────┐
                                                    │  Docker Stack :7860   │
                                                    │  - FastAPI Engine     │
                                                    │  - LiveKit WebRTC     │
                                                    │  - Gemini 3.1 Flash   │
                                                    │  - Tenant SQLite DBs  │
                                                    └───────────────────────┘
```

---

## 📦 Step 1: Push Code to GitHub Repository

1. From your local development machine:
   ```bash
   cd /Users/aziz/Documents/antigravity/hopeful-mendel
   
   # Add your GitHub remote (matches your other orxlabs repositories)
   git remote add origin https://github.com/AAlbadi/orx-agents.git 2>/dev/null || true
   
   # Commit and push
   git add .
   git commit -m "Production release: 2026 Admin Dashboard, Zero-Friction Booking, Google OAuth"
   git push -u origin master
   ```

---

## ☁️ Step 2: Provision Oracle Cloud Always Free VM (Phoenix, Arizona)

1. Log in to [Oracle Cloud Console](https://cloud.oracle.com/).
2. In the top-right region selector, choose **US West (Phoenix)** (Arizona data center for lowest North American voice latency).
3. Navigate to **Compute > Instances > Create Instance**:
   - **Name**: `orx-voice-agent-arizona`
   - **Image**: `Ubuntu 24.04 LTS (aarch64)`
   - **Shape**: `VM.Standard.A1.Flex` (Ampere ARM):
     - Select **4 OCPUs** and **24 GB RAM** (100% free under Oracle Always Free tier).
   - **SSH Keys**: Download or generate your SSH key pair.
   - **Boot Volume**: 50 GB to 100 GB (Always Free allows up to 200 GB across tenancy).
4. Click **Create** and copy the **Public IP Address** once running.

---

## ⚡ Step 3: Run the 1-Click Deployment Script on the Server

1. Connect via SSH:
   ```bash
   ssh -i ~/.ssh/oracle_key.pem ubuntu@<YOUR_ORACLE_VM_IP>
   ```

2. Clone your repository:
   ```bash
   git clone https://github.com/AAlbadi/orx-agents.git orx-agents
   cd orx-agents
   ```

3. Run the automated deployment script:
   ```bash
   chmod +x deploy/deploy_oracle.sh
   ./deploy/deploy_oracle.sh
   ```

4. Configure your production environment variables:
   ```bash
   cp deploy/production.env .env
   nano .env
   ```
   Add your:
   - `GEMINI_API_KEY`: Your Google AI Studio API key
   - `GROQ_API_KEY`: Your Groq LPU API key
   - `DEEPGRAM_API_KEY`: Speech-to-text key

5. Restart the containers with your production configuration:
   ```bash
   docker compose up -d
   ```

---

## 🌐 Step 4: Connect Subdomain `agents.orxlabs.com`

### Option A: Cloudflare Tunnel (Recommended - Zero Inbound Port Forwarding)
1. Go to [Cloudflare Zero Trust Dashboard](https://one.dash.cloudflare.com/) > **Networks > Tunnels**.
2. Click **Create a Tunnel**, choose **Cloudflared**, name it `orx-agents-arizona`.
3. Under **Public Hostname**:
   - Subdomain: `agents`
   - Domain: `orxlabs.com`
   - Service Type: `HTTP`
   - URL: `voice-agent:7860` (or `localhost:7860`)
4. Copy the `TUNNEL_TOKEN` provided by Cloudflare.
5. In your server `.env` on Oracle:
   ```env
   CLOUDFLARE_TUNNEL_TOKEN=eyJh...your_token_here
   ```
6. Start tunnel service:
   ```bash
   docker compose --profile with-tunnel up -d
   ```
7. `https://agents.orxlabs.com` is now instantly live with free, automatic SSL!

### Option B: Spaceship.com Direct DNS (If using Public IP & Caddy/Nginx)
In your [Spaceship.com](https://www.spaceship.com/) Domain Manager for `orxlabs.com`:

| Type | Host / Name | Value / Target | TTL | Description |
|---|---|---|---|---|
| **A** | `agents` | `<YOUR_ORACLE_VM_PUBLIC_IP>` | Automatic | Routes `agents.orxlabs.com` to Oracle Arizona |

---

## 🔍 Step 5: Verify Production End-to-End

Run the verification test suite directly on the Oracle instance:
```bash
docker compose exec voice-agent python scripts/test_orx_platform_e2e.py
```

All 11 checks will execute:
- ✅ Admin Dashboard: `https://agents.orxlabs.com/dashboard`
- ✅ Business Owner Portal: `https://agents.orxlabs.com/portal`
- ✅ Customer Onboarding & Polar Checkout: `https://agents.orxlabs.com/subscribe`
- ✅ LiveKit Voice Agent Lab: `https://agents.orxlabs.com/livekit`
- ✅ Google Calendar OAuth: `https://agents.orxlabs.com/api/auth/google/url`
