# Vehicle Inventory Intelligence System v3.0

Production dealership inventory analysis platform with AI vision, comp discovery, and daily probability curves.

## Features

1. **AI Photo Identification** — Paste image URL or upload photo → AI returns Year/Make/Model with confidence
2. **Spreadsheet Import** — CSV/XLSX upload with auto column mapping and preview
3. **Manual Entry** — Full vehicle data form with sample data loader
4. **Automated Comp Discovery** — Statistical comps + SerpAPI + Marketcheck integration
5. **Manual Comp Override** — Add/edit/delete comps manually
6. **Daily Probability Curve** — Day 1-90 hover chart with daily %, cumulative %, gross, floorplan
7. **Full Analysis Engine** — Pricing, exit path, action plan, risk assessment

## Quick Start (Local)

```bash
git clone https://github.com/YOUR_USERNAME/vehicle-intelligence.git
cd vehicle-intelligence
pip install -r requirements.txt
python main.py
# Open http://localhost:5000
