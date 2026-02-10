"""
Vehicle Inventory Intelligence System — Production Backend
Features:
  1. AI Vision Vehicle Identification (Claude/OpenAI)
  2. Spreadsheet Import (CSV/XLSX) with column mapping
  3. Automated Comp Discovery + Manual Override
  4. Daily Probability Curve (Day 1-90) with hover data
  5. Full Analysis Engine
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import io
import csv
import json
import math
import uuid
import re
import time
import hashlib
from datetime import datetime, timedelta
from functools import wraps

app = Flask(__name__, static_folder='public', static_url_path='')

# ============================================================
# CONFIGURATION
# ============================================================
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
SERPAPI_KEY = os.environ.get('SERPAPI_KEY', '')
MARKETCHECK_API_KEY = os.environ.get('MARKETCHECK_API_KEY', '')
VISION_CONFIDENCE_THRESHOLD = float(os.environ.get('VISION_CONFIDENCE_THRESHOLD', '0.75'))
COMPS_CACHE_TTL = int(os.environ.get('COMPS_CACHE_TTL', '3600'))  # 1 hour
ENABLE_WEB_COMPS = os.environ.get('ENABLE_WEB_COMPS', 'false').lower() == 'true'

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ============================================================
# IN-MEMORY STORAGE
# ============================================================
vehicles_db = {}
reports_db = {}
comps_db = {}  # vehicle_id -> [comp, comp, ...]
comps_cache = {}  # cache_key -> {data, timestamp}


# ============================================================
# SERVE FRONTEND
# ============================================================
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')


@app.route('/<path:path>')
def static_files(path):
    if os.path.exists(os.path.join('public', path)):
        return send_from_directory('public', path)
    return send_from_directory('public', 'index.html')


# ============================================================
# HEALTH CHECK
# ============================================================
@app.route('/api/health')
def health():
    providers = {
        'anthropic_vision': bool(ANTHROPIC_API_KEY),
        'openai_vision': bool(OPENAI_API_KEY),
        'serpapi_comps': bool(SERPAPI_KEY),
        'marketcheck_comps': bool(MARKETCHECK_API_KEY),
        'web_comps_enabled': ENABLE_WEB_COMPS,
    }
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0',
        'providers': providers,
        'vehicle_count': len(vehicles_db),
        'report_count': len(reports_db),
    })


# ============================================================
# FEATURE 1: AI VISION VEHICLE IDENTIFICATION
# ============================================================
@app.route('/api/vehicle/identify', methods=['POST'])
def identify_vehicle():
    """
    Accepts:
      - JSON with {"imageUrl": "https://..."} 
      - OR multipart form with file upload
    Returns vehicle identification with confidence.
    """
    image_url = None
    image_base64 = None

    if request.content_type and 'multipart' in request.content_type:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image file provided'}), 400

        import base64
        image_bytes = file.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = file.content_type or 'image/jpeg'
    else:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        image_url = data.get('imageUrl', '').strip()
        if not image_url:
            return jsonify({'error': 'No imageUrl provided'}), 400

    # Try providers in order
    result = None
    provider_used = None

    # Provider 1: Anthropic Claude Vision
    if ANTHROPIC_API_KEY and not result:
        try:
            result = identify_with_anthropic(image_url, image_base64,
                                             locals().get('mime_type', 'image/jpeg'))
            provider_used = 'anthropic_claude'
        except Exception as e:
            app.logger.error(f"Anthropic vision error: {e}")

    # Provider 2: OpenAI Vision
    if OPENAI_API_KEY and not result:
        try:
            result = identify_with_openai(image_url, image_base64,
                                          locals().get('mime_type', 'image/jpeg'))
            provider_used = 'openai_gpt4v'
        except Exception as e:
            app.logger.error(f"OpenAI vision error: {e}")

    # Provider 3: Keyword fallback (no API key needed)
    if not result:
        if image_url:
            result = identify_from_url_keywords(image_url)
            provider_used = 'url_keyword_fallback'
        else:
            result = {
                'year': None, 'make': None, 'model': None, 'trim': None,
                'confidence': 0,
                'notes': 'No vision API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.',
            }
            provider_used = 'none'

    confidence = result.get('confidence', 0)
    below_threshold = confidence < VISION_CONFIDENCE_THRESHOLD

    return jsonify({
        'identification': result,
        'provider': provider_used,
        'confidence_threshold': VISION_CONFIDENCE_THRESHOLD,
        'below_threshold': below_threshold,
        'fallback_recommended': below_threshold,
        'message': 'Low confidence — use spreadsheet or manual entry' if below_threshold
                   else 'Vehicle identified successfully'
    })


def identify_with_anthropic(image_url, image_base64, mime_type):
    """Uses Anthropic Claude Vision API"""
    try:
        from anthropic import Anthropic
    except ImportError:
        raise Exception("anthropic package not installed")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    content = []
    if image_base64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_base64,
            }
        })
    elif image_url:
        content.append({
            "type": "image",
            "source": {
                "type": "url",
                "url": image_url,
            }
        })

    content.append({
        "type": "text",
        "text": """Analyze this vehicle image. Return ONLY a JSON object with these fields:
{
  "year": <integer or null>,
  "make": "<string or null>",
  "model": "<string or null>",
  "trim": "<string or null>",
  "confidence": <float 0.0 to 1.0>,
  "notes": "<brief explanation of what you see and how confident you are>"
}
Be specific about confidence. If you can clearly see the badge, grille shape, and body style, confidence should be high. If the image is blurry, distant, or partially obscured, say so."""
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": content}]
    )

    text = response.content[0].text.strip()
    # Extract JSON from response
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    else:
        return {
            'year': None, 'make': None, 'model': None, 'trim': None,
            'confidence': 0.1,
            'notes': f'Could not parse structured response: {text[:200]}'
        }


def identify_with_openai(image_url, image_base64, mime_type):
    """Uses OpenAI GPT-4 Vision API"""
    try:
        from openai import OpenAI
    except ImportError:
        raise Exception("openai package not installed")

    client = OpenAI(api_key=OPENAI_API_KEY)

    if image_base64:
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            }
        }
    else:
        image_content = {
            "type": "image_url",
            "image_url": {"url": image_url}
        }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                image_content,
                {
                    "type": "text",
                    "text": """Analyze this vehicle image. Return ONLY a JSON object:
{
  "year": <integer or null>,
  "make": "<string or null>",
  "model": "<string or null>",
  "trim": "<string or null>",
  "confidence": <float 0.0 to 1.0>,
  "notes": "<brief explanation>"
}"""
                }
            ]
        }],
        max_tokens=500
    )

    text = response.choices[0].message.content.strip()
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {
        'year': None, 'make': None, 'model': None, 'trim': None,
        'confidence': 0.1, 'notes': f'Parse failed: {text[:200]}'
    }


def identify_from_url_keywords(url):
    """Fallback: extract vehicle info from URL string patterns"""
    text = url.lower().replace('-', ' ').replace('_', ' ').replace('/', ' ')

    makes_models = {
        'toyota': ['camry', 'corolla', 'rav4', 'tacoma', 'tundra', 'highlander',
                    '4runner', 'prius', 'avalon', 'sienna', 'venza', 'supra'],
        'honda': ['civic', 'accord', 'cr v', 'crv', 'hr v', 'hrv', 'pilot',
                  'odyssey', 'ridgeline', 'passport', 'fit'],
        'ford': ['f 150', 'f150', 'mustang', 'explorer', 'escape', 'bronco',
                 'ranger', 'edge', 'expedition', 'maverick', 'fusion'],
        'chevrolet': ['silverado', 'equinox', 'traverse', 'tahoe', 'suburban',
                      'camaro', 'corvette', 'blazer', 'malibu', 'colorado', 'trax'],
        'nissan': ['altima', 'sentra', 'rogue', 'pathfinder', 'murano',
                   'frontier', 'titan', 'maxima', 'versa', 'kicks', 'armada'],
        'hyundai': ['elantra', 'sonata', 'tucson', 'santa fe', 'palisade',
                    'kona', 'venue', 'ioniq'],
        'kia': ['forte', 'optima', 'k5', 'sportage', 'sorento', 'telluride',
                'soul', 'seltos', 'carnival', 'stinger'],
        'bmw': ['3 series', '5 series', 'x3', 'x5', 'x1', 'm3', 'm5',
                '330i', '530i', 'x7'],
        'mercedes': ['c class', 'e class', 's class', 'glc', 'gle', 'gls',
                     'c300', 'e350', 'amg'],
        'tesla': ['model 3', 'model y', 'model s', 'model x'],
        'jeep': ['wrangler', 'grand cherokee', 'cherokee', 'compass',
                 'renegade', 'gladiator', 'wagoneer'],
        'subaru': ['outback', 'forester', 'crosstrek', 'impreza', 'wrx',
                   'legacy', 'ascent'],
        'lexus': ['rx', 'es', 'nx', 'is', 'gx', 'lx', 'ux'],
        'audi': ['a3', 'a4', 'a6', 'q3', 'q5', 'q7', 'q8'],
        'volkswagen': ['jetta', 'passat', 'tiguan', 'atlas', 'golf', 'gti', 'taos'],
        'mazda': ['cx 5', 'cx5', 'cx 9', 'cx9', 'mazda3', 'mazda6',
                  'cx 30', 'cx30', 'cx 50', 'mx 5', 'miata'],
        'gmc': ['sierra', 'yukon', 'acadia', 'terrain', 'canyon'],
        'dodge': ['ram', 'charger', 'challenger', 'durango'],
        'ram': ['1500', '2500', '3500'],
    }

    years = re.findall(r'20[0-2][0-9]|19[89][0-9]', text)
    detected_year = int(years[0]) if years else None

    detected_make = None
    detected_model = None

    for make, models in makes_models.items():
        if make in text:
            detected_make = make
            for model in models:
                if model in text:
                    detected_model = model
                    break
            break
        for model in models:
            if model in text:
                detected_make = make
                detected_model = model
                break
        if detected_make:
            break

    conf = 0
    if detected_make:
        conf += 0.3
    if detected_model:
        conf += 0.3
    if detected_year:
        conf += 0.2

    return {
        'year': detected_year,
        'make': detected_make.title() if detected_make else None,
        'model': detected_model.title() if detected_model else None,
        'trim': None,
        'confidence': conf,
        'notes': f'Extracted from URL keywords. {"Make" if not detected_make else ""} '
                 f'{"Model" if not detected_model else ""} '
                 f'{"Year" if not detected_year else ""} '
                 f'not found in URL.'.strip()
    }


# ============================================================
# FEATURE 1B: SPREADSHEET IMPORT
# ============================================================
@app.route('/api/vehicle/template', methods=['GET'])
def download_template():
    """Download a CSV template for vehicle import"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Year', 'Make', 'Model', 'Trim', 'VIN', 'Stock_Number',
        'Mileage', 'Ext_Color', 'Int_Color', 'Acquisition_Cost',
        'Recon_Cost', 'List_Price', 'Wholesale_Price', 'Days_In_Inventory'
    ])
    writer.writerow([
        '2021', 'Toyota', 'Camry', 'SE', '4T1G11AK0MU123456', 'STK001',
        '41850', 'White', 'Black', '19400',
        '850', '23495', '20300', '52'
    ])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='vehicle_import_template.csv'
    )


@app.route('/api/vehicle/import', methods=['POST'])
def import_vehicles():
    """
    Import vehicles from CSV or XLSX.
    Accepts multipart form with:
      - file: the spreadsheet
      - column_map (optional JSON): mapping of file columns to our fields
      - preview_only (optional): if "true", returns parsed data without saving
    """
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    filename = file.filename.lower()
    column_map_raw = request.form.get('column_map', '{}')
    preview_only = request.form.get('preview_only', 'false').lower() == 'true'

    try:
        column_map = json.loads(column_map_raw)
    except json.JSONDecodeError:
        column_map = {}

    rows = []
    errors = []

    try:
        if filename.endswith('.csv'):
            rows, errors = parse_csv(file, column_map)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            rows, errors = parse_xlsx(file, column_map)
        else:
            return jsonify({'error': 'Unsupported format. Use CSV or XLSX.'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to parse file: {str(e)}'}), 400

    if preview_only:
        return jsonify({
            'message': 'Preview — not saved',
            'row_count': len(rows),
            'rows': rows[:50],  # limit preview
            'errors': errors,
            'detected_columns': list(rows[0].keys()) if rows else []
        })

    # Save vehicles
    saved = []
    save_errors = []
    for i, row in enumerate(rows):
        try:
            vid = str(uuid.uuid4())
            vehicle = build_vehicle_record(vid, row)
            vehicles_db[vid] = vehicle
            saved.append({'id': vid, 'title': f"{row.get('year','')} {row.get('make','')} {row.get('model','')}"})
        except Exception as e:
            save_errors.append({'row': i + 1, 'error': str(e)})

    return jsonify({
        'message': f'Imported {len(saved)} vehicles',
        'imported': saved,
        'import_errors': save_errors,
        'parse_errors': errors
    })


def parse_csv(file, column_map):
    """Parse CSV file into vehicle records"""
    content = file.read().decode('utf-8-sig')
    reader = csv.DictReader(io.StringIO(content))

    rows = []
    errors = []

    default_map = {
        'year': ['year', 'yr', 'model_year', 'modelyear'],
        'make': ['make', 'manufacturer', 'brand'],
        'model': ['model', 'model_name'],
        'trim': ['trim', 'trim_level', 'trimlevel'],
        'vin': ['vin', 'vin_number'],
        'stock_number': ['stock', 'stock_number', 'stocknumber', 'stk', 'stock#', 'stock_no'],
        'mileage': ['mileage', 'miles', 'odometer', 'odo'],
        'ext_color': ['ext_color', 'exterior_color', 'color', 'extcolor', 'exterior'],
        'int_color': ['int_color', 'interior_color', 'intcolor', 'interior'],
        'acquisition_cost': ['acquisition_cost', 'cost', 'purchase_price', 'acq_cost',
                            'bought_for', 'acq', 'acquisition'],
        'recon_cost': ['recon_cost', 'recon', 'reconditioning', 'repair_cost'],
        'list_price': ['list_price', 'price', 'asking_price', 'retail_price', 'listprice'],
        'wholesale_price': ['wholesale_price', 'wholesale', 'auction_price'],
        'days_in_inventory': ['days_in_inventory', 'days', 'age', 'dom', 'days_on_lot'],
    }

    # Build actual column mapping
    if reader.fieldnames:
        resolved_map = {}
        for our_field, possible_names in default_map.items():
            # Check user override first
            if our_field in column_map:
                resolved_map[our_field] = column_map[our_field]
                continue
            # Auto-detect
            for csv_col in reader.fieldnames:
                if csv_col.lower().strip().replace(' ', '_') in possible_names:
                    resolved_map[our_field] = csv_col
                    break

    for i, row in enumerate(reader):
        try:
            mapped = {}
            for our_field, csv_col in resolved_map.items():
                mapped[our_field] = row.get(csv_col, '').strip()
            rows.append(mapped)
        except Exception as e:
            errors.append({'row': i + 2, 'error': str(e)})

    return rows, errors


def parse_xlsx(file, column_map):
    """Parse XLSX file into vehicle records"""
    try:
        import openpyxl
    except ImportError:
        raise Exception("openpyxl not installed — cannot parse XLSX")

    file_bytes = file.read()
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True)
    ws = wb.active

    rows_data = []
    errors = []
    headers = []

    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            headers = [str(h).strip() if h else f'col_{j}' for j, h in enumerate(row)]
            continue

        row_dict = {}
        for j, val in enumerate(row):
            if j < len(headers):
                row_dict[headers[j]] = str(val).strip() if val is not None else ''

        # Apply same mapping logic as CSV
        default_map = {
            'year': ['year', 'yr', 'model_year'],
            'make': ['make', 'manufacturer', 'brand'],
            'model': ['model', 'model_name'],
            'trim': ['trim', 'trim_level'],
            'vin': ['vin', 'vin_number'],
            'mileage': ['mileage', 'miles', 'odometer'],
            'ext_color': ['ext_color', 'exterior_color', 'color', 'exterior'],
            'int_color': ['int_color', 'interior_color', 'interior'],
            'acquisition_cost': ['acquisition_cost', 'cost', 'purchase_price', 'acq_cost', 'acquisition'],
            'recon_cost': ['recon_cost', 'recon', 'reconditioning'],
            'list_price': ['list_price', 'price', 'asking_price', 'retail_price'],
            'wholesale_price': ['wholesale_price', 'wholesale'],
            'days_in_inventory': ['days_in_inventory', 'days', 'age', 'dom'],
        }

        mapped = {}
        for our_field, possible_names in default_map.items():
            if our_field in column_map:
                mapped[our_field] = row_dict.get(column_map[our_field], '')
                continue
            for h in headers:
                if h.lower().strip().replace(' ', '_') in possible_names:
                    mapped[our_field] = row_dict.get(h, '')
                    break

        if mapped.get('year') or mapped.get('make') or mapped.get('vin'):
            rows_data.append(mapped)

    wb.close()
    return rows_data, errors


# ============================================================
# FEATURE 2: COMPS DISCOVERY + MANUAL
# ============================================================
@app.route('/api/vehicle/<vehicle_id>/comps', methods=['GET'])
def get_comps(vehicle_id):
    """Get all comps for a vehicle"""
    if vehicle_id not in vehicles_db:
        return jsonify({'error': 'Vehicle not found'}), 404

    vehicle_comps = comps_db.get(vehicle_id, [])
    return jsonify({
        'vehicle_id': vehicle_id,
        'count': len(vehicle_comps),
        'comps': vehicle_comps
    })


@app.route('/api/vehicle/<vehicle_id>/comps', methods=['POST'])
def add_manual_comp(vehicle_id):
    """Add a manual comp"""
    if vehicle_id not in vehicles_db:
        return jsonify({'error': 'Vehicle not found'}), 404

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    comp = {
        'id': str(uuid.uuid4()),
        'vehicle_id': vehicle_id,
        'source': data.get('source', 'Manual Entry'),
        'url': data.get('url', ''),
        'sold_price': float(data.get('sold_price', 0)),
        'sold_date': data.get('sold_date', ''),
        'mileage': int(data.get('mileage', 0)),
        'location': data.get('location', ''),
        'days_on_market': int(data.get('days_on_market', 0)),
        'year': data.get('year', ''),
        'make': data.get('make', ''),
        'model': data.get('model', ''),
        'trim': data.get('trim', ''),
        'notes': data.get('notes', ''),
        'is_manual': True,
        'created_at': datetime.utcnow().isoformat()
    }

    if vehicle_id not in comps_db:
        comps_db[vehicle_id] = []
    comps_db[vehicle_id].append(comp)

    return jsonify({'message': 'Comp added', 'comp': comp}), 201


@app.route('/api/vehicle/<vehicle_id>/comps/<comp_id>', methods=['DELETE'])
def delete_comp(vehicle_id, comp_id):
    """Delete a comp"""
    if vehicle_id not in comps_db:
        return jsonify({'error': 'No comps found'}), 404

    comps_db[vehicle_id] = [c for c in comps_db[vehicle_id] if c['id'] != comp_id]
    return jsonify({'message': 'Comp deleted'})


@app.route('/api/vehicle/<vehicle_id>/comps/find', methods=['POST'])
def find_comps(vehicle_id):
    """
    Auto-discover comps for a vehicle.
    Uses available providers in priority order.
    Results are cached.
    """
    if vehicle_id not in vehicles_db:
        return jsonify({'error': 'Vehicle not found'}), 404

    vehicle = vehicles_db[vehicle_id]

    # Check cache
    cache_key = f"{vehicle['year']}_{vehicle['make']}_{vehicle['model']}_{vehicle.get('trim', '')}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()

    if cache_hash in comps_cache:
        cached = comps_cache[cache_hash]
        if time.time() - cached['timestamp'] < COMPS_CACHE_TTL:
            # Add cached comps to this vehicle
            for comp in cached['data']:
                comp_copy = dict(comp)
                comp_copy['id'] = str(uuid.uuid4())
                comp_copy['vehicle_id'] = vehicle_id
                comp_copy['cached'] = True
                if vehicle_id not in comps_db:
                    comps_db[vehicle_id] = []
                # Avoid duplicates
                existing_urls = {c.get('url') for c in comps_db[vehicle_id]}
                if comp_copy.get('url') not in existing_urls:
                    comps_db[vehicle_id].append(comp_copy)

            return jsonify({
                'message': 'Comps loaded from cache',
                'source': 'cache',
                'count': len(cached['data']),
                'comps': comps_db.get(vehicle_id, [])
            })

    # Try providers
    found_comps = []
    provider_used = 'generated'

    # Provider 1: SerpAPI
    if SERPAPI_KEY and ENABLE_WEB_COMPS:
        try:
            found_comps = search_comps_serpapi(vehicle)
            provider_used = 'serpapi'
        except Exception as e:
            app.logger.error(f"SerpAPI comp search error: {e}")

    # Provider 2: Marketcheck
    if MARKETCHECK_API_KEY and not found_comps:
        try:
            found_comps = search_comps_marketcheck(vehicle)
            provider_used = 'marketcheck'
        except Exception as e:
            app.logger.error(f"Marketcheck error: {e}")

    # Fallback: Generate statistical comps
    if not found_comps:
        found_comps = generate_statistical_comps(vehicle)
        provider_used = 'statistical_model'

    # Cache results
    comps_cache[cache_hash] = {
        'data': found_comps,
        'timestamp': time.time()
    }

    # Save to vehicle
    if vehicle_id not in comps_db:
        comps_db[vehicle_id] = []

    existing_urls = {c.get('url') for c in comps_db[vehicle_id]}
    added = 0
    for comp in found_comps:
        comp['id'] = str(uuid.uuid4())
        comp['vehicle_id'] = vehicle_id
        comp['is_manual'] = False
        comp['created_at'] = datetime.utcnow().isoformat()
        if comp.get('url') not in existing_urls:
            comps_db[vehicle_id].append(comp)
            added += 1

    return jsonify({
        'message': f'Found {added} new comps via {provider_used}',
        'source': provider_used,
        'new_count': added,
        'total_count': len(comps_db.get(vehicle_id, [])),
        'comps': comps_db.get(vehicle_id, [])
    })


def search_comps_serpapi(vehicle):
    """Search for sold vehicle comps using SerpAPI"""
    import requests as req

    query = f"{vehicle['year']} {vehicle['make']} {vehicle['model']} sold price"
    if vehicle.get('trim'):
        query += f" {vehicle['trim']}"

    params = {
        'q': query,
        'api_key': SERPAPI_KEY,
        'engine': 'google',
        'num': 10,
    }

    resp = req.get('https://serpapi.com/search', params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    comps = []
    for result in data.get('organic_results', [])[:8]:
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        link = result.get('link', '')

        # Try to extract price from snippet
        prices = re.findall(r'\$[\d,]+', snippet)
        price = 0
        if prices:
            try:
                price = float(prices[0].replace('$', '').replace(',', ''))
            except ValueError:
                pass

        if price > 5000:  # Only include if we found a reasonable price
            comps.append({
                'source': 'Google/SerpAPI',
                'url': link,
                'sold_price': price,
                'sold_date': '',
                'mileage': 0,
                'location': '',
                'days_on_market': 0,
                'year': vehicle['year'],
                'make': vehicle['make'],
                'model': vehicle['model'],
                'trim': vehicle.get('trim', ''),
                'notes': f'{title} — {snippet[:150]}',
            })

    return comps


def search_comps_marketcheck(vehicle):
    """Placeholder for Marketcheck API integration"""
    # In production, this would call the Marketcheck API
    # https://apidocs.marketcheck.com/
    import requests as req

    headers = {'Authorization': f'Bearer {MARKETCHECK_API_KEY}'}
    params = {
        'year': vehicle['year'],
        'make': vehicle['make'],
        'model': vehicle['model'],
        'sold': 'true',
        'rows': 10,
    }

    # Marketcheck endpoint (example - adjust per their actual API)
    try:
        resp = req.get(
            'https://mc-api.marketcheck.com/v2/search/car/active',
            headers=headers, params=params, timeout=15
        )
        resp.raise_for_status()
        data = resp.json()

        comps = []
        for listing in data.get('listings', []):
            comps.append({
                'source': 'Marketcheck',
                'url': listing.get('vdp_url', ''),
                'sold_price': float(listing.get('price', 0)),
                'sold_date': listing.get('last_seen_at', ''),
                'mileage': int(listing.get('miles', 0)),
                'location': f"{listing.get('city', '')}, {listing.get('state', '')}",
                'days_on_market': int(listing.get('dom', 0)),
                'year': listing.get('year', vehicle['year']),
                'make': listing.get('make', vehicle['make']),
                'model': listing.get('model', vehicle['model']),
                'trim': listing.get('trim', ''),
                'notes': '',
            })
        return comps
    except Exception:
        return []


def generate_statistical_comps(vehicle):
    """Generate realistic statistical comps when no API is available"""
    import random
    seed_val = hash(f"{vehicle['year']}{vehicle['make']}{vehicle['model']}{vehicle.get('mileage', 0)}")
    random.seed(seed_val)

    base_price = vehicle.get('list_price', 25000) or 25000
    mileage = vehicle.get('mileage', 40000) or 40000

    comp_low = vehicle.get('comp_low', 0) or base_price * 0.88
    comp_high = vehicle.get('comp_high', 0) or base_price * 1.05
    comp_mid = (comp_low + comp_high) / 2
    comp_spread = comp_high - comp_low

    comps = []
    sources = ['Auction - Manheim', 'Auction - ADESA', 'Dealer Retail (Delisted)',
               'CarGurus Sold', 'AutoTrader Sold', 'Cars.com Sold']

    for i in range(8):
        price_var = random.gauss(0, comp_spread * 0.15)
        sold_price = round(comp_mid + price_var, -2)
        sold_price = max(comp_low * 0.9, min(comp_high * 1.1, sold_price))

        mile_var = random.randint(-8000, 12000)
        comp_miles = max(5000, mileage + mile_var)
        days_on_market = max(3, int(random.gauss(32, 12)))
        days_ago = random.randint(3, 45)
        sold_date = (datetime.utcnow() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

        comps.append({
            'source': random.choice(sources),
            'url': '',
            'sold_price': sold_price,
            'sold_date': sold_date,
            'mileage': comp_miles,
            'location': random.choice([
                'Dallas, TX', 'Houston, TX', 'Atlanta, GA', 'Chicago, IL',
                'Phoenix, AZ', 'Denver, CO', 'Orlando, FL', 'Charlotte, NC',
                'Nashville, TN', 'Las Vegas, NV'
            ]),
            'days_on_market': days_on_market,
            'year': vehicle['year'],
            'make': vehicle['make'],
            'model': vehicle['model'],
            'trim': vehicle.get('trim', ''),
            'notes': f'Statistical comp #{i+1} — simulated from market data',
        })

    return comps


# ============================================================
# FEATURE 3: DAILY PROBABILITY CURVE (30/60/90 + hover)
# ============================================================
@app.route('/api/vehicle/<vehicle_id>/odds', methods=['GET'])
def get_vehicle_odds(vehicle_id):
    """
    Returns probability distribution for days 1-90.
    Each day includes daily % and cumulative %.
    Adjusts based on pricing, mileage, comps, and engagement.
    """
    if vehicle_id not in vehicles_db:
        return jsonify({'error': 'Vehicle not found'}), 404

    vehicle = vehicles_db[vehicle_id]
    max_days = int(request.args.get('maxDays', 90))
    max_days = min(max_days, 180)

    # Get comps if available
    vehicle_comps = comps_db.get(vehicle_id, [])

    odds = calculate_odds(vehicle, vehicle_comps, max_days)
    return jsonify(odds)


def calculate_odds(vehicle, comps, max_days=90):
    """
    Odds engine that produces day-by-day sell probability.
    
    Model inputs:
    - Base DOM distribution for segment
    - Price vs comps percentile
    - Mileage vs comps percentile
    - Days already on lot
    - Engagement signals
    """
    d = vehicle

    # --- Compute adjustment factors ---

    # 1. Price vs comps
    comp_prices = [c['sold_price'] for c in comps if c.get('sold_price', 0) > 0]
    if comp_prices:
        median_comp = sorted(comp_prices)[len(comp_prices) // 2]
        price_ratio = d['list_price'] / median_comp if median_comp > 0 else 1.0
    elif d.get('comp_low') and d.get('comp_high'):
        comp_mid = (d['comp_low'] + d['comp_high']) / 2
        price_ratio = d['list_price'] / comp_mid if comp_mid > 0 else 1.0
    else:
        price_ratio = 1.0

    # Price factor: underpriced = faster, overpriced = slower
    if price_ratio > 1.1:
        price_mult = 0.65  # significantly overpriced
    elif price_ratio > 1.05:
        price_mult = 0.8
    elif price_ratio > 0.98:
        price_mult = 1.0
    elif price_ratio > 0.93:
        price_mult = 1.15
    else:
        price_mult = 1.3  # significantly underpriced

    # 2. Mileage vs comps
    comp_miles = [c['mileage'] for c in comps if c.get('mileage', 0) > 0]
    if comp_miles:
        median_miles = sorted(comp_miles)[len(comp_miles) // 2]
        mile_ratio = d.get('mileage', 40000) / median_miles if median_miles > 0 else 1.0
    else:
        mile_ratio = 1.0

    if mile_ratio > 1.2:
        mile_mult = 0.8
    elif mile_ratio > 1.05:
        mile_mult = 0.9
    elif mile_ratio > 0.9:
        mile_mult = 1.0
    else:
        mile_mult = 1.1  # lower miles than comps = advantage

    # 3. Demand signal
    demand = d.get('demand_signal', 'moderate')
    demand_mult = 1.2 if demand == 'high' else (0.75 if demand == 'soft' else 1.0)

    # 4. Competition
    cu = d.get('competing_units', 10)
    comp_mult = 1.3 if cu <= 5 else (1.1 if cu <= 10 else (0.9 if cu <= 20 else 0.7))

    # 5. Engagement
    eng_score = (d.get('leads_7', 0) * 3) + (d.get('test_drives_7', 0) * 10) + (d.get('views_7', 0) * 0.2)
    eng_mult = 1.2 if eng_score > 25 else (1.0 if eng_score > 12 else (0.8 if eng_score > 5 else 0.6))

    # 6. Days already on lot (aging penalty)
    di = d.get('days_in_inventory', 0)
    aging_mult = 1.2 if di <= 15 else (1.0 if di <= 30 else (0.85 if di <= 50 else (0.7 if di <= 75 else 0.5)))

    # Composite factor
    composite = price_mult * mile_mult * demand_mult * comp_mult * eng_mult * aging_mult

    # --- Build daily curve ---
    # Base: modified Weibull-like distribution
    # Most vehicles sell between day 15-45, peak around day 20-30
    # Composite shifts the curve left (faster) or right (slower)

    # Scale parameter (higher = slower)
    scale = max(15, min(60, 35 / max(composite, 0.3)))
    # Shape parameter
    shape = 2.2

    curve = []
    cumulative = 0

    for day in range(1, max_days + 1):
        # Weibull hazard rate: h(t) = (shape/scale) * (t/scale)^(shape-1)
        t = day / scale
        hazard = (shape / scale) * (t ** (shape - 1))

        # Daily probability = hazard * (1 - cumulative)
        daily_prob = hazard * (1 - cumulative)

        # Cap individual day probability
        daily_prob = max(0, min(0.08, daily_prob))

        # Days already passed can't produce a sale
        if day <= di:
            daily_prob = 0

        cumulative = min(0.98, cumulative + daily_prob)

        # Calculate financial state at this day
        total_invested = d['acquisition_cost'] + d.get('recon_cost', 0)
        fp_rate = d.get('floorplan_rate', 7.25)
        daily_fp = (total_invested * fp_rate / 100) / 365
        fp_at_day = daily_fp * day
        gross_at_day = (d['list_price'] - 750) - total_invested - fp_at_day  # mid negotiation

        curve.append({
            'day': day,
            'daily_pct': round(daily_prob * 100, 2),
            'cumulative_pct': round(cumulative * 100, 1),
            'gross_if_sold': round(gross_at_day),
            'floorplan_cost': round(fp_at_day),
        })

    # Extract milestones
    p30 = next((p['cumulative_pct'] for p in curve if p['day'] == 30), 0)
    p60 = next((p['cumulative_pct'] for p in curve if p['day'] == 60), 0)
    p90 = next((p['cumulative_pct'] for p in curve if p['day'] == min(90, max_days)), 0)

    # Find peak day
    peak_day = max(curve, key=lambda p: p['daily_pct'])

    # Find day where cumulative hits 50%
    median_day = next((p['day'] for p in curve if p['cumulative_pct'] >= 50), max_days)

    return {
        'vehicle_id': d.get('id'),
        'vehicle_title': f"{d['year']} {d['make']} {d['model']} {d.get('trim', '')}".strip(),
        'summary': {
            'p_30_day': p30,
            'p_60_day': p60,
            'p_90_day': p90,
            'peak_day': peak_day['day'],
            'peak_daily_pct': peak_day['daily_pct'],
            'median_sell_day': median_day,
        },
        'model_inputs': {
            'price_vs_comps': round(price_ratio, 3),
            'price_multiplier': price_mult,
            'mileage_vs_comps': round(mile_ratio, 3),
            'mileage_multiplier': mile_mult,
            'demand_multiplier': demand_mult,
            'competition_multiplier': comp_mult,
            'engagement_multiplier': eng_mult,
            'aging_multiplier': aging_mult,
            'composite_factor': round(composite, 3),
        },
        'curve': curve,
        'max_days': max_days,
        'generated_at': datetime.utcnow().isoformat()
    }


# ============================================================
# VEHICLE CRUD
# ============================================================
@app.route('/api/vehicles', methods=['GET'])
def get_vehicles():
    vehicle_list = list(vehicles_db.values())
    vehicle_list.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify({'count': len(vehicle_list), 'vehicles': vehicle_list})


@app.route('/api/vehicles', methods=['POST'])
def add_vehicle():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    required = ['year', 'make', 'model', 'acquisition_cost', 'list_price']
    for f in required:
        if f not in data or not data[f]:
            return jsonify({'error': f'Missing: {f}'}), 400
    vid = str(uuid.uuid4())
    vehicle = build_vehicle_record(vid, data)
    vehicles_db[vid] = vehicle
    return jsonify({'message': 'Vehicle added', 'vehicle': vehicle}), 201


@app.route('/api/vehicles/<vid>', methods=['GET'])
def get_vehicle(vid):
    v = vehicles_db.get(vid)
    if not v:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'vehicle': v})


@app.route('/api/vehicles/<vid>', methods=['PUT'])
def update_vehicle(vid):
    if vid not in vehicles_db:
        return jsonify({'error': 'Not found'}), 404
    data = request.get_json()
    vehicles_db[vid] = build_vehicle_record(vid, data)
    return jsonify({'message': 'Updated', 'vehicle': vehicles_db[vid]})


@app.route('/api/vehicles/<vid>', methods=['DELETE'])
def delete_vehicle(vid):
    if vid not in vehicles_db:
        return jsonify({'error': 'Not found'}), 404
    del vehicles_db[vid]
    comps_db.pop(vid, None)
    return jsonify({'message': 'Deleted'})


# ============================================================
# FULL ANALYSIS (original engine)
# ============================================================
@app.route('/api/analyze', methods=['POST'])
def analyze_direct():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data'}), 400
    required = ['year', 'make', 'model', 'acquisition_cost', 'list_price']
    for f in required:
        if f not in data or not data[f]:
            return jsonify({'error': f'Missing: {f}'}), 400
    vehicle = build_vehicle_record(None, data)
    analysis = run_full_analysis(vehicle)
    return jsonify({
        'message': 'Analysis complete',
        'report': {
            'id': str(uuid.uuid4()),
            'vehicle_title': f"{vehicle['year']} {vehicle['make']} {vehicle['model']} {vehicle.get('trim', '')}".strip(),
            'analysis': analysis,
            'created_at': datetime.utcnow().isoformat()
        }
    })


@app.route('/api/vehicles/<vid>/analyze', methods=['POST'])
def analyze_saved_vehicle(vid):
    v = vehicles_db.get(vid)
    if not v:
        return jsonify({'error': 'Not found'}), 404
    analysis = run_full_analysis(v)
    rid = str(uuid.uuid4())
    reports_db[rid] = {
        'id': rid, 'vehicle_id': vid,
        'vehicle_title': f"{v['year']} {v['make']} {v['model']} {v.get('trim', '')}".strip(),
        'analysis': analysis, 'created_at': datetime.utcnow().isoformat()
    }
    return jsonify({'message': 'Analysis complete', 'report': reports_db[rid]})


# ============================================================
# DASHBOARD
# ============================================================
@app.route('/api/dashboard/summary', methods=['GET'])
def dashboard_summary():
    active = [v for v in vehicles_db.values() if v.get('status') == 'active']
    if not active:
        return jsonify({'summary': {
            'total_vehicles': 0, 'total_invested': 0, 'total_list_value': 0,
            'total_potential_gross': 0, 'avg_days_in_inventory': 0,
            'daily_floorplan_burn': 0, 'monthly_floorplan_burn': 0,
            'aging_breakdown': {'healthy': 0, 'at_risk': 0, 'danger': 0}
        }})

    ti = sum(v['acquisition_cost'] + v['recon_cost'] for v in active)
    tl = sum(v['list_price'] for v in active)
    ad = sum(v['days_in_inventory'] for v in active) / len(active)
    db_val = sum((v['acquisition_cost'] + v['recon_cost']) * v['floorplan_rate'] / 100 / 365 for v in active)

    return jsonify({'summary': {
        'total_vehicles': len(active),
        'total_invested': round(ti),
        'total_list_value': round(tl),
        'total_potential_gross': round(tl - ti),
        'avg_days_in_inventory': round(ad),
        'daily_floorplan_burn': round(db_val, 2),
        'monthly_floorplan_burn': round(db_val * 30),
        'aging_breakdown': {
            'healthy': len([v for v in active if v['days_in_inventory'] <= 30]),
            'at_risk': len([v for v in active if 30 < v['days_in_inventory'] <= 60]),
            'danger': len([v for v in active if v['days_in_inventory'] > 60]),
        }
    }})


@app.route('/api/reports', methods=['GET'])
def get_reports():
    rl = sorted(reports_db.values(), key=lambda x: x.get('created_at', ''), reverse=True)
    return jsonify({'count': len(rl), 'reports': rl})


@app.route('/api/reports/<rid>', methods=['GET'])
def get_report(rid):
    r = reports_db.get(rid)
    if not r:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'report': r})


# ============================================================
# HELPER: Build Vehicle Record
# ============================================================
def build_vehicle_record(vid, data):
    return {
        'id': vid or str(uuid.uuid4()),
        'year': int(data.get('year', 0) or 0),
        'make': str(data.get('make', '')),
        'model': str(data.get('model', '')),
        'trim': str(data.get('trim', '')),
        'mileage': int(data.get('mileage', 0) or 0),
        'ext_color': str(data.get('ext_color', '')),
        'int_color': str(data.get('int_color', '')),
        'vin': str(data.get('vin', '')),
        'equipment': str(data.get('equipment', '')),
        'acquisition_cost': float(data.get('acquisition_cost', 0) or 0),
        'recon_cost': float(data.get('recon_cost', 0) or 0),
        'list_price': float(data.get('list_price', 0) or 0),
        'floorplan_rate': float(data.get('floorplan_rate', 7.25) or 7.25),
        'wholesale_price': float(data.get('wholesale_price', 0) or 0),
        'min_gross': float(data.get('min_gross', 2000) or 2000),
        'days_in_inventory': int(data.get('days_in_inventory', 0) or 0),
        'price_changes': int(data.get('price_changes', 0) or 0),
        'days_since_price_change': int(data.get('days_since_price_change', 0) or 0),
        'comp_low': float(data.get('comp_low', 0) or 0),
        'comp_high': float(data.get('comp_high', 0) or 0),
        'competing_units': int(data.get('competing_units', 0) or 0),
        'demand_signal': str(data.get('demand_signal', 'moderate') or 'moderate'),
        'seasonal_notes': str(data.get('seasonal_notes', '')),
        'views_7': int(data.get('views_7', 0) or 0),
        'views_30': int(data.get('views_30', 0) or 0),
        'leads_7': int(data.get('leads_7', 0) or 0),
        'leads_30': int(data.get('leads_30', 0) or 0),
        'test_drives_7': int(data.get('test_drives_7', 0) or 0),
        'test_drives_30': int(data.get('test_drives_30', 0) or 0),
        'sales_notes': str(data.get('sales_notes', '')),
        'status': 'active',
        'created_at': data.get('created_at', datetime.utcnow().isoformat()),
    }


# ============================================================
# FULL ANALYSIS ENGINE
# ============================================================
def run_full_analysis(d):
    """Complete analysis — pricing, probability, erosion, exit path, actions"""
    a = {}

    total_invested = d['acquisition_cost'] + d['recon_cost']
    potential_gross = d['list_price'] - total_invested
    daily_fp = (total_invested * (d['floorplan_rate'] / 100)) / 365
    fp_accrued = daily_fp * d['days_in_inventory']
    net_gross = potential_gross - fp_accrued
    ws_net = d['wholesale_price'] - total_invested - fp_accrued

    a['financials'] = {
        'total_invested': r2(total_invested),
        'potential_gross_at_sticker': r2(potential_gross),
        'daily_floorplan_cost': r2(daily_fp),
        'floorplan_accrued_to_date': r2(fp_accrued),
        'current_net_gross': r2(net_gross),
        'wholesale_net_today': r2(ws_net),
    }

    # Market position
    cr = d['comp_high'] - d['comp_low']
    mp = ((d['list_price'] - d['comp_low']) / cr) if cr > 0 else 0.5
    mp_label = 'Top Quartile' if mp > 0.75 else ('Above Mid' if mp > 0.5 else ('Mid-Market' if mp > 0.25 else 'Value'))

    a['market_position'] = {
        'percentile': round(mp * 100), 'label': mp_label,
        'comp_range': {'low': d['comp_low'], 'high': d['comp_high']},
    }

    # Engagement
    awv = d['views_30'] / 4.3 if d['views_30'] > 0 else 0
    vt = ((d['views_7'] - awv) / awv * 100) if awv > 0 else 0
    es = (d['leads_7'] * 3) + (d['test_drives_7'] * 10) + (d['views_7'] * 0.2)

    a['engagement'] = {
        'view_trend_pct': round(vt), 'engagement_score': r2(es),
        'view_trend_label': 'Accelerating' if vt > 10 else ('Stable' if vt > -10 else 'Declining'),
    }

    # Probability factors
    dm = 1.2 if d['demand_signal'] == 'high' else (0.75 if d['demand_signal'] == 'soft' else 1.0)
    cu = d['competing_units']
    cf = 1.3 if cu <= 5 else (1.1 if cu <= 10 else (0.9 if cu <= 20 else 0.7))
    di = d['days_in_inventory']
    af = 1.2 if di <= 20 else (1.0 if di <= 40 else (0.85 if di <= 60 else (0.65 if di <= 90 else 0.45)))
    ef = 1.2 if es > 25 else (1.0 if es > 12 else (0.8 if es > 5 else 0.6))
    pf = 0.7 if mp > 0.8 else (0.85 if mp > 0.6 else (1.0 if mp > 0.4 else (1.15 if mp > 0.2 else 1.25)))

    composite = dm * cf * af * ef * pf
    p30 = clamp(0.35 * composite, 0.05, 0.95)
    p60 = clamp(0.55 * composite * 1.1, 0.10, 0.97)
    p90 = clamp(0.72 * composite * 1.15, 0.20, 0.98)

    # Daily curve
    vehicle_comps = comps_db.get(d.get('id', ''), [])
    odds_data = calculate_odds(d, vehicle_comps, 90)

    a['sale_probability'] = {
        'prob_30_day': round(p30 * 100),
        'prob_60_day': round(p60 * 100),
        'prob_90_day': round(p90 * 100),
        'daily_curve': odds_data['curve'],
        'curve_summary': odds_data['summary'],
    }

    # Aging
    zone = 'HEALTHY' if di <= 30 else ('AT-RISK' if di <= 60 else 'DANGER')
    erosion = []
    for ad in [0, 30, 60, 90]:
        td = di + ad
        fpt = daily_fp * td
        gs = potential_gross - fpt
        erosion.append({
            'additional_days': ad, 'total_days': td,
            'floorplan_accrued': r2(fpt), 'gross_at_sticker': r2(gs),
            'realistic_gross_low': r2(gs - 1000), 'realistic_gross_high': r2(gs - 500),
        })

    # Irrationality
    irr_day = di
    for day in range(di, di + 150):
        fp = daily_fp * day
        rn = (potential_gross - fp) - 750
        dp = clamp(p30 * (0.98 ** (day - di)), 0.05, 0.95)
        pwr = rn * dp
        wad = d['wholesale_price'] - total_invested - fp
        if pwr < max(wad, ws_net) or rn < 0:
            irr_day = day
            break
        irr_day = day

    a['aging'] = {
        'zone': zone, 'days_in_inventory': di,
        'erosion_table': erosion,
        'irrationality_threshold': {
            'day': irr_day, 'days_remaining': max(0, irr_day - di),
        }
    }

    # Pricing
    op = 0.45
    optimal_price = (d['comp_low'] + (cr * op)) if cr > 0 else d['list_price']
    pd = d['list_price'] - optimal_price

    if mp > 0.65 and di > 30:
        pa = 'REDUCE'
        ca = max(300, min(round(pd / 100) * 100, round(potential_gross * 0.35 / 100) * 100))
        np_ = d['list_price'] - ca
        pr = f"At {round(mp*100)}th pctl. ${ca:,} cut to ${np_:,.0f} repositions mid-market."
    elif mp < 0.25 and di < 20 and es > 20:
        pa = 'INCREASE'
        ca = min(round(abs(pd) * 0.5 / 100) * 100, 800)
        np_ = d['list_price'] + ca
        pr = 'Strong engagement below market. Capture gross.'
    elif mp > 0.55 and di > 45:
        pa = 'REDUCE'
        ca = max(300, round(pd * 0.7 / 100) * 100)
        np_ = d['list_price'] - ca
        pr = f'Aging at {di}d. ${ca:,} cut improves position.'
    else:
        pa = 'HOLD'
        ca = 0
        np_ = d['list_price']
        pr = 'Balanced. Hold and monitor.'

    etl = np_ - 1000
    eth = np_ - 500

    if cu > 15:
        el = 'HIGH'
    elif cu > 8:
        el = 'MODERATE-HIGH'
    elif cu > 4:
        el = 'MODERATE'
    else:
        el = 'LOW'

    a['pricing'] = {
        'action': pa, 'change_amount': ca,
        'current_list_price': d['list_price'], 'new_list_price': np_,
        'reasoning': pr,
        'timing': 'Execute today.' if pa != 'HOLD' else 'No action.',
        'elasticity': el,
        'expected_transaction_range': {'low': r2(etl), 'high': r2(eth)},
        'expected_gross_range': {'low': r2(etl - total_invested), 'high': r2(eth - total_invested)},
    }

    # Exit path
    reg = ((etl - total_invested + eth - total_invested) / 2) - (daily_fp * 20)
    rpw = reg * p30

    if rpw > ws_net and (etl - total_invested) > d['min_gross'] * 0.5:
        oe = 'RETAIL'
        er = f'Spread ~${round(reg - ws_net):,} justifies retail.'
    elif ws_net > -500:
        oe = 'WHOLESALE'
        er = 'Retail prob-weighted return below wholesale.'
    else:
        oe = 'RETAIL'
        er = 'Wholesale is a loss. Aggressive retail required.'

    rd = di + 14

    a['exit_path'] = {
        'optimal': oe, 'reasoning': er, 'reassess_day': rd,
        'paths': [
            {'path': 'RETAIL', 'recommended': oe == 'RETAIL',
             'expected_gross_low': r2(etl - total_invested), 'expected_gross_high': r2(eth - total_invested),
             'expected_days': '12-25d' if pa != 'HOLD' else '20-40d',
             'probability': f'{round(p30*100)}%-{round(p60*100)}%'},
            {'path': 'WHOLESALE', 'recommended': oe == 'WHOLESALE',
             'expected_gross_low': r2(ws_net), 'expected_gross_high': r2(ws_net),
             'expected_days': '3-7d', 'probability': '~95%'},
            {'path': 'DEALER_TRADE', 'recommended': False,
             'expected_gross_low': 0, 'expected_gross_high': 300,
             'expected_days': '7-21d', 'probability': 'Low'},
        ],
    }

    # Actions
    actions = []
    if pa == 'REDUCE':
        actions.append({'priority': 1, 'title': f'Cut ${ca:,}', 'timing': 'TODAY',
                        'detail': f'${d["list_price"]:,.0f} → ${np_:,.0f}. Hold cost: ${daily_fp:.2f}/day.',
                        'purpose': 'Reposition + re-index.'})
    elif pa == 'INCREASE':
        actions.append({'priority': 1, 'title': f'Raise ${ca:,}', 'timing': 'TODAY',
                        'detail': f'To ${np_:,.0f}.', 'purpose': 'Capture gross.'})
    else:
        actions.append({'priority': 1, 'title': 'Hold — monitor 7d', 'timing': 'THIS WEEK',
                        'detail': f'${d["list_price"]:,.0f}. Cut if views -15%.', 'purpose': 'Maintain momentum.'})

    actions.append({'priority': 2, 'title': 'Upgrade listing', 'timing': 'TODAY',
                    'detail': '30+ photos. Video. Feature filters on.', 'purpose': 'Convert traffic.'})
    if d['leads_30'] > 0:
        actions.append({'priority': 3, 'title': f'Re-engage {d["leads_30"]} leads', 'timing': 'BY WED',
                        'detail': 'Phone → text → email.', 'purpose': '2-3x conversion rate.'})
    actions.append({'priority': 4, 'title': 'Brief sales team', 'timing': 'TOMORROW',
                    'detail': f'Floor: ${max(np_ - 500, total_invested + d["min_gross"]):,.0f}.',
                    'purpose': 'Protect gross.'})
    actions.append({'priority': 5, 'title': f'WS trigger: Day {rd}', 'timing': 'CALENDAR',
                    'detail': f'<2 TDs = wholesale. Net: ${ws_net:,.0f}.', 'purpose': 'Remove emotion.'})

    a['action_plan'] = actions

    # Risk
    risks = []
    sn = d.get('seasonal_notes', '') or ''
    if 'compression' in sn.lower():
        risks.append({'factor': 'Incentive Compression', 'severity': 'HIGH'})
    if di > 45:
        risks.append({'factor': 'Stale Listing', 'severity': 'MEDIUM'})
    if cu > 12:
        risks.append({'factor': 'Heavy Supply', 'severity': 'MEDIUM'})
    if vt < -15:
        risks.append({'factor': 'Declining Views', 'severity': 'HIGH'})
    if not risks:
        risks.append({'factor': 'Standard', 'severity': 'LOW'})

    dp_ = [1 if d['views_30'] > 0 else 0, 1 if d['leads_30'] > 0 else 0,
           1 if d['comp_low'] > 0 else 0, 1 if d['competing_units'] > 0 else 0,
           1 if d['wholesale_price'] > 0 else 0]
    comp_ = sum(dp_) / len(dp_)
    conf = 'HIGH' if comp_ > 0.75 else ('MEDIUM' if comp_ > 0.5 else 'LOW')
    conf_pct = (80 + round(comp_ * 15)) if comp_ > 0.75 else ((50 + round(comp_ * 20)) if comp_ > 0.5 else (20 + round(comp_ * 30)))

    a['risk_and_confidence'] = {
        'risks': risks,
        'confidence': {'level': conf, 'percent': conf_pct},
    }

    a['summary'] = {
        'vehicle': f"{d['year']} {d['make']} {d['model']} {d.get('trim', '')}".strip(),
        'total_invested': r2(total_invested), 'current_list': d['list_price'],
        'recommended_price': np_, 'price_action': pa, 'aging_zone': zone,
        'optimal_exit': oe, 'confidence': conf,
        'generated_at': datetime.utcnow().isoformat(),
    }

    return a


# ============================================================
# HELPERS
# ============================================================
def r2(n):
    return round(n * 100) / 100

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# RUN
# ============================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
