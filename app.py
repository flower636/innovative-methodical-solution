from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps
import requests
import uuid
from datetime import datetime, timedelta

# Import database manager
from database import DatabaseManager

# Import blueprints
from features.feature1.routes import feature1_bp
from features.feature2.routes import feature2_bp
from features.feature3.routes import feature3_bp
from features.feature4.routes import feature4_bp
from features.feature5.routes import feature5_bp


app = Flask(__name__)
app.secret_key = 'military_webapp_secret_key_2024'

# Enhanced Session Configuration
app.permanent_session_lifetime = timedelta(hours=2)  # Extended session time
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_PERMANENT'] = True

# Load users from JSON file
def load_users():
    try:
        with open('config/users.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"users": {}}

# Load features configuration
def load_features():
    try:
        with open('config/features.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"features": {}}

# Enhanced weather function with comprehensive data
def get_weather(city="Delhi"):
    api_key = "f8031cf783ffad54fbeb740459022fe1"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200 and data.get("main"):
            weather = {
                "city": data.get("name", city),
                "country": data.get("sys", {}).get("country", ""),
                "temperature": round(data["main"]["temp"], 1),
                "feels_like": round(data["main"]["feels_like"], 1),
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "wind_speed": round(data["wind"]["speed"], 1),
                "wind_deg": data["wind"].get("deg", 0),
                "visibility": round(data.get("visibility", 0) / 1000, 1),  # meters â†’ km
                "weather_main": data["weather"][0]["main"],
                "weather_desc": data["weather"][0]["description"].title(),
                "clouds": data["clouds"]["all"],
                "sunrise": data["sys"]["sunrise"],
                "sunset": data["sys"]["sunset"],
                "status": "success"
            }
            return weather
        else:
            error_msg = data.get("message", "City not found")
            return {
                "city": city, "country": "", "temperature": "N/A", "feels_like": "N/A",
                "humidity": "N/A", "pressure": "N/A", "wind_speed": "N/A", "wind_deg": 0,
                "visibility": "N/A", "weather_main": "N/A", "weather_desc": error_msg,
                "clouds": "N/A", "sunrise": 0, "sunset": 0, "status": "error"
            }
    except Exception as e:
        print("Weather API error:", e)
        return {
            "city": city, "country": "", "temperature": "N/A", "feels_like": "N/A",
            "humidity": "N/A", "pressure": "N/A", "wind_speed": "N/A", "wind_deg": 0,
            "visibility": "N/A", "weather_main": "N/A", "weather_desc": "Connection Error",
            "clouds": "N/A", "sunrise": 0, "sunset": 0, "status": "error"
        }

# Get client IP address
def get_client_ip():
    if request.environ.get('HTTP_X_FORWARDED_FOR'):
        return request.environ['HTTP_X_FORWARDED_FOR'].split(',')[0]
    return request.environ.get('REMOTE_ADDR', 'Unknown')

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            print(f"âŒ Login required - redirecting to login page")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Role-based access decorator
def role_required(roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' not in session:
                print(f"âŒ Role check failed - no username in session")
                return redirect(url_for('login'))
            users_data = load_users()
            user_role = users_data['users'].get(session['username'], {}).get('role', '')
            if user_role not in roles:
                print(f"âŒ Role check failed - user role '{user_role}' not in {roles}")
                return render_template('403.html'), 403
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Activity logging decorator
def log_activity_decorator(action_type, feature_name=None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'username' in session:
                DatabaseManager.log_activity(
                    username=session.get('username'),
                    role=session.get('role'),
                    action_type=action_type,
                    feature_name=feature_name,
                    ip_address=get_client_ip(),
                    session_id=session.get('session_id'),
                    additional_data={'endpoint': request.endpoint, 'method': request.method}
                )
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users_data = load_users()
        user = users_data['users'].get(username)
        
        print(f"Login attempt - Username: {username}")
        
        if user and user['password'] == password:
            # Make session permanent and generate session ID
            session.permanent = True
            session_id = str(uuid.uuid4())
            
            session['username'] = username
            session['role'] = user['role']
            session['session_id'] = session_id
            session['login_time'] = datetime.now().isoformat()
            
            print(f"âœ… Login successful - User: {username}, Role: {user['role']}, Session: {session_id}")
            print(f"Session contents after login: {dict(session)}")
            
            # Log successful login
            DatabaseManager.log_login(
                username=username,
                role=user['role'],
                ip_address=get_client_ip(),
                session_id=session_id
            )
            
            return redirect(url_for('dashboard'))
        else:
            print(f"âŒ Login failed - Invalid credentials for username: {username}")
            # Log failed login attempt
            DatabaseManager.log_activity(
                username=username if username else 'Unknown',
                role='Unknown',
                action_type='login_failed',
                ip_address=get_client_ip(),
                additional_data={'attempted_username': username}
            )
            flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'Unknown')
    session_id = session.get('session_id')
    
    print(f"ðŸšª Logout - User: {username}, Session: {session_id}")
    
    if session_id:
        DatabaseManager.log_logout(session_id)
    
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
@log_activity_decorator('dashboard_access')
def dashboard():
    users_data = load_users()
    features_data = load_features()
    user_role = session.get('role')
    username = session.get('username')
    
    print(f"Dashboard access - User: {username}, Role: {user_role}")
    
    # Get available features for user role
    available_features = []
    for feature_id, feature_info in features_data['features'].items():
        if user_role in feature_info['roles'] and feature_info['active']:
            available_features.append({
                'id': feature_id,
                'name': feature_info['name']
            })
    
    # Get weather data - check for location parameter or use default
    location = request.args.get('location', 'Delhi')
    weather = get_weather(location)
    
    return render_template('dashboard.html',
                          username=username,
                          role=user_role,
                          features=available_features,
                          weather=weather)

# Session debugging route
@app.route('/debug-session')
@login_required
def debug_session():
    return jsonify({
        'session': dict(session),
        'username': session.get('username'),
        'role': session.get('role'),
        'session_keys': list(session.keys()),
        'session_id': session.get('session_id'),
        'login_time': session.get('login_time'),
        'permanent': session.permanent
    })

# Commander Dashboard Route
@app.route('/commander-dashboard')
@login_required
@role_required(['captain', 'commander'])
@log_activity_decorator('commander_dashboard_access')
def commander_dashboard():
    stats = DatabaseManager.get_dashboard_stats()
    return render_template(
        'commander_dashboard.html',
        stats=stats,
        username=session.get("username", "Guest")
    )

# API Routes for Commander Dashboard
@app.route('/api/activity-logs')
@login_required
@role_required(['captain', 'commander'])
def api_activity_logs():
    username = request.args.get('username')
    action_type = request.args.get('action_type')
    limit = int(request.args.get('limit', 50))
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    logs = DatabaseManager.get_activity_logs(
        username=username,
        action_type=action_type,
        limit=limit,
        date_from=date_from,
        date_to=date_to
    )
    
    # Convert datetime objects to strings for JSON serialization
    for log in logs:
        if log.get('timestamp'):
            log['timestamp'] = log['timestamp'].isoformat()
    
    return jsonify(logs)

@app.route('/api/dashboard-stats')
@login_required
@role_required(['captain', 'commander'])
def api_dashboard_stats():
    stats = DatabaseManager.get_dashboard_stats()
    
    # Convert datetime objects to strings
    for activity in stats.get('recent_activity', []):
        if activity.get('timestamp'):
            activity['timestamp'] = activity['timestamp'].isoformat()
    
    for soldier in stats.get('soldier_summary', []):
        if soldier.get('last_activity'):
            soldier['last_activity'] = soldier['last_activity'].isoformat()
    
    return jsonify(stats)

# AJAX endpoint for dynamic weather updates
@app.route('/api/weather')
@login_required
def api_weather():
    location = request.args.get('location', 'Delhi')
    weather = get_weather(location)
    
    # Log weather API usage
    DatabaseManager.log_activity(
        username=session.get('username'),
        role=session.get('role'),
        action_type='weather_api_usage',
        ip_address=get_client_ip(),
        session_id=session.get('session_id'),
        additional_data={'location': location}
    )
    
    return jsonify(weather)

# Feature access logging - Add this to each feature blueprint
@app.before_request
def log_feature_access():
    if request.endpoint and 'username' in session:
        # Log feature access for specific features
        if any(request.endpoint.startswith(f) for f in ['feature1', 'feature2', 'feature3', 'feature4','feature5']):
            feature_name = request.endpoint.split('.')[0] if '.' in request.endpoint else request.endpoint
            DatabaseManager.log_activity(
                username=session.get('username'),
                role=session.get('role'),
                action_type='feature_access',
                feature_name=feature_name,
                ip_address=get_client_ip(),
                session_id=session.get('session_id'),
                additional_data={'full_endpoint': request.endpoint}
            )

# Enhanced session debugging middleware
@app.before_request
def debug_session_middleware():
    if request.endpoint and request.endpoint.startswith('feature2'):
        print(f"ðŸ” Request to {request.endpoint}")
        print(f"Session data: {dict(session)}")
        print(f"Username in session: {'username' in session}")
        print(f"Role in session: {session.get('role', 'NOT_SET')}")

# Register blueprints
app.register_blueprint(feature1_bp)
app.register_blueprint(feature2_bp)
app.register_blueprint(feature3_bp)
app.register_blueprint(feature4_bp)
app.register_blueprint(feature5_bp)

# Ensure required directories exist
def create_required_directories():
    directories = [
        'config',
        'static/uploads/feature2', 
        'static/uploads/feature3',
        'static/processed/feature3',
        'uploads/feature3',
        'features/feature6',           # ADD THIS
        'features/feature6/templates'  # ADD THIS
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"âœ… Directory ensured: {directory}")

if __name__ == '__main__':
    # Create required directories
    create_required_directories()
    
    print("ðŸš€ Starting Flask application...")
    print(f"Session configuration:")
    print(f"  - Lifetime: {app.permanent_session_lifetime}")
    print(f"  - Secure: {app.config['SESSION_COOKIE_SECURE']}")
    print(f"  - HttpOnly: {app.config['SESSION_COOKIE_HTTPONLY']}")
    print(f"  - SameSite: {app.config['SESSION_COOKIE_SAMESITE']}")
    
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)