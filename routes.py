# Complete routes.py for Feature 3 - Advanced Object Detection System

from flask import Blueprint, render_template, request, jsonify, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
import uuid
import zipfile
import tempfile
from datetime import datetime
import threading
import time
import cv2
import base64
import numpy as np
from .models import AdvancedDetectionModel

feature3_bp = Blueprint('feature3', __name__, 
                       url_prefix='/feature3',
                       template_folder='templates',
                       static_folder='static')

# Configuration
UPLOAD_FOLDER = 'uploads/feature3'
PROCESSED_FOLDER = 'static/processed/feature3'
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.mp4', '.avi', '.mov', '.mkv', '.wmv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB total per session

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def cleanup_all_files():
    """Clean up all uploaded and processed files on startup"""
    try:
        import shutil
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        if os.path.exists(PROCESSED_FOLDER):
            shutil.rmtree(PROCESSED_FOLDER)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        print("All files cleaned up on startup")
    except Exception as e:
        print(f"Cleanup error: {e}")
# Initialize detection model
detection_model = AdvancedDetectionModel()

# Clean up all files on startup
cleanup_all_files()

# Store active processing sessions
active_sessions = {}

def allowed_file(filename):
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

def get_file_size(file_path):
    return os.path.getsize(file_path)

def cleanup_session_files(session_id):
    """Clean up files for a specific session"""
    try:
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        processed_folder = os.path.join(PROCESSED_FOLDER, session_id)
        
        if os.path.exists(session_folder):
            for file in os.listdir(session_folder):
                os.remove(os.path.join(session_folder, file))
            os.rmdir(session_folder)
        
        if os.path.exists(processed_folder):
            for file in os.listdir(processed_folder):
                os.remove(os.path.join(processed_folder, file))
            os.rmdir(processed_folder)
    except Exception as e:
        print(f"Cleanup error: {e}")

# =====================================================
# MAIN ROUTES
# =====================================================

@feature3_bp.route('/')
def index():
    if 'username' not in session or 'role' not in session:
        return redirect(url_for('login'))
    
    if session['role'] not in ['captain', 'soldier']:
        return render_template('403.html'), 403
    
    return render_template('feature3.html')

# =====================================================
# IMAGE PROCESSING ROUTES
# =====================================================

@feature3_bp.route('/upload_images', methods=['POST'])
def upload_images():
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        uploaded_files = []
        total_size = 0
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
            
            filename = secure_filename(file.filename)
            temp_path = os.path.join(session_folder, filename)
            file.save(temp_path)
            
            file_size = get_file_size(temp_path)
            
            if file_size > MAX_FILE_SIZE:
                os.remove(temp_path)
                cleanup_session_files(session_id)
                return jsonify({'error': f'File too large: {filename} ({file_size/1024/1024:.1f}MB > 50MB)'}), 400
            
            total_size += file_size
            if total_size > MAX_TOTAL_SIZE:
                cleanup_session_files(session_id)
                return jsonify({'error': f'Total upload size too large ({total_size/1024/1024:.1f}MB > 200MB)'}), 400
            
            uploaded_files.append(temp_path)
        
        if not uploaded_files:
            cleanup_session_files(session_id)
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'uploaded_count': len(uploaded_files),
            'total_size_mb': round(total_size / 1024 / 1024, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@feature3_bp.route('/process_images', methods=['POST'])
def process_images():
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        detection_filter = data.get('detection_filter', 'all')  # NEW: Get filter
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
        
        # Get uploaded files
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Session not found'}), 404
        
        image_files = []
        for filename in os.listdir(session_folder):
            if allowed_file(filename):
                image_files.append(os.path.join(session_folder, filename))
        
        if not image_files:
            return jsonify({'error': 'No valid images found'}), 400
        
        # Process images with filter
        results = detection_model.process_images_web(image_files, session_id, detection_filter)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =====================================================
# VIDEO PROCESSING ROUTES
# =====================================================

@feature3_bp.route('/upload_videos', methods=['POST'])
def upload_videos():
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        files = request.files.getlist('videos')
        if not files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        session_id = str(uuid.uuid4())
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_folder, exist_ok=True)
        
        uploaded_files = []
        total_size = 0
        
        for file in files:
            if file.filename == '':
                continue
            
            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
            
            filename = secure_filename(file.filename)
            temp_path = os.path.join(session_folder, filename)
            file.save(temp_path)
            
            file_size = get_file_size(temp_path)
            
            if file_size > MAX_FILE_SIZE:
                os.remove(temp_path)
                cleanup_session_files(session_id)
                return jsonify({'error': f'File too large: {filename} ({file_size/1024/1024:.1f}MB > 50MB)'}), 400
            
            total_size += file_size
            if total_size > MAX_TOTAL_SIZE:
                cleanup_session_files(session_id)
                return jsonify({'error': f'Total upload size too large ({total_size/1024/1024:.1f}MB > 200MB)'}), 400
            
            uploaded_files.append(temp_path)
        
        if not uploaded_files:
            cleanup_session_files(session_id)
            return jsonify({'error': 'No valid files uploaded'}), 400
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'uploaded_count': len(uploaded_files),
            'total_size_mb': round(total_size / 1024 / 1024, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@feature3_bp.route('/process_videos', methods=['POST'])
def process_videos():
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        detection_filter = data.get('detection_filter', 'all')  # NEW: Get filter
        
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
        
        # Get uploaded files
        session_folder = os.path.join(UPLOAD_FOLDER, session_id)
        if not os.path.exists(session_folder):
            return jsonify({'error': 'Session not found'}), 404
        
        video_files = []
        for filename in os.listdir(session_folder):
            if allowed_file(filename) and any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']):
                video_files.append(os.path.join(session_folder, filename))
        
        if not video_files:
            return jsonify({'error': 'No valid videos found'}), 400
        
        # Initialize session for progress tracking
        active_sessions[session_id] = {
            'status': 'processing',
            'progress': 0,
            'current_file': '',
            'cancel_requested': False
        }
        
        # Start background processing with filter
        def process_in_background():
            try:
                results = detection_model.process_videos_web(video_files, session_id, active_sessions, detection_filter)
                active_sessions[session_id]['status'] = 'completed'
                active_sessions[session_id]['results'] = results
                active_sessions[session_id]['progress'] = 100
            except Exception as e:
                active_sessions[session_id]['status'] = 'error'
                active_sessions[session_id]['error'] = str(e)
        
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({'success': True, 'session_id': session_id})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@feature3_bp.route('/video_progress/<session_id>')
def video_progress(session_id):
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    if session_id in active_sessions:
        return jsonify(active_sessions[session_id])
    else:
        return jsonify({'status': 'not_found'}), 404

@feature3_bp.route('/cancel_video/<session_id>', methods=['POST'])
def cancel_video(session_id):
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    if session_id in active_sessions:
        active_sessions[session_id]['cancel_requested'] = True
        return jsonify({'success': True})
    else:
        return jsonify({'error': 'Session not found'}), 404



# =====================================================
# DOWNLOAD ROUTES
# =====================================================

@feature3_bp.route('/download_image/<session_id>/<filename>')
def download_image(session_id, filename):
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        processed_folder = os.path.join(PROCESSED_FOLDER, session_id)
        file_path = os.path.join(processed_folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@feature3_bp.route('/download_all_images/<session_id>')
def download_all_images(session_id):
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        processed_folder = os.path.join(PROCESSED_FOLDER, session_id)
        
        if not os.path.exists(processed_folder):
            return jsonify({'error': 'Session not found'}), 404
        
        zip_path = os.path.join(tempfile.gettempdir(), f'detected_images_{session_id}.zip')
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for filename in os.listdir(processed_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                    file_path = os.path.join(processed_folder, filename)
                    zipf.write(file_path, filename)
        
        return send_file(zip_path, as_attachment=True, download_name=f'detected_images_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@feature3_bp.route('/download_video/<session_id>/<filename>')
def download_video(session_id, filename):
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        processed_folder = os.path.join(PROCESSED_FOLDER, session_id)
        file_path = os.path.join(processed_folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================
# CLEANUP ROUTES
# =====================================================

@feature3_bp.route('/cleanup_session/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    if 'username' not in session or session['role'] not in ['captain', 'soldier']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        cleanup_session_files(session_id)
        
        if session_id in active_sessions:
            del active_sessions[session_id]
        
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500