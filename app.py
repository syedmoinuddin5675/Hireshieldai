import os
import hashlib
import secrets
import smtplib
import ssl
import uuid
from email.message import EmailMessage
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask import send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from sqlalchemy import event, text

app = Flask(__name__)
app.secret_key = os.environ.setdefault('SECRET_KEY', secrets.token_hex(32))

app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///industry_ats.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'uploads'
PROFILE_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, 'profiles')
os.makedirs(PROFILE_UPLOAD_FOLDER, exist_ok=True)

# --- ENTERPRISE DATABASE MODELS ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(30), nullable=False) # candidate, recruiter, super_admin
    tc_accepted = db.Column(db.Boolean, default=False)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    recruiter_id = db.Column(db.Integer, nullable=True)
    title = db.Column(db.String(150), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='Active') # Active, Archived

class Application(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    resume_filename = db.Column(db.String(200), nullable=False)
    ai_match_score = db.Column(db.Float, nullable=False)
    stage = db.Column(db.String(50), default='Applied')
    job = db.relationship('Job')

class CandidateProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    headline = db.Column(db.String(160), default='')
    phone = db.Column(db.String(40), default='')
    location = db.Column(db.String(120), default='')
    skills = db.Column(db.Text, default='')
    summary = db.Column(db.Text, default='')
    resume_filename = db.Column(db.String(200), default='')

class ProfessionalProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    headline = db.Column(db.String(160), default='')
    phone = db.Column(db.String(40), default='')
    location = db.Column(db.String(120), default='')
    bio = db.Column(db.Text, default='')
    photo_filename = db.Column(db.String(255), default='')

class PasswordResetToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token_hash = db.Column(db.String(64), unique=True, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime)

class InterviewAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    technical_score = db.Column(db.Integer, nullable=False)
    communication_score = db.Column(db.Integer, nullable=False)
    overall_score = db.Column(db.Integer, nullable=False)
    completed_at = db.Column(db.String(100), nullable=False)

class InterviewSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, nullable=True)
    candidate_name = db.Column(db.String(100), nullable=False)
    job_title = db.Column(db.String(150), nullable=False)
    interview_date = db.Column(db.String(50), nullable=False)
    interviewer = db.Column(db.String(100), nullable=False)

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.String(100), nullable=False)
    previous_hash = db.Column(db.String(64), nullable=False, default='')
    record_hash = db.Column(db.String(64), nullable=False, default='')

@event.listens_for(AuditLog, 'before_insert')
def hash_audit_log(mapper, connection, target):
    previous_hash = connection.execute(db.select(AuditLog.record_hash).order_by(AuditLog.id.desc()).limit(1)).scalar_one_or_none() or ''
    target.previous_hash = previous_hash
    payload = f'{previous_hash}|{target.timestamp}|{target.action}'
    target.record_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()

with app.app_context():
    db.create_all()
    with db.engine.connect() as connection:
        connection.exec_driver_sql('BEGIN IMMEDIATE')
        columns = {row[1] for row in connection.exec_driver_sql('PRAGMA table_info(audit_log)').fetchall()}
        needs_hash_backfill = 'previous_hash' not in columns or 'record_hash' not in columns
        if 'previous_hash' not in columns:
            connection.exec_driver_sql("ALTER TABLE audit_log ADD COLUMN previous_hash VARCHAR(64) NOT NULL DEFAULT ''")
        if 'record_hash' not in columns:
            connection.exec_driver_sql("ALTER TABLE audit_log ADD COLUMN record_hash VARCHAR(64) NOT NULL DEFAULT ''")
        schedule_columns = {row[1] for row in connection.exec_driver_sql('PRAGMA table_info(interview_schedule)').fetchall()}
        if 'candidate_id' not in schedule_columns:
            connection.exec_driver_sql('ALTER TABLE interview_schedule ADD COLUMN candidate_id INTEGER')
            named_users = connection.exec_driver_sql('SELECT name, MIN(id), COUNT(*) FROM user GROUP BY name HAVING COUNT(*) = 1').fetchall()
            for name, user_id, _ in named_users:
                connection.execute(text('UPDATE interview_schedule SET candidate_id = :user_id WHERE candidate_name = :name'), {'user_id': user_id, 'name': name})
        job_columns = {row[1] for row in connection.exec_driver_sql('PRAGMA table_info(job)').fetchall()}
        if 'recruiter_id' not in job_columns:
            connection.exec_driver_sql('ALTER TABLE job ADD COLUMN recruiter_id INTEGER')
        legacy_jobs = connection.exec_driver_sql('SELECT id, title FROM job WHERE recruiter_id IS NULL').fetchall()
        recruiters = connection.exec_driver_sql("SELECT id, name FROM user WHERE role = 'recruiter'").fetchall()
        for job_id, title in legacy_jobs:
            matches = []
            for recruiter_id, name in recruiters:
                action = f'Recruiter {name} published job requisition: {title}'
                if connection.exec_driver_sql('SELECT 1 FROM audit_log WHERE action = ? LIMIT 1', (action,)).first():
                    matches.append(recruiter_id)
            if len(matches) == 1:
                connection.execute(text('UPDATE job SET recruiter_id = :recruiter_id WHERE id = :job_id'), {'recruiter_id': matches[0], 'job_id': job_id})
        if needs_hash_backfill:
            previous_hash = ''
            rows = connection.exec_driver_sql('SELECT id, action, timestamp FROM audit_log ORDER BY id').fetchall()
            for row in rows:
                payload = f'{previous_hash}|{row.timestamp}|{row.action}'
                record_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
                connection.execute(text('UPDATE audit_log SET previous_hash = :previous_hash, record_hash = :record_hash WHERE id = :id'), {'previous_hash': previous_hash, 'record_hash': record_hash, 'id': row.id})
                previous_hash = record_hash
        connection.exec_driver_sql("CREATE TRIGGER IF NOT EXISTS audit_log_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END")
        connection.exec_driver_sql("CREATE TRIGGER IF NOT EXISTS audit_log_no_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END")
        connection.commit()
    normalized_accounts = User.query.filter_by(role='hiring_manager').update({User.role: 'recruiter'}, synchronize_session=False)
    if normalized_accounts:
        db.session.add(AuditLog(action=f'Normalized {normalized_accounts} legacy hiring manager account(s) to recruiter role.', timestamp=str(datetime.now())))
        db.session.commit()

# --- ADVANCED AI NLP MATCHING ENGINE ---
def compute_ai_match(job_desc, resume_path):
    text = ""
    try:
        reader = PdfReader(resume_path)
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
    except Exception:
        pass
    if not text.strip():
        return 65.0
    vec = TfidfVectorizer(stop_words='english')
    mat = vec.fit_transform([job_desc, text])
    score = cosine_similarity(mat[0:1], mat[1:2])[0][0] * 100
    return round(score, 2)

def get_candidate_profile(user_id):
    profile = CandidateProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        profile = CandidateProfile(user_id=user_id)
        db.session.add(profile)
        db.session.commit()
    return profile

def extract_resume_text(path):
    try:
        reader = PdfReader(path)
        return '\n'.join(page.extract_text() or '' for page in reader.pages).strip()
    except Exception:
        return ''

def extract_resume_skills(text):
    known_skills = ('python', 'java', 'javascript', 'typescript', 'sql', 'flask', 'django', 'react', 'node.js', 'aws', 'azure', 'docker', 'kubernetes', 'machine learning', 'data analysis', 'communication', 'project management', 'html', 'css', 'git', 'c++', 'c#')
    lowered = text.lower()
    return ', '.join(skill for skill in known_skills if skill in lowered)

def require_role(*roles):
    user_id = session.get('user_id')
    user = db.session.get(User, user_id) if user_id else None
    if not user:
        session.clear()
        return redirect(url_for('index'))
    session['role'] = user.role
    session['user_name'] = user.name
    profile = ProfessionalProfile.query.filter_by(user_id=user.id).first()
    session['profile_photo'] = profile.photo_filename if profile else ''
    if user.role not in roles:
        return redirect(url_for('index'))
    return None

def send_email(recipient, subject, body):
    mail_server = os.environ.get('MAIL_SERVER')
    sender = os.environ.get('MAIL_DEFAULT_SENDER')
    if not mail_server or not sender:
        return False
    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = sender
    message['To'] = recipient
    message.set_content(body)
    try:
        port = int(os.environ.get('MAIL_PORT', '587'))
        with smtplib.SMTP(mail_server, port, timeout=15) as smtp:
            if os.environ.get('MAIL_USE_TLS', 'true').lower() in ('true', '1', 'yes'):
                smtp.starttls(context=ssl.create_default_context())
            username = os.environ.get('MAIL_USERNAME')
            password = os.environ.get('MAIL_PASSWORD')
            if username and password:
                smtp.login(username, password)
            smtp.send_message(message)
        return True
    except (OSError, smtplib.SMTPException, ValueError):
        app.logger.exception('Email delivery failed')
        return False

def get_professional_profile(user_id):
    profile = ProfessionalProfile.query.filter_by(user_id=user_id).first()
    if not profile:
        profile = ProfessionalProfile(user_id=user_id)
        db.session.add(profile)
        db.session.commit()
    return profile

def reset_token_digest(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()

def verify_audit_chain():
    previous_hash = ''
    records = AuditLog.query.order_by(AuditLog.id.asc()).all()
    for record in records:
        payload = f'{previous_hash}|{record.timestamp}|{record.action}'
        expected_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        if record.previous_hash != previous_hash or record.record_hash != expected_hash:
            return False
        previous_hash = record.record_hash
    return True

def redact_audit_identity(personal_values):
    connection = db.session.connection()
    connection.exec_driver_sql('DROP TRIGGER IF EXISTS audit_log_no_update')
    connection.exec_driver_sql('DROP TRIGGER IF EXISTS audit_log_no_delete')
    previous_hash = ''
    records = connection.exec_driver_sql('SELECT id, action, timestamp FROM audit_log ORDER BY id').fetchall()
    for record in records:
        action = record.action
        for value in personal_values:
            if value and len(value) >= 3:
                action = action.replace(value, '[deleted account]')
        action = action.replace(f'#{session["user_id"]}', '#deleted')
        payload = f'{previous_hash}|{record.timestamp}|{action}'
        record_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        connection.execute(text('UPDATE audit_log SET action = :action, previous_hash = :previous_hash, record_hash = :record_hash WHERE id = :id'), {'action': action, 'previous_hash': previous_hash, 'record_hash': record_hash, 'id': record.id})
        previous_hash = record_hash
    connection.exec_driver_sql("CREATE TRIGGER audit_log_no_update BEFORE UPDATE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END")
    connection.exec_driver_sql("CREATE TRIGGER audit_log_no_delete BEFORE DELETE ON audit_log BEGIN SELECT RAISE(ABORT, 'audit records are immutable'); END")

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        raw_password = request.form.get('password', '')
        if len(raw_password) < 8:
            flash('Choose a password with at least 8 characters.', 'danger')
            return redirect(url_for('register'))
        password = generate_password_hash(raw_password)
        role = request.form.get('role')
        if role not in ['candidate', 'recruiter']:
            flash('Choose a valid account type.', 'danger')
            return redirect(url_for('register'))
        if request.form.get('tc') != 'on':
            flash('You must review and accept the Terms & Conditions to register.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered in system database!', 'danger')
            return redirect(url_for('register'))

        new_user = User(name=name, email=email, password=password, role=role, tc_accepted=True)
        db.session.add(new_user)
        db.session.add(AuditLog(action=f"Registered user: {name} with role [{role}] & verified T&C acceptance.", timestamp=str(datetime.now())))
        db.session.commit()

        sent = send_email(email, 'Welcome to HireShield AI', f"Hello {name},\n\nYour HireShield AI account has been registered as a {role.replace('_', ' ')}. You can now sign in and complete your professional profile.\n\nIf you did not create this account, please contact the platform administrator.")
        if sent:
            flash('Registration complete. A welcome message has been sent to your email.', 'success')
        else:
            flash('Your account was created, but the welcome email could not be sent. Email delivery needs SMTP configuration.', 'danger')
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['user_name'] = user.name
        session['role'] = user.role

        db.session.add(AuditLog(action=f"Successful sign-in: {user.name} ({user.role})", timestamp=str(datetime.now())))
        db.session.commit()
        if send_email(user.email, 'New sign-in to HireShield AI', f"Hello {user.name},\n\nYour HireShield AI account was signed in at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. If this was not you, reset your password and contact the platform administrator."):
            flash('A sign-in notification was sent to your email.', 'success')

        if user.role == 'candidate':
            return redirect(url_for('candidate_dashboard'))
        elif user.role == 'recruiter':
            return redirect(url_for('recruiter_dashboard'))
        elif user.role == 'super_admin':
            return redirect(url_for('super_admin_dashboard'))

    db.session.add(AuditLog(action='Failed sign-in attempt.', timestamp=str(datetime.now())))
    db.session.commit()
    flash('Invalid credentials! Please check email or password.', 'danger')
    return redirect(url_for('index'))

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter(db.func.lower(User.email) == email).first()
        if user:
            token = secrets.token_urlsafe(32)
            reset_record = PasswordResetToken(user_id=user.id, token_hash=reset_token_digest(token), expires_at=datetime.now() + timedelta(hours=1))
            db.session.add(reset_record)
            db.session.commit()
            base_url = os.environ.get('APP_BASE_URL', request.url_root).rstrip('/')
            reset_link = f"{base_url}{url_for('reset_password', token=token)}"
            sent = send_email(user.email, 'Reset your HireShield AI password', f"Hello {user.name},\n\nUse this one-time link to reset your password. It expires in one hour.\n\n{reset_link}\n\nIf you did not request a password reset, you can ignore this message.")
            if not sent:
                db.session.delete(reset_record)
                db.session.commit()
        flash('If that email belongs to an account and email is configured, password reset instructions will arrive shortly.', 'success')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    reset_record = PasswordResetToken.query.filter_by(token_hash=reset_token_digest(token), used_at=None).first()
    if not reset_record or reset_record.expires_at < datetime.now():
        flash('This password reset link is invalid or expired. Request a new one.', 'danger')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        password = request.form.get('password', '')
        if len(password) < 8 or password != request.form.get('confirm_password', ''):
            flash('Use matching passwords with at least 8 characters.', 'danger')
            return render_template('reset_password.html', token=token)
        user = db.session.get(User, reset_record.user_id)
        if not user:
            flash('This password reset link is invalid or expired. Request a new one.', 'danger')
            return redirect(url_for('forgot_password'))
        user.password = generate_password_hash(password)
        now = datetime.now()
        PasswordResetToken.query.filter_by(user_id=user.id, used_at=None).update({'used_at': now})
        db.session.add(AuditLog(action=f"Password reset completed for account #{user.id}.", timestamp=str(now)))
        db.session.commit()
        flash('Password updated. Sign in with your new password.', 'success')
        return redirect(url_for('index'))
    return render_template('reset_password.html', token=token)

@app.route('/profile', methods=['GET', 'POST'])
def professional_profile():
    denied = require_role('candidate', 'recruiter', 'super_admin')
    if denied: return denied
    user = db.session.get(User, session['user_id'])
    profile = get_professional_profile(user.id)
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        if not name:
            flash('Name is required.', 'danger')
            return redirect(url_for('professional_profile'))
        photo = request.files.get('photo')
        extension = os.path.splitext(secure_filename(photo.filename))[1].lower() if photo and photo.filename else ''
        if extension and extension not in ['.jpg', '.jpeg', '.png', '.webp']:
            flash('Use a JPG, PNG or WebP profile photo.', 'danger')
            return redirect(url_for('professional_profile'))
        if extension:
            signature = photo.stream.read(12)
            photo.stream.seek(0)
            is_valid_image = (
                (extension in ['.jpg', '.jpeg'] and signature.startswith(b'\xff\xd8\xff'))
                or (extension == '.png' and signature.startswith(b'\x89PNG\r\n\x1a\n'))
                or (extension == '.webp' and signature.startswith(b'RIFF') and signature[8:12] == b'WEBP')
            )
            if not is_valid_image:
                flash('The selected file is not a valid JPG, PNG or WebP image.', 'danger')
                return redirect(url_for('professional_profile'))
        user.name = name
        profile.headline = request.form.get('headline', '').strip()
        profile.phone = request.form.get('phone', '').strip()
        profile.location = request.form.get('location', '').strip()
        profile.bio = request.form.get('bio', '').strip()
        if extension:
            filename = f"{user.id}_{uuid.uuid4().hex}{extension}"
            photo.save(os.path.join(PROFILE_UPLOAD_FOLDER, filename))
            profile.photo_filename = filename
        session['user_name'] = user.name
        session['profile_photo'] = profile.photo_filename
        db.session.add(AuditLog(action=f"Professional profile updated for account #{user.id}.", timestamp=str(datetime.now())))
        db.session.commit()
        flash('Professional profile saved.', 'success')
        return redirect(url_for('professional_profile'))
    return render_template('profile.html', user=user, profile=profile)

@app.route('/profile/photo')
def profile_photo():
    denied = require_role('candidate', 'recruiter', 'super_admin')
    if denied: return denied
    profile = ProfessionalProfile.query.filter_by(user_id=session['user_id']).first()
    if not profile or not profile.photo_filename:
        return redirect(url_for('professional_profile'))
    return send_from_directory(PROFILE_UPLOAD_FOLDER, profile.photo_filename)

@app.route('/profile/delete', methods=['POST'])
def delete_profile():
    denied = require_role('candidate', 'recruiter', 'super_admin')
    if denied: return denied
    user = db.session.get(User, session['user_id'])
    if not check_password_hash(user.password, request.form.get('password', '')):
        flash('Password confirmation failed. Your account was not deleted.', 'danger')
        return redirect(url_for('professional_profile'))
    if user.role == 'super_admin' and User.query.filter_by(role='super_admin').count() < 2:
        flash('The last super-admin account cannot be deleted.', 'danger')
        return redirect(url_for('professional_profile'))

    candidate_profile = CandidateProfile.query.filter_by(user_id=user.id).first()
    professional_profile = ProfessionalProfile.query.filter_by(user_id=user.id).first()
    resume_files = {application.resume_filename for application in Application.query.filter_by(candidate_id=user.id).all() if application.resume_filename}
    if candidate_profile and candidate_profile.resume_filename:
        resume_files.add(candidate_profile.resume_filename)
    photo_filename = professional_profile.photo_filename if professional_profile else ''
    personal_values = [user.name, user.email]
    if candidate_profile:
        personal_values.extend([candidate_profile.headline, candidate_profile.phone, candidate_profile.location, candidate_profile.skills, candidate_profile.summary])
    if professional_profile:
        personal_values.extend([professional_profile.headline, professional_profile.phone, professional_profile.location, professional_profile.bio])
    owned_jobs = Job.query.filter_by(recruiter_id=user.id).all()
    for job in owned_jobs:
        personal_values.extend([job.title, job.department, job.location, job.description])

    redact_audit_identity(personal_values)
    for job in owned_jobs:
        job.title = 'Former role'
        job.department = 'Archived'
        job.location = ''
        job.description = ''
        job.status = 'Archived'
        job.recruiter_id = None
    db.session.query(InterviewAssessment).filter_by(candidate_id=user.id).delete(synchronize_session=False)
    db.session.query(Application).filter_by(candidate_id=user.id).delete(synchronize_session=False)
    db.session.query(CandidateProfile).filter_by(user_id=user.id).delete(synchronize_session=False)
    db.session.query(ProfessionalProfile).filter_by(user_id=user.id).delete(synchronize_session=False)
    db.session.query(PasswordResetToken).filter_by(user_id=user.id).delete(synchronize_session=False)
    db.session.query(InterviewSchedule).filter_by(candidate_id=user.id).delete(synchronize_session=False)
    if User.query.filter_by(name=user.name).count() == 1:
        db.session.query(InterviewSchedule).filter_by(candidate_name=user.name, candidate_id=None).delete(synchronize_session=False)
    else:
        db.session.query(InterviewSchedule).filter_by(candidate_name=user.name, candidate_id=None).update({'candidate_name': 'Former candidate'}, synchronize_session=False)
    db.session.query(InterviewSchedule).filter_by(interviewer=user.name).update({'interviewer': 'Former account'}, synchronize_session=False)
    db.session.add(AuditLog(action=f'Account deletion completed for account #{user.id}; personal audit identifiers redacted.', timestamp=str(datetime.now())))
    db.session.delete(user)
    db.session.commit()
    session.clear()

    file_candidates = set(resume_files)
    if photo_filename:
        photo_path = os.path.join(PROFILE_UPLOAD_FOLDER, photo_filename)
        if os.path.isfile(photo_path):
            os.remove(photo_path)
    for filename in file_candidates:
        still_referenced = CandidateProfile.query.filter_by(resume_filename=filename).first() or Application.query.filter_by(resume_filename=filename).first()
        resume_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
        if not still_referenced and os.path.isfile(resume_path):
            os.remove(resume_path)
    flash('Your account and associated profile data have been deleted.', 'success')
    return redirect(url_for('index'))

@app.route('/profile/password', methods=['POST'])
def change_password():
    denied = require_role('candidate', 'recruiter', 'super_admin')
    if denied: return denied
    user = db.session.get(User, session['user_id'])
    current_password = request.form.get('current_password', '')
    new_password = request.form.get('new_password', '')
    if not check_password_hash(user.password, current_password):
        flash('Current password is incorrect.', 'danger')
        return redirect(url_for('professional_profile'))
    if len(new_password) < 8 or new_password != request.form.get('confirm_password', ''):
        flash('New passwords must match and contain at least 8 characters.', 'danger')
        return redirect(url_for('professional_profile'))
    user.password = generate_password_hash(new_password)
    db.session.add(AuditLog(action=f'Password changed for account #{user.id}.', timestamp=str(datetime.now())))
    db.session.commit()
    if send_email(user.email, 'Your HireShield AI password was changed', 'Your account password was changed. If you did not make this change, contact the platform administrator immediately.'):
        flash('Password changed. A security notification was sent to your email.', 'success')
    else:
        flash('Password changed. Configure email delivery to receive security notifications.', 'success')
    return redirect(url_for('professional_profile'))

@app.route('/candidate/dashboard')
def candidate_dashboard():
    denied = require_role('candidate')
    if denied: return denied
    search = request.args.get('q', '').strip()
    jobs_query = Job.query.filter_by(status='Active')
    if search:
        term = f'%{search}%'
        jobs_query = jobs_query.filter(db.or_(Job.title.ilike(term), Job.department.ilike(term), Job.location.ilike(term), Job.description.ilike(term)))
    jobs = jobs_query.order_by(Job.id.desc()).all()
    apps = Application.query.filter_by(candidate_id=session['user_id']).all()
    profile = get_candidate_profile(session['user_id'])
    assessments = InterviewAssessment.query.filter_by(candidate_id=session['user_id']).order_by(InterviewAssessment.id.desc()).all()
    applied_job_ids = {application.job_id for application in apps}
    return render_template('candidate_dash.html', jobs=jobs, applications=apps, profile=profile, assessments=assessments, applied_job_ids=applied_job_ids, search=search)

@app.route('/candidate/profile', methods=['POST'])
def candidate_save_profile():
    denied = require_role('candidate')
    if denied: return denied
    profile = get_candidate_profile(session['user_id'])
    profile.headline = request.form.get('headline', '').strip()
    profile.phone = request.form.get('phone', '').strip()
    profile.location = request.form.get('location', '').strip()
    profile.skills = request.form.get('skills', '').strip()
    profile.summary = request.form.get('summary', '').strip()
    db.session.commit()
    flash('Profile changes saved.', 'success')
    return redirect(url_for('candidate_dashboard'))

@app.route('/candidate/upload-resume', methods=['POST'])
def candidate_upload_resume():
    denied = require_role('candidate')
    if denied: return denied
    file = request.files.get('resume')
    if file and file.filename and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(f"{session['user_id']}_{file.filename}")
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        text = extract_resume_text(path)
        profile = get_candidate_profile(session['user_id'])
        profile.resume_filename = filename
        if not profile.summary and text:
            profile.summary = text[:1200]
        if not profile.skills:
            profile.skills = extract_resume_skills(text)
        db.session.add(profile)
        db.session.add(AuditLog(action=f"Candidate {session.get('user_name')} uploaded and parsed master resume.", timestamp=str(datetime.now())))
        db.session.commit()
        flash('Resume uploaded. Review the extracted profile details and save any edits.', 'success')
    else:
        flash('Choose a PDF resume to upload.', 'danger')
    return redirect(url_for('candidate_dashboard'))

@app.route('/candidate/apply/<int:job_id>', methods=['POST'])
def candidate_apply(job_id):
    denied = require_role('candidate')
    if denied: return denied
    profile = get_candidate_profile(session['user_id'])
    job = db.session.get(Job, job_id)
    if not job or job.status != 'Active':
        flash('This job is no longer available.', 'danger')
    elif not profile.resume_filename:
        flash('Upload and save a PDF resume before applying.', 'danger')
    elif Application.query.filter_by(candidate_id=session['user_id'], job_id=job.id).first():
        flash('You have already applied to this role.', 'danger')
    else:
        path = os.path.join(UPLOAD_FOLDER, profile.resume_filename)
        score = compute_ai_match(job.description, path)
        db.session.add(Application(candidate_id=session['user_id'], job_id=job.id, resume_filename=profile.resume_filename, ai_match_score=score))
        db.session.add(AuditLog(action=f"Candidate #{session['user_id']} applied to Job ID {job.id}. Match score: {score}%", timestamp=str(datetime.now())))
        db.session.commit()
        flash(f'Application submitted. Resume-to-role match: {score}%.', 'success')
    return redirect(url_for('candidate_dashboard'))

@app.route('/candidate/mock-interview', methods=['GET', 'POST'])
def mock_interview():
    denied = require_role('candidate')
    if denied: return denied
    rounds = [
        ("1", "Coding & Data Structures", "Implement an optimized algorithm to detect a cycle in a linked list."),
        ("2", "System Design", "Design a distributed caching architecture handling high throughput."),
        ("3", "HR & Culture Fit", "Describe how you manage conflicting priorities under tight timelines."),
        ("4", "Aptitude & Logical Reasoning", "Solve probability and permutation metrics for scaling servers."),
        ("5", "Behavioral Evaluation", "Share an instance where you resolved a critical technical roadblock."),
        ("6", "Communication Mock", "Deliver a structured technical summary of your final-year project.")
    ]
    if request.method == 'POST':
        r_num = int(request.form.get('round_num'))
        answer = request.form.get('answer', '').strip()
        answers = session.get('mock_answers', [])
        answers.append(answer)
        session['mock_answers'] = answers
        if r_num < 6:
            next_r = r_num + 1
            return render_template('ai_interview.html', round_num=next_r, round_name=rounds[next_r-1][1], question=rounds[next_r-1][2])
        else:
            full_answer = ' '.join(answers)
            words = full_answer.split()
            technical_terms = ['algorithm', 'complexity', 'database', 'api', 'cache', 'testing', 'trade-off', 'because', 'design', 'solution']
            technical = min(100, 25 + min(len(words), 90) // 2 + min(sum(term in full_answer.lower() for term in technical_terms) * 3, 25))
            communication = min(100, 25 + min(len(words), 100) // 2 + min(full_answer.count('.') + full_answer.count('!') + full_answer.count('?'), 10) * 2)
            overall = round((technical + communication) / 2)
            db.session.add(InterviewAssessment(candidate_id=session['user_id'], technical_score=technical, communication_score=communication, overall_score=overall, completed_at=str(datetime.now())))
            db.session.add(AuditLog(action=f"Candidate {session.get('user_name')} completed the six-round practice assessment. Rubric score: {overall}/100.", timestamp=str(datetime.now())))
            db.session.commit()
            session.pop('mock_answers', None)
            flash(f'Practice assessment complete. Local rubric score: {overall}/100 (technical {technical}, communication {communication}).', 'success')
            return redirect(url_for('candidate_dashboard'))

    return render_template('ai_interview.html', round_num=1, round_name=rounds[0][1], question=rounds[0][2])

@app.route('/recruiter/dashboard')
def recruiter_dashboard():
    denied = require_role('recruiter')
    if denied: return denied
    jobs = Job.query.all()
    stage_filter = request.args.get('stage', '').strip()
    search = request.args.get('q', '').strip()
    apps_query = Application.query.join(Job).join(User, Application.candidate_id == User.id)
    if stage_filter:
        apps_query = apps_query.filter(Application.stage == stage_filter)
    if search:
        term = f'%{search}%'
        apps_query = apps_query.filter(db.or_(User.name.ilike(term), User.email.ilike(term), Job.title.ilike(term)))
    apps = apps_query.order_by(Application.ai_match_score.desc()).all()
    schedules = InterviewSchedule.query.all()
    candidates = {user.id: user for user in User.query.filter_by(role='candidate').all()}
    assessments = InterviewAssessment.query.order_by(InterviewAssessment.id.desc()).all()
    latest_assessments = {}
    for assessment in assessments:
        latest_assessments.setdefault(assessment.candidate_id, assessment)
    return render_template('recruiter_dash.html', jobs=jobs, applications=apps, schedules=schedules, candidates=candidates, latest_assessments=latest_assessments, stage_filter=stage_filter, search=search)

@app.route('/recruiter/post-job', methods=['POST'])
def recruiter_post_job():
    denied = require_role('recruiter')
    if denied: return denied
    db.session.add(Job(recruiter_id=session['user_id'], title=request.form.get('title'), department=request.form.get('department'), location=request.form.get('location'), description=request.form.get('description')))
    db.session.add(AuditLog(action=f"Recruiter {session.get('user_name')} published job requisition: {request.form.get('title')}", timestamp=str(datetime.now())))
    db.session.commit()
    flash('Job requisition published successfully!', 'success')
    return redirect(url_for('recruiter_dashboard'))

@app.route('/recruiter/edit-job/<int:job_id>', methods=['POST'])
def recruiter_edit_job(job_id):
    denied = require_role('recruiter')
    if denied: return denied
    job = db.session.get(Job, job_id)
    if job:
        job.title = request.form.get('title', job.title).strip()
        job.department = request.form.get('department', job.department).strip()
        job.location = request.form.get('location', job.location).strip()
        job.description = request.form.get('description', job.description).strip()
        db.session.add(AuditLog(action=f'Job requisition ID {job_id} edited by {session.get("user_name")}.', timestamp=str(datetime.now())))
        db.session.commit()
        flash('Job listing updated.', 'success')
    return redirect(url_for('recruiter_dashboard'))

@app.route('/recruiter/archive-job/<int:job_id>', methods=['POST'])
def recruiter_archive_job(job_id):
    denied = require_role('recruiter')
    if denied: return denied
    job = db.session.get(Job, job_id)
    if job:
        job.status = 'Archived'
        db.session.add(AuditLog(action=f"Job requisition ID {job_id} archived by {session.get('user_name')}", timestamp=str(datetime.now())))
        db.session.commit()
        flash('Job successfully archived!', 'success')
    return redirect(url_for('recruiter_dashboard'))

@app.route('/recruiter/update-stage/<int:app_id>', methods=['POST'])
def recruiter_update_stage(app_id):
    denied = require_role('recruiter')
    if denied: return denied
    app_entry = db.session.get(Application, app_id)
    stage = request.form.get('stage', '')
    if app_entry and stage in ['Applied', 'Assessment', 'Interview', 'Offer', 'Rejected']:
        app_entry.stage = stage
        db.session.add(AuditLog(action=f'Application ID {app_id} moved to [{stage}] by {session.get("user_name")}.', timestamp=str(datetime.now())))
        db.session.commit()
        flash('Candidate stage updated.', 'success')
    return redirect(url_for('recruiter_dashboard'))

@app.route('/recruiter/schedule/<int:app_id>', methods=['POST'])
def recruiter_schedule(app_id):
    denied = require_role('recruiter')
    if denied: return denied
    app_entry = db.session.get(Application, app_id)
    if app_entry:
        app_entry.stage = 'Interview'
        candidate = db.session.get(User, app_entry.candidate_id)
        job = db.session.get(Job, app_entry.job_id)
        db.session.add(InterviewSchedule(candidate_id=app_entry.candidate_id, candidate_name=candidate.name if candidate else f"Candidate #{app_entry.candidate_id}", job_title=job.title if job else 'Role', interview_date=request.form.get('interview_date', 'To be confirmed'), interviewer=request.form.get('interviewer', session.get('user_name'))))
        db.session.add(AuditLog(action=f"Interview scheduled for candidate #{app_entry.candidate_id} by {session.get('user_name')}", timestamp=str(datetime.now())))
        db.session.commit()
        flash('Interview slot recorded.', 'success')
    return redirect(url_for('recruiter_dashboard'))

@app.route('/super-admin/dashboard')
def super_admin_dashboard():
    denied = require_role('super_admin')
    if denied: return denied
    logs = AuditLog.query.order_by(AuditLog.id.desc()).all()
    role_counts = {
        'candidates': User.query.filter_by(role='candidate').count(),
        'recruiters': User.query.filter_by(role='recruiter').count(),
        'administrators': User.query.filter_by(role='super_admin').count()
    }
    security_logs = AuditLog.query.filter(db.or_(AuditLog.action.ilike('%sign-in%'), AuditLog.action.ilike('%password%'), AuditLog.action.ilike('%role%'))).order_by(AuditLog.id.desc()).limit(100).all()
    return render_template('super_admin_dash.html', logs=logs, security_logs=security_logs, users=User.query.all(), jobs=Job.query.all(), applications=Application.query.all(), interviews=InterviewSchedule.query.count(), assessments=InterviewAssessment.query.count(), role_counts=role_counts, audit_valid=verify_audit_chain(), email_configured=bool(os.environ.get('MAIL_SERVER') and os.environ.get('MAIL_DEFAULT_SENDER')))

@app.route('/super-admin/user-role/<int:user_id>', methods=['POST'])
def super_admin_update_role(user_id):
    denied = require_role('super_admin')
    if denied: return denied
    user = db.session.get(User, user_id)
    role = request.form.get('role', '')
    if user and role in ['candidate', 'recruiter', 'super_admin']:
        if user.id == session['user_id'] and role != 'super_admin':
            flash('You cannot remove your own super-admin access.', 'danger')
            return redirect(url_for('super_admin_dashboard'))
        if user.role == 'super_admin' and role != 'super_admin' and User.query.filter_by(role='super_admin').count() < 2:
            flash('The last super-admin account cannot be demoted.', 'danger')
            return redirect(url_for('super_admin_dashboard'))
        user.role = role
        db.session.add(AuditLog(action=f"Super admin updated account #{user.id} role to [{role}].", timestamp=str(datetime.now())))
        db.session.commit()
        flash('Account role updated.', 'success')
    return redirect(url_for('super_admin_dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=False, port=5000)