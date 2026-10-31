# Email Setup

Account registration notices, sign-in alerts, and password-reset links use SMTP. The app does not send mail until these environment variables are configured.

For a local PowerShell session, set them before starting Flask:

```powershell
$env:MAIL_SERVER = "smtp.gmail.com"
$env:MAIL_PORT = "587"
$env:MAIL_USE_TLS = "true"
$env:MAIL_USERNAME = "your-mailbox@example.com"
$env:MAIL_PASSWORD = "provider-issued-app-password"
$env:MAIL_DEFAULT_SENDER = "HireShield AI <your-mailbox@example.com>"
$env:APP_BASE_URL = "http://127.0.0.1:5000"
$env:SECRET_KEY = "replace-with-a-long-random-secret"
python app.py
```

Use your provider's SMTP host and credentials. For Gmail, use an app password rather than your regular account password. Do not commit real credentials. Environment variables set this way apply only to that PowerShell session.

For deployment, set `APP_BASE_URL` to the public HTTPS address of the app. Password-reset links expire after one hour and can only be used once. If SMTP is missing or unavailable, account registration and sign-in still work, but messages are not delivered.
