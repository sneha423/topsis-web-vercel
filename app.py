from flask import Flask, request, render_template_string, send_file
import numpy as np
import pandas as pd
import io
import os
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

# in‑memory buffer for last result
LAST_RESULT_CSV = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>TOPSIS Web Service</title>
    <style>
        body { background:#000; color:#000; font-family: Arial, sans-serif; }
        .wrapper { max-width: 700px; margin: 40px auto; background:#fff; padding:30px; box-shadow:0 0 10px rgba(0,0,0,0.4);}
        h1 { text-align:center; letter-spacing:2px; }
        label { display:block; margin-top:15px; font-weight:bold; }
        input[type="text"], input[type="email"] {
            width:100%; padding:8px; margin-top:5px; box-sizing:border-box;
        }
        input[type="file"] { margin-top:5px; }
        button {
            margin-top:20px; width:100%; padding:12px;
            background:#000; color:#fff; border:none; cursor:pointer;
            font-weight:bold; letter-spacing:1px;
        }
        button:hover { background:#222; }
        .msg { margin-top:15px; padding:10px; background:#f2f2f2; }
        table { border-collapse:collapse; width:100%; margin-top:20px; }
        th, td { border:1px solid #000; padding:8px; text-align:center; }
        .error { color:#b00020; }
        .success { color:#006400; }
        .example-box {
            margin-top:30px; background:#000; color:#fff; padding:15px;
            font-size:13px;
        }
        .download-link {
            display:inline-block; margin-top:10px; padding:8px 12px;
            background:#000; color:#fff; text-decoration:none; font-weight:bold;
            border:1px solid #000;
        }
    </style>
</head>
<body>
<div class="wrapper">
    <h1>TOPSIS WEB SERVICE</h1>
    <p style="text-align:center;">MULTI-CRITERIA DECISION ANALYSIS TOOL</p>

    {% if message %}
      <div class="msg {{ 'error' if error else 'success' }}">{{ message }}</div>
    {% endif %}

    <form method="post" enctype="multipart/form-data">
        <label>UPLOAD CSV FILE *</label>
        <input type="file" name="file" />
        <small>First column: option names, remaining columns: numeric criteria values</small>

        <label>WEIGHTS *</label>
        <input type="text" name="weights" placeholder="e.g., 1,1,1,2" value="{{ weights }}" />

        <label>IMPACTS *</label>
        <input type="text" name="impacts" placeholder="e.g., +,+,-,+" value="{{ impacts }}" />

        <label>SEND RESULTS TO EMAIL (OPTIONAL)</label>
        <input type="email" name="email" placeholder="you@example.com" value="{{ email }}" />

        <button type="submit">CALCULATE TOPSIS</button>
    </form>

    {% if result_table is not none %}
      <h2 style="margin-top:25px;">Result</h2>
      <table>
        <thead>
          <tr>
          {% for col in result_table.columns %}
            <th>{{ col }}</th>
          {% endfor %}
          </tr>
        </thead>
        <tbody>
        {% for _, row in result_table.iterrows() %}
          <tr>
          {% for col in result_table.columns %}
            <td>{{ row[col] }}</td>
          {% endfor %}
          </tr>
        {% endfor %}
        </tbody>
      </table>

      <a href="/download" class="download-link">DOWNLOAD RESULT CSV</a>
    {% endif %}

    <div class="example-box">
        <strong>EXAMPLE CSV FORMAT:</strong><br/><br/>
        Model,Price,Storage,Camera,Battery<br/>
        P1,250,64,12,4000<br/>
        P2,200,32,8,3500<br/>
        P3,300,128,16,4500<br/>
    </div>
</div>
</body>
</html>
"""

def run_topsis(df, weights, impacts):
    data = df.iloc[:, 1:].astype(float).values
    weights = np.array(weights, dtype=float)
    impacts = np.array(impacts)

    norm = np.sqrt((data ** 2).sum(axis=0))
    norm_data = data / norm
    weighted = norm_data * weights

    ideal_best = np.zeros(weighted.shape[1])
    ideal_worst = np.zeros(weighted.shape[1])
    for i, impact in enumerate(impacts):
        if impact == '+':
            ideal_best[i] = weighted[:, i].max()
            ideal_worst[i] = weighted[:, i].min()
        else:
            ideal_best[i] = weighted[:, i].min()
            ideal_worst[i] = weighted[:, i].max()

    s_pos = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    s_neg = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    score = s_neg / (s_pos + s_neg)

    df_out = df.copy()
    df_out["Topsis Score"] = score
    df_out["Rank"] = df_out["Topsis Score"].rank(ascending=False).astype(int)
    df_out = df_out.sort_values("Rank")
    return df_out

def send_email_with_csv(to_email, csv_bytes, filename="topsis_result.csv"):
    smtp_user = os.getenv("MAIL_USERNAME")
    smtp_pass = os.getenv("MAIL_PASSWORD")
    if not smtp_user or not smtp_pass:
        return False, "Email not sent: MAIL_USERNAME or MAIL_PASSWORD not set."

    msg = EmailMessage()
    msg["Subject"] = "TOPSIS Result"
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.set_content("Please find attached the TOPSIS result file.")

    msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename=filename)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Email sent successfully."
    except Exception as e:
        return False, f"Email error: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    global LAST_RESULT_CSV

    message = None
    error = False
    result_table = None
    email = ""
    weights_str = ""
    impacts_str = ""

    if request.method == "POST":
        file = request.files.get("file")
        weights_str = (request.form.get("weights") or "").strip()
        impacts_str = (request.form.get("impacts") or "").strip()
        email = (request.form.get("email") or "").strip()

        if not file or file.filename == "":
            message = "Please upload a CSV file."
            error = True
            df = None
        else:
            try:
                file_bytes = file.read()
                if not file_bytes:
                    raise ValueError("Uploaded file is empty.")
                buffer = io.BytesIO(file_bytes)
                df = pd.read_csv(buffer, encoding="utf-8", engine="python")
            except Exception as e:
                message = f"CSV error: {e}"
                error = True
                df = None

        if not error:
            try:
                weights = [float(x) for x in weights_str.split(",")]
                impacts = [x.strip() for x in impacts_str.split(",")]
            except Exception:
                message = "Invalid weights or impacts format."
                error = True
                weights = []
                impacts = []

        if not error:
            if len(weights) != len(impacts):
                message = "Weights and impacts must have the same length."
                error = True
            elif df is not None and df.shape[1] - 1 != len(weights):
                message = "Number of weights/impacts must match criteria columns."
                error = True
            elif any(i not in ["+", "-"] for i in impacts):
                message = "Impacts must be + or - only."
                error = True

        if not error and df is not None:
            try:
                result_table = run_topsis(df, weights, impacts)

                buf = io.StringIO()
                result_table.to_csv(buf, index=False)
                csv_bytes = buf.getvalue().encode("utf-8")
                LAST_RESULT_CSV = csv_bytes

                if email:
                    ok, mail_msg = send_email_with_csv(email, csv_bytes)
                    message = mail_msg
                    error = not ok
                else:
                    message = "TOPSIS calculated successfully."
            except Exception as e:
                message = f"Error during TOPSIS calculation: {e}"
                error = True

    return render_template_string(
        HTML_TEMPLATE,
        message=message,
        error=error,
        result_table=result_table,
        email=email,
        weights=weights_str,
        impacts=impacts_str,
    )

@app.route("/download")
def download():
    global LAST_RESULT_CSV
    if not LAST_RESULT_CSV:
        # nothing calculated yet
        return "No result available to download.", 400

    return send_file(
        io.BytesIO(LAST_RESULT_CSV),
        mimetype="text/csv",
        as_attachment=True,
        download_name="result.csv",
    )

if __name__ == "__main__":
    app.run(debug=True)
