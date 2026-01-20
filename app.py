from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
import io
import os
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

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
            background:#fff; color:#000; text-decoration:none; font-weight:bold;
            border:1px solid #fff;
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

    <!-- IMPORTANT: correct method + enctype, button is inside form -->
    <form method="post" enctype="multipart/form-data">
        <label>UPLOAD CSV FILE *</label>
        <input type="file" name="file" />
        <small>First column: option names, remaining columns: numeric criteria values</small>

        <label>WEIGHTS *</label>
        <input type="text" name="weights" placeholder="e.g., 1,1,1,2" />

        <label>IMPACTS *</label>
        <input type="text" name="impacts" placeholder="e.g., +,+,-,+" />

        <label>SEND RESULTS TO EMAIL (OPTIONAL)</label>
        <input type="email" name="email" placeholder="you@example.com" />

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
    {% endif %}

    <div class="example-box">
        <strong>EXAMPLE CSV FORMAT:</strong><br/><br/>
        Model,Price,Storage,Camera,Battery<br/>
        P1,250,64,12,4000<br/>
        P2,200,32,8,3500<br/>
        P3,300,128,16,4500<br/><br/>
        <a href="/sample.csv" class="download-link">DOWNLOAD SAMPLE CSV</a>
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
    df_out['Topsis Score'] = score
    df_out['Rank'] = df_out['Topsis Score'].rank(ascending=False).astype(int)
    df_out = df_out.sort_values('Rank')
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
    # DEBUG: first, just prove POST is received
    if request.method == "POST":
        # When this works, replace this block with full TOPSIS processing.
        return "Got POST (form submitted). Next step: re‑enable TOPSIS logic.", 200

    # For GET, just render the form
    return render_template_string(
        HTML_TEMPLATE,
        message=None,
        error=False,
        result_table=None,
        email="",
        weights="",
        impacts="",
    )

if __name__ == "__main__":
    app.run(debug=True)
