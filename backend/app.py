from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os

# Configure Flask to use the frontend folder for templates
template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'frontend')
app = Flask(__name__, template_folder=template_dir)
app.secret_key = "skillorbit_secret"

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
