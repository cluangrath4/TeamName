from flask import Flask
from flask import render_template, jsonify, request
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', title='Welcome to Flask')

@app.route('/run-script', methods=['POST'])
def run_script():
    # Run your Python script logic here
    input_string = request.get_json()
    input_string = input_string['searchInput']

    process = subprocess.run(['python', '../test.py', input_string], capture_output=True, text=True)

    if process.returncode != 0:
        result = {"status": "error", "message": process.stderr}
    else:
        result = {"status": "success", "message": process.stdout}
    result = {"status": "success", "message": "Script executed successfully!"}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)