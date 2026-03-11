from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    try:
        # Replace 'python detect.py' with the appropriate command for running detect.py
        result = subprocess.run(['python', 'detect.py'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error executing script: {result.stderr}", 500
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
