import asyncio
from flask import Flask, request, jsonify, Response
from app import aimane
from flask import request, render_template

app = Flask(__name__)

aimane = aimane.AiMane()

# Route will be
# /api/version

@app.route('/api/version', methods=['GET'])
def api_version():
    version = aimane.get_version()
    return version

# Route will be
# /api/status
@app.route('/api/status', methods=['GET'])
def getstatus():
    status = aimane.get_status()
    return status

# Server Sent Event
# Route will be
# /api/sse/status
prev_status = None

@app.route('/api/sse/status', methods=['GET'])
def sse_status():
    def gen():
        global prev_status  # Use the previously defined prev_status variable
        while True:
            status = aimane.get_status()
            if status is not None and status != prev_status:
                yield 'data: {}\n\n'.format(status)
                prev_status = status  # Update the previous status
            else:
                yield ''
            asyncio.sleep(0.5)

    return Response(gen(), mimetype='text/event-stream')




    


# Route will be /api/prepare
# Receive 1 parameter: force
# force = 1: force to prepare dataset
# force = 0: only prepare dataset if it is not prepared

@app.route('/api/prepare', methods=['GET'])
def prepare():
    
    # Get force from the query string
    force = request.args.get('force')
    # Start the preparation process in a separate task
    # aimane.prepare_dataset(force)
    aimane.stage_1()
    # Prepare dataset
    # result = 
    # Return progress status
    return "Preparation process has started."



@app.route('/api/repairdataset', methods=['GET'])
def repair_dataset():
    # Remove .prepare_lock
    aimane.repair_dataset()

    # Return progress status
    return "Preparation process has started."



@app.route('/api/train', methods=['GET'])
def train():
    # Start the training process in a separate task
    aimane.train()
    # Return progress status
    return "Training process has started."


@app.route('/api/trainstatus', methods=['GET'])
def train_status():
    # Get training status
    status = aimane.get_train_status()
    # Return status
    return status




@app.route('/api/predict', methods=['POST'])
def predict():
    # Get image from the POST request
    # Image is encoded in base64 format
    if request.method == 'POST' and request.data is not None:
        image_data = request.data.decode('utf-8')
        
    else:
        return jsonify({'error': 'Invalid request', 'data': {}}), 400


    prediction = aimane.test_model_live(image_data)
    
    # Return prediction result
    return jsonify({'prediction': prediction}), 200



@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')


@app.route('/', methods=['GET'])
def index():
    return render_template('test.html')


# Initialize the app
if __name__ == '__main__':
    app.run(debug=True)






