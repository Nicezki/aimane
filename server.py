import time
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
        prev_status = None  # Initialize prev_status within the generator function
        while True:
            status = aimane.get_status()
            if status is not None and status != prev_status:
                yield 'data: {}\n\n'.format(status)
                prev_status = status  # Update the previous status
            time.sleep(0.5)

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
    image = request.files['image']

    # Make prediction
    result = aimane.test_model_live(image)

    # Convert ndarray to list
    result = result.tolist()

    # Return result in JSON format
    return jsonify(result)



@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')


@app.route('/', methods=['GET'])
def index():
    return render_template('test.html')


# # Initialize the app
# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)






