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


@app.route('/api/trainstatus', methods=['GET'])
def gettrainstatus():
    status = aimane.get_train_status()
    return status



# Route will be
# /api/copyuc
@app.route('/api/restouc', methods=['GET'])
def copyuc():
    status = aimane.copy_result_to_usercontent()
    return "Copy result to usercontent successful!"




# Server Sent Event
# Route will be
# /api/sse/status
prev_status = None
train_prev_status = None

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


@app.route('/api/sse/trainstatus', methods=['GET'])
def sse_trainstatus():
    def gen():
        train_prev_status = None
        while True:
            status = aimane.get_train_status()
            if status is not None and status != train_prev_status:
                yield 'data: {}\n\n'.format(status)
                train_prev_status = status
            time.sleep(0.5)

    return Response(gen(), mimetype='text/event-stream')

    


# Route will be /api/prepare
# Receive 1 parameter: force
# force = 1: force to prepare dataset
# force = 0: only prepare dataset if it is not prepared

@app.route('/api/prepare', methods=['GET'])
def prepare():
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

    # Check if the result is string then return error message
    if isinstance(result, str):
        # Return error message as JSON
        error = {
            'error': result
        }
        return jsonify(error)
    
    # Convert ndarray to list
    result = result.tolist()

    # Return result in JSON format
    return jsonify(result)


# Route will be /api/definelastpredict?label=xxx
@app.route('/api/definelastpredict', methods=['GET'])
def definepredict():
    # Get label from the GET request
    label = request.args.get('label')

    # Start the training process in a separate task
    res = aimane.define_last_prediction(label)

    if res == True and label != None:
        return "Define last prediction to " + label + " successful!"
    elif res == False and label != None:
        return "Define last prediction to " + label + " failed!"
    else:
        return "Define last prediction failed because label is not provided or not correct!"
    


# Route will be /api/definepredict?label=xxx&image=xxx
@app.route('/api/definepredict', methods=['GET'])
def definepredict2():
    # Get label from the GET request
    label = request.args.get('label')
    # Get image from the GET request
    image = request.args.get('image')


    if label == None or image == None:
        return "Define prediction failed because label or image is not provided!"


    # Check if the image is in the correct format // the format is 1--aabbccddee
    if image.find("--") == -1:
        return "Define prediction failed because image is not in the correct format!"
    
    
    # Saparate image and label from the string 
    # the format is 1--aabbccddee
    group = image.split("--")[0]
    image = image.split("--")[1]



    # Start the training process in a separate task
    res = aimane.define_prediction(label, group, image)

    if res == True and label != None:
        return "Define prediction of " + image + " from group " + group + " to " + label + " successful!"
    elif res == False and label != None:
        return "Define prediction of " + image + " from group " + group + " to " + label + " failed!"
    else:
        return "Define prediction failed because label or image is not provided or not correct!"
    
    



@app.route('/api/iknowwhatimdoing/rewrite_filename', methods=['GET'])
def rewrite_filename():
    aimane.rewrite_filename()
    return "Rewrite filename process has started."



# @app.route('/api/predict-lab', methods=['GET'])
# # Same as predict but user provide the label also to check if the prediction is correct
# def predict_lab():



@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')


@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')



# # Initialize the app
# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == '__main__':
    # Run as local server with debug mode on
    #app.run(debug=True, use_reloader=False)
    # Run as production server
    aimane.sysmane.write_status("Server is started")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)



    






