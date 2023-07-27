import time
from flask import Flask, request, jsonify, Response, send_from_directory
from app import aimane
from flask import request, render_template
from flask_cors import CORS, cross_origin
from waitress import serve


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})
# Set the allowed origins in a list Now not used
# allowed_origins = [
#     'https://me.nicezki.com/',
#     'https://nicezki.com/',
#     'https://localhost/',
#     'https://127.0.0.1/',
#     'https://192.168.1.2/'
# ]

# CORS(app, resources={r"/*": {"origins": allowed_origins, "supports_credentials": True}})

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


@app.route('/api/predictresults', methods=['GET'])
def getpredictstatus():
    status = aimane.get_prediction_result()
    
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
predict_prev_result = None

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


@app.route('/api/sse/predictresults', methods=['GET'])
def sse_predictresults():
    def gen():
        predict_prev_result = None
        while True:
            status = aimane.get_prediction_result()
            if status is not None and status != predict_prev_result:
                yield 'data: {}\n\n'.format(status)
                predict_prev_result = status
            time.sleep(0.5)

    return Response(gen(), mimetype='text/event-stream')



@app.route('/api/sse/events', methods=['GET'])
# Show all events
# Saparate events by group
# custom events: status, trainstatus, predictresults

def sse_events():
    def gen():
        prev_status = None
        train_prev_status = None
        predict_prev_result = None
        while True:
            status = aimane.get_status()
            train_status = aimane.get_train_status()
            predict_result = aimane.get_prediction_result()

            # Custom events : Status
            if status is not None and status != prev_status:
                yield 'event: status\ndata: {}\n\n'.format(status)
                prev_status = status

            # Custom events : Train Status
            if train_status is not None and train_status != train_prev_status:
                yield 'event: trainstatus\ndata: {}\n\n'.format(train_status)
                train_prev_status = train_status
            
            # Custom events : Prediction Result
            if predict_result is not None and predict_result != predict_prev_result:
                yield 'event: predictresults\ndata: {}\n\n'.format(predict_result)
                predict_prev_result = predict_result

            time.sleep(0.2)

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
    

# Set training config
# Route will be /api/setconfig?epochs=xxx&uc=xxx&img=xxx&model=xxx
# epochs: number of epochs
# uc: User Content
# img: Image saving
# model: Model saving
# No need to have all parameters
@app.route('/api/setconfig', methods=['GET'])
def setconfig():
    if request.args.get('epochs') != None:
        epochs = request.args.get('epochs')
        if epochs != None and epochs != "" and epochs.isdigit() and int(epochs) > 0:
            epochs = int(epochs)
            aimane.set_training_config(epochs=epochs)
    if request.args.get('uc') != None:
        uc = request.args.get('uc')
        if uc == "true" or uc == "True" or uc == "1":
            uc = True
        elif uc == "false" or uc == "False" or uc == "0":
            uc = False
        aimane.set_training_config(usercontent=uc)
    if request.args.get('img') != None:
        img = request.args.get('img')
        if img == "true" or img == "True" or img == "1":
            img = True
        elif img == "false" or img == "False" or img == "0":
            img = False
        aimane.set_training_config(save_image=img)
    if request.args.get('model') != None:
        model = request.args.get('model')
        if model == "true" or model == "True" or model == "1":
            model = True
        elif model == "false" or model == "False" or model == "0":
            model = False
        aimane.set_training_config(save_model=model)
    return aimane.get_training_config()



    


    
    







@app.route('/api/iknowwhatimdoing/rewrite_filename', methods=['GET'])
def rewrite_filename():
    aimane.rewrite_filename()
    return "Rewrite filename process has started."



@app.route('/predict', methods=['GET'])
def predict_page():
    return render_template('predict.html')


@app.route('/', methods=['GET'])
def index():
    return render_template('main.html')


@app.route('/test', methods=['GET'])
def test1():
    return render_template('test.html')


# Add route for script/main.js
@app.route('/script/main.js', methods=['GET'])
def script():
    return render_template('script/main.js')






# for /app/v1
# New AIMANE app for production

# for /app/v0
# Old just for testing purpose app before v1 was created
# The purpose of this app is to test the api path and see if it works

@app.route('/app/<path:path>')
def appv0(path):
    return send_from_directory('app/frontend', path)



# # Initialize the app
# if __name__ == '__main__':
#     app.run(debug=True)


if __name__ == '__main__':
    ####### FOR DEV ########

    # FOR DEV WITHOUT SSL 
    # Professors can use this also If you want to see and debug the code

    # app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)



    # FOR DEV WITH SSL
    # Professors can not use this without the valid SSL certificate
    # Professors need to have SSL and put ssl files in app/ssl folder
    # Professors can generate SSL certificate using openssl or use certbot to get free SSL certificate

    #app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False, ssl_context=('app/ssl/site-cert.pem', 'app/ssl/site-key.pem'))
    



    ####### FOR PRODUCTION ########

    # FOR PRODUCTION WITHOUT SSL 
    # This is the default setting for production
    # Professors, please use this
    # This will use waitress as the production server instead of flask

    
    aimane.sysmane.write_status("Use http://localhost:5000/app/v1/index.html to access the app version 1")
    aimane.sysmane.write_status("Use http://localhost:5000/app/v0/index.html to access the app version prototype")
    aimane.sysmane.write_status("Use http://localhost:5000/api/<path> to access the api")
    aimane.sysmane.write_status("Use http://localhost:5000/api/sse/<path> to access the Server sent event api")
    aimane.sysmane.write_status("Server is started")
    serve(app, host='0.0.0.0', port=5000)







