from flask import Flask, send_from_directory
from flask_cors import CORS

client = Flask(__name__)
CORS(client, resources={r"/*": {"origins": "*", "supports_credentials": True}})

# for /app/v1
# New AIMANE app for production

# for /app/v0
# Old just for testing purpose app before v1 was created
# The purpose of this app is to test the api path and see if it works


# Handle index.html
@client.route('/')
def index():
    return send_from_directory('app/frontend/v2', 'index.html')

# Handle v1 index.html
@client.route('/v1/')
def indexv1():
    return send_from_directory('app/frontend/v1', 'index.html')


# Handle v0 index.html
@client.route('/v0/')
def indexv0():
    return send_from_directory('app/frontend/v0', 'index.html')

@client.route('/v0/<path:path>')
def appv0(path):
    return send_from_directory('app/frontend/v0', path)

@client.route('/v1/<path:path>')
def appv1(path):
    return send_from_directory('app/frontend/v1', path)

@client.route('/<path:path>')
def appv2(path):
    return send_from_directory('app/frontend/v2', path)



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

    print("[CLIENT] Use http://localhost:8080/ to access the app version 2")
    print("[CLIENT] Use http://localhost:8080/v1/ to access the app version 1")
    print("[CLIENT] Use http://localhost:8080/v0/ to access the app version prototype")

    print("AIMANE Client is started")
    print(" [!] You also need a server to use this app [!]")

    from waitress import serve
    serve(client, host='0.0.0.0', port=8080, threads=16)

else :
    # call with 
    # waitress-serve --listen=*:8080 client:client_app
    print("[CLIENT] Use http://localhost:8080/ to access the app version 1")
    print("[CLIENT] Use http://localhost:8080/v0/ to access the app version prototype")

    print("AIMANE Client is started in production mode")
    print(" [!] You also need a server to use this app [!]")

    client_app = client
    

    







