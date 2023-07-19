// app.js

class AIManeUI {
    constructor() {
        this.serverProtocol = "https";
        this.serverAddress = "192.168.1.2"
        this.serverPort = "5000";
        this.serverSSERoute = "/api/sse/events";
        this.serverURL = this.serverProtocol + "://" + this.serverAddress + ":" + this.serverPort;
        this.connected = false;
        this.status = null;
        this.laststatus = null;
        this.trainstatus = null;
        this.lasttrainstatus = null;
        this.predictresults = null;
        this.lastpredictresults = null;
        this.trainingtimerstart = null;
        this.source = null;
        this.finishedTraining = true;
        this.createdEpochProgressBox = false;
        this.correctPredictionButtonCreated = false;
        this.canvas = null;
        this.ui_elements = {
            "aimane-main" : document.querySelector(".aimane-main"), //.style.display = "Flex" or "None"
                "logmane": document.querySelector(".logmane"), //.className = "fas fa-file-download"
                    "lmn-icon" : document.querySelector(".lmn-icon div div div i"), //.textContent = "Loading Dataset"
                    "lmn-title" : document.querySelector(".lmn-title div h2"), //.textContent = "Saving validate image"
                    "lmn-subtitle" : document.querySelector(".lmn-subtitle div h5"), //.textContent
                    "lmn-percentage" : document.querySelector(".lmn-percentage div h1"), //.textContent = "0%"
                    "lmn-progress" : document.querySelector(".lmn-progress div div div"), //.style.width = "0%"
                "smallmane" : document.querySelector(".smallmane"), //.style.display = "Flex" or "None"
                    "smn-icon" : document.querySelector(".smn-icon div div div i"), //.className = "fas fa-atom"
                    "smn-title" : document.querySelector(".smn-title div h3"), //.textContent = "Training In-progress​"
                    "smn-time" : document.querySelector(".smn-time div h2"), //.textContent = "00:00:00"
                    "smn-box-progresstop" : document.querySelector(".smn-box-progress"), //For add epoch progress box
                    "smn-box-progress" : document.querySelector(".smn-box-progress div div div"), //.style.width = "0%"
                        "smn-box-progress-elem" : null,
                        "smn-template-passive" : document.querySelector(".slot-passive"), //.style.display = "Flex" or "None"
                        "smn-template-active" : document.querySelector(".slot-active"), //.style.display = "Flex" or "None"
                        "smn-template-done" : document.querySelector(".slot-done"), //.style.display = "Flex" or "None"
                    "smn-trainstats" : document.querySelector(".smn-trainstats"), //.style.display = "Flex" or "None"
                        "smn-acc" : document.querySelector(".smn-acc div h1"), //.textContent = "0%"
                        "smn-loss" : document.querySelector(".smn-loss div h1"), //.textContent = "0%"
                        "smn-batch" : document.querySelector(".smn-batch div h1"), //.textContent = "0%
                        "smn-epoch" : document.querySelector(".smn-epoch div h1"), //.textContent = "1"
                "resultmane" : document.querySelector(".resultmane"),
                    "rmn-icon" : document.querySelector(".rmn-icon div div div i"), //.className = "fas fa-atom"
                    "rmn-title" : document.querySelector(".rmn-title div h3"), //.textContent = "Prediction​"
                    "rmn-guess" : document.querySelector(".rmn-guess div h4"), //.textContent = "I think you write..."
                    "rmn-number" : document.querySelector(".rmn-number div h1"), //.textContent = "0"
                    "rmn-teachtext" : document.querySelector(".rmn-teachtext div h2"), //.textContent = "Teach me if I'm wrong"
                    "rmn-trytext" : document.querySelector(".rmn-trytext div h4"), //.textContent = "Try writing your number"
                    "rmn-teachbox" : document.querySelector(".rmn-teachbox"), //.style.display = "Flex" or "None" //Trigger click event
                    "rmn-teachclasses" : document.querySelector(".rmn-teach"), //.style.display = "Flex" or "None"
                        "rmn-template-retrain" : document.querySelector(".btn-retrain"),
                    "rmn-btn-prepare" : document.querySelector(".btn-prepare"), //Trigger click event
                    "rmn-btn-repair" : document.querySelector(".btn-repair"), //Trigger click event
                    "rmn-btn-rtuc" : document.querySelector(".btn-rtuc"), //Trigger click event
                    "rmn-btn-train" : document.querySelector(".btn-train"), //Trigger click event
                    "rmn-btn-reset" : document.querySelector(".btn-reset"), //Trigger click event
                    "rmn-btn-guess" : document.querySelector(".btn-guess"), //Trigger click event
                    "rmn-btn-wrong" : document.querySelector(".btn-wrong"), //Trigger click event
                        "box-disconnect" : document.querySelector(".box-disconnect"),
                    "rmn-canvas" : document.querySelector("canvas#rmn-canvas"), //For add canvas
                "disc-header" : document.querySelector(".disc-header div h2"), //.textContent = "Disconnected from AIMANE Server​"
                "disc-subheader" : document.querySelector(".disc-subheader div h5"), //.textContent = "Please check your connection"
                "disc-reconnect" : document.querySelector(".disc-reconnect"), //Trigger click event
                "disc-selectserver" : document.querySelector(".disc-selectserver"), //Trigger click event
            "box-connect" : document.querySelector(".box-connect"), //.style.display = "Flex" or "None"
                "conn-serverlist-box" : document.querySelector(".conn-serverlist-box div"), // For add server list button
                "conn-serverlist-button" : document.querySelector(".conn-serverlist-button"), //.style.display = "Flex" or "None"
                "conn-address-field" : document.querySelector("#form-field-srvaddress"), //.value = "192.168.1.2:5000"
                "conn-ok-button" : document.querySelector(".connect") //Trigger click event
        };
        this.server_list = [
            {
                "name": "192.168.1.2:5000",
                "address": "192.168.1.2",
                "port": "5000",
                "protocol": "https"
            },
            {
                "name": "localhost:5000",
                "address": "localhost",
                "port": "5000",
                "protocol": "https"
            }
        ];

        this.init();

    }
    
    showElement(element_name) {
        this.ui_elements[element_name].style.display = "Flex";
    }

    hideElement(element_name) {
        this.ui_elements[element_name].style.display = "none";
    }

    changeText(element_name, text) {
        this.ui_elements[element_name].textContent = text;
    }

    getText(element_name) {
        return this.ui_elements[element_name].textContent;
    }

    changeIcon(element_name, icon) {
        let icon_name = "fas fa-" + icon;
        this.ui_elements[element_name].className = icon_name;
    }

    getIcon(element_name) {
        return this.ui_elements[element_name].className;
    }

    changeProgress(element_name, progress) {
        this.ui_elements[element_name].style.width = progress + "%";
    }

    getProgress(element_name) {
        return this.ui_elements[element_name].style.width;
    }

    init() {
        this.consoleLog("「AIMANE」 by Nattawut Manjai-araya  v1.0.0");
        // Init UI

        // Prepare server list
        this.createServerSelectionButton();

        // Add event listener to button
        this.buttonTriggerSetup();
        this.showConnectScreen();

        // Setup canvas
        this.setupDrawCanvas();
    }

    buttonTriggerSetup() {
        this.ui_elements["conn-ok-button"].addEventListener("click", () => {
            // Called function to change server
            //getServerURLFromFields return [address, port, protocol];
            let serverdata = this.getServerURLFromFields();
            this.changeServer(serverdata[0], serverdata[1], serverdata[2]);
        });        
        this.ui_elements["disc-reconnect"].addEventListener("click", () => {
            // Called function to change server
            //getServerURLFromFields return [address, port, protocol];
            this.reconnect();
        });
        this.ui_elements["disc-selectserver"].addEventListener("click", () => {

            this.showConnectScreen();
        });
        this.ui_elements["rmn-btn-prepare"].addEventListener("click", () => {
            this.aimane_prepare();
        });
        this.ui_elements["rmn-btn-repair"].addEventListener("click", () => {
            this.aimane_repair();
        });
        this.ui_elements["rmn-btn-rtuc"].addEventListener("click", () => {
            this.aimane_rtuc();
        });
        this.ui_elements["rmn-btn-train"].addEventListener("click", () => {
            this.aimane_train();
        });
        this.ui_elements["rmn-btn-reset"].addEventListener("click", () => {
            this.resetCanvas();
        });
        this.ui_elements["rmn-btn-guess"].addEventListener("click", () => {
            this.predict();
        });
        this.ui_elements["rmn-btn-wrong"].addEventListener("click", () => {
            this.aimane_wrong();
        });
    }

    aimane_prepare() {
        // send GET api to {server}/api/prepare
        fetch(this.serverURL + "/api/prepare", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                this.consoleLog("Prepare request sent successfully", "SUCCESS");
            }
            else {
                this.consoleLog("Prepare request sent failed", "ERROR");
            }
        });
    }

    aimane_repair() {
        // send GET api to {server}/api/repairdataset
        fetch(this.serverURL + "/api/repairdataset", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                this.consoleLog("Repair request sent successfully", "SUCCESS");
            }
            else {
                this.consoleLog("Repair request sent failed", "ERROR");
            }
        });
    }

    aimane_rtuc() {
        // send GET api to {server}/api/rtuc
        fetch(this.serverURL + "/api/rtuc", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                this.consoleLog("Resource to Usercontent request sent successfully", "SUCCESS");
            }
            else {
                this.consoleLog("Resources to Usercontents request sent failed", "ERROR");
            }
        });
    }

    aimane_train() {
        // send GET api to {server}/api/train
        fetch(this.serverURL + "/api/train", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                this.consoleLog("Train request sent successfully", "SUCCESS");
            }
            else {
                this.consoleLog("Train request sent failed", "ERROR");
            }
        });
    }

    aimane_correctprediction(defclass) {
        // send GET api to {server}/api//api/definelastpredict?label={defclass}
        fetch(this.serverURL + "/api/definelastpredict?label=" + defclass, {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                this.consoleLog("Correct prediction to " + defclass + " request sent successfully", "SUCCESS");
            }
            else {
                this.consoleLog("Correct prediction to " + defclass + " request sent failed", "ERROR");
            }
        });
    }

    setupDrawCanvas(){
        this.canvas = new fabric.Canvas(this.ui_elements["rmn-canvas"]);
        this.canvas.backgroundColor = "#000000";
        this.canvas.renderAll();
        this.canvas.isDrawingMode = true;
        this.canvas.freeDrawingBrush.width = 10;
        this.canvas.freeDrawingBrush.color = "white";

    }

    resetCanvas() {
        this.canvas.clear();
        this.canvas.backgroundColor = "#000000";
        this.canvas.renderAll();
    }


    showConnectScreen() {
        this.showElement("box-connect");
        this.hideElement("box-disconnect");
        this.hideElement("aimane-main");
    }
    
    showDisconnectScreen() {
        this.hideElement("box-connect");
        this.showElement("box-disconnect");
        this.hideElement("aimane-main");
    }

    showMainScreen() {
        this.hideElement("box-connect");
        this.hideElement("box-disconnect");
        this.showElement("aimane-main");
    }

    logmane(header = null, subheader = null, percentage = null, icon = null) {
        if (header != null) {
            if(this.getText("lmn-title") != header) { this.changeText("lmn-title", header);}
        }
        if (subheader != null) {
            if(this.getText("lmn-subtitle") != subheader) { this.changeText("lmn-subtitle", subheader);}
        }
        if (percentage != null) {
            let rpercentage = this.roundDown(percentage, 2);
            this.changeText("lmn-percentage", rpercentage+"%");
            if(this.getProgress("lmn-progress") != percentage) { this.changeProgress("lmn-progress", percentage);}
        }

        if (icon != null) {
            if(this.getIcon("lmn-icon") != icon) { this.changeIcon("lmn-icon", icon);}
        }
    }

    smallmane(title = null, icon = null, time = null) {
        if (title != null) {
            if(this.getText("smn-title") != title) { this.changeText("smn-title", title);}
        }
        if (icon != null) {
            if(this.getIcon("smn-icon") != icon) { this.changeIcon("smn-icon", icon);}
        }
        if (time != null) {
            if(this.getText("smn-time") != time) { this.changeText("smn-time", time);}
        }
    }

    smtrainstatus(acc = null, loss = null, batch = null, epoch = null, total_epoch = null) {
        if (acc != null) {
            if(this.getText("smn-acc") != acc) { this.changeText("smn-acc", acc);}
        }
        if (loss != null) {
            if(this.getText("smn-loss") != loss) { this.changeText("smn-loss", loss);}
        }
        if (batch != null) {
            if(this.getText("smn-batch") != batch) { this.changeText("smn-batch", batch);}
        }
        if (epoch != null) {
            epoch = epoch + "/" + total_epoch;
            if(this.getText("smn-epoch") != epoch) { this.changeText("smn-epoch", epoch);}
        }
    }

    resultmane(number = null) {
        if (number != null) {
            if(this.getText("rmn-number") != number) { this.changeText("rmn-number", number);}
        }
        let guessText = [
            "I think it's a ",
            "Hmm, I think it's a ",
            "Wow it's a ",
            "Let me guess, it's a ",
            "From my experience, it's a ",
            "I'm pretty sure it's a ",
            "I'm guessing it's a ",
            "Maybe it's a ",
            "I think today lotto is"
        ]
        let guess = guessText[Math.floor(Math.random() * guessText.length)];
        if(this.getText("rmn-guess") != guess) { this.changeText("rmn-guess", guess);}

        let teachText = [
            "Teach me if I'm wrong",
            "You can teach me",
            "I'm still learning",
            "I'm wrong?",
            "Oh, I'm wrong?",
            "Teach me!!",
        ]
        let teach = teachText[Math.floor(Math.random() * teachText.length)];
        if(this.getText("rmn-teachtext") != teach) { this.changeText("rmn-teachtext", teach);}

        let tryText = [
            "Try writing your number",
            "It's your turn",
            "Let's try",
            "Let's write",
            "Write your number",
            "Write your number here",
            "Give me your number",
            "It's lotto time",
        ]
        let tryy = tryText[Math.floor(Math.random() * tryText.length)];
        if(this.getText("rmn-trytext") != tryy) { this.changeText("rmn-trytext", tryy);}

    }


    createServerSelectionButton() {
        for (var i = 0; i < this.server_list.length; i++) {
            let newdiv = this.ui_elements["conn-serverlist-button"].cloneNode(true);
            newdiv.className = "elementor-element elementor-element-380916d elementor-align-center conn-serverlist-button elementor-widget elementor-widget-button serverlist-b" + i;
            newdiv.querySelector(".elementor-button-text").textContent = this.server_list[i]["name"];
            newdiv.style.display = "flex";
            // Add property to button for access server list
            newdiv.address = this.server_list[i]["address"];
            newdiv.port = this.server_list[i]["port"];
            newdiv.protocol = this.server_list[i]["protocol"];
            // Add event listener to button
            newdiv.addEventListener("click", () => {
                // Called function to change server
                this.changeServer(newdiv.address, newdiv.port, newdiv.protocol);
            });
            this.ui_elements["conn-serverlist-box"].appendChild(newdiv);
        }
    }

    createCorrectPredictionButton(classes) {
        if(this.correctPredictionButtonCreated) {
            this.consoleLog("Correct prediction button already created", "WARN");
            return;
        }
        //classes= ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Star']

        for (var i = 0; i < classes.length; i++) {
            let newdiv = this.ui_elements["rmn-template-retrain"].cloneNode(true);
            newdiv.className = "elementor-element elementor-element-d05252b elementor-align-center elementor-widget__width-initial btn-retrain elementor-widget elementor-widget-button btn-retrain-" + classes[i];
            newdiv.querySelector(".elementor-button-text").textContent = classes[i];
            newdiv.style.display = "flex";
            // Add property to button for easy access class
            newdiv.trainclass = i
            // Add event listener to button
            newdiv.addEventListener("click", () => {
                // Called function to correct prediction
                this.aimane_correctprediction(newdiv.trainclass);
            });
            this.ui_elements["rmn-teachclasses"].appendChild(newdiv);
        }
        this.correctPredictionButtonCreated = true;
    }


    deleteCorrectPredictionButton() {
        if(!this.correctPredictionButtonCreated) {
            this.consoleLog("Correct prediction button not created", "WARN");
            return;
        }
        // Delete all button with loop
        for (var i = 0; i < this.ui_elements["rmn-teachclasses"].childElementCount; i++) {
            this.ui_elements["rmn-teachclasses"].removeChild(this.ui_elements["rmn-teachclasses"].children[0]);
        }
        this.correctPredictionButtonCreated = false;
    }


    predict() {
        let dataURL = this.canvas.toDataURL();
        let blob = this.dataURLtoBlob(dataURL);
    
        let formData = new FormData();
        formData.append('image', blob, 'image.png');
    
        fetch(this.serverProtocol + "://" + this.serverAddress + ":" + this.serverPort + "/api/predict", {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                this.consoleLog("Error: " + response.status + " " + response.statusText, "ERROR");
            }
            return response.json();
        })
        .then(data => {
            this.consoleLog("Prediction is: " + data[0], "SUCCESS");
            this.resultmane(data[0]);
        })
        .catch(error => {
            this.consoleLog("Error: " + error.message, "ERROR");
        });
    }

    dataURLtoBlob(dataURL) {
        let parts = dataURL.split(';base64,');
        let contentType = parts[0].split(':')[1];
        let raw = window.atob(parts[1]);
        let rawLength = raw.length;
        let uInt8Array = new Uint8Array(rawLength);

        for (let i = 0; i < rawLength; ++i) {
            uInt8Array[i] = raw.charCodeAt(i);
        }

        return new Blob([uInt8Array], { type: contentType });
    }
        

        
    

    changeServer(address, port, protocol) {
        // Change server address and port
        this.serverAddress = address;
        this.serverPort = port;
        this.serverProtocol = protocol;
        this.serverURL = this.serverProtocol + "://" + this.serverAddress + ":" + this.serverPort;
        this.consoleLog("Server changed to " + this.serverProtocol + "://" + this.serverAddress + ":" + this.serverPort,"INFO");
        // If connected, disconnect and reconnect
        if (this.connected){
            this.disconnect();
        }else{
            this.connect();
        }
        
    }

    reconnect() {
        this.disconnect();
        this.connect();
    }


    createEpochProgressBox(total_epoch) {
        // If already created, not create again
        if (this.createdEpochProgressBox) {
            this.consoleLog("Epoch progress box already created","WARN");
            return;
        }
        // Append copy of slot-passive to topdiv and add new class ep-xxx (loop from total_epoch start from 1)
        this.consoleLog("Create epoch progress box with " + total_epoch + " epoch","INFO");

        for (var i = 1; i <= total_epoch; i++) {
            var newdiv = this.ui_elements["smn-template-passive"].cloneNode(true);
            newdiv.className = "elementor-element elementor-element-aaa4191 e-con-full slot-passive e-con epochbox ep-" + i;
            newdiv.querySelector(".elementor-heading-title").textContent = i;
            newdiv.style.display = "flex";
            this.ui_elements["smn-box-progresstop"].appendChild(newdiv);
            // Add class ep-xxx to smn-box-progress-elem
            this.ui_elements["smn-box-progress-elem"] = document.querySelector(".ep-" + i);

        }
        this.createdEpochProgressBox = true;
    }


    deleteAllEpochProgressBox() {
        this.consoleLog("Delete all epoch progress box","INFO");
        // Delete all element with class epochbox
        var epochbox = document.querySelectorAll(".epochbox");
        for (var i = 0; i < epochbox.length; i++) {
            epochbox[i].remove();
        }
        this.createdEpochProgressBox = false;
    }


    setEpochActive(epoch, text = epoch) {
        // Select slot-passive with class ep-xxx and change to slot-active
        var olddiv = document.querySelector(".ep-" + epoch);
        var newdiv = this.ui_elements["smn-template-active"].cloneNode(true);
        newdiv.className = "elementor-element elementor-element-2a948e0 e-con-full slot-active e-con epochbox ep-" + epoch;
        newdiv.querySelector(".elementor-heading-title").textContent = text;
        newdiv.style.display = "flex";
        olddiv.parentNode.replaceChild(newdiv, olddiv);
        // Change smn-box-progress-elem to new slot-active
        this.ui_elements["smn-box-progress-elem"] = document.querySelector(".ep-" + epoch);
    }


    setEpochPassive(epoch,text = epoch) {
        var olddiv = document.querySelector(".ep-" + epoch);
        var newdiv = this.ui_elements["smn-template-passive"].cloneNode(true);
        newdiv.className = "elementor-element elementor-element-aaa4191 e-con-full slot-passive e-con epochbox ep-" + epoch;
        newdiv.querySelector(".elementor-heading-title").textContent = text;
        newdiv.style.display = "flex";
        olddiv.parentNode.replaceChild(newdiv, olddiv);
        this.ui_elements["smn-box-progress-elem"] = document.querySelector(".ep-" + epoch);
    }

    setEpochDone(epoch,text = epoch) {
        var olddiv = document.querySelector(".ep-" + epoch);
        var newdiv = this.ui_elements["smn-template-done"].cloneNode(true);
        newdiv.className = "elementor-element elementor-element-0ade0b9 e-con-full slot-done e-con epochbox ep-" + epoch;
        newdiv.querySelector(".elementor-heading-title").textContent = text;
        newdiv.style.display = "flex";
        olddiv.parentNode.replaceChild(newdiv, olddiv);
        this.ui_elements["smn-box-progress-elem"] = document.querySelector(".ep-" + epoch);
    }
        

    getServerURLFromFields() {
        var address = this.ui_elements["conn-address-field"].value;
        // saparate address and port if include protocol also
        var protocol = "https";
        var port = "5000";
        if (address.includes("https://")) {
            protocol = "https";
            address = address.replace("https://", "");
        }

        if (address.includes("http://")) {
            protocol = "http";
            address = address.replace("http://", "");
        }

        if (address.includes(":")) {
            var split = address.split(":");
            address = split[0];
            port = split[1];
        }

        if (protocol == "") {
            protocol = "https";
        }

        if (address == "") {
            address = "localhost";
        }

        if (port == "") {
            port = "5000";
        }

        console.log("[INFO] Server protocol: " + protocol + "| address: " + address + "| port: " + port);
        return [address, port, protocol];

    }


    roundDown(num, precision) {
        precision = Math.pow(10, precision)
        return Math.floor(num * precision) / precision
    }
    
    

    // Setup SSE
    async setupSSE() {
        let sse_url = `${this.serverProtocol}://${this.serverAddress}:${this.serverPort}${this.serverSSERoute}`;
        // Check if SSE is already connected
        if (this.connected || this.source != null) {
            // Disconnect first
            this.disconnect();
            this.connected = false;
        }

        this.source = await (new EventSource(sse_url));
        // Check SSE connection
        this.source.onopen = (e) => {
            console.log("[INFO] SSE connected to " + sse_url);
            this.connected = true;
        };
        this.source.onerror = (e) => {
            console.log("[ERROR] SSE connection error");
            this.connected = false;
            this.disconnect();
            this.showDisconnectScreen();
        };
        console.log("[INFO] SSE setup at " + sse_url);
        // Listen for messages from server type: status
        this.source.addEventListener('status', (e) => {
            var data = JSON.parse(e.data);
            console.log("[INFO] SSE status: ");
            console.log(data);
            this.status = data;
        
            // Handle Status
            this.handleStatus(this.status);
            // document.getElementById("status").innerHTML = this.status;
        }, false);
        // Listen for messages from server type: trainstatus
        this.source.addEventListener('trainstatus', (e) => {
            var data = JSON.parse(e.data);
            console.log("[INFO] SSE trainstatus: ");
            console.log(data);
            this.trainstatus = data;
            this.handleTrainStatus(this.trainstatus);
            //document.getElementById("trainstatus").innerHTML = this.trainstatus;
        }, false);
        // Listen for messages from server type: predictresults
        this.source.addEventListener('predictresults', (e) => {
            var data = JSON.parse(e.data);
            console.log("[INFO] SSE predictresults: ");
            console.log(data);
            this.predictresults = data;
            this.handlePredictResults(this.predictresults);
            
            //document.getElementById("predictresults").innerHTML = this.predictresults;
        }, false);

        return this.source;
    }

    disconnectSSE() {
        this.source.close();
        this.consoleLog("[INFO] SSE disconnected from " + this.serverAddress + ":" + this.serverPort,"WARN");
    }


    connect() {
        let ssesource = this.setupSSE();
        if (ssesource) {
            this.consoleLog("Connected to server at " + this.serverAddress + ":" + this.serverPort,"SUCCESS");
            this.connected = true;
            this.deleteAllEpochProgressBox();
            this.showMainScreen();
        }else{
            this.consoleLog("Connection to server at " + this.serverAddress + ":" + this.serverPort + " failed", "ERROR");
            this.connected = false;
            this.showConnectScreen();
        }
    }

    disconnect() {
        this.connected = false;
        this.consoleLog("[INFO] Disconnected from server at " + this.serverAddress + ":" + this.serverPort,"WARN");
    }



    consoleLog(Text, Type = "", Color = "", Bold = false, Italic = false) {
        let logStyle = "";
    
        // 「AIMANE」
        switch (Type.toUpperCase()) {
            case "INFO":
                logStyle = "background-color: #E60962; color: white;";
                break;
            case "SUCCESS":
                logStyle = "background-color: green; color: white;";
                break;
            case "WARN":
                logStyle = "background-color: orange; color: white;";
                break;
            case "ERROR":
                logStyle = "background-color: red; color: white;";
                break;
            default:
                break;
        }
    
        if (Color) {
            logStyle = `background-color: ${Color}; color: white;`;
        }
    
        // Apply Bold and Italic styles if specified
        if (Bold && Italic) {
            Text = `<b><i>${Text}</i></b>`;
        } else if (Bold) {
            Text = `<b>${Text}</b>`;
        } else if (Italic) {
            Text = `<i>${Text}</i>`;
        }
    
        const logMessage = `%c${Type ? " [" + Type + "] " : ""}${Text}`;
    
        console.log(logMessage+" ", logStyle);
    }



    handleStatus(status) {

        // {
        //     "time": "2023-07-20 00:01:52",
        //     "status": "Server is started",
        //     "stage": "Starting",
        //     "percentage": 0
        // }

        if (status != null && status != this.lastStatus) {
            console.log("[INFO] Time: " + status.time + " | Status: " + status.status + " | Stage: " + status.stage + " | Percentage: " + status.percentage);
            // If stage have this string "Preparing" 
            let icon = "";

                if(status.stage.includes("Preparing")){
                    icon = "file-download"
                }
                else if(status.stage.includes("Starting")){
                    icon = "play"
                }
                else if(status.stage.includes("Repair")){
                    icon = "wrench"
                }
                else if(status.stage.includes("Training")){
                    icon = "robot"
                }
                else if(status.stage.includes("Predict")){
                    icon = "magic"
                }
                else if(status.stage.includes("Done")){
                    icon = "check"
                }
                else{
                    icon = "greater-than-equal"
                }

   
            this.logmane(status.stage, status.status, status.percentage, icon);

            this.lastStatus = status;
        }
    }

    handleTrainHistory(data,finished = false,live_acc = null) {
        // {
            // "e1": 0.8696097135543823, 
            // "e2": 0.9497954249382019
        //},
        // Check if data is not null and have at least 1 element
        if (data != null && Object.keys(data).length > 0) {
            // If finished training before
            if (this.finishedTraining && this.trainingtimerstart == null) {
                // Clear all
                this.deleteAllEpochProgressBox();
                //Setup new epoch progress box
                this.createEpochProgressBox(this.trainstatus.total_epoch);
            }

            // Loop through each element in data
            for (const [key, value] of Object.entries(data)) {
                // setEpochPassive(epoch, acc)
                // epoch is a number from 0 to total_epoch
                let epoch = parseInt(key.replace("e", ""));
                let rvalue = this.roundDown(value, 3);
            // If finished the last epoch will use  this.setEpochDone(epoch, value);
            // If not finished the last epoch will +1 from object entries and use this.setEpochActive(epoch,live_acc);
            // Other will use  this.setEpochPassive(epoch, value); 
                // Check if createdEpochProgressBox is false
                if (!this.createdEpochProgressBox) {
                    this.createEpochProgressBox(this.trainstatus.total_epoch);
                }

                if (finished) {
                    // If this is the last epoch
                    // use object entries to get the last epoch instead of using the this.trainstatus.total_epoch
                    if (epoch == Object.keys(data).length) {
                        this.setEpochDone(epoch, rvalue);
                    }
                    else {
                        this.setEpochActive(epoch, rvalue);
                    }
                }
                else {
                    // If this is the last epoch + 1
                    if (epoch == Object.keys(data).length + 1) {
                        if (live_acc != null) {
                            live_acc = this.roundDown(live_acc, 3);
                        }
                        this.setEpochDone(epoch,live_acc);
                    }
                    else {
                        this.setEpochActive(epoch, rvalue);
                    }
                    
                }
            }
        }
    }            

        

    handleTrainStatus(data) {
        // {"time": "2023-07-20 01:56:12",
        // "status": "Not training",
        // "stage": "Not running",
        // "percentage": -1,
        // "epoch": -1,
        // "total_epoch": 25,
        // "batch": -1,
        // "loss": -1,
        // "acc": -1,
        // "history": {
            // "e1": 0.8696097135543823, 
            // "e2": 0.9497954249382019
        // },
        // "finished": false,
        // "result": "Not ready"}

        if (data != null && data != this.lastTrainStatus) {
            // If this.createEpochProgressBox(this.trainstatus.total_epochs); is not called
            if (!this.createdEpochProgressBox){
                this.createEpochProgressBox(data.total_epoch);
            }

            // If status is "Training model"
            if (data.status == "Training model") {
                // If last status is not "Training model"
                if (this.lastTrainStatus == null || this.lastTrainStatus.status != "Training model") {
                    // Clear all
                    this.deleteAllEpochProgressBox();
                }
                //If this.trainingtimerstart is null, start the training timer
                if (this.trainingtimerstart == null) {
                    this.handleTrainingTimer("start");
                }
                this.finishedTraining = false;
                //convert from 0.123456 to 12.345% with rounddown 3 decimal 
                let acctext = this.roundDown((data.acc * 100),3) + "%";
                let losstext = this.roundDown((data.loss * 100),3) + "%";

                this.smtrainstatus(acctext, losstext, data.batch, data.epoch, data.total_epoch);
                // handleTrainHistory
                // If data.acc is -1 then use null
                let real_acc = data.acc == -1 ? null : data.acc;
                this.handleTrainHistory(data.history,data.finished,real_acc);
            }

            // If status is "Finished training"
            if (data.status == "Finished training") {
                //If this.trainingtimerstart is not null, stop the training timer
                if (this.trainingtimerstart != null) {
                    this.handleTrainingTimer("stop");
                }
                this.finishedTraining = true;
                
                //convert from 0.123456 to 12.345% with rounddown 3 decimal 
                let acctext = this.roundDown((data.acc * 100),3) + "%";
                let losstext = this.roundDown((data.loss * 100),3) + "%";

                this.smtrainstatus(acctext, losstext, data.batch, data.epoch, data.total_epoch);
                // handleTrainHistory
                // If data.acc is -1 then use null
                let real_acc = data.acc == -1 ? null : data.acc;
                this.handleTrainHistory(data.history,data.finished,real_acc);
            }
                

            this.lastTrainStatus = data;
        }
    }

    handleTrainingTimer(state = "stop") {
        // When state is start, start the timer record start timer time to this.trainingtimerstart
        // and update the timer every second by calculating the difference between current time and this.trainingtimerstart
        if(state == "start"){
            // Check if this.trainingtimerstart is not null
            if(this.trainingtimerstart != null){
                this.consoleLog("Training timer is already started", "WARN");
            }
            else{
                this.trainingtimerstart = new Date();
                this.trainingtimer = setInterval(() => {
                    let now = new Date();
                    let diff = now - this.trainingtimerstart;
                    let diffsec = Math.floor(diff / 1000);
                    let diffmin = Math.floor(diffsec / 60);
                    let diffhour = Math.floor(diffmin / 60);
                    diffsec = diffsec % 60;
                    diffmin = diffmin % 60;
                    let diffstr = diffhour.toString().padStart(2, "0") + ":" + diffmin.toString().padStart(2, "0") + ":" + diffsec.toString().padStart(2, "0");
                    this.changeText("smn-time", diffstr);
                }, 1000);
            }
        // When state is stop, stop the timer clear the timer interval and clear the this.trainingtimerstart to null
        }else if(state == "stop"){
            // Check if this.trainingtimerstart is null
            if(this.trainingtimerstart == null){
                this.consoleLog("Training timer is already stopped", "WARN");
            }
            else{
                clearInterval(this.trainingtimer);
                this.trainingtimerstart = null;
                this.changeText("smn-time", "00:00:00");
            }
        }
    }

    handlePredictResults(data) {
        // {
        //     "result": null,
        //     "percentage": null,
        //     "other_result": null,
        //     "other_percentage": null,
        //     "uuid": null,
        //     "image_data": null,
        //     "numpy_array": null,
        //     "classes": 11,
        //     "class_names": [
        //         "0",
        //         "1",
        //         "2",
        //         "3",
        //         "4",
        //         "5",
        //         "6",
        //         "7",
        //         "8",
        //         "9",
        //         "Star"
        //     ]
        // }
        if (data != null && data != this.lastPredictResults) {
            this.resultmane(data.result)
            this.lastPredictResults = data;

        }

        if (this.correctPredictionButtonCreated == false) {
            this.createCorrectPredictionButton(data.class_names)
        }

    }

}


  const aiManeUI = new AIManeUI();
  