// app.js

class AIManeUI {
    constructor() {
        this.serverProtocol = "https";
        this.serverAddress = "192.168.1.2"
        this.serverPort = "5000";
        this.serverSSERoute = "/api/sse/events";
        this.connected = false;
        this.status = null;
        this.laststatus = null;
        this.trainstatus = null;
        this.lasttrainstatus = null;
        this.predictresults = null;
        this.lastpredictresults = null;
        this.source = null;
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
                    "rmn-teachbox" : document.querySelector(".rmn-teachbox"), //.style.display = "Flex" or "None"
            "box-disconnect" : document.querySelector(".box-disconnect"), //.style.display = "Flex" or "None"
                "disc-header" : document.querySelector(".disc-header div h2"), //.textContent = "Disconnected from AIMANE Server​"
                "disc-subheader" : document.querySelector(".disc-subheader div h5"), //.textContent = "Please check your connection"
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
    }

    buttonTriggerSetup() {
        this.ui_elements["conn-ok-button"].addEventListener("click", () => {
            // Called function to change server
            //getServerURLFromFields return [address, port, protocol];
            let serverdata = this.getServerURLFromFields();
            this.changeServer(serverdata[0], serverdata[1], serverdata[2]);
        });        
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
            this.changeText("lmn-percentage", percentage+"%");
            if(this.getProgress("lmn-progress") != percentage) { this.changeProgress("lmn-progress", percentage);}
        }

        if (icon != null) {
            if(this.getIcon("lmn-icon") != icon) { this.changeIcon("lmn-icon", icon);}
        }
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
    

    changeServer(address, port, protocol) {
        // Change server address and port
        this.serverAddress = address;
        this.serverPort = port;
        this.serverProtocol = protocol;
        this.consoleLog("Server changed to " + this.serverProtocol + "://" + this.serverAddress + ":" + this.serverPort,"INFO");
        // If connected, disconnect and reconnect
        if (this.connected){
            this.disconnect();
        }else{
            this.connect();
        }
        
    }

    createEpochProgressBox(total_epoch) {
        // Append copy of slot-passive to topdiv and add new class ep-xxx (loop from total_epoch start from 1)

        for (var i = 1; i <= total_epoch; i++) {
            var newdiv = this.ui_elements["smn-template-passive"].cloneNode(true);
            newdiv.className = "elementor-element elementor-element-aaa4191 e-con-full slot-passive e-con epochbox ep-" + i;
            newdiv.querySelector(".elementor-heading-title").textContent = i;
            newdiv.style.display = "flex";
            this.ui_elements["smn-box-progresstop"].appendChild(newdiv);
            // Add class ep-xxx to smn-box-progress-elem
            this.ui_elements["smn-box-progress-elem"] = document.querySelector(".ep-" + i);

        }
    }


    deleteAllEpochProgressBox() {
        // Delete all element with class epochbox
        var epochbox = document.querySelectorAll(".epochbox");
        for (var i = 0; i < epochbox.length; i++) {
            epochbox[i].remove();
        }
    }


    setEpochActive(epoch, text = epoch) {
        // Select slot-passive with class ep-xxx and change to slot-active
        var olddiv = document.querySelector(".ep-" + epoch);
        var newdiv = this.ui_elements["smn-slot-active"].cloneNode(true);
        newdiv.className = "elementor-element elementor-element-2a948e0 e-con-full slot-active e-con epochbox ep-" + epoch;
        newdiv.querySelector(".elementor-heading-title").textContent = text;
        newdiv.style.display = "flex";
        olddiv.parentNode.replaceChild(newdiv, olddiv);
        // Change smn-box-progress-elem to new slot-active
        this.ui_elements["smn-box-progress-elem"] = document.querySelector(".ep-" + epoch);
    }


    setEpochPassive(epoch,text = epoch) {
        var olddiv = document.querySelector(".ep-" + epoch);
        var newdiv = this.ui_elements["smn-slot-passive"].cloneNode(true);
        newdiv.className = "elementor-element elementor-element-aaa4191 e-con-full slot-passive e-con epochbox ep-" + epoch;
        newdiv.querySelector(".elementor-heading-title").textContent = text;
        newdiv.style.display = "flex";
        olddiv.parentNode.replaceChild(newdiv, olddiv);
        this.ui_elements["smn-box-progress-elem"] = document.querySelector(".ep-" + epoch);
    }

    setEpochDone(epoch,text = epoch) {
        var olddiv = document.querySelector(".ep-" + epoch);
        var newdiv = this.ui_elements["smn-slot-done"].cloneNode(true);
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
        this.source.onopen = function(e) {
            console.log("[INFO] SSE connected to " + sse_url);
            this.connected = true;
        };
        this.source.onerror = function(e) {
            console.log("[ERROR] SSE connection error");
            this.connected = false;
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
            //document.getElementById("trainstatus").innerHTML = this.trainstatus;
        }, false);
        // Listen for messages from server type: predictresults
        this.source.addEventListener('predictresults', (e) => {
            var data = JSON.parse(e.data);
            console.log("[INFO] SSE predictresults: ");
            console.log(data);
            this.predictresults = data;
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
            let istatus = "";
            let icon = "";

            if(status.stage.includes("Preparing")){
                istatus = "Preparing"
            }else if(status.stage.includes("Repair")){
                istatus = "Repair"
            }else if(status.stage.includes("Training")){
                istatus = "Training"
            }else if(status.stage.includes("Predict")){
                istatus = "Predict"
            }else if(status.stage.includes("Done")){
                istatus = "Done"
            }else{
                istatus = "Unknown"
            }

            switch(istatus){
                case "Preparing":
                    icon = "file-download"
                    break;
                case "Repair":
                    icon = "wrench"
                    break;
                case "Training":
                    icon = "robot"
                    break;
                case "Predict":
                    icon = "magic"
                    break;
                case "Done":
                    icon = "check"
                    break;
                default:
                    icon = "greater-than-equal"
            }

   
            this.logmane(status.stage, status.status, status.percentage, icon);

            this.lastStatus = status;
        }
    }

}


  const aiManeUI = new AIManeUI();
  