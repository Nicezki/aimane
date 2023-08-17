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
        this.on_selectserver = true;
        this.lastmodel = null;
        this.ui_elements = {
            "aimane-main" : document.querySelector(".aimane-main"), //.style.display = "Flex" or "None"
                "logmane": document.querySelector(".logmane"), //.className = "fas fa-file-download"
                "box-log" : document.querySelector(".log-box"), //.style.display = "Flex" or "None"
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
                    "btn-configtrain" : document.querySelector(".btn-configtrain"), //Trigger click event
                    "smn-training-config-box" : document.querySelector(".smn-training-config-box"), //.style.display = "Flex" or "None"

                        "conf-cur-facc" : document.querySelector(".conf-cur-facc div h6"), //.textContent = "Current: 0.0000"
                        "form-field-facc" : document.querySelector("#form-field-facc"), // Text Field
                        "conf-cur-uc" : document.querySelector(".conf-cur-uc div h6"), // .textContent = "Current: true"
                        "form-field-uc" : document.querySelector("#form-field-uc"), // Dropdown
                        "conf-cur-saveimg" : document.querySelector(".conf-cur-saveimg div h6"), // .textContent = "Current: true"
                        "form-field-saveimg" : document.querySelector("#form-field-img"), // Dropdown
                        "conf-cur-savemodel" : document.querySelector(".conf-cur-savemodel div h6"), // .textContent = "Current: true"
                        "form-field-savemodel" : document.querySelector("#form-field-model"), // Dropdown
                        "conf-cur-advanced" : document.querySelector(".conf-cur-advanced div h6"), // .textContent = "Current: true"
                        "form-field-advanced" : document.querySelector("#form-field-advanced"), // Dropdown
                    "smn-advanced-config-box" : document.querySelector(".smn-advanced-config-box"), //.style.display = "Flex" or "None"
                    //Advanced config
                        "conf-cur-epoch" : document.querySelector(".conf-cur-epoch div h6"), //.textContent = "Current: 25"
                        "form-field-epoch" : document.querySelector("#form-field-epoch"), // Text Field
                        "conf-cur-selmodel" : document.querySelector(".conf-cur-selmodel div h6"), // .textContent = "Current: true"
                        "form-field-selmodel" : document.querySelector("#form-field-selmodel"), // Dropdown

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
                    "rmn-btn-transfer" : document.querySelector(".btn-transfer"), //Trigger click event
                    "rmn-btn-combine" : document.querySelector(".btn-combine"), //Trigger click event
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
                "name": "localhost:5000 (Use this, Prof.)",
                "address": "localhost",
                "port": "5000",
                "protocol": "http"
            },
            {
                "name": "Nicezki HTTPS (For Dev)",
                "address": "miri.network.nicezki.com",
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

    reshowElement(element_name) {
        this.ui_elements[element_name].style.display = "none";
        setTimeout(() => {
            this.ui_elements[element_name].style.display = "Flex";
        }, 100);
    }

    reshowElementByTime(element_name, time) {
        this.ui_elements[element_name].style.display = "none";
        setTimeout(() => {
            this.ui_elements[element_name].style.display = "Flex";
        }, time);
    }

    hideElementByTime(element_name, time) {
        this.ui_elements[element_name].style.display = "Flex";
        setTimeout(() => {
            this.ui_elements[element_name].style.display = "none";
        }, time);
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
        this.consoleLog("「AIMANE」 by Nattawut Manjai-araya  v2.0.0");
        // Init UI

        // Prepare server list
        this.createServerSelectionButton();

        // Add event listener to button
        this.buttonTriggerSetup();
        this.showConnectScreen();

        // Setup canvas
        this.setupDrawCanvas();
        this.logmane("Application started", "Ready..", 0, "server");
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
        this.ui_elements["btn-configtrain"].addEventListener("click", () => {
            this.toggleTrainConfig();
        });

        this.ui_elements["form-field-epoch"].addEventListener("change", () => {
            this.consoleLog("Epoch changed to " + this.ui_elements["form-field-epoch"].value);
            this.setTrainConfig("epochs", this.ui_elements["form-field-epoch"].value);
        });

        this.ui_elements["form-field-uc"].addEventListener("change", () => {
            this.consoleLog("UC changed to " + this.ui_elements["form-field-uc"].value);
            this.setTrainConfig("uc", this.ui_elements["form-field-uc"].value);
        });

        this.ui_elements["form-field-saveimg"].addEventListener("change", () => {
            this.consoleLog("Save image changed to " + this.ui_elements["form-field-saveimg"].value);
            this.setTrainConfig("img", this.ui_elements["form-field-saveimg"].value);
        });

        this.ui_elements["form-field-savemodel"].addEventListener("change", () => {
            this.consoleLog("Save model changed to " + this.ui_elements["form-field-savemodel"].value);
            this.setTrainConfig("savemodel", this.ui_elements["form-field-savemodel"].value);
        });

        this.ui_elements["form-field-facc"].addEventListener("change", () => {
            this.consoleLog("Finish Acc changed to " + this.ui_elements["form-field-facc"].value);
            this.setTrainConfig("facc", this.ui_elements["form-field-facc"].value);
        });

        this.ui_elements["form-field-selmodel"].addEventListener("change", () => {
            this.consoleLog("Selected model changed to " + this.ui_elements["form-field-selmodel"].value);
            this.setTrainConfig("model", this.ui_elements["form-field-selmodel"].value);
        });

        this.ui_elements["form-field-advanced"].addEventListener("change", () => {
            this.consoleLog("Advanced option changed to " + this.ui_elements["form-field-advanced"].value);
            this.setAdvancedConfig(this.ui_elements["form-field-advanced"].value);
        });

        this.ui_elements["rmn-btn-transfer"].addEventListener("click", () => {
            this.aimane_transferlearning();
        }
        );

        this.ui_elements["rmn-btn-combine"].addEventListener("click", () => {
            this.aimane_combine();
        }
        );
        


    }

    toggleTrainConfig() {
        if (this.ui_elements["smn-training-config-box"].style.display == "none") {
            this.ui_elements["smn-training-config-box"].style.display = "flex";
            this.setAdvancedConfig(this.ui_elements["form-field-advanced"].value);
        }
        else {
            this.ui_elements["smn-training-config-box"].style.display = "none";
            this.ui_elements["smn-advanced-config-box"].style.display = "none";
        }
    }

    setAdvancedConfig(status) {
        if (status == "true" || status == true) {
            this.showElement("smn-advanced-config-box");
            this.changeText("conf-cur-advanced", "Current: Show");
        }
        else {
            this.hideElement("smn-advanced-config-box");
            this.changeText("conf-cur-advanced", "Current: Hide");
        }
    }



    getTrainConfig() {
        // send GET api to {server}/api/setconfig
        fetch(this.serverURL + "/api/setconfig", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                response.json().then((data) => {
                    this.configCurrentShow(data);

                });
            }
            else {
                this.consoleLog("Get config request sent failed", "ERROR");
            }
        });
    }


    setTrainConfig(config,value) {
        // send GET api to {server}/api/setconfig
        fetch(this.serverURL + "/api/setconfig?"+config+"=" + value, {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                response.json().then((data) => {
                    this.configCurrentShow(data);
                    this.deleteAllEpochProgressBox();
                    this.createEpochProgressBox(data["epochs"]);
            });
            }
            else {
                this.consoleLog("Set config request sent failed", "ERROR");
            }
        });
    }


    configCurrentShow(data){
        this.ui_elements["conf-cur-epoch"].textContent = "Current: " + data["epochs"];
        this.ui_elements["form-field-epoch"].value = data["epochs"];
        // If finish acc is 0.0000 or 0 then showing Disable text instead of 0
        let cur_facc_text = data["stop_on_acc"] == 0.0000 || data["stop_on_acc"] == 0 ? "Disable" : data["stop_on_acc"];
        this.ui_elements["conf-cur-facc"].textContent = "Current: " + cur_facc_text;
        this.ui_elements["form-field-facc"].value = data["stop_on_acc"];
        this.ui_elements["conf-cur-uc"].textContent = "Current: " + data["usercontent"];
        this.ui_elements["form-field-uc"].value = data["usercontent"];
        this.ui_elements["conf-cur-saveimg"].textContent = "Current: " + data["save_image"];
        this.ui_elements["form-field-saveimg"].value = data["save_image"];

        // If saving image is disabled, disable the wrong button
        if (data["save_image"] == "false" ||  data["save_image"] == 0 || data["save_image"] == false) {
            this.hideElement("rmn-btn-wrong");
            this.hideElement("rmn-teachbox");
        }else{
            this.showElement("rmn-btn-wrong");
        }
        this.ui_elements["conf-cur-savemodel"].textContent = "Current: " + data["save_model"];
        this.ui_elements["form-field-savemodel"].value = data["save_model"];

        if (data["model"] == 0 || data["model"] == "0") {
            var cur_mode_text = "MNIST";
        }
        else if (data["model"] == 1 || data["model"] == "1") {
            var cur_mode_text = "Custom";
        }
        else if (data["model"] == 2 || data["model"] == "2") {
            var cur_mode_text = "Combine";
        }

        this.ui_elements["conf-cur-selmodel"].textContent = "Current: " + cur_mode_text;
        this.ui_elements["form-field-selmodel"].value = data["model"];
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
        fetch(this.serverURL + "/api/restouc", {
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


    aimane_transferlearning() {
        // send GET api to {server}/api/transferlearn
        fetch(this.serverURL + "/api/transferlearn", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                this.consoleLog("Transfer learning request sent successfully", "SUCCESS");
            }
            else {
                this.consoleLog("Transfer learning request sent failed", "ERROR");
            }
        });
    }

    aimane_combine() {
        // send GET api to {server}/api/combine
        fetch(this.serverURL + "/api/combine", {
            method: "GET",
            headers: {
                "Content-Type": "application/json"
            }
        }).then((response) => {
            if (response.status == 200) {
                this.consoleLog("Combine request sent successfully", "SUCCESS");
            }
            else {
                this.consoleLog("Combine request sent failed", "ERROR");
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
                // Hide rmn-teachbox
                this.hideElement("rmn-teachbox");
                // Change rmn-guess and rmn-number
                this.changeText("rmn-guess", "Now corrected to");
                // Mapping to this.predictresults.class_names[defclass]
                let result = this.predictresults.class_names[defclass];
                this.changeText("rmn-number", result);
            }
            else {
                this.consoleLog("Correct prediction to " + defclass + " request sent failed", "ERROR");
            }
        });
    }

    aimane_wrong() {
        //Show rmn-teachbox
        this.hideElementByTime("rmn-teachbox",5000);
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
        this.hideElement("box-log");
        this.on_selectserver = true;
    }
    
    showDisconnectScreen() {
        if (this.on_selectserver == false) {
            this.showElement("box-disconnect");
            this.hideElement("box-connect");
            this.hideElement("box-log");
        }
    }

    showMainScreen() {
        this.hideElement("box-connect");
        this.hideElement("box-disconnect");
        this.showElement("aimane-main");
        this.showElement("box-log");
        this.on_selectserver = false;
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
            // Mapping to this.predictresults.class_names[number]
            let result = this.predictresults.class_names[number];
            if(this.getText("rmn-number") != result) { this.changeText("rmn-number", result);}
        }
        let guessText = [
            "I think it's a ",
            "Hmm, I think it's a ",
            "Wow, it's a ",
            "Let me guess, it's a ",
            "From my experience, it's a ",
            "I'm pretty sure it's a ",
            "I'm guessing it's a ",
            "Maybe it's a ",
            "I think today's lotto is",
            "Nailed it! It's obviously a ",
            "No doubt about it, it's a ",
            "I've got this: It's a ",
            "Piece of cake! It's a ",
            "Oh, it's definitely a ",
            "Watch and learn, it's a ",
            "Easy peasy, it's a ",
            "My intuition says it's a ",
            "Locking it in: It's a ",
            "Voila! It's a ",
            "Hold your applause, it's a ",
            "This one's a no-brainer: It's a ",
            "Tada! It's a ",
            "I spy with my little eye, it's a ",
            "There you have it, it's a ",
            "Eureka! It's a ",
            "Count on me, it's a ",
            "Magic eight ball says it's a ",
            "It's written in the stars, it's a ",
            "Confidently stating, it's a ",
            "Cross my heart, it's a "
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
            "Teach me if I'm mistaken",
            "I stand corrected, please teach me",
            "Educate me if I err",
            "Help me learn if I'm off track",
            "Oh, my bad. Teach me!",
            "Show me the right path if I'm wrong",
            "I'm open to learning, correct me",
            "If I misspoke, please teach me",
            "In case I'm mistaken, teach me",
            "I welcome corrections, teach me",
            "I'm all ears, show me the right way",
            "If there's an error, teach me",
            "I'm a student, please correct me",
            "Point out my mistakes and teach me",
            "I'm still growing, educate me",
            "Don't hesitate to correct me",
            "I'm here to learn, show me",
            "Please guide me if I'm wrong",
            "Correct me if I'm mistaken",
            "I'm humble enough to learn, teach me",
            "Open to corrections, show me",
            "Teach me if I miss the mark",
            "I appreciate feedback, correct me",
            "If I'm mistaken, please teach me",
            "I'm learning, guide me",
            "If I'm off-base, teach me",
            "I'm willing to learn, show me",
            "I'm not perfect, correct me",
            "Please point out my errors and teach me",
            "I seek knowledge, educate me"
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
            "Your turn to jot down the number",
            "Put your number on paper",
            "Time to pen your number",
            "Let's see your written number",
            "Ink your chosen number",
            "Go ahead, write the number",
            "Waiting for your written number",
            "The stage is yours, write the number",
            "Write down your lucky number",
            "Write it up, your number is up",
            "Let's get your number on record",
            "Time to scribble your number",
            "Ready, set, write your number",
            "Don't be shy, write your number",
            "We need your written number",
            "Give us your number in writing",
            "Writing time, your number please",
            "Numbers on paper, write yours",
            "I'm all ears for your written number",
            "Write it in bold, your number",
            "Write your number boldly",
            "Waiting for your number on paper",
            "Your number, your writing",
            "It's showtime, write your number",
            "The moment has come, write your number",
            "Write your winning number",
            "Inscribe your number here"
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
            newdiv.className = "elementor-element elementor-element-d05252b elementor-align-center elementor-widget__width-initial btn-retrain elementor-widget elementor-widget-button btn-retrained btn-retrain-" + classes[i];
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
        // Delete all button with loop id btn-retrain-xxx
        var btnretrain = document.querySelectorAll(".btn-retrained");
        for (var i = 0; i < btnretrain.length; i++) {
            btnretrain[i].remove();
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
        this.source = await (async () => {
            if (this.source) {
                this.source.close();
            }
            return new EventSource(sse_url);
        })();
        // Check SSE connection
        this.source.onopen = (e) => {
            console.log("[INFO] SSE connected to " + sse_url);
            this.connected = true;
            this.getTrainConfig();
        };
        this.source.onerror = (e) => {
            console.log("[ERROR] SSE connection error");
            this.connected = false;
            this.disconnect();
            this.showDisconnectScreen();
            this.consoleLog("Connection to server at " + this.serverAddress + ":" + this.serverPort + " failed", "ERROR");
            this.source.close();
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


    epochProgressBox(epoch) {
        let epochProgressBox = document.querySelector(".ep-" + epoch);
        return epochProgressBox;
    }

    epochProgressBoxTextContent(epoch) {
        let epochProgressBoxText = document.querySelector(".ep-" + epoch + " div div h4").textContent;
        return epochProgressBoxText;
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

                // epoch is number so convert it to string before using it
                if (this.epochProgressBoxTextContent(epoch) != "" + epoch) {
                    if(this.epochProgressBoxTextContent(epoch) == ""+rvalue){
                        this.consoleLog("[Mane Detect Debug] Epoch " + epoch + " seem to already have OK data", "SUCESS");
                        continue;
                    }

                    this.consoleLog("[Mane Detect] Epoch " + epoch + " seem to already have data in it. Trying to resetting it...", "WARN");
                    this.deleteAllEpochProgressBox();
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
            this.createCorrectPredictionButton(data.class_names);
        }

        if (this.lastmodel != data.model && this.correctPredictionButtonCreated == true) {
            this.deleteCorrectPredictionButton();
            this.createCorrectPredictionButton(data.class_names);
        }

    }

}


if (window.location.href.indexOf("elementor-preview") > -1) {
    console.log("[INFO] AIMane not loaded in this page because this is development environment");
}else{
    console.log("URL: "+window.location.href);
    const aiManeUI = new AIManeUI();
}