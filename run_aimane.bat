@echo off

SET "ESC_CHAR=" & FOR /F %%A IN ('echo prompt $E ^| cmd') DO SET "ESC_CHAR=%%A"
SET "GREEN=%ESC_CHAR%[92m"
SET "YELLOW=%ESC_CHAR%[93m"
SET "RED=%ESC_CHAR%[91m"
SET "NC=%ESC_CHAR%[0m"


call python -c "import sys; exit(1 if sys.version_info < (3, 11) else 0)" && (
    set requirements=requirements_3_11.txt
    set python_text = Detected Python 3.11.x or newer, using requirements_3_11.txt
) || (
    set requirements=requirements.txt
    set python_text = Detected Python 3.10.x, using requirements.txt
)



if "%1"=="" (
    
    
    rem detecting python ver if not exist return error and close
    rem If it's not 3.10.x or newer
    python -c "import sys; exit(1 if sys.version_info < (3, 10) else 0)" || (
        echo :"Professor, Python 3.10.x or newer is required to run this server."
        echo :"Please install it from https://www.python.org/downloads/"
        pause
        exit
    )
    
    echo: Good!!, you have Python 3.10.x or newer
    
    rem If 3.10.x use requirements.txt
    rem If 3.11.x or newer use requirements_3_11.txt
    echo: %python_text%

    
    rem If venv directory doesn't exist
    if not exist venv (
        echo: Virtual environment doesn't exist, creating one...
        echo: Updating pip...
        call python -m pip install --user --upgrade pip 
        call python -m venv venv
        echo: Activating virtual environment...
        call venv\Scripts\activate
        echo: Installing requirements...
        call pip install -r %requirements%
        echo: Installed requirements
    )
    if exist venv (
        echo: Noice~! Virtual environment exists, activating...
        echo: 
        start /b run_aimane.bat server-seemless
        start /b run_aimane.bat client-seemless
        start http://localhost:8080

        echo %GREEN%This is AIMANE Server and Client%NC%
        echo %YELLOW%by Nattawut Manjai-araya%NC%
        echo %RED%PLEASE DO NOT CLOSE THIS WINDOW!!%NC%
        echo %RED%Closing this window will stop the server and client [User Interface] from running%NC%
        echo %GREEN% You can access the server at http://localhost:8080  [recommended]%NC%
        echo %GREEN% or on the server alternatively at http://localhost:5000/app/v1/%NC%
        rem echo "Everything is done, you can close this window now"
        rem timeout /t 10 /nobreak > NUL
    )

)


if /i "%1"=="client" (
    echo %GREEN%This is AIMANE Client%NC%
    echo %YELLOW%by Nattawut Manjai-araya%NC%
    echo %RED%PLEASE DO NOT CLOSE THIS WINDOW!!%NC%
    echo %RED%Closing this window will stop the client [Interface] from running%NC%

    call venv\Scripts\activate
    echo Running client...
    echo: 
    start /b waitress-serve --listen=*:8080 client:client_app
    start http://localhost:8080

)

if /i "%1"=="server" (
    
    echo %GREEN%This is AIMANE Server%NC%
    echo %YELLOW%by Nattawut Manjai-araya%NC%
    echo %RED%PLEASE DO NOT CLOSE THIS WINDOW!!%NC%
    echo %RED%Closing this window will stop the server from running%NC%

    call venv\Scripts\activate
    echo Running server...
    echo: 
    start /b waitress-serve --listen=*:5000 server:server_app
    start http://localhost:5000/app/v1/
    
)

if /i "%1"=="client-seemless" (
    call venv\Scripts\activate
    echo "Running client..."
    start /b waitress-serve --listen=*:8080 client:client_app
    
)

if /i "%1"=="server-seemless" (
    call venv\Scripts\activate
    echo Running server...
    start /b waitress-serve --listen=*:5000 server:server_app
)



if /i "%1"=="stop" (
    echo Stopping server...
    taskkill /f /im python.exe
    taskkill /f /im pythonw.exe
    @pause
)




