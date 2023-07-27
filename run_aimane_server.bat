@echo off

rem detecting python ver if not exist return error and close
rem If it's not 3.10.x or newer
python -c "import sys; exit(1 if sys.version_info < (3, 10) else 0)" || (
    echo :"Professor, Python 3.10.x or newer is required to run this server."
    echo :"Please install it from https://www.python.org/downloads/"
    pause
    exit
)

echo: Good!!, you have Python 3.10.x or newer
   


rem If venv directory doesn't exist
if not exist venv (
    echo: Virtual environment doesn't exist, creating one...
    python -m venv venv
    echo: Activating virtual environment...
    call venv\Scripts\activate
    echo: Installing requirements...
    call pip install -r requirements.txt
    echo: Installed requirements
)
if exist venv (
    echo: Noice~! Virtual environment exists, activating...
    call venv\Scripts\activate
    call python server.py
)

@pause 



