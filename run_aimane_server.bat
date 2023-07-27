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

rem If 3.10.x use requirements.txt
rem If 3.11.x or newer use requirements_3_11.txt
python -c "import sys; exit(1 if sys.version_info < (3, 11) else 0)" && (
    echo: Detected Python 3.11.x or newer, using requirements_3_11.txt
    set requirements=requirements_3_11.txt
) || (
    echo: Detected Python 3.10.x, using requirements.txt
    set requirements=requirements.txt
)
   


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
    call venv\Scripts\activate
    call python server.py
)

@pause 



