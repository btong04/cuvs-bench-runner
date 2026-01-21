# Installation Instructions
Navigate to fast storage disk (e.g., `/raid` on local machine). Clone git repository:
```
git clone https://github.com/btong04/cuvs-bench-runner.git
cd ./cuvs-bench-runner
```

Create a pip virtual environment and install required packages.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For subsequent runs, you can navigate to the script directory and run `source venv/bin/activate`. Launch jupyter lab to start an interactive notebook sessions: 
```
jupyter lab
```
