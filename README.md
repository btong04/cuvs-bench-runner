# Navigate to folder with scripts
cd /raid/cuvs-bench-runner

# Create pip virtual environment and install required packages.
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For subsequent runs, you can navigate to the script directory and run "source venv/bin/activate".

# Launch jupyter lab with root priveleges
jupyter lab --allow-root
