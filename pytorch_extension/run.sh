# Uninstall the 'sparse-mm' package using pip
pip uninstall sparse-mm -y

# Remove the 'build' directory
rm -rf build

# Remove the 'dist' directory
rm -rf dist

# Remove the 'sparse_mm.egg-info' directory
rm -rf sparse_mm.egg-info

# Install the package using the 'setup.py' script
python setup.py install