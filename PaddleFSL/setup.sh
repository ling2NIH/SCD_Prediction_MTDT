conda create -n paddleFSL1 python=3.7 -y
source activate paddleFSL1
pip install paddlepaddle-gpu==2.4.1
pip install rdkit-pypi
pip install pgl
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install paddlehelix
pip install paddlefsl
pip install scikit-learn==1.0.2
pip install ujson
pip install paddlenlp==2.0.6
pip install transformers==3.3.1
pip install Pillow==8.2.0