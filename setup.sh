#!/bin/bash

#crea el entorno e instala los requerimientos 

# Crea un entorno virtual
python -m venv venv

# Activa el entorno virtual
source venv/bin/activate

# Instala los requisitos
pip install -r requirements.txt

# Desactiva el entorno
deactivate
# ejecutar con bash setup.sh
