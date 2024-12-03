# Este repositorio contiene un algoritmo de Python para generar videos a partir de un texto dado

*Descripción del algoritmo*

El algoritmo se utiliza para generar un video basado en el texto proporcionado. El modelo está pre-entrenado para generar videos de alta calidad con una variedad de estilos y contenido.

## Pre-requisitos

• Python 3.7 o superior
• Librerías de Python:
    • Transformers
    • diffusers
    • torch

## Instalación

bash```
pip install -r requirements.txt
```

## Configuración (config.json)

El archivo config.json contiene la configuración del proceso de generación de video. Se pueden modificar los parámetros de la configuración de acuerdo a las necesidades del usuario.

Parámetros:

• name_model: Nombre del modelo de generación de video.
• settings: Lista de configuraciones para la generación de video.
    • num_videos_per_prompt: Número de videos a generar por prompt.
    • num_inference_steps: Número de pasos de inferencia para generar el video.
    • num_frames: Número de frames del video.
    • guidance_scale: Escala de orientación para la generación del video.
    • fps: Frecuencia de frames por segundo del video.
• prompt: Texto que describe el video a generar.

## Ejecución

1. Asegúrate de que el archivo config.json esté configurado correctamente.
2. Ejecuta el train.py para que se realice el entrenamiento
3. Ejecute el main.py para generar el video

## Ejemplo

Para generar un video de un perro jugando en una ciudad, el archivo config.json podría contener la siguiente configuración:

json```
{
  "name_model":"THUDM/CogVideoX-2b",
  "settings":[{
    "num_videos_per_prompt":1,
    "num_inference_steps":50,
    "num_frames":49,
      "guidance_scale":6,
      "fps":8
  }],
  "prompt":"Un perro jugando en una ciudad"
}
```

El script de generación creará un video de 49 frames, con una frecuencia de 8 fps.

Notas

• El rendimiento del modelo y la calidad del video generado dependen de la configuración del proceso de generación.
• Se recomienda ajustar los parámetros de la configuración para obtener el mejor resultado.
• La generación de videos requiere una gran cantidad de recursos computacionales.
Licencia