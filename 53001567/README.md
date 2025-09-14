# Control Óptimo y Adaptativo

Este curso contiene implementaciones de los conceptos y técnicas de control óptimo y adaptativo, utilizando herramientas de programación y simulación como OpenModelica.

## Instalación de OpenModelica

Sólo haré hincapié en la instalación de OpenModelica para macOS.
Windows y Linux son sistemas operativos donde la instalación es mucho más sencilla.

### Prerrequisitos

- Instalar XQuartz.
- Instalar Docker Desktop.
- Instalar `socat` con Homebrew.

### Imagen de Docker

La mejor forma de obtener la imagen de Docker es a través de la página oficial de OpenModelica.
Asegurarse de tener la última versión de la imagen, y ejecutar el siguiente comando en la terminal:

```sh
docker pull openmodelica/openmodelica:<version>-gui
```

> [!NOTE]
> Revisar [Docker Hub](https://hub.docker.com/r/openmodelica/openmodelica/tags) para las versiones disponibles.

### Iniciar OpenModelica GUI

Asumiendo que la instalación de XQuartz es exitosa, y la imagen se encuentra en el registry local, procedemos a ejecutar
el siguiente script de bash que permite iniciar OpenModelica.

```sh
source ./openmodelica-setup.sh && openmodelica-gui
```

> [!WARNING]
> En caso de que no se encuentre la imagen de Docker, asegurarse que el script haga mención a la versión que se haya
> descargado.
