# Transformaciones

Las transformaciones 2D implementadas en esta práctica son:

- Translación
- Rotación
- Euclídea
- Similitud
- Afín
- Proyectiva

## Parámetros utilizados

- Traslación: dos parámetros, siendo éstos dos valores para la traslación en ambos ejes (en
  píxeles). 100 px en el eje y, con 0 px en el eje x.
- Rotación: un único parámetro utilizado; el ángulo de rotación en radianes, equivalente a 45
  grados.
- Euclidea: tres parámetros, combinando los dos previos en una sola transformación.
- Similitud: cuatro parámetros, combinando los dos previos en una sola transformación, más la
  constante de escalado que 0.75; haciendo más pequeña la imagen.
- Afín: seis parámetros, los valores de traslación y los valores de la matriz de rotación de
  forma arbitraria.
- Proyectiva: ocho parámetros, cobinando los valores de la transformación afín más dos valores
  más. Esos valores se varió sólo uno para generar una perspectiva.

## Transformación Inversa

Al aplicar la transformación, se aplicó la transformación inversa ya que ayuda con la
interpolación de valores en la imagen resultante. Esto evita que tengamos "agujeros" dentro de la
imagen, en transformaciones como la euclidea.
