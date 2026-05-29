# MITEX Steam

Steam es una plataforma de distribución de vidiojuegos que cuenta con una API para obtener información sobre juegos. En este proyecto se obtiene información para usar en un rag para que un modelo de lenguaje pueda responder preguntas sobre los juegos.

Usamos Steam porque tiene mucha información disponible gracias a la ideología de Valve y [Gabe Newell](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/106d800f-15b4-44d4-aa16-972239963d2d/d4rnffi-c828868d-089e-4fd5-b38b-33c6c6b6f8da.png/v1/fill/w_800,h_1107,q_80,strp/gabe_newell_portrait_by_freddre_d4rnffi-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTEwNyIsInBhdGgiOiIvZi8xMDZkODAwZi0xNWI0LTQ0ZDQtYWExNi05NzIyMzk5NjNkMmQvZDRybmZmaS1jODI4ODY4ZC0wODllLTRmZDUtYjM4Yi0zM2M2YzZiNmY4ZGEucG5nIiwid2lkdGgiOiI8PTgwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.pG6EORYPxTTlYSLRZ7SHHuHJtjbSuGNjMj1E-NONigk).

## Elegir juegos

Para elegir los videojuegos se usa el enlace <https://store.steampowered.com/search/results/?filter=globaltopsellers&hidef2p=1&json=1>. Iterando por 4 páginas se obtienen los 100 juegos más vendidos no gratuítos en Steam. Con esta información se guarda en el fichero [`videogames.json`](../videogames.json) el nombre del juego y su id en Steam.

Pero si se abre el fichero se ve que hay más de 100 juegos. Esto se debe a que cada día aparecen nuevos juegos entre los top 100 mientras que desaparecen otros. Para mantener nuestra lista actualizada con los nuevos y viejos mejores videojuegos, se ejecuta una acción de GitHub que actualiza el fichero cada día (`0 0 * * *`) y en cada actualización se añaden los nuevos juegos a la lista.

Otro fichero que se crea es [`topSellers.csv`](../topSellers.csv) que contiene en orden el id para los cien juegos de cada día. Esto no se va a usar en el proyecto pero se puede ver el momento que se lanzan nuevos juegos, se anuncian, sale una expansión o se rebajan. También se puede ver que cuando se lanza/anuncia un juego de una franquicia, el resto de juegos de la franquicia tienen más ventas.

## Obtener información de los juegos

Para cada juego queremos obtener su información general y sus reseñas. Los guardamos en la carpeta [`rawData`](../rawData/).

### Información general

Para esto usamos la API de Steam con el endpoint <https://store.steampowered.com/api/appdetails?appids=70>. En este caso el id del juego es 70, que corresponde a [`Half-Life`](https://store.steampowered.com/app/70/HalfLife/). Con un request simple ya obtenemos toda la información en formato JSON. Se guarda como el nombre del juego más `Info.json`. En el caso de Half-Life se guarda como `Half-LifeInfo.json`. Básicamente, se trata de toda la información que cualquier usuario puede ver en la página del juego en Steam.

### Reseñas

Para las reseñas cogemos las 100 primeras positivas y las 100 negativas. Para las positivas se usa <https://store.steampowered.com/appreviews/70?json=1&filter=all&language=english&num_per_page=100&review_type=positive> y para las negativas <https://store.steampowered.com/appreviews/70?json=1&filter=all&language=english&num_per_page=100&review_type=negative>.

La documentación de esta url es esta [User Reviews - Get List](https://partner.steamgames.com/doc/store/getreviews). Usamos los parámetros `json=1` para obtener la respuesta en formato JSON, `filter=all` para obtener las reseñas ordenadas por su utilidad, `language=english` para obtener solo reseñas en inglés, `num_per_page=100` para obtener 100 reseñas por página y `review_type` para especificar si queremos reseñas positivas o negativas. Lo importante es saber que puede no devolver ninguna reseña, por ejemplo si el juego es muy nuevo o no tiene suficientes reseñas en inglés. El idioma puede fallar y devolvernos reseñas en otros idiomas que solucionaremos en la limpieza de datos.

Las reseñas se guardan con el nombre del juego más `Positive.jsonl` para las positivas y `Negative.jsonl` para las negativas. Usamos `.jsonl` porque al ser cada línea un JSON, sería más fácil de procesar después si hay problemas en una única reseña. Aunque esto nunca pasó porque todo está bien formateado. De cada reseña guardamos la utilidad que le da STEAM y el texto.

## Limpieza de datos

En esta parte del pipeline se limpian los datos obtenidos en la parte anterior. Se guardan en la carpeta [`cleanData`](../cleanData/). Para cada juego, se crea un nuevo fichero con el nombre del juego más `Info.json`que contiene la información general del juego que se va a usar en el rag. Elimina datos redundantes y solo se queda con 6 campos clave: name, description, platforms (Windows, Linux o Mac), pegi, genres y price.

En cuanto a las reseñas, el script abre un `ProcessPoolExecutor` para poder procesarlas en paralelo con varios procesadores de la cpu. Para cada reseña, se eliminan las que no están en inglés, se eliminan los saltos de línea y se eliminan las reseñas que no tienen texto. Se guardan en un nuevo fichero con el nombre del juego más `Negative.jsonl` o `Positive.jsonl`. Si en el archivo no queda reseña en inglés, se borra.

Además, con el fin de evitar repetir todo el proceso al tener nuevos juegos, se guarda un fichero llamado `languageLog.json` que contiene el número de reseñas que no están en inglés para cada juego. Esto para poder recuperar el conteo de idiomas que ya se había calculado en ejecuciones anteriores.

## Modelado de temas

En esta fase del pipeline agrupamos las reseñas limpias en diferentes tópicos utilizando **BERTopic** para descubrir cuáles son los temas de conversación dominantes entre los jugadores (por ejemplo, problemas de optimización, calidad de la historia, microtransacciones, etc.). Los resultados se guardan en la carpeta [`topicsData`](../topicsData/).

### Extracción de tópicos y clustering

El script lee todos los documentos disponibles en la carpeta `cleanData`. Para evitar que palabras genéricas del ámbito de los videojuegos contaminen los resultados, se aplica un filtro exhaustivo que combina las _stopwords_ en inglés de `NLTK` junto con una lista personalizada de términos comunes (como _game_, _gameplay_, _play_, _fun_ o _story_).

El proceso de modelado sigue los siguientes pasos secuenciales:

1. **Reducción de dimensionalidad:** Se utiliza `UMAP` para proyectar los vectores de las reseñas en un espacio de 5 dimensiones basándose en la similitud del coseno.
2. **Clustering:** El algoritmo `HDBSCAN` identifica los clústeres de reseñas compartidas, exigiendo un tamaño mínimo de 80 reseñas por grupo.
3. **Optimización de palabras clave:** Se aplica `MaximalMarginalRelevance (MMR)` con una diversidad del 0.3 para asegurar que las palabras que describen cada tema no sean redundantes entre sí.
4. **Reducción de Outliers:** Aquellas reseñas marcadas inicialmente como ruido (clúster `-1`) se reasignan a los temas principales mediante una estrategia basada en `c-TF-IDF`, recalculando después las palabras clave para mantener la precisión. Siempre se mantiene un umbral de 0.05 para evitar asignaciones forzadas de reseñas que realmente no encajan en ningún tema.

Durante la ejecución, se calcula y muestra la **coherencia del modelo ($C_v$)** utilizando `Gensim` para evaluar matemáticamente la calidad semántica de los grupos generados. Esto se hace antes del paso de reducción de outliers y después para mostrar la mejora obtenida. Además, se muestra el porcentaje de reseñas que quedan como outliers antes y después de la reasignación. Este es el resultado:

```Tópicos: 24 | Outliers: 41.22% | Coherencia C_v: 0.4921
Tópicos: 24 | Outliers: 17.04% | Coherencia C_v: 0.5990
```

### Enriquecimiento con Modelos de Lenguaje (LLM)

Una vez definidos los clústeres con sus palabras clave y guardados en el archivo [`videogameClusters.json`](../topicsData/videogameClusters.json), el script interactúa con un modelo de lenguaje local gestionado por `LLMManager`.

El LLM analiza de forma automática las palabras clave de cada clúster y genera una estructura JSON formateada mediante restricciones de tokens (`TokenSequenceConstraint`). De esta manera, se añade a cada tópico:

- **`name_topic`:** Un título corto y descriptivo (de 3 a 5 palabras) que resume la temática.
- **`description`:** Una frase explicativa orientada a lo que este tema significa desde la perspectiva de un jugador o desarrollador.

### Estructuración de datos para el RAG

Finalmente, el pipeline transforma la estructura orientada a clústeres en una vista centrada en los videojuegos mediante la función `create_game_centric_json`. El resultado se almacena en el archivo [`gameTopics.json`](../topicsData/gameTopics.json).

Este archivo final organiza la información en dos grandes bloques: una lista detallada de los tópicos enriquecidos por el LLM y un mapeo por cada videojuego, indicando en qué porcentaje y volumen aparecen dichos temas en sus reseñas de Steam (filtrando únicamente aquellos juegos que representen al menos el 2.0% del total del clúster). Esto permite al RAG cruzar de manera eficiente los metadatos generales del juego con el contexto semántico de la opinión de su comunidad.

### Resultados del modelado de temas

Nos encontramos con 24 tópicos que abarcan diferentes aspectos de los videojuegos y que permiten clasificar los videojuegos yendo más allá de su género o descripción general. Por ejemplo, el tópico 16 se centra en la mecánica de combate, con palabras clave como "parry", "attack", "dodge" o "boss". Este tópico podría incluir juegos que tienen un sistema de combate desafiante y que requieren habilidades de parry y dodge para superar a los enemigos. También nos permite conocer aquellos juegos que tiene problemas de optimización, como el tópico 5 que se centra en la estabilidad del juego y la integridad de los archivos, con palabras clave como "crash", "unplayable", "file", "error" o "loading screen". Este tópico podría incluir juegos que tienen problemas frecuentes de estabilidad, como caídas del juego, corrupción de archivos o errores en las pantallas de carga.

Sin embargo, el cluster 0 almacena una gran parte de las reseñas, lo que indica que es un cluster genérico que engloba muchos comentarios. Por otra parte, tenemos cluster mucho más específicos con muy pocas reseñas.

## Resumen extractivo de reseñas

El pipeline incluye una fase de resumen extractivo basado en grafos que selecciona las 5 frases más representativas de las reseñas positivas y negativas. Los resúmenes resultantes se almacenan en la carpeta [`summaryData`](../summaryData/) y sirven para que el RAG conozca la opinión pública de un juego sin saturar la ventana de contexto del modelo de lenguaje con miles de tokens repetitivos.

Primero, el tokenizador de `NLTK` divide las reseñas en frases individuales, que luego se convierten en vectores densos mediante un modelo de embeddings (`models.EMBEDDING_MODEL`). Utilizando `NetworkX`, se construye un grafo no dirigido donde los nodos son las frases y las aristas representan la similitud del coseno entre ellas, descartando con un umbral de `0.3` cualquier conexión débil o ruidosa.

Para que el resumen priorice las opiniones más valiosas, el algoritmo ejecuta un PageRank Personalizado utilizando las puntuaciones de utilidad (_scores_) de Steam. Además, para evitar que las reseñas largas dominen el grafo por el simple hecho de tener más texto, se aplica una amortiguación logarítmica dividiendo la puntuación de cada frase entre uno más el logaritmo del total de frases de su review. Así, el sistema equilibra de forma justa el peso de los análisis cortos y los detallados, seleccionando las 5 frases con mayor centralidad y guardándolas en archivos como `Half-LifePositive.json`.
