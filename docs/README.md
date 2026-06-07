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

Para las reseñas se guarda un fichero llamado `languageLog.json` que contiene el número de reseñas que no están en inglés para cada juego. Se va actualizando cada vez que se añaden nuevos juegos o se vuelven a procesar los juegos existentes. Podemos ver que en la mayoría de los juegos, Steam filtra por idioma correctamente sin muchos errores.

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

## Documentos para el RAG

En esta etapa, el pipeline unifica toda la información procesada para construir el motor de búsqueda que alimentará al RAG. La función addDocsToCollection se encarga de consolidar los datos de cada videojuego, combinando la información general de `cleanData`, los resúmenes de opiniones de `summaryData` y los porcentajes de temas de `gameTopics.json`. Con este texto estructurado se genera un documento final que se convierte en vector mediante nuestro modelo de embeddings y se almacena en una colección de ChromaDB. Para identificar cada juego de forma única y evitar duplicados, se genera un identificador aplicando un hash SHA256 sobre su nombre.

Por cada documento, se crea un único bloque de texto formateado de la siguiente manera:

- Título: El nombre del videojuego.

- Géneros: Los géneros a los que pertenece el juego.

- Descripción: La descripción general del juego proveniente de STEAM.

- Resumen de reseñas positivas: Las 5 frases más representativas extraídas de las opiniones positivas (o un texto por defecto si no hay).

- Resumen de reseñas negativas: Las 5 frases más representativas extraídas de las opiniones negativas (o un texto por defecto si no hay).

- Temas asociados: Una lista con los tópicos detectados por BERTopic extraído de las reseñas, detallando el título del tema, su descripción y el porcentaje de relevancia que tiene dentro de las reseñas de ese juego concreto.

Además, se añaden metadatos adicionales a cada documento, como el precio, las plataformas disponibles y la clasificación PEGI, para enriquecer aún más la información que el RAG puede utilizar para responder preguntas específicas sobre cada videojuego.

### Búsqueda Híbrida

Para recuperar la información de la manera más precisa posible, implementamos una estrategia de búsqueda híbrida mediante la función rrf. Este método combina los puntos fuertes de la búsqueda léxica y la semántica a través del algoritmo Reciprocal Rank Fusion. Primero, el script calcula las coincidencias exactas de términos sobre los documentos tokenizados usando un modelo BM25Okapi para obtener los 50 mejores resultados. En paralelo, se genera el embedding de la consulta y se interroga a ChromaDB para extraer los 50 resultados con mayor similitud conceptual. Ambas listas se fusionan otorgando a cada juego una puntuación basada en su posición de rango dentro de cada búsqueda, utilizando la fórmula `score = 1/(k + rank)`, donde `k` es un factor de amortiguación (en este caso, 60) que equilibra la influencia de ambos métodos. Finalmente, se ordenan los resultados combinados por su puntuación total y se devuelven los 3 juegos más relevantes.

## Arquitectura Multi Agente

Para coordinar todo el flujo desde que el usuario introduce una consulta hasta que recibe la respuesta final, implementamos una clase Orchestrator que gestiona un sistema multi-agente basado en nodos secuenciales y guarda un registro completo de cada ejecución en [`completeExecutionLog.json`](../completeExecutionLog.json). En este json se pueden ver los outputs de cada agente, los documentos recuperados, las consultas a los LLMs y las respuestas generadas. Esto es fundamental para entender el proceso completo y detectar posibles errores o áreas de mejora.

### Nodo 1: Query Understanding

El primer paso del pipeline consiste en determinar si la consulta del usuario requiere extraer contexto de nuestra base de datos vectorial o no. El Nodo 1 actúa como el cerebro clasificador del sistema utilizando un prompt del sistema estricto. Su único objetivo es analizar la lista de mensajes y decidir entre varias acciones posibles:

- `rag`, si la duda está relacionada con videojuegos y necesita el contexto de ChromaDB.

- `nothing`, si es una consulta genérica o ajena al dataset.

- `search`, si se puede buscar por el nombre del juego sin necesidad de usar el contexto de ChromaDB.

Como el usuario final puede responder y seguir hablando, este nodo tiene su prompt de sistema seguido con toda la conversación previa (incluyendo los documentos recuperados en iteraciones anteriores) para que el modelo tenga toda la información necesaria para tomar una decisión informada.

Para garantizar que este nodo no rompa el flujo de ejecución, forzamos al modelo a responder exclusivamente en un formato JSON estructurado mediante una restricción de tokens (`TokenSequenceConstraint`). El JSON resultante contiene obligatoriamente un campo `Thinking` con el razonamiento del modelo y un campo `Action` restringido únicamente a las dos opciones válidas.

- Si se elige `rag`, se va al [Nodo 3](#nodo-3-recuperación-de-documentos).

- Si se elige `nothing`, se va al [Nodo 2](#nodo-2-recuperación-rag-y-respuesta).

- Si se elige `search`, se va al [Nodo 4](#nodo-4-extracción-del-nombre-del-juego).

### Nodo 2: Recuperación RAG y Respuesta

La consulta y los documentos recuperados (en caso de que los haya) se envían al Nodo 2. Este último componente actúa como el asistente final, mantiene la conversación hasta el momento y añade la consulta del usuario con sus documentos. Utiliza un decodificador de muestreo tradicional (SamplingDecoder) y genera la respuesta que se devuelve al usuario.

Si el usuario responde algo distinto de `:q`, se vuelve al [Nodo 1](#nodo-1-query-understanding).

### Nodo 3: Recuperación de documentos

Invoca la función de búsqueda híbrida `rag.rrf` para recuperar los 3 documentos más relevantes de la colección y se va al [Nodo 2](#nodo-2-recuperación-rag-y-respuesta).

### Nodo 4: Extracción del nombre del juego

Este nodo solo analiza la conversación hasta el momento y extraer el nombre del videojuego adecuado. Para ello utiliza un LLM con una salida JSON estricta con la forma:

`{"Game": "[Name of the game]"}`

Al igual que en el Nodo 1, se aplica `TokenSequenceConstraint` para forzar la estructura y evitar respuestas fuera de formato. El resultado se registra en el log de ejecución y se pasa directamente al [Nodo 5](#nodo-5-recuperación-del-documento-del-videojuego).

### Nodo 5: Recuperación del documento del videojuego

Este nodo recibe el nombre extraído por el Nodo 4 y es el responsable de recuperar el documento final del juego.

1. Primero intenta encontrar una coincidencia aproximada en [`videogames.json`](../videogames.json) usando `difflib.get_close_matches` con `cutoff=0.6`.
2. Si encuentra coincidencia, genera el documento con `rag.generateDocument` y lo devuelve.
3. Si no encuentra coincidencia:
   1. Hace una búsqueda web en Steam (`https://store.steampowered.com/search/results/`) con `BeautifulSoup`.
   2. Detecta juegos nuevos, actualiza [`videogames.json`](../videogames.json).
   3. Ejecuta `updateData()` para incorporar esos nuevos juegos al pipeline completo (limpieza, tópicos, resumen extractivo y colección RAG).
   4. Vuelve a intentar la coincidencia sobre los juegos nuevos y, si encuentra uno válido, genera su documento.

Finalmente, el documento recuperado se envía al [Nodo 2](#nodo-2-recuperación-rag-y-respuesta), que genera la respuesta final para el usuario. Si no se encuentra, se devuelve una lista vacía.

### Nodo Sintético

Este es un nodo adicional que se activa solo si `useSynthetic` es `True`. Su función es simular una posible respuesta del usuario después de recibir la respuesta del Nodo 2. Esto permite probar el sistema de manera autónoma sin necesidad de interacción humana constante, generando nuevas consultas basadas en la conversación previa. El prompt del sistema instruye al modelo a ser creativo pero relevante, y a finalizar la simulación con `:q` para indicar que no hay más preguntas.
