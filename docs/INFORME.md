## DISEÑO Y ARQUITECTURA DEL SISTEMA

El diseño de este sistema nace de la necesidad de resolver el problema asociado a construir un chatbot con un RAG que sea capaz de responder preguntas complejas sobre un catálogo de videojuegos enorme y cambiante como el de Steam sin saturar la ventana de contexto de los modelos de lenguaje ni incurrir en costes de computación prohibitivos. La arquitectura propuesta aborda este desafío mediante una división estructural clara y unificada en dos fases: un pipeline de procesamiento o fase offline, dedicado a la construcción y enriquecimiento de la base de conocimiento, y una fase online, encargada de la resolución de consultas en tiempo real. Esta bifurcación técnica garantiza que el sistema disponga de información estructurada y de alta densidad semántica antes de que el usuario interactúe con la interfaz.

En la fase offline, el flujo de información avanza de manera secuencial. El proceso se inicia con la captura automatizada de datos comerciales y técnicos de la plataforma Steam, los cuales se someten a un proceso de limpieza profunda para erradicar el ruido. La información depurada sufre dos procesamientos: por un lado, se ejecuta un modelado de tópicos global que identifica las corrientes de opinión y las temáticas latentes de la comunidad y, por otro lado, se aplica un algoritmo de resumen extractivo basado en grafos de utilidad para sintetizar las reseñas. Ambos flujos convergen en la estructuración de los Documentos RAG, que consolidan la ficha técnica de cada videojuego, inyectándose finalmente en una base de datos vectorial.

Esta base de conocimiento es la que permite el correcto funcionamiento de la fase online, la cual está gobernada por una infraestructura multi-agente coordinada por una clase centralizadora denominada Organizador u Orchestrator. Cuando el usuario realiza una consulta, este componente orquesta el flujo a través de cinco nodos secuenciales. El Nodo Uno actúa como el cerebro clasificador, analizando el historial de la conversación para determinar si la consulta del usuario requiere recuperar un documento de la base vectorial, una búsqueda léxica directa por el nombre del título del videojuego, o una interacción genérica para la que no precisa información extra. Si la consulta es directa sobre un título, se activa el Nodo Cuatro para extraer el nombre exacto del videojuego bajo restricciones estrictas de salida. Posteriormente, el Nodo Cinco gestiona la recuperación del documento del videojuego aplicando algoritmos de emparejamiento difuso y, en caso de no hallar el título en el registro local, ejecuta de manera autónoma un proceso de extracción web en caliente, forzando la actualización inmediata de la base de datos a través del pipeline offline. Si la duda requiere un análisis profundo, el flujo se redirige al Nodo Tres para ejecutar la recuperación híbrida de documentos en la base vectorial. Toda la información recuperada por cualquiera de las vías converge finalmente en el Nodo De Generación o Nodo Dos, que procesa el contexto depurado y genera la respuesta definitiva, cerrando un ciclo, modular y adaptativo.

## IMPLEMENTACIÓN Y ROBUSTEZ TÉCNICA DEL PIPELINE

### Obtención de los datos

La robustez técnica de la implementación se sostiene sobre un pipeline de procesamiento de datos dividido en módulos independientes donde cada uno resuelve una problemática específica del ciclo de vida de la información. En la fase inicial de obtención de datos, el problema principal radica en la volatilidad del mercado de videojuegos, donde los títulos más vendidos cambian diariamente. La solución implementada consiste en un script automatizado mediante una acción de GitHub que se ejecuta diariamente a medianoche para iterar sobre las listas de los más vendidos de Steam, almacenando los datos en un registro estructurado que crece de manera incremental. El resultado son dos archivos, uno dónde almacenamos el nombre del videojuego con su id (videogames.json):
![videogames.json](videogames.png)

Y otro csv con los ids de los juegos más vendidos por cada día (topSellers.csv):
![topSellers.csv](topSellers.png)

De esta manera, obtenemos un flujo de datos actualizado diariamente que alimenta el sistema con los títulos más relevantes del mercado, asegurando que el chatbot siempre tenga acceso a información actualizada.

Con los ids de los juegos más vendidos, se ejecuta un proceso de extracción masiva de datos utilizando la API de Steam para obtener la ficha técnica de cada juego y sus reseñas. El resultado es un conjunto de archivos JSON. Tres por cada juego, los cuales uno contiene toda la información que la página principal del título en Steam proporciona; otra almacena las reseñas negativas del juego en inglés y el último, las positivas. Estos json se encuentran en la carpeta rawData:
![rawData](rawData.png)

Además, para cada reseña almacenamos la puntuación de utilidad que los usuarios de Steam han otorgado a cada opinión, lo cual se convierte en un dato crucial para el posterior proceso de resumen extractivo basado en grafos, permitiendo priorizar las opiniones más valoradas por la comunidad.

### Limpieza y preprocesamiento de datos

Respecto a la limpieza de datos, el problema técnico identificado fue la alta redundancia de los datos originales de la API de Steam y la contaminación de las reseñas por comentarios en idiomas distintos al inglés. La solución adoptada fue el desarrollo de un pipeline de limpieza que utiliza un ejecutor de procesos en paralelo para aprovechar al máximo los múltiples núcleos del procesador de la computadora. Este componente limpia el texto de las opiniones, elimina aquellas reseñas que carecen de contenido o que pertenecen a otros idiomas, y reduce la ficha técnica del juego a únicamente seis campos clave esenciales para el sistema de recuperación: nombre, descripción, plataformas compatibles, clasificación por edades, géneros y precio. El resultado es un conjunto de archivos optimizados almacenados en la carpeta cleanData y que aceleran drásticamente las fases de lectura y vectorización posteriores. Adjuntamos tres imágenes de ejemplo que muestran los archivos resultantes de este proceso de limpieza para un juego concreto:

![Expedition33 Info](expedition33info.png)

![Expedition33 Negatives](image-1.png)

![Expedition33 Positives](image-2.png)

### Modelado de tópicos

En el módulo de modelado de temas, el problema era descubrir las corrientes de opinión dominantes y los aspectos críticos de los videojuegos entre miles de comentarios heterogéneos sin depender de categorías manuales. Además de poder asociar videojuegos no solo por sus géneros sino por despertar opiniones similares entre el público; ya sea positivas o negativas.

La solución consistió en integrar la librería BERTopic para procesar todas las reviews limpias a la vez. No se realiza para cada juego por separado, sino en conjunto para poder determinar patrones en general entre ellos. Primero, se aplica un filtro exhaustivo de exclusión de palabras comunes que combina los términos genéricos del lenguaje junto con una lista personalizada de vocabulario propio de la industria de los videojuegos (game, play, like...). A continuación, el proceso realiza una reducción de dimensionalidad mediante la técnica UMAP basándose en la similitud del coseno, seguida de un agrupamiento por densidad con el algoritmo HDBSCAN, exigiendo un volumen mínimo por grupo para garantizar la relevancia estadística.

Para evitar redundancias en las palabras clave de cada tema, se aplica un principio de máxima relevancia marginal, y finalmente se implementa una estrategia de reasignación de elementos atípicos o ruidosos utilizando la frecuencia de término inversa. Con este proceso obtenemos una extracción limpia de veinticuatro tópicos semánticos (sin contar el cluster de ruído) que caracterizan desde mecánicas de combate hasta problemas de optimización técnica de los juegos.

Un problema adicional que surgió fue el número de reseñas en el clúster de ruido, el cual era muy elevado. Para solucionarlo, se reasignan a los temas principales mediante una estrategia basada en c-TF-IDF, recalculando después las palabras clave para mantener la precisión.

Una vez definidos estos clústeres, se utiliza un LLM local para procesar las palabras clave de cada clúster. Esto permite generar de forma automática un título descriptivo corto y una explicación orientada a la experiencia del usuario para cada tópico.

Posteriormente, se reorganiza esta estructura orientada a clústeres en una vista centrada en el videojuego, mapeando qué porcentaje y volumen de cada tema aparece en las opiniones de un juego concreto.

Con el objetivo de seleccionar los mejores hiperparámetros para el medelo de tópicos, se realizaron diferentes pruebas variando los valores y comparando los clústers obtenidos además de buscar la mayor coherencia semántica posible. Para poder medir esta coherencia se utilizó el CoherenceModel de la librería Gensim. Con esto, hemos seleccionado esta configuración:

Primero, en la fase inicial de vectorización, se establece un umbral de frecuencia mínima de cinco reviews para omitir términos anecdóticos y un límite superior del ochenta por ciento para neutralizar vocabulario excesivamente generales. Además, se incluyen n-gramas de hasta dos palabras para capturar expresiones compuestas comunes en el ámbito de los videojuegos. En la reducción de dimensionalidad, se opta por un número de vecinos de treinta para equilibrar la preservación de la estructura local y global del espacio vectorial. En el algoritmo HDBSCAN, se establece un tamaño mínimo de clúster de ochenta para garantizar que cada tema tenga suficiente representación estadística, junto con un umbral de selección de clústeres de cero punto uno para controlar la granularidad de los grupos formados. Para las palabras clave, se utilizó una diversidad de 0.3 para asegurar que las palabras que describen cada tema no sean redundantes entre sí. También, en la reducción del clúster -1, encontramos que un umbral de 0.05 evita las reasignaciones forzadas y aumenta la coherencia reduciendo el ruído. Finalmente, se limita el número total de tópicos a veinticinco para mantener un equilibrio entre la diversidad temática y la interpretabilidad.

En el JSON gamesTopics.json de la carpeta topicsData se pueden observar los tópicos extraídos para cada juego con el porcentaje de reviews que pertenecen a cada uno de ellos. Además, al final del archivo se encuentran los títulos y explicaciones generados por el LLM para cada tópico:

![Ejemplo de tópicos](topics.png)

### Resumen extractivo

Por último, el módulo de resumen extractivo aborda el problema de condensar el valor de miles de opiniones positivas y negativas sin saturar el contexto del modelo lingüístico con ideas repetitivas. La solución fue diseñar un algoritmo de resumen basado en grafos informáticos. El sistema divide las reseñas en frases individuales a través de herramientas de procesamiento de lenguaje natural y las convierte en vectores densos. Utilizando una librería de redes, se construye un grafo no dirigido donde los nodos representan las frases y las conexiones miden su similitud conceptual, descartando cualquier enlace por debajo de un umbral de ruido establecido. Para priorizar las opiniones más valiosas, el algoritmo ejecuta una variante personalizada del algoritmo PageRank que utiliza las puntuaciones de utilidad nativas de Steam como peso, aplicando una amortiguación logarítmica para evitar que las reseñas excesivamente largas dominen la centralidad del grafo de manera injusta. El resultado es la selección matemática de las cinco frases más representativas y densas en información para el espectro positivo y negativo de cada título.

Los resultados se pueden ver en la carpeta summaryData. Este es un ejemplo de resumen extractivo para el juego "Baldur's Gate 3":

![Resumen Negativo bg3](negative_bg3.png)
![Resumen Positivo bg3](positive_bg3.png)

### Estructuración de los documentos RAG

El problema final de la fase offline era cómo unificar los datos técnicos limpios, la distribución porcentual de los tópicos globales y los resúmenes extractivos de opiniones en un único formato de documento estructurado que fuera óptimo tanto para la indexación vectorial como para la asimilación conceptual por parte del modelo de lenguaje en la fase online. La solución fue diseñar una plantilla de generación de documentos RAG estandarizada. En esta etapa, el pipeline unifica toda la información procesada para construir el motor de búsqueda que alimentará al RAG. La función addDocsToCollection se encarga de consolidar los datos de cada videojuego, combinando la información general de cleanData, los resúmenes de opiniones de summaryData y los porcentajes de temas de gameTopics.json. Con este texto estructurado se genera un documento final que se convierte en vector mediante nuestro modelo de embeddings y se almacena en una colección de ChromaDB. Para identificar cada juego de forma única y evitar duplicados, se genera un identificador aplicando un hash SHA256 sobre su nombre.

Por cada documento, se crea un único bloque de texto formateado de la siguiente manera:

- Título: El nombre del videojuego.

- Géneros: Los géneros a los que pertenece el juego.

- Descripción: La descripción general del juego proveniente de STEAM.

- Resumen de reseñas positivas: Las 5 frases más representativas extraídas de las opiniones positivas (o un texto por defecto si no hay).

- Resumen de reseñas negativas: Las 5 frases más representativas extraídas de las opiniones negativas (o un texto por defecto si no hay).

- Temas asociados: Una lista con los tópicos detectados por BERTopic extraído de las reseñas, detallando el título del tema, su descripción y el porcentaje de relevancia que tiene dentro de las reseñas de ese juego concreto.

Además, se añaden metadatos adicionales a cada documento, como el precio, las plataformas disponibles y la clasificación PEGI, para enriquecer aún más la información que el RAG puede utilizar para responder preguntas específicas sobre cada videojuego.

### Orquestador y nodos de consulta

El motor interactivo en tiempo real del sistema está gobernado por la clase Orchestrator, la cual gestiona el ciclo de vida completo de la consulta del usuario. El problema principal de este componente era cómo garantizar un enrutamiento determinista y seguro de los diálogos abiertos de los usuarios sin que las respuestas ambiguas rompieran la lógica de la aplicación. La solución consistió en implementar el Nodo Uno, un agente clasificador con un decodificador restringido que obliga al modelo a devolver exclusivamente un formato estructurado con la acción de recuperación, búsqueda léxica directa o interacción vacía. El resultado es la erradicación absoluta de errores sintácticos en la toma de decisiones, permitiendo que el organizador derive la consulta al nodo técnico correspondiente de forma infalible.

Cuando la acción determinada es el uso del rag, el flujo se transfiere al Nodo Tres, el cual emplea una búsquea híbrida para obtener los tres documentos más relevantes. Por el contrario, si el usuario interroga al sistema sobre un videojuego específico, se activan de forma secuencial el Nodo Cuatro y el Nodo Cinco. El problema en esta ruta era aislar con exactitud el nombre del videojuego y mitigar el problema de juegos recientes ausentes en la base de datos. La solución fue utilizar un extractor lingüístico en el Nodo Cuatro y un algoritmo de emparejamiento difuso con un umbral del sesenta por ciento en el Nodo Cinco; si el título no se encuentra en el índice local, el nodo ejecuta de forma reactiva una petición sobre la tienda oficial de Steam, importando la información y forzando la actualización del pipeline offline en el momento.

Finalmente, toda la información contextual recuperada converge en el Nodo De Generación o Nodo Dos. El problema técnico en esta fase de síntesis era evitar la alucinación del modelo de lenguaje, la repetición literal de textos y la latencia provocada por reevaluar el historial de la conversación en cada turno. La solución fue diseñar un agente experto dotado de reglas severas de reformulación. El resultado es un modelo que genera respuesta coherentes y con una buena latencia.

## DISCUSIÓN, VALIDACIÓN Y EVALUACIÓN CRÍTICA

La calidad del sistema se evidencia en la validación de sus componentes. Con ese fin, primero evaluamos el modelo de tópicos, para el cual se midió la coherencia semántica del modelo antes y después del proceso de reasignación de comentarios ruidosos. En el estado inicial, el modelo presentaba una tasa de elementos atípicos del cuarenta y un por ciento y un valor de coherencia de 0,49. Tras la aplicación de la estrategia basada en la frecuencia de término inversa, la tasa de elementos atípicos se redujo drásticamente al diecisiete por ciento, mientras que la coherencia semántica se elevó hasta alcanzar un valor de 0,59. Este último valor demuestra un resultado más que aceptable.

También, si nos disponemos a evaluar los tópicos y su interpretación, podemos observar resultados muy interesantes. Por ejemplo, el tópico 8 se centra en la mecánica de combate, con palabras clave como "parry", "attack", "dodge" o "boss". Este tópico podría incluir juegos que tienen un sistema de combate desafiante y que requieren habilidades de parry y dodge para superar a los enemigos. Esto permite relacionar juegos como Elden Ring o Expedition 33, pese a que uno es un Souls y el otro un juego por turnos respectivamente. Sus géneros son muy diferentes, pero comparten similitudes en la mecánica de combate. También nos permite conocer aquellos juegos que tiene problemas de optimización, como el tópico 11 que se centra en el problema de los tramposos en los videojuegos multijugador y los fallos que generan los anticheats.

Sin embargo, tampoco es perfecto, pues el cluster 0 almacena una gran parte de las reseñas, lo que indica que es un cluster genérico que engloba muchos comentarios. Mientras que en el lado contrario, presenta clusters mucho más específicos con muy pocas reseñas. Esto son problemas que no se han podido resolver con la configuración actual de hiperparámetros y que podrían ser objeto de mejora en futuras iteraciones del proyecto. Reducir la cantidad de reseñas en el cluster genéricos aumenta los clústers específicos, mientras que reducir estos últimos aumenta el genérico, lo que sugiere que el modelo tiene dificultades para encontrar un equilibrio óptimo entre la generalidad y la especificidad de los temas extraídos.

Con el fin de evaluar la calidad del sistema multiagente hemos creado una serie de preguntas de prueba y un nodo sintético. Para cada pregunta, el modelo genera su respuesta y el nodo sintético toma el rol del usuario para continuar la conversación. De esta manera, podemos evaluar no solo la respuesta a la pregunta inicial, sino también la capacidad del sistema para mantener un diálogo coherente y relevante a lo largo de múltiples turnos de conversación. Estas conversaciones las almacenamos en el json completeExecution.json, donde se pueden observar la fecha en la que se inició la conversación, el prompt inicial, los outputs internos de cada nodo, la respuesta final del sistema y la contestación del nodo sintético:

![Ejecución sobre Expedition 33](execution_expedition33.png)

Gracias a estos logs, nos damos cuenta del correcto funcionamiento general del sistema. Es capaz de buscar con éxito la información de un videojuego específico preguntado por el usuario y de generar una respuesta coherente y relevante utilizando la información recuperada. También busca documentos relacionados con preguntas más generales de videojuegos, por ejemplo recomendaciones de juegos de rol, y es capaz de generar respuestas que integran información de múltiples documentos recuperados.

A pesar de la solidez general del pipeline, la evaluación crítica del sistema ha permitido identificar fallos puntuales en el comportamiento del asistente final que representan áreas de mejora. Se ha observado que, en escenarios específicos de generación de respuestas, el modelo tiende a volcar fragmentos del documento de contexto de manera literal en lugar de realizar un proceso de síntesis e integración orgánica, lo cual afecta la naturalidad de la interacción. Asimismo, se han detectado redundancias operativas donde el sistema inicia procesos de búsqueda sobre documentos que ya se encuentran cargados en la memoria del historial de la conversación. Estos comportamientos sugieren la necesidad de refinar las directrices de los prompts de sistema para optimizar el consumo de recursos y mejorar la fluidez de las respuestas entregadas al usuario. Sin emabargo, pese a la mejora de los system prompts, no se ha conseguido erradicar completamente este problema, lo que indica que podría ser necesario implementar algún mecanismo de control adicional o emplear un modelo de lenguaje más avanzado capaz de adaptarse mejor a las reglas de generación establecidas.

## Cambios aplicados desde la entrega inicial

- Modoficaciones en los sistem prompts.

- Cambio de toda el informe
