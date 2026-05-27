# MITEX Steam

Steam es una plataforma de distribución de vidiojuegos que cuenta con una API para obtener información sobre juegos. En este proyecto se obtiene información para usar en un rag para que un modelo de lenguaje pueda responder preguntas sobre los juegos.

Usamos Steam porque tiene mucha información disponible gracias a la ideología de Valve y [Gabe Newell](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/106d800f-15b4-44d4-aa16-972239963d2d/d4rnffi-c828868d-089e-4fd5-b38b-33c6c6b6f8da.png/v1/fill/w_800,h_1107,q_80,strp/gabe_newell_portrait_by_freddre_d4rnffi-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9MTEwNyIsInBhdGgiOiIvZi8xMDZkODAwZi0xNWI0LTQ0ZDQtYWExNi05NzIyMzk5NjNkMmQvZDRybmZmaS1jODI4ODY4ZC0wODllLTRmZDUtYjM4Yi0zM2M2YzZiNmY4ZGEucG5nIiwid2lkdGgiOiI8PTgwMCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.pG6EORYPxTTlYSLRZ7SHHuHJtjbSuGNjMj1E-NONigk).

## Elegir juegos

Para elegir los videojuegos se usa el enlace <https://store.steampowered.com/search/results/?filter=globaltopsellers&hidef2p=1&json=1>. Iterando por 4 páginas se obtienen los 100 juegos más vendidos no gratuítos en Steam. Con esta información se guarda en el fichero [`videogames.json`](../videogames.json) el nombre del juego y su id en Steam.

Pero si se abre el fichero se ve que hay más de 100 juegos. Esto se debe a que hay una acción de GitHub que actualiza el fichero cada día (`0 0 * * *`) y en cada actualización se añaden los nuevos juegos a la lista.

Otro fichero que se crea es [`topSellers.csv`](../topSellers.csv) que contiene en orden el id para los cien juegos de cada día. Esto no se va a usar en el proyecto pero se puede ver el momento que se lanzan nuevos juegos, se anuncian, sale una expansión o se rebajan. También se puede ver que cuando se lanza/anuncia un juego de una franquicia, el resto de juegos de la franquicia tienen más ventas.

## Obtener información de los juegos

Para cada juego queremos obtener su información general y sus reseñas. Los guardamos en la carpeta [`rawData`](../rawData/).

### Información general

Para esto usamos la API de Steam con el endpoint <https://store.steampowered.com/api/appdetails?appids=70>. En este caso el id del juego es 70, que corresponde a [`Half-Life`](https://store.steampowered.com/app/70/HalfLife/). Con un request simple ya obtenemos toda la información en formato JSON. Se guarda como el nombre del juego más `Info.json`. En el caso de Half-Life se guarda como `Half-LifeInfo.json`.

### Reseñas

Para las reseñas cogemos las 100 primeras positivas y las 100 negativas. Para las positivas se usa <https://store.steampowered.com/appreviews/70?json=1&filter=all&language=english&num_per_page=100&review_type=positive> y para las negativas <https://store.steampowered.com/appreviews/70?json=1&filter=all&language=english&num_per_page=100&review_type=negative>.

La documentación de esta url es esta [User Reviews - Get List](https://partner.steamgames.com/doc/store/getreviews). Usamos los parámetros `json=1` para obtener la respuesta en formato JSON, `filter=all` para obtener las reseñas ordenadas por su utilidad, `language=english` para obtener solo reseñas en inglés, `num_per_page=100` para obtener 100 reseñas por página y `review_type` para especificar si queremos reseñas positivas o negativas. Lo importante es saber que puede no devolver ninguna reseña, por ejemplo si el juego es muy nuevo o no tiene suficientes reseñas en inglés. El idioma puede fallar y devolvernos reseñas en otros idiomas que solucionaremos en la limpieza de datos.

Las reseñas se guardan con el nombre del juego más `Positive.jsonl` para las positivas y `Negative.jsonl` para las negativas. Usamos `.jsonl` porque al ser cada línea un JSON, sería más fácil de procesar después si hay problemas en una única reseña. Aunque esto nunca pasó porque todo está bien formateado. De cada reseña guardamos la utilidad que le da STEAM y el texto.
