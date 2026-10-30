# Capítulo 3 · Proyecto: preparar los datos del Titanic 🚢

El 15 de abril de 1912 el Titanic se hundió en su viaje inaugural. De las 891 personas de este dataset, solo sobrevivió un 38 %. ¿Fue cuestión de suerte, o hubo factores (sexo, edad, clase del billete…) que influyeron en quién se salvó?

En el próximo capítulo vas a entrenar tu **primer modelo** para responder esa pregunta: predecir si un pasajero sobrevivió (`survived`). Pero un modelo solo es tan bueno como los datos con los que aprende. En este proyecto vas a hacer el trabajo que ocupa la mayor parte del tiempo de cualquier científico de datos: **entender, limpiar y preparar los datos**.

> ⚠️ **Este documento no tiene código.** Tú decides cómo resolver cada paso con lo que aprendiste en el capítulo 3 (NumPy, pandas, matplotlib, limpieza, outliers, transformación, codificación y escalado).

---

## 🎯 Objetivos

Al terminar el proyecto serás capaz de:

1. Hacer un **análisis exploratorio** (EDA) de un dataset real y sacar conclusiones de él.
2. Detectar problemas de calidad: duplicados, valores faltantes, columnas redundantes y outliers.
3. **Tomar decisiones de limpieza y justificarlas** por escrito. Esta es la parte más importante.
4. Transformar, codificar y escalar las variables.
5. Dejar los datos **listos para entrenar un modelo**, separados en train y test.

---

## 📦 Entregables

| Entregable | Dónde | Descripción |
|---|---|---|
| Notebook del proyecto | `workspace/chapter_03/proyecto_titanic.ipynb` | Todo tu análisis, de principio a fin, con celdas de Markdown que expliquen lo que haces. |
| Bitácora de decisiones | Dentro del notebook (sección final) | Una tabla con **cada decisión** de limpieza y transformación que tomaste y **por qué**. |
| Conclusiones | Dentro del notebook (sección final) | Un resumen de lo que aprendiste de los datos y de cómo quedaron preparados. |

Reglas del notebook (las mismas de siempre):

- Trabaja **solo** dentro de `workspace/`. No modifiques archivos del curso.
- Tiene que funcionar con **Restart & Run All**, de arriba abajo, con todos los imports al principio.
- Usa rutas relativas para leer los datos (`data/titanic.csv`).
- Fija las semillas aleatorias (`random_state=42`).
- Escribe en Markdown **qué** haces y **por qué** antes o después de cada bloque de código. Un notebook solo con código no se considera completo.
- Limpia las salidas antes de hacer commit.

---

## 🗂️ Los datos

Si aún no tienes el archivo `data/titanic.csv`, descárgalo con el script del curso (ver [docs/installation.md](../../docs/installation.md)).

El dataset tiene **891 filas y 15 columnas**. Cada fila es un pasajero.

| Columna | Descripción |
|---|---|
| `survived` | ¿Sobrevivió? (0 = no, 1 = sí). **Es la variable objetivo.** |
| `pclass` | Clase del billete (1 = primera, 2 = segunda, 3 = tercera). |
| `sex` | Sexo del pasajero. |
| `age` | Edad en años. |
| `sibsp` | Número de hermanos o cónyuges a bordo. |
| `parch` | Número de padres o hijos a bordo. |
| `fare` | Precio del billete (en libras de la época). |
| `embarked` | Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton). |
| `class` | Clase del billete, en texto. |
| `who` | ¿Hombre, mujer o niño? |
| `adult_male` | ¿Es un hombre adulto? |
| `deck` | Cubierta del camarote. |
| `embark_town` | Puerto de embarque, en texto. |
| `alive` | ¿Sobrevivió?, en texto. |
| `alone` | ¿Viajaba solo? |

> 💡 Lee bien esta tabla antes de empezar. Algunas columnas esconden sorpresas que tendrás que descubrir en la fase 1.

---

## Fase 1 · Primer contacto con los datos

Antes de cambiar nada, **mira**. El objetivo de esta fase es conocer el dataset, no arreglarlo.

### Tareas

1. Carga el dataset y muestra las primeras y las últimas filas.
2. Averigua cuántas filas y columnas tiene, y el tipo de dato de cada columna.
3. Obtén un resumen estadístico de las columnas numéricas y otro de las categóricas.
4. Cuenta los valores faltantes por columna, en número y en porcentaje.
5. Cuenta cuántas filas están duplicadas.
6. Para cada columna categórica, mira cuántos valores distintos tiene y cuáles son.

### Preguntas

- ¿Qué tipo de problema de Machine Learning es este: regresión o clasificación? ¿Por qué?
- ¿Qué columnas son numéricas y cuáles categóricas? ¿Hay alguna numérica que en realidad se comporte como categórica? ¿Alguna categórica que tenga un **orden** natural?
- ¿Qué columnas tienen valores faltantes? ¿Cuál te preocupa más y por qué?
- ¿Ves columnas que parezcan decir **lo mismo** que otra? Haz una lista de esas parejas y comprueba con los datos si de verdad contienen la misma información.
- Mira la columna `alive`. ¿Qué pasaría si se la dieras a un modelo que tiene que predecir `survived`? ¿Ese modelo serviría en la vida real?

---

## Fase 2 · Análisis exploratorio (EDA)

Ahora toca hacer preguntas a los datos con gráficas y tablas. Cada gráfica debe ir acompañada de **una o dos frases** con lo que observas. Una gráfica sin interpretación no cuenta.

### 2.1 La variable objetivo

- ¿Qué porcentaje de pasajeros sobrevivió? Represéntalo con una gráfica.
- ¿Las clases están equilibradas? ¿Por qué podría importar esto cuando entrenes un modelo?

### 2.2 Variables una a una (análisis univariado)

- Haz un histograma de `age` y otro de `fare`. Describe su forma: ¿son simétricas o asimétricas?
- Haz un diagrama de caja (*boxplot*) de `age` y otro de `fare`. ¿Ves puntos alejados del resto?
- Haz gráficas de barras de `sex`, `pclass` y `embarked`. ¿Qué categoría domina en cada una?
- Mira la distribución de `sibsp` y `parch`. ¿Qué es lo más habitual?

### 2.3 Variables frente a la supervivencia (análisis bivariado)

- ¿Qué porcentaje sobrevivió según el **sexo**? ¿Y según la **clase**?
- ¿Sobrevivieron más los niños que los adultos? Compara la distribución de `age` entre los que sobrevivieron y los que no.
- ¿Los billetes más caros tenían más probabilidad de sobrevivir?
- ¿Influyó el puerto de embarque? Antes de concluir nada, mira qué clase de billete compraba la gente en cada puerto. ¿Puede ser que el puerto no importe y lo que importe sea otra cosa?
- ¿Viajar solo te ayudaba o te perjudicaba?
- Combina dos variables: ¿cómo fue la supervivencia de las mujeres de tercera clase comparada con la de los hombres de primera?

### Preguntas

- Si tuvieras que adivinar si un pasajero sobrevivió **conociendo solo una columna**, ¿cuál elegirías? Justifícalo con tus gráficas.
- Escribe **tres hallazgos** del EDA que te hayan llamado la atención.
- ¿Hay algo en el EDA que te haga sospechar de errores en los datos?

---

## Fase 3 · Limpieza de datos ⭐

**Esta es la fase más importante del proyecto.** Casi nunca hay una única respuesta correcta: lo que se evalúa es que **tomes una decisión, la justifiques y midas su efecto**.

Para cada decisión, anota en tu bitácora (ver más abajo):

- **Qué** problema encontraste.
- **Qué opciones** tenías.
- **Cuál elegiste y por qué.**
- **Cuántas filas o valores** se vieron afectados.

Antes de empezar, **guarda una copia** de los datos originales. Al final de la fase compararás el antes y el después.

### 3.1 Columnas redundantes y fuga de información

En la fase 1 encontraste columnas que repiten información, y al menos una que "chiva" la respuesta.

- Decide qué columnas eliminas y cuáles conservas de cada pareja redundante. ¿Con qué criterio eliges cuál quedarte?
- `who` y `adult_male` se calcularon a partir de `sex` y `age`. ¿Te aportan algo nuevo? ¿Las conservarías?
- Explica con tus palabras qué es la **fuga de información** (*data leakage*) y qué columna(s) la provocan aquí.

### 3.2 Duplicados

El dataset tiene filas duplicadas. Pero ojo: este dataset **no tiene el nombre ni el número de billete** de cada pasajero.

- ¿Esas filas son errores (la misma persona registrada dos veces) o pueden ser **personas distintas** que comparten las mismas características?
- Piensa en un ejemplo concreto: dos hombres de tercera clase, de 20 años, que viajan solos, embarcaron en Southampton y pagaron lo mismo. ¿Es posible?
- Decide si eliminas los duplicados o no, y justifícalo. ¿Qué perderías si te equivocas en cada caso?

### 3.3 Valores faltantes

Trata cada columna con faltantes por separado.

**`deck`**

- ¿Qué porcentaje de valores le falta?
- ¿Tiene sentido imputar una columna a la que le faltan tantos datos? ¿Qué opciones tienes: eliminarla, imputarla, crear una categoría "desconocido", crear un indicador de si se conoce la cubierta…?
- Antes de decidir: ¿el hecho de que falte la cubierta tiene relación con la supervivencia o con la clase? Compruébalo. ¿Cambia eso tu decisión?

**`age`**

- ¿Cuántos valores faltan? ¿Eliminar esas filas sería buena idea? ¿Cuántos datos perderías?
- ¿Imputarías con la media o con la mediana? Mira la forma de la distribución para decidir.
- ¿Sería mejor imputar por grupos (por ejemplo, la mediana de edad según la clase, el sexo o ambos)? Compara las medianas de cada grupo: ¿son parecidas o muy distintas?
- ¿Vale la pena crear una columna indicadora de "edad desconocida"? ¿Por qué podría ayudar al modelo?
- Después de imputar, vuelve a hacer el histograma de `age`. ¿Cómo cambió la distribución? ¿Ves algún efecto raro?

**`embarked`**

- Solo faltan unos pocos valores. ¿Qué harías y por qué?
- ¿La moda es una opción razonable aquí?

### 3.4 Outliers

Para `fare`, `age`, `sibsp` y `parch`:

1. Detecta posibles outliers con al menos **dos métodos** vistos en clase (por ejemplo, IQR y Z-score). ¿Cuántos encuentra cada método? ¿Coinciden?
2. Para cada caso, pregúntate: ¿es un **error** o es un valor **raro pero real**?

Preguntas guía:

- `fare`: hay billetes de más de 500 libras. ¿Es un error o había pasajeros muy ricos? ¿Cómo lo comprobarías con los propios datos (por ejemplo, mirando su clase)?
- `fare`: hay billetes que costaron **0**. ¿Qué pudo pasar? ¿Es un error, un valor faltante disfrazado, o puede ser real (tripulación, invitados…)? ¿Qué harías con ellos?
- `age`: hay edades con decimales y menores de 1 año. ¿Son errores?
- `sibsp`: algunos pasajeros tienen 8 hermanos o cónyuges a bordo. ¿Es posible? ¿Qué te dice eso de la familia?
- ¿Por qué el método de Z-score puede fallar con una variable tan asimétrica como `fare`?
- Elige un tratamiento para cada variable: no hacer nada, eliminar filas, winsorizar (*capping*) o transformar. Justifícalo.

> 💡 Recuerda lo que vimos en clase: un outlier **no es automáticamente un error**. Eliminar datos reales puede hacer que el modelo aprenda un mundo que no existe.

### 3.5 Antes y después

- Compara el dataset original con el limpio: número de filas, de columnas y de valores faltantes.
- Compara las distribuciones de `age` y `fare` antes y después de la limpieza.
- ¿Cambió el porcentaje de supervivientes después de limpiar? Si cambió, ¿por qué? ¿Es un problema?

---

## Fase 4 · Transformación, codificación y escalado

Con los datos limpios, prepáralos para que un modelo pueda usarlos. Un modelo solo entiende **números**.

### 4.1 Crear variables nuevas (*feature engineering*)

Crea **al menos dos** variables nuevas a partir de las que ya tienes. Algunas ideas para que pienses (no tienes que hacerlas todas):

- El **tamaño de la familia** a bordo, a partir de `sibsp` y `parch`.
- **Tramos de edad** (niño, adolescente, adulto, mayor). ¿Dónde pondrías los cortes y por qué?
- Si conservaste algo de `deck`, ¿qué información útil puedes extraer?

Para cada variable nueva:

- Explica por qué crees que puede ayudar a predecir la supervivencia.
- Compruébalo con una gráfica o tabla frente a `survived`.

### 4.2 Transformar variables asimétricas

- ¿Qué variables numéricas son muy asimétricas?
- Aplica una transformación logarítmica a la que lo necesite. Compara el histograma antes y después.
- ¿Qué problema tendrías al aplicar un logaritmo si la variable tiene valores 0? ¿Cómo lo resuelves?

### 4.3 Codificar las variables categóricas

Para cada variable categórica que conserves:

- ¿Es **nominal** (sin orden) u **ordinal** (con orden)?
- ¿Qué codificación usarías: one-hot u ordinal? ¿Por qué?
- `pclass` ya es un número. ¿Lo dejas así o lo codificas con one-hot? Piensa en qué le "dice" cada opción al modelo. ¿Es la diferencia entre primera y segunda clase igual que entre segunda y tercera?
- Las columnas booleanas (`alone`, `adult_male`, si las conservas), ¿necesitan alguna codificación?
- En one-hot, ¿qué significa eliminar una de las columnas (`drop="first"`)? ¿Cuándo tiene sentido?

### 4.4 Separar en train y test

- Separa los datos en entrenamiento (80 %) y prueba (20 %), con `random_state=42`.
- Como las clases no están equilibradas, investiga qué significa hacer una separación **estratificada**. ¿Debería mantenerse el mismo porcentaje de supervivientes en train y en test? Compruébalo.
- **Pregunta clave:** ¿en qué orden deben ir la separación y el escalado? ¿Y la imputación? ¿Qué pasa si calculas la mediana de `age` o la media de `fare` con **todos** los datos, incluidos los de test? Relaciónalo con la fuga de información de la fase 3.

### 4.5 Escalar las variables numéricas

- ¿Qué variables necesitan escalarse? ¿Y cuáles no?
- ¿Usarías estandarización (`StandardScaler`) o normalización (`MinMaxScaler`)? Justifícalo según la forma de cada variable y la presencia de outliers.
- ¿Por qué el escalador se ajusta (`fit`) **solo con train** y luego se aplica a test?
- ¿Escalarías también las columnas que salen del one-hot? ¿Por qué?

### 4.6 Todo junto (opcional, recomendado 🚀)

Reúne la imputación, la codificación y el escalado en un `ColumnTransformer` dentro de un `Pipeline`, como vimos en clase.

- ¿Qué ventaja tiene esto frente a hacerlo todo paso a paso?
- Imagina que mañana llega un pasajero nuevo. ¿Qué tendrías que hacer para prepararlo con tu solución paso a paso? ¿Y con el pipeline?

### 4.7 Comprobación final

Antes de terminar, verifica que:

- [ ] No queda **ningún** valor faltante en `X_train` ni en `X_test`.
- [ ] Todas las columnas son **numéricas**.
- [ ] `X_train` y `X_test` tienen **las mismas columnas**, en el mismo orden.
- [ ] La variable objetivo `survived` **no** está dentro de `X`, ni ninguna columna que la revele.
- [ ] Las variables numéricas escaladas tienen la media y la desviación (o el mínimo y el máximo) que esperabas en train.
- [ ] El notebook funciona con **Restart & Run All**.

Muestra la forma final (`shape`) de `X_train`, `X_test`, `y_train` e `y_test`, y las primeras filas de `X_train`.

---

## 📝 Bitácora de decisiones

Al final del notebook, incluye una tabla en Markdown con **todas** tus decisiones. Este es el formato:

| # | Columna | Problema detectado | Opciones consideradas | Decisión | Justificación | Filas / valores afectados |
|---|---|---|---|---|---|---|
| 1 | `ejemplo` | 5 % de valores faltantes | Eliminar filas, media, mediana | Mediana | La distribución es asimétrica, la media se ve arrastrada por valores altos | 45 valores |
| 2 | … | … | … | … | … | … |

Debe tener, como mínimo, una fila por cada:

- Columna eliminada.
- Decisión sobre duplicados.
- Columna con valores faltantes.
- Variable revisada por outliers.
- Variable nueva creada.
- Transformación, codificación y escalado aplicados.

---

## 🧠 Conclusiones

Termina el notebook con una sección de conclusiones que responda:

1. ¿Qué tres factores crees que más influyeron en la supervivencia? Apóyate en tu EDA.
2. ¿Cuál fue la decisión de limpieza **más difícil** que tomaste? ¿Qué te hizo dudar?
3. Si un compañero hubiera tomado una decisión distinta en esa misma situación, ¿cómo crees que afectaría al modelo?
4. ¿Qué información te hubiera gustado tener que no está en el dataset?
5. ¿Con cuántas filas y columnas terminaste? ¿Qué columnas va a ver el modelo?

---

## ✅ Cómo se evalúa

| Criterio | Peso | Qué se busca |
|---|---|---|
| Exploración (fases 1 y 2) | 20 % | Gráficas adecuadas, **cada una interpretada**; preguntas respondidas. |
| Limpieza y decisiones (fase 3) | 35 % | Decisiones **justificadas con datos**, efecto de cada una medido, comparación antes/después. |
| Transformación y escalado (fase 4) | 25 % | Codificación y escalado correctos, separación train/test sin fuga de información. |
| Bitácora y conclusiones | 15 % | Bitácora completa y clara; conclusiones con argumentos. |
| Calidad del notebook | 5 % | Funciona con *Restart & Run All*, ordenado, con Markdown explicativo. |

> 💬 **No existe "la" limpieza correcta del Titanic.** Dos notebooks con decisiones distintas pueden tener la máxima nota si ambas están bien justificadas. Lo que no vale es tomar una decisión sin explicar por qué.

---

## 🆘 Si te atascas

- Repasa los notebooks de clase del capítulo 3, en especial [04_limpieza_y_outliers](../../notebooks/lectures/chapter_03/04_limpieza_y_outliers.ipynb) y [05_transformacion_y_escalado](../../notebooks/lectures/chapter_03/05_transformacion_y_escalado.ipynb).
- La actividad [04_actividad_decisiones_preprocesamiento](../../notebooks/exercises/chapter_03/04_actividad_decisiones_preprocesamiento.ipynb) tiene casos muy parecidos a los de este proyecto.
- Si usas un asistente de IA, pídele **pistas y explicaciones**, no la solución completa. Las decisiones del proyecto tienen que ser tuyas.

### 🎓 ¿Qué sigue?

En el capítulo 4 usarás los datos que preparaste aquí para entrenar tu primer modelo de clasificación. Guarda bien tu notebook: ¡lo vas a necesitar!
