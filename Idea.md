# ⚽️ Proyecto: **Team Shape Analyzer – Understanding Team Structure Across Phases of Play**

## 🎯 Idea general

Desarrollar una herramienta **open-source** que analice la **estructura colectiva de un equipo** a lo largo de un partido, utilizando datos de **tracking** y **eventos dinámicos**.

El objetivo es cuantificar y visualizar **cómo cambia la forma del equipo** (_compactness, width y depth_) en las distintas fases del juego:

- **Sin posesión** (defensa y presión)
- **Con posesión** (construcción y ataque)
- **Transiciones** (tras pérdida y tras recuperación)

Todo dentro de una **app ligera en Streamlit**, con visualizaciones interactivas que puedan servir tanto para **análisis propio** como de **rivales**.

---

## 🧠 Motivación

En entornos profesionales, los cuerpos técnicos necesitan identificar patrones de comportamiento espacial:  
cuán compacto es el equipo sin balón, cómo se expande en ataque o cómo reacciona al cambio de posesión.

Este proyecto busca ofrecer un enfoque **descriptivo, visual y reproducible**, aplicable a cualquier equipo o partido, **sin requerir conocimiento previo del modelo de juego**.

---

## 🧩 Objetivos específicos

1. **Medir la estructura del equipo** a través de tres métricas principales:

   - 🌀 **Compactness** → área del _Convex Hull_ de los jugadores.
   - ↔️ **Width** → anchura máxima (diferencia lateral Y).
   - ↕️ **Depth** → profundidad (diferencia longitudinal X).

2. **Analizar el comportamiento según la fase de juego**  
   Aprovechar las etiquetas de _in possession / out of possession_ para comparar cómo se reorganiza el equipo.

3. **Contextualizar espacialmente**  
   Dividir el campo en una **rejilla 3×3** y evaluar las métricas por zona del balón.

4. **Incorporar las transiciones**  
   Medir la respuesta del equipo tras perder o recuperar el balón (repliegue, presión o avance rápido).

5. **Visualizar y comparar**

   - Timelines de las métricas a lo largo del partido.
   - Heatmaps 3×3 (por fase y métrica).
   - Gráficos claros y exportables (útiles para informes tácticos).

6. **Entregar un software open-source reproducible**
   - Implementado en **Python + Streamlit**.
   - Configurable mediante un archivo **YAML** (paths, equipo, rejilla, parámetros temporales).
   - Reutilizable con cualquier partido del dataset.

---

## 🖥️ Estructura esperada de la app

Tres pestañas principales:

| Fase             | Qué muestra                                                                     | Ejemplo de visuales                |
| ---------------- | ------------------------------------------------------------------------------- | ---------------------------------- |
| **Sin posesión** | Altura del bloque, compactness, width/depth, mapa de presiones o recuperaciones | Timeline + heatmap defensivo       |
| **Con posesión** | Anchura y profundidad ofensiva, compactness, patrones de ocupación del campo    | Timeline + heatmap ofensivo        |
| **Transiciones** | Presión tras pérdida, recuperación y progresión tras recuperar                  | Barras o heatmap con tasa de éxito |

> 💡 _La pestaña de transiciones puede ser opcional o parcial en la primera versión._

---

## 🧾 Resultado final

Un módulo **open-source** capaz de:

- Calcular y visualizar **métricas espaciales** por fase y zona.
- Comparar el comportamiento del equipo **en posesión y sin posesión**.
- Generar **gráficos reproducibles** (timeline y heatmaps).
- Servir como base para futuros **análisis tácticos o scouting de rivales**.

---

## ⚙️ Stack técnico

- 🐍 **Python** (`pandas`, `numpy`, `scipy`, `matplotlib` / `mplsoccer`)
- 💻 **Streamlit** para visualización interactiva
- ⚙️ **YAML** para configuración paramétrica
- 🌐 **GitHub (MIT License)** como entrega final del hackathon

---

## 🧭 Cronograma (21 oct – 29 dic)

| Semana         | Fecha           | Fase                       | Objetivos principales                                                                    | Entregable                           |
| -------------- | --------------- | -------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------ |
| **1**          | 21–27 oct       | 🟩 Setup & exploración     | Cargar datos, validar coordenadas, elegir partido y equipo. Crear notebook base.         | `01_exploracion.ipynb`               |
| _(Vacaciones)_ | 31 oct – 16 nov | 🌴 Lectura conceptual      | Leer sobre compactness/width/depth, análisis de bloque y fases. Tomar ideas de visuales. | Notas o referencias                  |
| **2**          | 18–24 nov       | ⚙️ Métricas base           | Implementar compactness, width, depth. Calcular por frame y validar.                     | Funciones listas y 1° timeline       |
| **3**          | 25 nov – 1 dic  | 📈 Timeline temporal       | Agregar resample (cada 5s), smoothing, anotaciones de eventos.                           | Timeline de métricas                 |
| **4**          | 2–8 dic         | 🧭 Segmentación espacial   | Dividir campo (3×3), calcular promedios por zona y fase (in/out).                        | Heatmaps básicos                     |
| **5**          | 9–15 dic        | 🔀 Fases de juego          | Crear pestañas _in/out possession_ en la app y conectar con cálculos.                    | Tabs “Sin posesión” y “Con posesión” |
| **6**          | 16–22 dic       | ⚡ Transiciones y visuales | Añadir cálculo de presión tras pérdida / avance tras recuperación. Pulir visuales.       | Tab “Transiciones” (simple)          |
| **7**          | 23–29 dic       | 🏁 Documentación y entrega | README, instrucciones, capturas, revisión final.                                         | Repositorio completo                 |

---

## ⏱️ Dedicación estimada

- 5–10 h/semana promedio
- Total aproximado: **55–60 h efectivas**  
  (incluyendo setup, desarrollo y documentación)

---

## 🌱 Extensiones posibles (post-hackathon)

- Clasificación de líneas (defensa, medio, ataque) para medir distancias entre ellas.
- Orientación de la presión (hacia adentro o hacia banda).
- Análisis de redes de pase en posesión.
- Comparaciones entre partidos o rivales.
- Dashboard interactivo con filtros por mitad, marcador o zona.

---

## 📄 Resumen corto

> **Team Shape Analyzer** es una herramienta open-source para explorar cómo un equipo se organiza durante las distintas fases del juego.  
> A partir de datos de tracking, mide _compactness_, _width_ y _depth_, segmentadas por posesión y zonas del campo, y visualiza cómo cambia la estructura del equipo en defensa, ataque y transición.  
> El objetivo es ofrecer un análisis táctico **descriptivo, visual y reutilizable**, útil tanto para análisis propio como para scouting de rivales.
