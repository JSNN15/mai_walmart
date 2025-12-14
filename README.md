# 🚛 Walmart - Herramienta Avanzada de Carga de Camiones

Sistema inteligente de optimización 3D para la carga de camiones con algoritmo avanzado de **Bin Packing 3D**.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-red.svg)

## 🎯 Características Principales

### ✨ Funcionalidades Avanzadas

- **Algoritmo 3D Bin Packing Inteligente**: Optimización automática de espacios
- **Visualización 3D Interactiva**: Vista en tiempo real del camión cargado usando Plotly
- **Múltiples Restricciones**:
  - Límites de peso y volumen
  - Reglas de apilamiento por tipo de carga
  - Soporte estructural (mínimo 60% de área de contacto)
  - Restricciones de altura para cargas pesadas
  - Protección de cargas frágiles
- **Sistema de Prioridades**: Carga optimizada según importancia
- **Análisis Detallado**: Métricas de utilización, distribución y recomendaciones
- **Reportes Exportables**: CSV y TXT con información completa

### 📦 Tipos de Carga Soportados

- 🔵 **Normal**: Productos estándar
- 🟡 **Frágil**: Requiere cuidado especial, sin apilamiento
- ⚫ **Pesado**: Va en la base del camión
- 🟢 **Refrigerado**: Productos que requieren temperatura controlada
- 🔷 **Líquido**: Productos líquidos con restricciones de apilamiento

### 🚚 Flota de Camiones

1. **Camión Grande 53'**: 1300x250x270 cm, 20,000 kg
2. **Camión Mediano 26'**: 750x240x250 cm, 12,000 kg
3. **Camión Pequeño 16'**: 480x230x230 cm, 6,000 kg

### 🛒 Catálogo de Productos (23 productos)

El sistema incluye productos típicos de Walmart:
- Electrodomésticos (TVs, refrigeradores, lavadoras, microondas)
- Muebles (sofás, mesas, sillas)
- Alimentos y bebidas (palets, cajas, vinos, aceites)
- Productos de limpieza
- Textiles y ropa
- Electrónica (laptops, tablets)
- Juguetes y deportes
- Ferretería y pinturas

## 🚀 Instalación y Uso

### Requisitos

- Python 3.8 o superior
- pip

### Instalación

```bash
# Clonar el repositorio
git clone <repository-url>
cd mai_walmart

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar la Aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Guía de Uso

### Paso 1: Seleccionar Camión
En el panel lateral, elige el tipo de camión según tus necesidades de capacidad.

### Paso 2: Seleccionar Productos
- Filtra por tipo de carga si lo deseas
- Expande el catálogo de productos
- Ingresa la cantidad de cada producto que deseas cargar

### Paso 3: Optimizar
Presiona el botón **"🚀 Optimizar Carga"** para ejecutar el algoritmo.

### Paso 4: Analizar Resultados
Revisa:
- **Métricas principales**: utilización de volumen y peso
- **Visualización 3D**: posición exacta de cada producto
- **Productos cargados**: lista detallada con posiciones
- **Productos no cargados**: items que no cupieron
- **Análisis detallado**: gráficos y recomendaciones

### Paso 5: Exportar
Descarga los reportes en formato CSV o TXT para documentación.

## 🧮 Algoritmo de Optimización

### Estrategia de Carga

1. **Ordenamiento Inteligente**:
   - Prioridad alta primero
   - Productos pesados antes (para base estable)
   - Productos más grandes primero
   - Frágiles al final (para protección)

2. **Posicionamiento**:
   - Búsqueda de posición óptima por score
   - Preferencia por posiciones bajas y frontales
   - Prueba de 6 rotaciones posibles por producto

3. **Validaciones**:
   - Verificación de colisiones
   - Chequeo de soporte estructural (60% mínimo)
   - Validación de peso sobre productos apilados
   - Restricciones especiales por tipo

### Restricciones Implementadas

- ✅ Límites de dimensiones del camión
- ✅ Peso máximo del camión
- ✅ Detección de colisiones 3D
- ✅ Soporte estructural obligatorio
- ✅ Restricciones de apilamiento por tipo
- ✅ Peso máximo sobre cada producto
- ✅ Productos frágiles sin peso encima
- ✅ Productos pesados en zona baja (<30% altura)

## 🏗️ Arquitectura del Código

```
app.py
├── Modelos de Datos
│   ├── CargoType (Enum)
│   ├── Position (Dataclass)
│   ├── Cargo (Dataclass)
│   └── Truck (Dataclass)
│
├── Algoritmo de Bin Packing
│   └── BinPacking3D (Class)
│       ├── can_place()
│       ├── find_best_position()
│       ├── pack()
│       └── métodos auxiliares
│
├── Motor de Optimización
│   └── LoadingOptimizer (Class)
│       ├── optimize_loading()
│       ├── optimize_multi_truck()
│       └── _sort_cargos()
│
├── Visualización
│   └── TruckVisualizer (Class)
│       ├── create_3d_view()
│       ├── _add_truck_frame()
│       └── _add_box()
│
├── Datos Hardcodeados
│   ├── get_walmart_products()
│   └── get_walmart_trucks()
│
└── Interfaz Streamlit
    └── main()
```

## 📊 Métricas y Análisis

El sistema proporciona:

- **Utilización de Volumen**: % del espacio del camión ocupado
- **Utilización de Peso**: % de la capacidad de peso utilizada
- **Distribución por Tipo**: Cantidad y peso de cada tipo de carga
- **Distribución de Altura**: Items por rango de altura
- **Balance de Peso**: Análisis frente/atrás
- **Recomendaciones**: Sugerencias automáticas de optimización

## 🔧 Personalización

### Agregar Nuevos Productos

Edita la función `get_walmart_products()` en `app.py`:

```python
Cargo(
    id="PROD001",
    name="Mi Producto",
    length=100,  # cm
    width=80,    # cm
    height=60,   # cm
    weight=50,   # kg
    cargo_type=CargoType.NORMAL,
    priority=3,
    stackable=True,
    max_stack_weight=200.0
)
```

### Agregar Nuevos Camiones

Edita la función `get_walmart_trucks()`:

```python
Truck(
    id="TRK004",
    name="Mi Camión",
    length=1000,  # cm
    width=250,    # cm
    height=300,   # cm
    max_weight=15000  # kg
)
```

## 🎨 Mejoras Implementadas

Este MVP incluye varias mejoras sobre un sistema básico:

1. **Algoritmo Sofisticado**: Bin packing 3D con múltiples rotaciones
2. **Física Realista**: Soporte estructural y restricciones de apilamiento
3. **UI Moderna**: Interfaz intuitiva con Streamlit
4. **Visualización Profesional**: Gráficos 3D interactivos
5. **Análisis Completo**: Dashboard con métricas y recomendaciones
6. **Datos Realistas**: Catálogo basado en productos reales de Walmart
7. **Exportación**: Reportes en múltiples formatos

## 🚀 Próximas Mejoras (Roadmap)

- [ ] Soporte para múltiples camiones simultáneos
- [ ] Integración con base de datos
- [ ] API REST para integración externa
- [ ] Optimización por ruta de entrega
- [ ] ML para predicción de tiempos de carga
- [ ] Modo histórico y analytics
- [ ] Soporte para pallets estándar
- [ ] Generación automática de instrucciones de carga

## 📝 Licencia

MIT License - ver archivo [LICENSE](LICENSE)

## 👤 Autor

Proyecto desarrollado para Walmart Logistics

## 🤝 Contribuciones

Este es un MVP. Las contribuciones son bienvenidas mediante pull requests.

## 📞 Soporte

Para preguntas o issues, por favor abre un issue en el repositorio.

---

**Desarrollado con ❤️ usando Python, Streamlit y Plotly**