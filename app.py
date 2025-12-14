"""
Walmart Advanced Truck Loading Tool
Herramienta Avanzada de Carga de Camiones para Walmart
MVP con datos hardcodeados
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import copy


# ============================================================================
# MODELOS DE DATOS / DATA MODELS
# ============================================================================

class CargoType(Enum):
    """Tipos de mercancía"""
    NORMAL = "Normal"
    FRAGIL = "Frágil"
    PESADO = "Pesado"
    REFRIGERADO = "Refrigerado"
    LIQUIDO = "Líquido"


@dataclass
class Position:
    """Posición 3D de un item en el camión"""
    x: float
    y: float
    z: float


@dataclass
class Cargo:
    """Representa una carga/paquete"""
    id: str
    name: str
    length: float  # largo (cm)
    width: float   # ancho (cm)
    height: float  # alto (cm)
    weight: float  # peso (kg)
    cargo_type: CargoType
    priority: int = 1  # 1-5, mayor = más prioritario
    stackable: bool = True
    max_stack_weight: float = 500.0  # kg máximos encima
    position: Optional[Position] = None
    rotation: Tuple[int, int, int] = (0, 0, 0)  # rotación aplicada

    @property
    def volume(self) -> float:
        """Volumen en cm³"""
        return self.length * self.width * self.height

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Dimensiones actuales (l, w, h)"""
        return (self.length, self.width, self.height)

    def get_rotations(self) -> List[Tuple[float, float, float]]:
        """Retorna todas las rotaciones posibles"""
        l, w, h = self.length, self.width, self.height
        return [
            (l, w, h),
            (l, h, w),
            (w, l, h),
            (w, h, l),
            (h, l, w),
            (h, w, l)
        ]


@dataclass
class Truck:
    """Representa un camión"""
    id: str
    name: str
    length: float  # cm
    width: float   # cm
    height: float  # cm
    max_weight: float  # kg
    loaded_cargo: List[Cargo] = field(default_factory=list)

    @property
    def volume(self) -> float:
        """Volumen total en cm³"""
        return self.length * self.width * self.height

    @property
    def used_volume(self) -> float:
        """Volumen usado"""
        return sum(c.volume for c in self.loaded_cargo)

    @property
    def used_weight(self) -> float:
        """Peso usado"""
        return sum(c.weight for c in self.loaded_cargo)

    @property
    def volume_utilization(self) -> float:
        """% de utilización de volumen"""
        return (self.used_volume / self.volume) * 100 if self.volume > 0 else 0

    @property
    def weight_utilization(self) -> float:
        """% de utilización de peso"""
        return (self.used_weight / self.max_weight) * 100 if self.max_weight > 0 else 0


# ============================================================================
# ALGORITMO DE BIN PACKING 3D
# ============================================================================

class BinPacking3D:
    """Algoritmo avanzado de empaquetado 3D"""

    def __init__(self, truck: Truck):
        self.truck = truck
        self.space_map: List[Dict] = []  # espacios disponibles
        self._init_space()

    def _init_space(self):
        """Inicializa el espacio disponible"""
        self.space_map = [{
            'position': Position(0, 0, 0),
            'length': self.truck.length,
            'width': self.truck.width,
            'height': self.truck.height
        }]

    def can_place(self, cargo: Cargo, position: Position,
                  dimensions: Tuple[float, float, float]) -> bool:
        """Verifica si se puede colocar una carga en una posición"""
        l, w, h = dimensions

        # Verificar límites del camión
        if (position.x + l > self.truck.length or
            position.y + w > self.truck.width or
            position.z + h > self.truck.height):
            return False

        # Verificar peso total
        if self.truck.used_weight + cargo.weight > self.truck.max_weight:
            return False

        # Verificar colisiones con cargas existentes
        for loaded in self.truck.loaded_cargo:
            if self._check_collision(
                position, dimensions,
                loaded.position, loaded.dimensions
            ):
                return False

        # Verificar soporte (la carga debe estar apoyada)
        if position.z > 0:
            if not self._has_support(position, dimensions):
                return False

        # Verificar restricciones de apilamiento
        if not self._check_stacking_rules(cargo, position, dimensions):
            return False

        return True

    def _check_collision(self, pos1: Position, dim1: Tuple[float, float, float],
                        pos2: Position, dim2: Tuple[float, float, float]) -> bool:
        """Verifica si dos cajas colisionan"""
        if pos2 is None:
            return False

        return not (
            pos1.x + dim1[0] <= pos2.x or pos2.x + dim2[0] <= pos1.x or
            pos1.y + dim1[1] <= pos2.y or pos2.y + dim2[1] <= pos1.y or
            pos1.z + dim1[2] <= pos2.z or pos2.z + dim2[2] <= pos1.z
        )

    def _has_support(self, position: Position,
                     dimensions: Tuple[float, float, float]) -> bool:
        """Verifica que la carga tenga soporte debajo"""
        l, w, h = dimensions
        support_area = 0
        total_area = l * w

        # Buscar cajas debajo
        for loaded in self.truck.loaded_cargo:
            if loaded.position is None:
                continue

            # Verificar si está justo debajo
            if abs(loaded.position.z + loaded.dimensions[2] - position.z) < 1:
                # Calcular área de soporte
                overlap = self._calculate_overlap_area(
                    position, dimensions,
                    loaded.position, loaded.dimensions
                )
                support_area += overlap

        # Requerir al menos 60% de soporte
        return support_area >= total_area * 0.6

    def _calculate_overlap_area(self, pos1: Position, dim1: Tuple[float, float, float],
                               pos2: Position, dim2: Tuple[float, float, float]) -> float:
        """Calcula el área de superposición en el plano XY"""
        x_overlap = max(0, min(pos1.x + dim1[0], pos2.x + dim2[0]) - max(pos1.x, pos2.x))
        y_overlap = max(0, min(pos1.y + dim1[1], pos2.y + dim2[1]) - max(pos1.y, pos2.y))
        return x_overlap * y_overlap

    def _check_stacking_rules(self, cargo: Cargo, position: Position,
                             dimensions: Tuple[float, float, float]) -> bool:
        """Verifica reglas de apilamiento"""
        # Verificar peso encima de otras cajas
        for loaded in self.truck.loaded_cargo:
            if loaded.position is None:
                continue

            # Si esta carga está encima de otra
            if abs(position.z - (loaded.position.z + loaded.dimensions[2])) < 1:
                overlap_area = self._calculate_overlap_area(
                    position, dimensions,
                    loaded.position, loaded.dimensions
                )

                if overlap_area > 0:
                    # Verificar si la caja de abajo puede soportar el peso
                    if not loaded.stackable:
                        return False
                    if cargo.weight > loaded.max_stack_weight:
                        return False

        # Reglas especiales por tipo de carga
        if cargo.cargo_type == CargoType.FRAGIL:
            # Frágiles no pueden tener nada encima (verificar que no haya nada planeado)
            # Solo aplicable si es frágil y no stackable
            if not cargo.stackable:
                return True

        if cargo.cargo_type == CargoType.PESADO:
            # Pesados deben ir en el fondo (nivel z bajo)
            if position.z > self.truck.height * 0.3:
                return False

        return True

    def find_best_position(self, cargo: Cargo) -> Optional[Tuple[Position, Tuple[float, float, float]]]:
        """Encuentra la mejor posición para una carga"""
        best_position = None
        best_rotation = None
        best_score = float('inf')

        # Probar todas las rotaciones posibles
        rotations = cargo.get_rotations()

        # Generar posiciones candidatas
        positions = self._generate_candidate_positions()

        for pos in positions:
            for rot in rotations:
                if self.can_place(cargo, pos, rot):
                    # Calcular score (preferir posiciones bajas y hacia el frente)
                    score = pos.z * 2 + pos.x + pos.y * 0.5

                    if score < best_score:
                        best_score = score
                        best_position = pos
                        best_rotation = rot

        if best_position:
            return (best_position, best_rotation)
        return None

    def _generate_candidate_positions(self) -> List[Position]:
        """Genera posiciones candidatas inteligentes"""
        positions = [Position(0, 0, 0)]

        # Agregar posiciones basadas en cargas existentes
        for loaded in self.truck.loaded_cargo:
            if loaded.position is None:
                continue

            # Posición a la derecha
            positions.append(Position(
                loaded.position.x + loaded.dimensions[0],
                loaded.position.y,
                loaded.position.z
            ))

            # Posición atrás
            positions.append(Position(
                loaded.position.x,
                loaded.position.y + loaded.dimensions[1],
                loaded.position.z
            ))

            # Posición encima
            positions.append(Position(
                loaded.position.x,
                loaded.position.y,
                loaded.position.z + loaded.dimensions[2]
            ))

        return positions

    def pack(self, cargo: Cargo) -> bool:
        """Intenta empaquetar una carga"""
        result = self.find_best_position(cargo)

        if result:
            position, rotation = result
            cargo.position = position
            cargo.length, cargo.width, cargo.height = rotation
            self.truck.loaded_cargo.append(cargo)
            return True

        return False


# ============================================================================
# MOTOR DE OPTIMIZACIÓN
# ============================================================================

class LoadingOptimizer:
    """Motor de optimización de carga"""

    @staticmethod
    def optimize_loading(cargos: List[Cargo], truck: Truck) -> Tuple[Truck, List[Cargo]]:
        """
        Optimiza la carga de un camión
        Retorna: (camión_cargado, items_no_cargados)
        """
        # Copiar datos para no modificar originales
        truck_copy = copy.deepcopy(truck)
        cargos_copy = copy.deepcopy(cargos)

        # Ordenar cargas por prioridad y características
        sorted_cargos = LoadingOptimizer._sort_cargos(cargos_copy)

        # Crear algoritmo de packing
        packer = BinPacking3D(truck_copy)

        # Intentar cargar cada item
        not_loaded = []
        for cargo in sorted_cargos:
            if not packer.pack(cargo):
                not_loaded.append(cargo)

        return truck_copy, not_loaded

    @staticmethod
    def _sort_cargos(cargos: List[Cargo]) -> List[Cargo]:
        """Ordena cargas por criterio óptimo"""
        def sort_key(cargo: Cargo):
            # Prioridad alta primero
            priority_score = -cargo.priority * 10000

            # Pesados primero (para base)
            type_score = 0
            if cargo.cargo_type == CargoType.PESADO:
                type_score = -5000
            elif cargo.cargo_type == CargoType.FRAGIL:
                type_score = 3000  # Frágiles después

            # Más grandes primero
            volume_score = -cargo.volume

            return priority_score + type_score + volume_score

        return sorted(cargos, key=sort_key)

    @staticmethod
    def optimize_multi_truck(cargos: List[Cargo],
                            trucks: List[Truck]) -> List[Tuple[Truck, List[Cargo]]]:
        """Optimiza carga en múltiples camiones"""
        results = []
        remaining_cargos = copy.deepcopy(cargos)

        for truck in trucks:
            if not remaining_cargos:
                break

            loaded_truck, not_loaded = LoadingOptimizer.optimize_loading(
                remaining_cargos, truck
            )
            results.append((loaded_truck, not_loaded))
            remaining_cargos = not_loaded

        return results


# ============================================================================
# VISUALIZACIÓN 3D
# ============================================================================

class TruckVisualizer:
    """Visualizador 3D de camiones"""

    @staticmethod
    def create_3d_view(truck: Truck) -> go.Figure:
        """Crea visualización 3D del camión"""
        fig = go.Figure()

        # Dibujar estructura del camión (wireframe)
        TruckVisualizer._add_truck_frame(fig, truck)

        # Dibujar cada carga
        colors = {
            CargoType.NORMAL: 'lightblue',
            CargoType.FRAGIL: 'yellow',
            CargoType.PESADO: 'darkgray',
            CargoType.REFRIGERADO: 'lightgreen',
            CargoType.LIQUIDO: 'cyan'
        }

        for cargo in truck.loaded_cargo:
            if cargo.position is None:
                continue

            color = colors.get(cargo.cargo_type, 'lightblue')
            TruckVisualizer._add_box(
                fig, cargo.position, cargo.dimensions,
                color, cargo.name
            )

        # Configurar layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Largo (cm)', range=[0, truck.length]),
                yaxis=dict(title='Ancho (cm)', range=[0, truck.width]),
                zaxis=dict(title='Alto (cm)', range=[0, truck.height]),
                aspectmode='data'
            ),
            title="Vista 3D del Camión",
            showlegend=True,
            height=700
        )

        return fig

    @staticmethod
    def _add_truck_frame(fig: go.Figure, truck: Truck):
        """Agrega el marco del camión"""
        # Líneas del marco
        edges = [
            # Base
            ([0, truck.length], [0, 0], [0, 0]),
            ([0, truck.length], [truck.width, truck.width], [0, 0]),
            ([0, 0], [0, truck.width], [0, 0]),
            ([truck.length, truck.length], [0, truck.width], [0, 0]),
            # Techo
            ([0, truck.length], [0, 0], [truck.height, truck.height]),
            ([0, truck.length], [truck.width, truck.width], [truck.height, truck.height]),
            ([0, 0], [0, truck.width], [truck.height, truck.height]),
            ([truck.length, truck.length], [0, truck.width], [truck.height, truck.height]),
            # Verticales
            ([0, 0], [0, 0], [0, truck.height]),
            ([truck.length, truck.length], [0, 0], [0, truck.height]),
            ([0, 0], [truck.width, truck.width], [0, truck.height]),
            ([truck.length, truck.length], [truck.width, truck.width], [0, truck.height]),
        ]

        for edge in edges:
            fig.add_trace(go.Scatter3d(
                x=edge[0], y=edge[1], z=edge[2],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

    @staticmethod
    def _add_box(fig: go.Figure, position: Position,
                 dimensions: Tuple[float, float, float],
                 color: str, name: str):
        """Agrega una caja 3D"""
        l, w, h = dimensions
        x0, y0, z0 = position.x, position.y, position.z

        # Vértices del cubo
        vertices = np.array([
            [x0, y0, z0], [x0+l, y0, z0], [x0+l, y0+w, z0], [x0, y0+w, z0],  # base
            [x0, y0, z0+h], [x0+l, y0, z0+h], [x0+l, y0+w, z0+h], [x0, y0+w, z0+h]  # techo
        ])

        # Caras del cubo
        faces = [
            [0, 1, 2, 3],  # base
            [4, 5, 6, 7],  # techo
            [0, 1, 5, 4],  # frente
            [2, 3, 7, 6],  # atrás
            [0, 3, 7, 4],  # izquierda
            [1, 2, 6, 5]   # derecha
        ]

        # Crear mesh3d
        i, j, k = [], [], []
        for face in faces:
            i.extend([face[0], face[0]])
            j.extend([face[1], face[2]])
            k.extend([face[2], face[3]])

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=i, j=j, k=k,
            color=color,
            opacity=0.7,
            name=name,
            hovertemplate=f"<b>{name}</b><br>" +
                         f"Posición: ({x0:.0f}, {y0:.0f}, {z0:.0f})<br>" +
                         f"Dimensiones: {l:.0f}x{w:.0f}x{h:.0f} cm<br>" +
                         "<extra></extra>"
        ))


# ============================================================================
# DATOS HARDCODEADOS - PRODUCTOS WALMART
# ============================================================================

def get_walmart_products() -> List[Cargo]:
    """Retorna catálogo de productos típicos de Walmart"""
    return [
        # Electrodomésticos
        Cargo("TV001", "TV 55 pulgadas", 140, 85, 15, 18, CargoType.FRAGIL,
              priority=3, stackable=False),
        Cargo("TV002", "TV 43 pulgadas", 110, 70, 12, 12, CargoType.FRAGIL,
              priority=3, stackable=False),
        Cargo("REF001", "Refrigerador", 180, 80, 70, 85, CargoType.REFRIGERADO,
              priority=5, stackable=False),
        Cargo("LAV001", "Lavadora", 85, 60, 60, 65, CargoType.PESADO,
              priority=4, stackable=False),
        Cargo("MIC001", "Microondas", 50, 45, 30, 15, CargoType.NORMAL,
              priority=2, stackable=True),

        # Muebles
        Cargo("SOF001", "Sofá 3 plazas (caja)", 220, 100, 80, 55, CargoType.NORMAL,
              priority=3, stackable=False),
        Cargo("MES001", "Mesa comedor (caja)", 160, 100, 15, 35, CargoType.PESADO,
              priority=2, stackable=True, max_stack_weight=100),
        Cargo("SIL001", "Set 4 sillas (caja)", 80, 60, 80, 25, CargoType.NORMAL,
              priority=2, stackable=True),

        # Alimentos y bebidas
        Cargo("BEB001", "Palet bebidas (24 cajas)", 120, 100, 145, 380, CargoType.PESADO,
              priority=5, stackable=True, max_stack_weight=400),
        Cargo("BEB002", "Caja refrescos (12 unid)", 40, 30, 25, 8, CargoType.NORMAL,
              priority=3, stackable=True, max_stack_weight=50),
        Cargo("ALC001", "Caja vinos (6 botellas)", 35, 25, 30, 12, CargoType.FRAGIL,
              priority=2, stackable=True, max_stack_weight=25),
        Cargo("ACE001", "Caja aceite (12 litros)", 40, 30, 35, 14, CargoType.LIQUIDO,
              priority=3, stackable=True, max_stack_weight=30),

        # Productos de limpieza
        Cargo("DET001", "Caja detergente (6 unid)", 45, 35, 30, 9, CargoType.LIQUIDO,
              priority=2, stackable=True, max_stack_weight=40),
        Cargo("LIM001", "Productos limpieza mix", 50, 40, 35, 11, CargoType.NORMAL,
              priority=1, stackable=True),

        # Textiles
        Cargo("ROB001", "Caja ropa (surtida)", 60, 40, 40, 8, CargoType.NORMAL,
              priority=1, stackable=True, max_stack_weight=60),
        Cargo("SAB001", "Paquete sábanas (10 sets)", 70, 50, 30, 12, CargoType.NORMAL,
              priority=2, stackable=True),

        # Electrónica
        Cargo("LAP001", "Laptop (caja)", 45, 35, 8, 3, CargoType.FRAGIL,
              priority=5, stackable=False),
        Cargo("TAB001", "Tablets (caja 5 unid)", 35, 25, 15, 4, CargoType.FRAGIL,
              priority=4, stackable=True, max_stack_weight=10),

        # Juguetes
        Cargo("JUG001", "Juguetes surtidos", 80, 60, 60, 15, CargoType.NORMAL,
              priority=2, stackable=True),
        Cargo("BIC001", "Bicicleta (caja)", 150, 80, 30, 18, CargoType.NORMAL,
              priority=3, stackable=False),

        # Deportes
        Cargo("DEP001", "Equipo deportivo mix", 100, 60, 50, 22, CargoType.NORMAL,
              priority=2, stackable=True),

        # Ferretería
        Cargo("HER001", "Herramientas (caja)", 60, 40, 30, 25, CargoType.PESADO,
              priority=2, stackable=True, max_stack_weight=200),
        Cargo("PIN001", "Pinturas (12 galones)", 50, 40, 35, 45, CargoType.LIQUIDO,
              priority=3, stackable=True, max_stack_weight=100),
    ]


def get_walmart_trucks() -> List[Truck]:
    """Retorna flota de camiones Walmart"""
    return [
        Truck("TRK001", "Camión Grande 53'", 1300, 250, 270, 20000),
        Truck("TRK002", "Camión Mediano 26'", 750, 240, 250, 12000),
        Truck("TRK003", "Camión Pequeño 16'", 480, 230, 230, 6000),
    ]


# ============================================================================
# INTERFAZ STREAMLIT
# ============================================================================

def main():
    st.set_page_config(
        page_title="Walmart Truck Loading Tool",
        page_icon="🚛",
        layout="wide"
    )

    # Header
    st.title("🚛 Walmart - Herramienta Avanzada de Carga de Camiones")
    st.markdown("**Sistema de Optimización 3D con Algoritmo Inteligente de Bin Packing**")
    st.divider()

    # Sidebar - Configuración
    with st.sidebar:
        st.header("⚙️ Configuración")

        # Selección de camión
        trucks = get_walmart_trucks()
        truck_names = [f"{t.name} ({t.length}x{t.width}x{t.height}cm, {t.max_weight}kg)"
                      for t in trucks]
        selected_truck_idx = st.selectbox(
            "Selecciona el camión:",
            range(len(trucks)),
            format_func=lambda i: truck_names[i]
        )
        selected_truck = trucks[selected_truck_idx]

        st.divider()

        # Selección de productos
        st.subheader("📦 Selecciona productos a cargar")
        all_products = get_walmart_products()

        # Filtros
        cargo_types = st.multiselect(
            "Filtrar por tipo:",
            [ct.value for ct in CargoType],
            default=[ct.value for ct in CargoType]
        )

        selected_cargos = []

        with st.expander("🛒 Catálogo de productos", expanded=True):
            for product in all_products:
                if product.cargo_type.value in cargo_types:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        label = f"{product.name} ({product.cargo_type.value})"
                        st.caption(f"📐 {product.length}x{product.width}x{product.height}cm | ⚖️ {product.weight}kg")
                    with col2:
                        quantity = st.number_input(
                            "Cant.",
                            min_value=0,
                            max_value=20,
                            value=0,
                            key=product.id,
                            label_visibility="collapsed"
                        )

                    # Agregar copias según cantidad
                    for i in range(quantity):
                        cargo_copy = copy.deepcopy(product)
                        cargo_copy.id = f"{product.id}_{i+1}"
                        selected_cargos.append(cargo_copy)

        st.divider()

        # Botón optimizar
        optimize_button = st.button("🚀 Optimizar Carga", type="primary", use_container_width=True)

    # Main content
    if not optimize_button:
        # Pantalla inicial
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Camión seleccionado", selected_truck.name)
        with col2:
            st.metric("Capacidad", f"{selected_truck.max_weight:,.0f} kg")
        with col3:
            st.metric("Volumen", f"{selected_truck.volume/1_000_000:.1f} m³")

        st.info("👈 Selecciona productos en el panel lateral y presiona 'Optimizar Carga'")

        # Mostrar camión vacío
        st.subheader("Vista 3D del Camión")
        empty_truck = copy.deepcopy(selected_truck)
        fig = TruckVisualizer.create_3d_view(empty_truck)
        st.plotly_chart(fig, use_container_width=True)

    else:
        if not selected_cargos:
            st.warning("⚠️ No has seleccionado ningún producto para cargar.")
        else:
            # Ejecutar optimización
            with st.spinner("🔄 Optimizando carga del camión..."):
                loaded_truck, not_loaded = LoadingOptimizer.optimize_loading(
                    selected_cargos,
                    copy.deepcopy(selected_truck)
                )

            # Métricas principales
            st.subheader("📊 Resultados de la Optimización")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(
                    "Productos cargados",
                    len(loaded_truck.loaded_cargo),
                    delta=f"{len(selected_cargos)} solicitados"
                )
            with col2:
                st.metric(
                    "Utilización volumen",
                    f"{loaded_truck.volume_utilization:.1f}%"
                )
            with col3:
                st.metric(
                    "Utilización peso",
                    f"{loaded_truck.weight_utilization:.1f}%"
                )
            with col4:
                st.metric(
                    "Peso cargado",
                    f"{loaded_truck.used_weight:,.0f} kg",
                    delta=f"{loaded_truck.max_weight - loaded_truck.used_weight:,.0f} kg disponible"
                )
            with col5:
                st.metric(
                    "No cargados",
                    len(not_loaded),
                    delta="items rechazados" if not_loaded else "¡Todo cargado!",
                    delta_color="inverse"
                )

            # Visualización 3D
            st.subheader("🎯 Vista 3D del Camión Cargado")
            fig = TruckVisualizer.create_3d_view(loaded_truck)
            st.plotly_chart(fig, use_container_width=True)

            # Detalles de carga
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("✅ Productos Cargados")
                if loaded_truck.loaded_cargo:
                    cargo_data = []
                    for cargo in loaded_truck.loaded_cargo:
                        cargo_data.append({
                            "ID": cargo.id,
                            "Producto": cargo.name,
                            "Tipo": cargo.cargo_type.value,
                            "Dimensiones (cm)": f"{cargo.length:.0f}x{cargo.width:.0f}x{cargo.height:.0f}",
                            "Peso (kg)": f"{cargo.weight:.1f}",
                            "Posición (x,y,z)": f"({cargo.position.x:.0f}, {cargo.position.y:.0f}, {cargo.position.z:.0f})" if cargo.position else "N/A",
                            "Prioridad": cargo.priority
                        })

                    df_loaded = pd.DataFrame(cargo_data)
                    st.dataframe(df_loaded, use_container_width=True, hide_index=True)
                else:
                    st.info("No se cargaron productos.")

            with col_right:
                st.subheader("❌ Productos No Cargados")
                if not_loaded:
                    not_loaded_data = []
                    for cargo in not_loaded:
                        not_loaded_data.append({
                            "ID": cargo.id,
                            "Producto": cargo.name,
                            "Tipo": cargo.cargo_type.value,
                            "Dimensiones (cm)": f"{cargo.length:.0f}x{cargo.width:.0f}x{cargo.height:.0f}",
                            "Peso (kg)": f"{cargo.weight:.1f}",
                            "Volumen (m³)": f"{cargo.volume/1_000_000:.3f}",
                            "Prioridad": cargo.priority
                        })

                    df_not_loaded = pd.DataFrame(not_loaded_data)
                    st.dataframe(df_not_loaded, use_container_width=True, hide_index=True)

                    st.warning(f"⚠️ {len(not_loaded)} productos no pudieron ser cargados debido a restricciones de espacio, peso o apilamiento.")
                else:
                    st.success("✅ ¡Todos los productos fueron cargados exitosamente!")

            # Análisis adicional
            st.divider()
            st.subheader("📈 Análisis Detallado")

            tab1, tab2, tab3 = st.tabs(["Distribución por Tipo", "Distribución de Peso", "Recomendaciones"])

            with tab1:
                # Gráfico de distribución por tipo
                type_counts = {}
                type_weights = {}
                for cargo in loaded_truck.loaded_cargo:
                    ct = cargo.cargo_type.value
                    type_counts[ct] = type_counts.get(ct, 0) + 1
                    type_weights[ct] = type_weights.get(ct, 0) + cargo.weight

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Cantidad por Tipo de Carga**")
                    if type_counts:
                        df_types = pd.DataFrame({
                            "Tipo": list(type_counts.keys()),
                            "Cantidad": list(type_counts.values())
                        })
                        st.bar_chart(df_types.set_index("Tipo"))

                with col_b:
                    st.write("**Peso por Tipo de Carga (kg)**")
                    if type_weights:
                        df_weights = pd.DataFrame({
                            "Tipo": list(type_weights.keys()),
                            "Peso": list(type_weights.values())
                        })
                        st.bar_chart(df_weights.set_index("Tipo"))

            with tab2:
                # Distribución de alturas
                st.write("**Distribución de Cargas por Altura**")
                if loaded_truck.loaded_cargo:
                    height_ranges = {"0-90cm": 0, "90-180cm": 0, "180-270cm": 0}
                    for cargo in loaded_truck.loaded_cargo:
                        if cargo.position:
                            z = cargo.position.z
                            if z < 90:
                                height_ranges["0-90cm"] += 1
                            elif z < 180:
                                height_ranges["90-180cm"] += 1
                            else:
                                height_ranges["180-270cm"] += 1

                    df_heights = pd.DataFrame({
                        "Rango de Altura": list(height_ranges.keys()),
                        "Cantidad": list(height_ranges.values())
                    })
                    st.bar_chart(df_heights.set_index("Rango de Altura"))

            with tab3:
                st.write("**💡 Recomendaciones del Sistema**")

                recommendations = []

                # Analizar utilización
                if loaded_truck.volume_utilization < 60:
                    recommendations.append("🔸 La utilización de volumen es baja (<60%). Considera agregar más productos pequeños.")
                elif loaded_truck.volume_utilization > 90:
                    recommendations.append("🔸 ¡Excelente! Utilización de volumen >90%.")

                if loaded_truck.weight_utilization < 50:
                    recommendations.append("🔸 La utilización de peso es baja (<50%). Puedes agregar productos más pesados.")
                elif loaded_truck.weight_utilization > 85:
                    recommendations.append("🔸 ¡Muy bien! Utilización de peso >85%.")

                # Analizar no cargados
                if not_loaded:
                    fragile_not_loaded = sum(1 for c in not_loaded if c.cargo_type == CargoType.FRAGIL)
                    if fragile_not_loaded > 0:
                        recommendations.append(f"🔸 {fragile_not_loaded} productos frágiles no pudieron cargarse. Considera usar un camión adicional o reconfigurar el orden de carga.")

                    heavy_not_loaded = sum(1 for c in not_loaded if c.cargo_type == CargoType.PESADO)
                    if heavy_not_loaded > 0:
                        recommendations.append(f"🔸 {heavy_not_loaded} productos pesados no cupieron. Verifica las dimensiones y considera distribuirlos en varios camiones.")

                # Verificar balance
                total_front_weight = sum(c.weight for c in loaded_truck.loaded_cargo
                                        if c.position and c.position.x < loaded_truck.length / 2)
                total_back_weight = loaded_truck.used_weight - total_front_weight

                if abs(total_front_weight - total_back_weight) > loaded_truck.used_weight * 0.3:
                    recommendations.append("⚠️ El peso está desbalanceado entre frente y parte trasera. Considera redistribuir la carga.")
                else:
                    recommendations.append("✅ La distribución de peso está balanceada.")

                if not recommendations:
                    recommendations.append("✅ ¡La carga está óptimamente configurada!")

                for rec in recommendations:
                    st.write(rec)

            # Botón de descarga
            st.divider()
            col_download1, col_download2, col_download3 = st.columns([1, 1, 2])

            with col_download1:
                # Preparar CSV de carga
                if loaded_truck.loaded_cargo:
                    csv_data = df_loaded.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Reporte CSV",
                        data=csv_data,
                        file_name=f"carga_{loaded_truck.id}.csv",
                        mime="text/csv"
                    )

            with col_download2:
                # Generar resumen
                summary = f"""
REPORTE DE CARGA - WALMART
{'='*50}
Camión: {loaded_truck.name}
Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

MÉTRICAS:
- Productos cargados: {len(loaded_truck.loaded_cargo)}/{len(selected_cargos)}
- Utilización volumen: {loaded_truck.volume_utilization:.1f}%
- Utilización peso: {loaded_truck.weight_utilization:.1f}%
- Peso total: {loaded_truck.used_weight:.0f} kg / {loaded_truck.max_weight:.0f} kg
- Volumen usado: {loaded_truck.used_volume/1_000_000:.2f} m³ / {loaded_truck.volume/1_000_000:.2f} m³

PRODUCTOS NO CARGADOS: {len(not_loaded)}
"""
                st.download_button(
                    label="📄 Descargar Resumen TXT",
                    data=summary,
                    file_name=f"resumen_{loaded_truck.id}.txt",
                    mime="text/plain"
                )


if __name__ == "__main__":
    main()
