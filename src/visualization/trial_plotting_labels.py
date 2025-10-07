import matplotlib.pyplot as plt
from neurotask.tmt.model.tmt_model import TMTTrial


def heigth_to_pixels(x, y):
    x = ((x + 8 / 9) / (16 / 9)) * 1920
    y = (0.5 - y) * 1080
    return x, y

def height_radius_to_pixels(r):
    return r * 1080  # porque el eje height total es 1.0 -> 1080px

def plot_with_labels_scatter(trial: TMTTrial, target_radius: float, labels: list[str], labels_title="Labels",
                                   title="Cursor tracking during a TMT trial", cmap_name='tab10', plot_start=False):

    # Validación
    cursor_trail = trial.get_cursor_trail_from_start()
    if len(labels) != len(cursor_trail):
        raise ValueError(
            "La cantidad de etiquetas debe coincidir con la cantidad de puntos en la trayectoria del cursor.")

    # Convertir posiciones del cursor
    cursor_coords = [heigth_to_pixels(p.position.x, p.position.y) for p in cursor_trail]
    cursor_x, cursor_y = zip(*cursor_coords)

    # Asignar colores únicos a cada etiqueta
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap(cmap_name) if len(unique_labels) <= 10 else plt.get_cmap('tab20')
    label_to_color = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}
    point_colors = [label_to_color[label] for label in labels]

    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))  # proporción 16:9
    ax.scatter(cursor_x, cursor_y, c=point_colors, s=20, zorder=4)

    # Dibujar los targets (convertidos también)
    for target in trial.stimuli:
        tx, ty = heigth_to_pixels(target.position.x, target.position.y)
        circle = plt.Circle((tx, ty), height_radius_to_pixels(target_radius), color='steelblue', alpha=0.3, zorder=5)
        ax.add_patch(circle)
        ax.text(tx, ty, target.content, color='black', fontsize=8, ha='center', va='center', zorder=6)

    # Marcar el primer clic
    if plot_start and trial.start:
        sx, sy = heigth_to_pixels(trial.start.position.x, trial.start.position.y)
        ax.scatter(sx, sy, color='cyan', edgecolor='black', s=100, marker='o', alpha=0.3, label='First Click', zorder=7)

    # Leyenda
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                          markerfacecolor=color, markersize=8)
               for label, color in label_to_color.items()]
    ax.legend(handles=handles, title=labels_title)

    # Estética
    ax.set_xlabel('X screen coordinate (pixels)')
    ax.set_ylabel('Y screen coordinate (pixels)')
    ax.set_title(title)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)  # invertir eje Y (como en pantalla)
    ax.set_aspect('equal', adjustable='box')

    return fig

def plot_with_labels(trial: TMTTrial, target_radius: float, labels: list[str], labels_title="Labels",
                     title="TMT Trial Trajectory", cmap_name='tab10'):
    """
    Dibuja la trayectoria del cursor de un trial del TMT, coloreando cada segmento
    según una etiqueta categórica (ej: "TRAVEL", "SEARCH", etc.).

    Parameters:
    - trial: TMTTrial, contiene los targets y la trayectoria del cursor.
    - target_radius: float, radio de los círculos que rodean los targets.
    - labels: list[str], una etiqueta por punto de la trayectoria del cursor.
    """

    # Validar que la cantidad de etiquetas coincida con los puntos de la trayectoria
    cursor_trail = trial.get_cursor_trail_from_start()
    if len(labels) != len(cursor_trail):
        raise ValueError(
            "La cantidad de etiquetas debe coincidir con la cantidad de puntos en la trayectoria del cursor.")

    # Extraer posiciones de los targets
    target_x = [target.position.x for target in trial.stimuli]
    target_y = [target.position.y for target in trial.stimuli]
    target_contents = [target.content for target in trial.stimuli]

    # Extraer posiciones del cursor
    cursor_x = [point.position.x for point in cursor_trail]
    cursor_y = [point.position.y for point in cursor_trail]

    # Generar colores para cada etiqueta única
    unique_labels = list(sorted(set(labels)))
    cmap = plt.get_cmap(cmap_name) if len(unique_labels) <= 10 else plt.get_cmap('tab20')
    label_to_color = {label: cmap(i % cmap.N) for i, label in enumerate(unique_labels)}

    # Crear figura
    fig, ax = plt.subplots(figsize=(8, 8))

    # Dibujar la trayectoria por tramos, coloreados por etiqueta
    for i in range(len(cursor_x) - 1):
        label = labels[i]
        color = label_to_color[label]
        ax.plot([cursor_x[i], cursor_x[i + 1]],
                [cursor_y[i], cursor_y[i + 1]],
                color=color, linewidth=2, zorder=4)

    # Dibujar targets
    for x, y, content in zip(target_x, target_y, target_contents):
        circle = plt.Circle((x, y), target_radius, color='steelblue', alpha=0.3, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, content, color='black', fontsize=8, ha='center', va='center', zorder=6)

    # Marcar primer clic
    if trial.start:
        fc_x = trial.start.position.x
        fc_y = trial.start.position.y
        ax.scatter(fc_x, fc_y, color='cyan', edgecolor='black', s=100, label='First Click', zorder=7,
                   marker='o', alpha=0.3)

    # Crear leyenda para etiquetas
    handles = [plt.Line2D([0], [0], color=color, lw=2, label=label) for label, color in label_to_color.items()]
    ax.legend(handles=handles, title=labels_title)

    # Estética
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')

    return fig
