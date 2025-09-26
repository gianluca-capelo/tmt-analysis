from matplotlib import pyplot as plt
from neurotask.tmt.metrics.speed_metrics import calculate_accelerations_between_cursor_positions
from neurotask.tmt.metrics.speed_metrics import calculate_speeds_between_cursor_positions
from neurotask.tmt.model.tmt_model import TMTTrial


def plot_with_color(trial: TMTTrial, target_radius: float, color_by=None, show_start=False, cmap_name='Blues'):
    # Extraer la posición de los targets
    target_x = [target.position.x for target in trial.stimuli]
    target_y = [target.position.y for target in trial.stimuli]
    target_contents = [target.content for target in trial.stimuli]

    cursol_trail = trial.cursor_trail  # .get_cursor_trail_from_start()

    # Extraer la trayectoria del cursor
    cursor_x = [cursor_info.position.x for cursor_info in cursol_trail]
    cursor_y = [cursor_info.position.y for cursor_info in cursol_trail]
    cursor_times = [cursor_info.time for cursor_info in cursol_trail]

    if color_by is not None:
        colors, norm = get_cursor_colors(cmap_name, color_by, cursor_times, trial)
    else:
        colors, norm = None, None

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 8))

    # Dibujar la trayectoria del cursor
    for i in range(len(cursor_x) - 1):
        plt.plot(
            [cursor_x[i], cursor_x[i + 1]],
            [cursor_y[i], cursor_y[i + 1]],
            color=colors[i] if colors is not None else "black",
            linewidth=1,
            zorder=4
        )

    # Solo agregar barra de color si hay colores
    if colors is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
        sm.set_array([])
        cbar_label = (
            'Time' if color_by == 'time'
            else 'Speed' if color_by == 'speed'
            else 'Acceleration'
        )
        fig.colorbar(sm, ax=ax, label=cbar_label)

    # Dibujar los targets
    for x, y, content in zip(target_x, target_y, target_contents):
        circle = plt.Circle((x, y), target_radius, color='steelblue', alpha=0.3, zorder=5)
        ax.add_patch(circle)
        plt.text(x, y, content, color='black', fontsize=8, ha='center', va='center', zorder=6)

    # Labels y formato
    ax.set_xlabel('X screen coordinate (pixels)')
    ax.set_ylabel('Y screen coordinate (pixels)')
    ax.set_title('Cursor tracking during a TMT trial')

    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Destacar el primer clic
    if trial.start and show_start:
        plot_trial_start(ax, trial)

    return fig


def plot_trial_start(ax, trial):
    fc_x = trial.start.position.x
    fc_y = trial.start.position.y
    plt.scatter(fc_x, fc_y, color='cyan', edgecolor='black', s=100,
                label='Trial start', zorder=7, marker='o', alpha=0.3)
    # Add legend
    ax.legend()


def get_cursor_colors(cmap_name, color_by, cursor_times, trial):
    if color_by == 'time':
        norm = plt.Normalize(min(cursor_times), max(cursor_times))
        colors = plt.get_cmap(cmap_name)(norm(cursor_times))
    elif color_by == 'speed':
        speeds = calculate_speeds_between_cursor_positions(trial)
        speeds = [0] + speeds  # Para igualar el número de puntos con las velocidades calculadas
        norm = plt.Normalize(min(speeds), max(speeds))
        colors = plt.get_cmap(cmap_name)(norm(speeds))  # Usar un mapa de colores para la velocidad
    elif color_by == 'acceleration':
        accelerations = calculate_accelerations_between_cursor_positions(trial)
        accelerations = [0, 0] + accelerations  # Igualar el número de puntos (2 primeros puntos sin aceleración)
        norm = plt.Normalize(min(accelerations), max(accelerations))
        colors = plt.get_cmap(cmap_name)(norm(accelerations))
    else:
        raise ValueError("El parámetro color_by debe ser 'time', 'speed', 'acceleration'")
    return colors, norm
