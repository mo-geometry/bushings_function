"""Bushings Function — GLFW + PyOpenGL + Dear ImGui renderer.

Renders surfaces of revolution from Bushing's function with Blinn-Phong
shading, a ground-plane grid, mouse-driven orbit camera, and a Dear ImGui
control panel for adjusting parameters (η, λ, nF) in real time.

This uses raw PyOpenGL calls (not ModernGL) so the code maps directly
to C/C++ OpenGL tutorials and is suitable for GPU shader learning.

Run with:
    python -m bushings_function
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import glfw  # type: ignore[import-untyped]
import imgui  # type: ignore[import-untyped]
import numpy as np
from imgui.integrations.glfw import GlfwRenderer  # type: ignore[import-untyped]
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_BACK,
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_FRONT,
    GL_LINES,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_SRC_ALPHA,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glBindVertexArray,
    glBlendFunc,
    glClear,
    glClearColor,
    glCullFace,
    glDrawArrays,
    glDrawElements,
    glEnable,
    glGetUniformLocation,
    glUniform3fv,
    glUseProgram,
    glViewport,
)

from bushings_function.assets.grid import create_grid
from bushings_function.assets.sor_mesh import create_sor_mesh
from bushings_function.core.bushings import SurfaceType
from bushings_function.core.orbit_camera import OrbitCamera
from bushings_function.io.export_mesh import save_as_obj, save_as_ply
from bushings_function.rendering.gl_utils import (
    create_line_vao,
    create_mesh_vao,
    create_program,
    set_uniform_float,
    set_uniform_int,
    set_uniform_mat4,
    set_uniform_vec3,
)
from bushings_function.rendering.shaders import (
    FRAGMENT_SHADER,
    LINE_FRAGMENT_SHADER,
    LINE_VERTEX_SHADER,
    VERTEX_SHADER,
)

# Window dimensions
WIDTH = 1280
HEIGHT = 720
TITLE = "Bushings Function — PyOpenGL + ImGui"
CONFIG_FILE = Path.home() / ".bushings_function_config.json"


def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:  # type: ignore[type-arg]
    """Build a 4x4 perspective projection matrix (row-major)."""
    f = 1.0 / np.tan(fov_y / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[3, 2] = -1.0
    proj[2, 3] = (2.0 * far * near) / (near - far)
    return proj


def _load_config() -> dict[str, object]:
    """Load user configuration from disk."""
    try:
        if CONFIG_FILE.exists():
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_config(config: dict[str, object]) -> None:
    """Save user configuration to disk."""
    with contextlib.suppress(OSError):
        CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _get_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _get_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _get_list(config: dict[str, object], key: str, default: Sequence[object]) -> list[object]:
    value = config.get(key, default)
    if isinstance(value, list):
        return value
    return list(default)


def _sync_export_extension(filename: str, format_idx: int, formats: list[str]) -> str:
    """Return filename with extension updated to match the selected format.

    Preserves the user-typed stem; only the suffix changes.
    """
    stem = Path(filename).stem
    ext = "." + formats[format_idx].lower()
    return stem + ext


def _get_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def main() -> None:
    """Launch the windowed renderer with ImGui controls."""
    # --- GLFW initialisation ---
    if not glfw.init():
        sys.exit("Failed to initialise GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.SAMPLES, 4)

    window = glfw.create_window(WIDTH, HEIGHT, TITLE, None, None)
    if not window:
        glfw.terminate()
        sys.exit("Failed to create GLFW window")

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Vsync

    # --- Dear ImGui initialisation ---
    imgui.create_context()  # type: ignore[attr-defined]
    impl = GlfwRenderer(window)

    # --- OpenGL state ---
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_CULL_FACE)

    fb_width, fb_height = glfw.get_framebuffer_size(window)
    glViewport(0, 0, fb_width, fb_height)

    print(f"Framebuffer: {fb_width}x{fb_height}")

    # --- Compile shaders ---
    prog = create_program(VERTEX_SHADER, FRAGMENT_SHADER)
    line_prog = create_program(LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER)

    # --- Initial user configuration ---
    config = _load_config()

    eta = _get_float(config, "eta", 5.0)
    lam = _get_int(config, "lam", 11)
    n_f = _get_int(config, "n_f", 23)
    surface_idx = _get_int(config, "surface_idx", 0)
    surface_types = [
        SurfaceType.PANCAKE,
        SurfaceType.CIGAR,
        SurfaceType.SPHERE,
    ]
    surface_labels = [
        "Bushing pancake",
        "Bushing cigar",
        "Sphere",
    ]
    color_scheme_idx = _get_int(config, "color_scheme_idx", 0)
    color_schemes = ["uniform", "theta_gradient", "radius_gradient", "height_gradient"]
    color_scheme_labels = ["Uniform", "Theta Gradient", "Radius Gradient", "Height Gradient"]
    alpha_val = _get_float(config, "alpha_val", 0.85)
    mesh_slices = _get_int(config, "mesh_slices", 64)
    mesh_stacks = _get_int(config, "mesh_stacks", 32)

    surface_idx = max(0, min(surface_idx, len(surface_types) - 1))
    color_scheme_idx = max(0, min(color_scheme_idx, len(color_schemes) - 1))

    needs_rebuild = True
    show_imgui = _get_bool(config, "show_imgui", True)

    mesh_vao = 0
    mesh_index_count = 0

    # Keep the last generated mesh in memory for export
    last_positions = None
    last_normals = None
    last_colors = None
    last_indices = None

    # Export UI state
    export_formats = ["PLY", "OBJ"]
    export_format_idx = 0
    export_filename = "bushings_mesh.ply"
    export_status = ""

    # --- Create ground grid ---
    grid_positions, grid_colors = create_grid(size=5.0, divisions=10)
    grid_vao, grid_vertex_count = create_line_vao(grid_positions, grid_colors)

    # --- Camera setup ---
    camera = OrbitCamera(distance=4.0, azimuth=0.5, elevation=0.4)

    aspect = fb_width / fb_height
    fov_y = float(np.radians(60.0))
    near, far = 0.1, 100.0
    proj = _perspective(fov_y, aspect, near, far)
    model = np.eye(4, dtype=np.float32)

    # --- GLFW callbacks (only active when ImGui doesn't capture input) ---
    def _mouse_button_cb(
        _win: object, button: int, action: int, mods: int
    ) -> None:
        nonlocal show_imgui
        if button == glfw.MOUSE_BUTTON_MIDDLE and action == glfw.PRESS:
            show_imgui = not show_imgui
            return

        if not imgui.get_io().want_capture_mouse:  # type: ignore[attr-defined]
            camera.on_mouse_button(button, action, mods)

    def _cursor_pos_cb(_win: object, x: float, y: float) -> None:
        if not imgui.get_io().want_capture_mouse:  # type: ignore[attr-defined]
            camera.on_cursor_pos(x, y)
        else:
            # Still track position so drag doesn't jump when leaving ImGui
            camera._last_x = x
            camera._last_y = y

    def _scroll_cb(_win: object, x_offset: float, y_offset: float) -> None:
        if not imgui.get_io().want_capture_mouse:  # type: ignore[attr-defined]
            camera.on_scroll(x_offset, y_offset)

    def _framebuffer_size_cb(_win: object, width: int, height: int) -> None:
        nonlocal proj, aspect
        if height == 0:
            return
        glViewport(0, 0, width, height)
        aspect = width / height
        proj = _perspective(fov_y, aspect, near, far)

    glfw.set_mouse_button_callback(window, _mouse_button_cb)
    glfw.set_cursor_pos_callback(window, _cursor_pos_cb)
    glfw.set_scroll_callback(window, _scroll_cb)
    glfw.set_framebuffer_size_callback(window, _framebuffer_size_cb)

    # --- Light setup (4 lights) ---
    default_lights_pos = [
        [5.0, 8.0, 5.0],
        [-5.0, 6.0, -5.0],
        [0.0, 3.0, 0.0],
        [8.0, 2.0, 0.0],
    ]
    default_lights_color = [
        [1.0, 1.0, 1.0],
        [0.6, 0.6, 1.0],
        [1.0, 0.7, 0.5],
        [1.0, 0.8, 0.6],
    ]
    lights_pos = [
        np.array(v, dtype=np.float32)
        for v in _get_list(config, "lights_pos", default_lights_pos)
    ]
    if len(lights_pos) != 4:
        lights_pos = [np.array(v, dtype=np.float32) for v in default_lights_pos]

    lights_color = [
        np.array(v, dtype=np.float32)
        for v in _get_list(config, "lights_color", default_lights_color)
    ]
    if len(lights_color) != 4:
        lights_color = [np.array(v, dtype=np.float32) for v in default_lights_color]

    lights_enabled = config.get("lights_enabled", [True, False, False, False])
    if not isinstance(lights_enabled, list) or len(lights_enabled) != 4:
        lights_enabled = [True, False, False, False]
    num_lights = sum(bool(v) for v in lights_enabled)
    if num_lights == 0:
        lights_enabled[0] = True
        num_lights = 1

    # --- Render loop ---
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            impl.process_inputs()

            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(window, True)

            # --- ImGui frame ---
            if show_imgui:
                imgui.new_frame()  # type: ignore[attr-defined]

                imgui.set_next_window_position(10, 10, condition=imgui.FIRST_USE_EVER)  # type: ignore[attr-defined]
                imgui.set_next_window_size(320, 420, condition=imgui.FIRST_USE_EVER)  # type: ignore[attr-defined]
                imgui.begin("Bushing's Function Parameters")  # type: ignore[attr-defined]

                opened, _ = imgui.collapsing_header("Surface type")  # type: ignore[attr-defined]
                if opened:
                    changed_type, surface_idx = imgui.combo(  # type: ignore[attr-defined]
                        "Surface type##surface_type", surface_idx, surface_labels
                    )
                    if changed_type:
                        needs_rebuild = True

                    imgui.separator()  # type: ignore[attr-defined]
                    changed_scheme, color_scheme_idx = imgui.combo(  # type: ignore[attr-defined]
                        "Color scheme##color_scheme", color_scheme_idx, color_scheme_labels
                    )
                    if changed_scheme:
                        needs_rebuild = True

                    _, alpha_val = imgui.slider_float("Alpha", alpha_val, 0.1, 1.0)  # type: ignore[attr-defined]

                opened_b, _ = imgui.collapsing_header("Bushings Function parameters")  # type: ignore[attr-defined]
                if opened_b:
                    if surface_idx < 2:
                        changed_eta, eta = imgui.slider_float("eta (spread)", eta, 0.5, 15.0)  # type: ignore[attr-defined]
                        if changed_eta:
                            needs_rebuild = True

                        changed_lam, lam = imgui.slider_int("lambda (anisotropy)", lam, 1, 30)  # type: ignore[attr-defined]
                        if changed_lam:
                            needs_rebuild = True

                        changed_nf, n_f = imgui.slider_int("nF (occupancy)", n_f, 0, 50)  # type: ignore[attr-defined]
                        if changed_nf:
                            needs_rebuild = True
                    else:
                        imgui.text("Sphere mode does not use Bushing parameters.")  # type: ignore[attr-defined]

                opened_l, _ = imgui.collapsing_header("Lighting parameters")  # type: ignore[attr-defined]
                if opened_l:
                    for i in range(4):
                        if imgui.tree_node(f"Light {i + 1}", flags=0):  # type: ignore[attr-defined]
                            _, lights_enabled[i] = imgui.checkbox(  # type: ignore[attr-defined]
                                f"Enabled##light{i}", lights_enabled[i]
                            )
                            if lights_enabled[i]:
                                _, lights_pos[i][0] = imgui.slider_float(  # type: ignore[attr-defined]
                                    f"X##light{i}x", lights_pos[i][0], -15.0, 15.0
                                )
                                _, lights_pos[i][1] = imgui.slider_float(  # type: ignore[attr-defined]
                                    f"Y##light{i}y", lights_pos[i][1], -15.0, 15.0
                                )
                                _, lights_pos[i][2] = imgui.slider_float(  # type: ignore[attr-defined]
                                    f"Z##light{i}z", lights_pos[i][2], -15.0, 15.0
                                )
                                _, lights_color[i][0] = imgui.slider_float(  # type: ignore[attr-defined]
                                    f"R##light{i}r", lights_color[i][0], 0.0, 1.0
                                )
                                _, lights_color[i][1] = imgui.slider_float(  # type: ignore[attr-defined]
                                    f"G##light{i}g", lights_color[i][1], 0.0, 1.0
                                )
                                _, lights_color[i][2] = imgui.slider_float(  # type: ignore[attr-defined]
                                    f"B##light{i}b", lights_color[i][2], 0.0, 1.0
                                )
                            imgui.tree_pop()  # type: ignore[attr-defined]

                opened_m, _ = imgui.collapsing_header("Meshgrid density")  # type: ignore[attr-defined]
                if opened_m:
                    changed_slices, mesh_slices = imgui.slider_int("Slices", mesh_slices, 8, 128)  # type: ignore[attr-defined]
                    changed_stacks, mesh_stacks = imgui.slider_int("Stacks", mesh_stacks, 4, 64)  # type: ignore[attr-defined]
                    if changed_slices or changed_stacks:
                        needs_rebuild = True

                    imgui.separator()  # type: ignore[attr-defined]
                    imgui.text(f"Vertices: {(mesh_stacks + 1) * (mesh_slices + 1)}")  # type: ignore[attr-defined]
                    imgui.text(f"Triangles: {mesh_stacks * mesh_slices * 2}")  # type: ignore[attr-defined]

                opened_e, _ = imgui.collapsing_header("Export mesh")  # type: ignore[attr-defined]
                if opened_e:
                    changed_fmt, export_format_idx = imgui.combo(  # type: ignore[attr-defined]
                        "Format##export", export_format_idx, export_formats
                    )
                    if changed_fmt:
                        export_filename = _sync_export_extension(
                            export_filename, export_format_idx, export_formats
                        )
                    _, export_filename = imgui.input_text("Filename##export", export_filename, 256)  # type: ignore[attr-defined]
                    if imgui.button("Export mesh##export"):  # type: ignore[attr-defined]
                        try:
                            # ensure all mesh arrays are available
                            if (
                                last_positions is None
                                or last_normals is None
                                or last_indices is None
                                or last_colors is None
                            ):
                                export_status = "No mesh to export"
                            else:
                                path = Path(export_filename)
                                fmt = export_formats[export_format_idx]
                                # localize variables to appease static checkers
                                lp = last_positions
                                ln = last_normals
                                lc = last_colors
                                li = last_indices
                                if fmt == "PLY":
                                    if path.suffix.lower() != ".ply":
                                        path = path.with_suffix(".ply")
                                    save_as_ply(path, lp, ln, lc, li)
                                else:
                                    if path.suffix.lower() not in (".obj",):
                                        path = path.with_suffix(".obj")
                                    save_as_obj(path, lp, ln, li)
                                export_status = f"Saved {path}"
                        except Exception as e:
                            export_status = f"Export failed: {e}"
                    if export_status:
                        imgui.text(export_status)  # type: ignore[attr-defined]

                imgui.end()  # type: ignore[attr-defined]

            if needs_rebuild:
                positions, normals, colors, indices = create_sor_mesh(
                    eta=eta,
                    lam=lam,
                    n_f=n_f,
                    surface_type=surface_types[surface_idx],
                    slices=mesh_slices,
                    stacks=mesh_stacks,
                    color_scheme=color_schemes[color_scheme_idx],
                )
                mesh_vao, mesh_index_count = create_mesh_vao(
                    positions, normals, colors, indices
                )
                # store last generated mesh for export
                last_positions = positions
                last_normals = normals
                last_colors = colors
                last_indices = indices
                needs_rebuild = False

            # --- Clear and draw ---
            glClearColor(0.10, 0.10, 0.12, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # type: ignore[operator]

            view = camera.view_matrix.astype(np.float32)
            eye_pos = camera.eye.astype(np.float32)

            # --- Draw grid ---
            glUseProgram(line_prog)
            set_uniform_mat4(line_prog, "view", view)
            set_uniform_mat4(line_prog, "projection", proj)
            glBindVertexArray(grid_vao)
            glDrawArrays(GL_LINES, 0, grid_vertex_count)

            # --- Draw surface (back faces first for transparency) ---
            glUseProgram(prog)
            set_uniform_mat4(prog, "model", model)
            set_uniform_mat4(prog, "view", view)
            set_uniform_mat4(prog, "projection", proj)

            # Set light arrays (only active lights)
            active_count = 0
            for i in range(4):
                if lights_enabled[i]:
                    pos_loc = glGetUniformLocation(prog, f"lights[{active_count}].position")
                    col_loc = glGetUniformLocation(prog, f"lights[{active_count}].color")
                    if pos_loc >= 0:
                        glUniform3fv(pos_loc, 1, lights_pos[i])
                    if col_loc >= 0:
                        glUniform3fv(col_loc, 1, lights_color[i])
                    active_count += 1

            if active_count == 0:
                pos_loc = glGetUniformLocation(prog, "lights[0].position")
                col_loc = glGetUniformLocation(prog, "lights[0].color")
                if pos_loc >= 0:
                    glUniform3fv(pos_loc, 1, lights_pos[0])
                if col_loc >= 0:
                    glUniform3fv(col_loc, 1, lights_color[0])
                active_count = 1

            set_uniform_int(prog, "num_lights", active_count)
            set_uniform_vec3(prog, "view_pos", eye_pos)
            set_uniform_float(prog, "alpha", alpha_val)

            glBindVertexArray(mesh_vao)
            glCullFace(GL_FRONT)
            glDrawElements(GL_TRIANGLES, mesh_index_count, GL_UNSIGNED_INT, None)
            glCullFace(GL_BACK)
            glDrawElements(GL_TRIANGLES, mesh_index_count, GL_UNSIGNED_INT, None)

            # --- Draw ImGui ---
            if show_imgui:
                imgui.render()  # type: ignore[attr-defined]
                impl.render(imgui.get_draw_data())  # type: ignore[attr-defined]

            glfw.swap_buffers(window)
    finally:
        # --- Save configuration ---
        _save_config(
            {
                "eta": float(eta),
                "lam": int(lam),
                "n_f": int(n_f),
                "surface_idx": int(surface_idx),
                "color_scheme_idx": int(color_scheme_idx),
                "alpha_val": float(alpha_val),
                "mesh_slices": int(mesh_slices),
                "mesh_stacks": int(mesh_stacks),
                "lights_pos": [
                    pos.tolist() if hasattr(pos, "tolist") else list(pos) for pos in lights_pos
                ],
                "lights_color": [
                    col.tolist() if hasattr(col, "tolist") else list(col) for col in lights_color
                ],
                "lights_enabled": [bool(v) for v in lights_enabled],
                "show_imgui": bool(show_imgui),
            }
        )

        # --- Cleanup ---
        impl.shutdown()
        glfw.terminate()


if __name__ == "__main__":
    main()
