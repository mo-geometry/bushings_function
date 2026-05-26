"""Low-level OpenGL helpers — shader compilation, program linking, VAO/VBO creation.

These map directly to the C OpenGL API and are intentionally explicit
(no abstraction layers) so the code reads like an OpenGL tutorial and
transfers cleanly to C/C++.
"""

from __future__ import annotations

from ctypes import c_void_p

import numpy as np
import numpy.typing as npt
from OpenGL.GL import (  # type: ignore[import-untyped]
    GL_ARRAY_BUFFER,
    GL_COMPILE_STATUS,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_LINK_STATUS,
    GL_STATIC_DRAW,
    GL_VERTEX_SHADER,
    glAttachShader,
    glBindBuffer,
    glBindVertexArray,
    glBufferData,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteShader,
    glEnableVertexAttribArray,
    glGenBuffers,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniform1i,
    glUniform3fv,
    glUniformMatrix4fv,
    glVertexAttribPointer,
)


def compile_shader(source: str, shader_type: int) -> int:
    """Compile a GLSL shader from source.

    Args:
        source: GLSL source code string.
        shader_type: GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.

    Returns:
        OpenGL shader handle.

    Raises:
        RuntimeError: If compilation fails.
    """
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)

    if glGetShaderiv(shader, GL_COMPILE_STATUS) == GL_FALSE:
        info = glGetShaderInfoLog(shader).decode("utf-8")
        kind = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        msg = f"{kind} shader compilation failed:\n{info}"
        raise RuntimeError(msg)

    return int(shader)  # type: ignore[arg-type]


def create_program(vertex_source: str, fragment_source: str) -> int:
    """Compile and link a vertex + fragment shader program.

    Args:
        vertex_source: GLSL vertex shader source.
        fragment_source: GLSL fragment shader source.

    Returns:
        OpenGL program handle.

    Raises:
        RuntimeError: If linking fails.
    """
    vs = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_source, GL_FRAGMENT_SHADER)

    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS) == GL_FALSE:
        info = glGetProgramInfoLog(program).decode("utf-8")
        msg = f"Shader program linking failed:\n{info}"
        raise RuntimeError(msg)

    # Shaders are linked — no longer needed individually
    glDeleteShader(vs)
    glDeleteShader(fs)

    return int(program)  # type: ignore[arg-type]


def create_mesh_vao(
    positions: npt.NDArray[np.float32],
    normals: npt.NDArray[np.float32],
    colors: npt.NDArray[np.float32],
    indices: npt.NDArray[np.uint32],
) -> tuple[int, int]:
    """Create a VAO with position/normal/color VBOs and an index buffer.

    Layout matches the Blinn-Phong vertex shader:
        location 0: in_position (vec3)
        location 1: in_normal   (vec3)
        location 2: in_color    (vec3)

    Args:
        positions: (N, 3) float32 vertex positions.
        normals: (N, 3) float32 per-vertex normals.
        colors: (N, 3) float32 per-vertex RGB colours.
        indices: (M,) uint32 triangle indices.

    Returns:
        Tuple (vao, index_count) — the VAO handle and number of indices.
    """
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    # Position buffer — location 0
    vbo_pos = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(0)

    # Normal buffer — location 1
    vbo_norm = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_norm)
    glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(1)

    # Color buffer — location 2
    vbo_col = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(2)

    # Element (index) buffer
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glBindVertexArray(0)

    return int(vao), len(indices)


def create_line_vao(
    positions: npt.NDArray[np.float32],
    colors: npt.NDArray[np.float32],
) -> tuple[int, int]:
    """Create a VAO for line rendering (grid, wireframe).

    Layout matches the line vertex shader:
        location 0: in_position (vec3)
        location 1: in_color    (vec3)

    Args:
        positions: (N, 3) float32 line endpoints.
        colors: (N, 3) float32 per-vertex colours.

    Returns:
        Tuple (vao, vertex_count).
    """
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo_pos = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos)
    glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(0)

    vbo_col = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_col)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, c_void_p(0))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)

    return int(vao), len(positions)


def set_uniform_mat4(program: int, name: str, matrix: npt.NDArray[np.float32]) -> None:
    """Upload a 4x4 matrix uniform (column-major for OpenGL).

    Args:
        program: Shader program handle.
        name: Uniform name in the shader.
        matrix: 4x4 float32 matrix in row-major order (transposed on upload).
    """
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        # Transpose from row-major (NumPy) to column-major (OpenGL)
        glUniformMatrix4fv(loc, 1, GL_FALSE, matrix.T.copy())


def set_uniform_vec3(program: int, name: str, vec: npt.NDArray[np.float32]) -> None:
    """Upload a vec3 uniform.

    Args:
        program: Shader program handle.
        name: Uniform name in the shader.
        vec: 3-element float32 array.
    """
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        glUniform3fv(loc, 1, vec)


def set_uniform_float(program: int, name: str, value: float) -> None:
    """Upload a float uniform.

    Args:
        program: Shader program handle.
        name: Uniform name in the shader.
        value: Float value.
    """
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        glUniform1f(loc, value)


def set_uniform_int(program: int, name: str, value: int) -> None:
    """Upload an int uniform.

    Args:
        program: Shader program handle.
        name: Uniform name in the shader.
        value: Int value.
    """
    loc = glGetUniformLocation(program, name)
    if loc >= 0:
        glUniform1i(loc, value)
