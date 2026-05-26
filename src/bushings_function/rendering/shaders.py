"""Built-in GLSL shader sources for the PyOpenGL rendering pipeline.

These are the same Blinn-Phong shaders used in the sibling projects,
but compiled via raw OpenGL calls (glCreateShader, glCompileShader)
rather than ModernGL's ctx.program() wrapper.
"""

# --- Blinn-Phong shading with per-vertex colour ---

VERTEX_SHADER = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

out vec3 frag_pos;
out vec3 frag_normal;
out vec3 frag_color_v;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    frag_pos = world_pos.xyz;
    frag_normal = mat3(transpose(inverse(model))) * in_normal;
    frag_color_v = in_color;
    gl_Position = projection * view * world_pos;
}
"""

FRAGMENT_SHADER = """
#version 330 core

struct Light {
    vec3 position;
    vec3 color;
};

uniform Light lights[4];
uniform int num_lights;
uniform vec3 view_pos;
uniform float alpha;

in vec3 frag_pos;
in vec3 frag_normal;
in vec3 frag_color_v;

out vec4 frag_color;

void main() {
    vec3 norm = normalize(frag_normal);
    vec3 view_dir = normalize(view_pos - frag_pos);

    // Global ambient
    vec3 result = vec3(0.15);

    // Accumulate contribution from all active lights
    for (int i = 0; i < num_lights && i < 4; i++) {
        vec3 light_dir = normalize(lights[i].position - frag_pos);

        // Diffuse
        float diff = max(dot(norm, light_dir), 0.0);
        vec3 diffuse = diff * lights[i].color;

        // Specular (Blinn-Phong)
        vec3 halfway = normalize(light_dir + view_dir);
        float spec = pow(max(dot(norm, halfway), 0.0), 64.0);
        vec3 specular = 0.4 * spec * lights[i].color;

        result += (diffuse + specular);
    }

    vec3 final_color = result * frag_color_v;
    frag_color = vec4(final_color, alpha);
}
"""

# --- Simple line shader for grid and wireframe ---

LINE_VERTEX_SHADER = """
#version 330 core

uniform mat4 view;
uniform mat4 projection;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

out vec3 frag_color_v;

void main() {
    frag_color_v = in_color;
    gl_Position = projection * view * vec4(in_position, 1.0);
}
"""

LINE_FRAGMENT_SHADER = """
#version 330 core

in vec3 frag_color_v;
out vec4 frag_color;

void main() {
    frag_color = vec4(frag_color_v, 1.0);
}
"""
