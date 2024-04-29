#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_video.h>
#include <stdio.h>

#define GLSL_PROGRAM_LOG_SIZE 2028

char glsl_log[GLSL_PROGRAM_LOG_SIZE];

GLuint createShader(GLenum type, const char *file_name) {
    printf("Loading %-32s", file_name);
    GLuint shader = glCreateShader(type);

    SDL_RWops *file = SDL_RWFromFile(file_name, "r");
    if (file == NULL) {
        fprintf(stderr, "%s\n", SDL_GetError());
        return 0;
    }
    size_t file_size = SDL_RWsize(file);
    char *source = (char *)malloc(file_size + 1);

    SDL_RWread(file, source, file_size, file_size);
    SDL_RWclose(file);
    source[file_size] = '\0';

    glShaderSource(shader, 1, (const char **)(&source), NULL);
    glCompileShader(shader);

    GLint compile_ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_ok);
    if (compile_ok == GL_FALSE) {
        int length;
        glGetShaderInfoLog(shader, 2048, &length, glsl_log);
        fprintf(stderr, "failed!\n");
        glDeleteShader(shader);
        fprintf(stderr, "%s\n", glsl_log);
        return 0;
    }
    printf("ok!\n");
    return shader;
}

int main() {
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("N body simulation", SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED, 512, 512, SDL_WINDOW_OPENGL);
    if (window == NULL) {
        fprintf(stderr, "Failed to create window: %s", SDL_GetError());
        return 1;
    }

    int width, height;
    SDL_GetWindowSize(window, &width, &height);
    printf("%d x %d\n", width, height);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    GLenum status = glewInit();
    if (status != GLEW_OK) {
        fprintf(stderr, "Failed to init GLEW: %s", (const char *)glewGetErrorString(status));
        return 1;
    }

    glEnable(GL_POINT_SPRITE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_DEPTH_TEST);

    GLuint vertex_shader = createShader(GL_VERTEX_SHADER, "09-n-body-problem-vertex.glsl");
    GLuint fragment_shader = createShader(GL_FRAGMENT_SHADER, "09-n-body-problem-fragment.glsl");
    GLuint gl_program = glCreateProgram();
    glAttachShader(gl_program, vertex_shader);
    glAttachShader(gl_program, fragment_shader);
    glLinkProgram(gl_program);

    GLint gl_program_status;
    glGetProgramiv(gl_program, GL_LINK_STATUS, &gl_program_status);
    if (gl_program_status == GL_FALSE) {
        int length;
        glGetProgramiv(gl_program, GL_INFO_LOG_LENGTH, &length);
        printf("Some error %d\n", length);
        if (length > 0) {
            char* error_log = (char*)malloc(length);
            glGetProgramInfoLog(gl_program, GLSL_PROGRAM_LOG_SIZE, &length, glsl_log);
            fprintf(stderr, "Linking failed: %s\n", error_log);
        } else {
            fprintf(stderr, "Linking failed\n");
        }
        return 1;
    }

    glUseProgram(gl_program);

    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);

    GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glEnableVertexAttribArray(0);

    SDL_Event event;
    bool is_running = true;
    int x_location = 0;
    while (is_running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_QUIT:
                is_running = 0;
                break;
            }
        }

        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        x_location++;
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        GLfloat vertices[] = {((GLfloat)x_location - 256) / 256.0f, 0.0f, 0.0f, 0.0f};
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
        glDrawArrays(GL_POINTS, 0, 2);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            printf("GL error: %d\n", error);
        }

        SDL_GL_SwapWindow(window);
    }
    return 0;
}
