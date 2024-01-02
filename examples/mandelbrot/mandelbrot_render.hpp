//
// Created by david on 02-01-2024.
//

#ifndef LMARROW_MANDELBROT_RENDER_HPP
#define LMARROW_MANDELBROT_RENDER_HPP

#include <SDL.h>
#include <cmath>

#include "lmarrow/lmarrow.hpp"


using namespace lmarrow;

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 800;

void renderData(SDL_Renderer* renderer, vector<int>& data, int max) {

    int size = std::sqrt(data.size());

    int cellWidth = SCREEN_WIDTH / size;
    int cellHeight = SCREEN_HEIGHT / size;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            int val = data[i*size + j] * 255 / max;
            SDL_Rect rect = { j * cellWidth, i * cellHeight, cellWidth, cellHeight };
            SDL_SetRenderDrawColor(renderer, val, val, val, 255);
            SDL_RenderFillRect(renderer, &rect);
        }
    }
}

void render(vector<int>& data, int max) {

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window* window = SDL_CreateWindow("Mandelbrot Render", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    renderData(renderer, data, max);

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

#endif //LMARROW_MANDELBROT_RENDER_HPP
