#include "helper.h"
#include <iostream>
#include <SDL2/SDL.h>



void display_data(int N, int n, float* data, int* labels) {
    float min_x = -3;
    float max_x = 3;
    float min_y = -3;
    float max_y = 3;

    int screen_width = 1000;
    int screen_height = 1000;

    if(SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cout << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return;
    }
    if(n != 2) {
        std::cout << "Only 2D data can be displayed" << std::endl;
        return;
    }  
    SDL_Window* window = SDL_CreateWindow("Data", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screen_width, screen_height, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Event e;
    bool quit = false;
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderClear(renderer);
        
        for (int i = 0; i < N; i++) {
            int x = (data[i] - min_x) / (max_x - min_x) * screen_width;
            int y = (data[i + N] - min_y) / (max_y - min_y) * screen_height;
            int label = labels[i];
            int r = 0;
            int g = 0;
            int b = 0;
            if (label == 0) {
                r = 255;
            } else if (label == 1) {
                g = 255;
            } else {
                b = 255;
            }
            SDL_SetRenderDrawColor(renderer, r, g, b, 255);
            SDL_Rect rect = {x, y, 10, 10};
            SDL_RenderFillRect(renderer, &rect);
        }

        SDL_RenderPresent(renderer);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}