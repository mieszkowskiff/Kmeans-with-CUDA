#include "helper.h"
#include <iostream>
#include <SDL2/SDL.h>

int colors[10][3] = {
    {255, 0, 0},
    {0, 255, 0},
    {0, 0, 255},
    {255, 255, 0},
    {255, 0, 255},
    {0, 255, 255},
    {128, 0, 0},
    {0, 128, 0},
    {0, 0, 128},
    {128, 128, 0}
};


void display_data(int N, int n, float* data, int* labels) {

    float size = 3;
    float min_x = -size;
    float max_x = size;
    float min_y = -size;
    float max_y = size;

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
            int r = colors[label][0];
            int g = colors[label][1];
            int b = colors[label][2];
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


void display_data_with_centroids(int N, int n, float* data, int* labels, float* centroids, int k) {
    
    float size = 3;


    float min_x = -size;
    float max_x = size;
    float min_y = -size;
    float max_y = size;

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
            int r = colors[label][0];
            int g = colors[label][1];
            int b = colors[label][2];
            SDL_SetRenderDrawColor(renderer, r, g, b, 255);
            SDL_Rect rect = {x, y, 10, 10};
            SDL_RenderFillRect(renderer, &rect);
        }

        for(int i = 0; i < k; i++) {
            int x = (centroids[i] - min_x) / (max_x - min_x) * screen_width;
            int y = (centroids[i + k] - min_y) / (max_y - min_y) * screen_height;
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_Rect rect = {x, y, 15, 15};
            SDL_RenderFillRect(renderer, &rect);
        }

        SDL_RenderPresent(renderer);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

