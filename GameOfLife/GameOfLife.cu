#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SFML/Graphics.hpp>
#include <stdio.h>
#include <time.h>
#include <random>
#include <unordered_map>
#include "GameOfLife.h"
#define window_area state.window_w * state.window_h
#define pixels_size 4 * (window_area)

struct GOL_params {
	const int WINDOW_WIDTH = 512;
	const int WINDOW_HEIGHT = 512;
	const int BOARD_SIZE = 8192;
	const int BOARD_AREA = BOARD_SIZE * BOARD_SIZE;
	const std::string WINDOW_TITLE = "Geim of Life";
	const int TARGET_FPS = 120;
	const double ZOOM_FACTOR = 1.25;

	uint8_t* pixels;
	uint8_t* board;
	uint8_t* dev_board_IN;
	uint8_t* dev_board_OUT;
	uint8_t* dev_pixels;

	sf::RenderWindow window;
	sf::ContextSettings settings;
	sf::Sprite mainSprite;
	sf::Texture mainTexture;
	sf::Vector2i MouseMoveStartPos;

	int window_w = WINDOW_WIDTH;
	int window_h = WINDOW_HEIGHT;

	double simulation_fps = 5.0;

	bool paused = false;
	bool LMousePressed = false;
	bool MMousePressed = false;
	bool resetRect = false;
	bool upload_and_redraw_board = false;
	bool redraw_board = false;
	bool fullScreen = false;
	bool showSettings = false;

	int drawMode = -1;
	int lastBoardIndx = -1;

	double xs = 0.0;
	double ys = 0.0;
	double dx = 1.0;
	double dy = dx * window_h / window_w;

	clock_t last_frame_time = clock();
	clock_t cur_time = clock();

	sf::Font font;
	sf::RectangleShape settings_button;
	sf::Text settings_button_text;

	std::uniform_int_distribution<> distributed_random;
	std::mt19937 random_generator_function;

	const std::unordered_map<sf::Keyboard::Key, int> keyToPatternMap = {
		{sf::Keyboard::R, 0},
		{sf::Keyboard::Num0, 0},
		{sf::Keyboard::Num1, 1},
		{sf::Keyboard::Num2, 2},
		{sf::Keyboard::Num3, 3},
		{sf::Keyboard::Num4, 4}
	};
};

__global__ void get_next_frame_kernel(uint8_t* board_1, uint8_t* board_2, int BOARD_SIZE) {
	int neigh = 0;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int cx = x;
	int cy = y;

	for (int dy = -1; dy <= 1; dy++) {
		cy = y + dy;
		for (int dx = -1; dx <= 1; dx++) {
			cx = x + dx;
			if (cx >= 0 && cx <= BOARD_SIZE - 1 && cy >= 0 && cy <= BOARD_SIZE - 1 && (dx != 0 || dy != 0)) {
				neigh += board_1[cy * BOARD_SIZE + cx] != 0;
			}
		}
	}

	int tid = BOARD_SIZE * y + x;
	uint8_t cellStatus = board_1[tid];
	if (cellStatus == 0 && neigh == 3) {
		board_2[tid] = 1;
	}
	else if (cellStatus == 1 && (neigh < 2 || neigh > 3)) {
		board_2[tid] = 0;
	}
	else {
		board_2[tid] = cellStatus;
	}
}
__global__ void update_pixels_kernel(uint8_t* pixels, uint8_t* board, int BOARD_SIZE, int w_width, int w_height, double xs, double ys, double dx, double dy) {
	int wx = blockIdx.x * blockDim.x + threadIdx.x;
	int wy = blockIdx.y * blockDim.y + threadIdx.y;

	int pixel_index = 4 * (wy * w_width + wx);

	if (wx >= w_width || wy >= w_height) {
		return;
	}

	uint8_t red;
	uint8_t blue;
	uint8_t green;

	double ww = w_width;
	double wh = w_height;

	double xb = xs + (double)wx / ww * dx; // [0 : 1]
	double yb = ys + (double)wy / wh * dy; // [0 : 1]

	if (xb < 0.0 || yb < 0.0 || xb >= 1.0 || yb >= 1.0) {
		pixels[pixel_index] = 0;
		pixels[pixel_index + 1] = 0;
		pixels[pixel_index + 2] = 0;
		pixels[pixel_index + 3] = 255;
		return;
	}

	int board_index_x = xb * BOARD_SIZE;
	int board_index_y = yb * BOARD_SIZE;
	int board_index = board_index_y * BOARD_SIZE + board_index_x;

	uint8_t status = board[board_index];

	switch (status) {
	case 0:
		red = 250;
		green = 250;
		blue = 250;
		break;
	case 1:
		red = 0;
		green = 0;
		blue = 0;
		break;
	default:
		red = status;
		green = status;
		blue = status;
		break;
	}

	pixels[pixel_index] = red;
	pixels[pixel_index + 1] = green;
	pixels[pixel_index + 2] = blue;
	pixels[pixel_index + 3] = 255;
}

cudaError_t get_next_frame_cuda(GOL_params& state) {
	cudaError_t cudaStatus;

	dim3 blockSize(16, 16);
	dim3 gridSize((state.BOARD_SIZE + blockSize.x - 1) / blockSize.x, (state.BOARD_SIZE + blockSize.y - 1) / blockSize.y);

	get_next_frame_kernel << <gridSize, blockSize >> > (state.dev_board_IN, state.dev_board_OUT, state.BOARD_SIZE);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaGetLastError\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(state.board, state.dev_board_OUT, state.BOARD_AREA, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaMemcpy\n");
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaDeviceSynchronize\n");
		return cudaStatus;
	}

	uint8_t* cuda_board_IN_temp = state.dev_board_IN;
	state.dev_board_IN = state.dev_board_OUT;
	state.dev_board_OUT = cuda_board_IN_temp;

	return cudaStatus;
}
cudaError_t update_pixels_cuda(GOL_params& state) {
	cudaError_t cudaStatus;

	dim3 blockSize(16, 16);
	dim3 gridSize((state.window_w + blockSize.x - 1) / blockSize.x, (state.window_h + blockSize.y - 1) / blockSize.y);

	update_pixels_kernel << <gridSize, blockSize >> > (state.dev_pixels, state.dev_board_IN, state.BOARD_SIZE, state.window_w, state.window_h, state.xs, state.ys, state.dx, state.dy);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("error in update_pixels_kernel\n");
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaDeviceSynchronize update_pixels_cuda\n");
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(state.pixels, state.dev_pixels, pixels_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaMemcpy state.pixels\n");
		return cudaStatus;
	}

	return cudaStatus;
}
void resize_window(GOL_params& state, int w, int h) {
	state.window_w = w;
	state.window_h = h;
	state.window.setView(sf::View(sf::FloatRect(0, 0, (float)state.window_w, (float)state.window_h)));
}
void make_window(GOL_params& state) {
	state.window.close();
	sf::VideoMode screenSize;
	if (state.fullScreen) {
		screenSize = sf::VideoMode::getDesktopMode();
		state.window.create(screenSize, state.WINDOW_TITLE, sf::Style::Fullscreen, state.settings);
	}
	else {
		screenSize = sf::VideoMode(state.window_w, state.window_h, 24);
		state.window.create(screenSize, state.WINDOW_TITLE, sf::Style::Resize | sf::Style::Close, state.settings);
	}
	resize_window(state, screenSize.width, screenSize.height);
	state.window.setFramerateLimit(state.TARGET_FPS);
	state.window.setKeyRepeatEnabled(false);
	state.window.requestFocus();
}
void sfml_setup_gui(GOL_params& state) {
	state.settings_button = sf::RectangleShape(sf::Vector2f(60, 25));
	if (!state.font.loadFromFile("arial.ttf")) {
		printf("Font file not found!");
	}
	state.settings_button_text = sf::Text("settings", state.font, 15);
	state.settings_button.setFillColor(sf::Color(128, 128, 128, 128));
	state.settings_button.setPosition(0, 0);
}
cudaError_t cuda_setup(GOL_params& state) {
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)&state.dev_board_IN, state.BOARD_AREA);
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaMalloc state.dev_board_IN cuda_setup\n");
	}

	cudaStatus = cudaMalloc((void**)&state.dev_board_OUT, state.BOARD_AREA);
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaMalloc state.dev_board_OUT cuda_setup\n");
	}

	cudaStatus = cudaMalloc((void**)&state.dev_pixels, pixels_size);
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaMalloc state.dev_pixels cuda_setup\n");
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaDeviceSynchronize cuda_setup\n");
	}

	return cudaStatus;
}
void sfml_setup(GOL_params& state) {
	state.settings.depthBits = 24;
	state.settings.stencilBits = 8;
	state.settings.antialiasingLevel = 4;
	state.settings.majorVersion = 3;
	state.settings.minorVersion = 0;

	make_window(state); 
	sfml_setup_gui(state);
}
void setup(GOL_params& state) {
	std::random_device rd;
	state.random_generator_function = std::mt19937(rd());
	state.distributed_random = std::uniform_int_distribution<>(0, 1);

	state.board = (uint8_t*)malloc(state.BOARD_AREA);
	if (state.board == NULL) {
		printf("error in malloc state.board\n");
	}
	state.pixels = (uint8_t*)malloc(pixels_size);
	if (state.pixels == NULL) {
		printf("error in malloc state.pixels\n");
	}

	sfml_setup(state);
	cuda_setup(state);
}
cudaError cuda_resize(GOL_params& state) {
	cudaError_t cudaStatus;
	cudaFree(state.dev_pixels);

	cudaStatus = cudaMalloc((void**)&state.dev_pixels, pixels_size);
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaMalloc state.dev_pixels cuda_resize\n");
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("error in cudaDeviceSynchronize cuda_resize\n");
	}
	return cudaStatus;
}
int mouse_cords_to_board_indx(GOL_params& state, sf::Vector2i& mousePos) {
	double xm = (double)mousePos.x / state.window_w;
	double ym = (double)mousePos.y / state.window_h;
	int board_x = (state.xs + xm * state.dx) * state.BOARD_SIZE;
	int board_y = (state.ys + ym * state.dy) * state.BOARD_SIZE;
	int board_indx = board_y * state.BOARD_SIZE + board_x;

	return board_indx;
}
void board_reset_and_draw_pattern(GOL_params& state, int indx) {
	memset(state.board, 0, state.BOARD_AREA);
	switch (indx)
	{
	case 0:
		break;
	case 1:
		state.board[state.BOARD_SIZE * 100 + 100] = 1;
		state.board[state.BOARD_SIZE * 100 + 101] = 1;
		state.board[state.BOARD_SIZE * 100 + 102] = 1;
		break;
	case 2:
		for (int iy = state.BOARD_SIZE / 3; iy < 2 * state.BOARD_SIZE / 3; iy++) {
			for (int ix = state.BOARD_SIZE / 3; ix < 2 * state.BOARD_SIZE / 3; ix++) {
				state.board[iy * state.BOARD_SIZE + ix] = state.distributed_random(state.random_generator_function);
			}
		}
		break;
	case 3:
		for (int ix = state.BOARD_SIZE / 3; ix < 2 * state.BOARD_SIZE / 3; ix++) {
			state.board[state.BOARD_SIZE * state.BOARD_SIZE / 2 + ix] = 1;
		}
		break;
	case 4:
		for (int iy = 0; iy < state.BOARD_SIZE; iy++) {
			state.board[iy * state.BOARD_SIZE + state.BOARD_SIZE / 2] = 1;
		}
		break;
	}
	state.upload_and_redraw_board = true;
}
void handle_board_reset_pattern(GOL_params& state, sf::Keyboard::Key key) {
	auto it = state.keyToPatternMap.find(key);
	if (it != state.keyToPatternMap.end()) {
		board_reset_and_draw_pattern(state, it->second);
	}
}
void handle_window_resize(GOL_params& state, int width, int height, bool recreate_window) {
	if (recreate_window) {
		state.window_w = width;
		state.window_h = height;
		make_window(state);
	}
	else {
		resize_window(state, width, height);
	}
	cuda_resize(state);

	state.dy = state.dx * state.window_h / state.window_w;

	state.pixels = (uint8_t*)realloc(state.pixels, pixels_size);
	state.mainTexture.create(state.window_w, state.window_h);
	state.resetRect = true;
}
void toggle_fullscreen(GOL_params& state) {
	state.fullScreen = !state.fullScreen;
	handle_window_resize(state, state.WINDOW_HEIGHT, state.WINDOW_WIDTH, true);
}
void handle_mouse_press(GOL_params& state, sf::Mouse::Button mouse_button) {
	switch (mouse_button) {
	case sf::Mouse::Left: {
		sf::Vector2i mousePos = sf::Mouse::getPosition(state.window);

		if (state.settings_button.getGlobalBounds().contains((sf::Vector2f)mousePos)) {
			printf("Button clicked!\n");
			state.showSettings = !state.showSettings;
			break;
		}

		state.LMousePressed = true;

		int board_indx = mouse_cords_to_board_indx(state, mousePos);
		state.lastBoardIndx = board_indx;

		state.drawMode = (int)(state.board[board_indx] == 0);

		state.board[board_indx] = state.drawMode;
		state.upload_and_redraw_board = true;

		break;
	}
	case sf::Mouse::Middle:
		state.MMousePressed = true;
		state.MouseMoveStartPos = sf::Mouse::getPosition(state.window);
		break;
	}
}
void handle_mouse_release(GOL_params& state, sf::Mouse::Button mouse_button) {
	switch (mouse_button) {
	case sf::Mouse::Left:
		state.LMousePressed = false;
		state.upload_and_redraw_board = true;
		state.drawMode = -1;
		break;
	case sf::Mouse::Middle:
		state.MMousePressed = false;
		state.redraw_board = true;
		break;
	}
}
void handle_mouse_move(GOL_params& state) {
	if (state.MMousePressed) {
		sf::Vector2i mousePos = sf::Mouse::getPosition(state.window);
		state.xs += (double)(state.MouseMoveStartPos.x - mousePos.x) / state.window_w * state.dx;
		state.ys += (double)(state.MouseMoveStartPos.y - mousePos.y) / state.window_h * state.dy;
		state.MouseMoveStartPos = mousePos;
		state.redraw_board = true;
	}
	if (state.LMousePressed) {
		sf::Vector2i mousePos = sf::Mouse::getPosition(state.window);
		int board_indx = mouse_cords_to_board_indx(state, mousePos);
		if (board_indx != state.lastBoardIndx) {
			state.lastBoardIndx = board_indx;
			state.board[board_indx] = state.drawMode;
			state.upload_and_redraw_board = true;
		}
	}
}
void handle_mouse_scroll(GOL_params& state, int scroll) {
	sf::Vector2i mousePos = sf::Mouse::getPosition(state.window);
	double xm = (double)mousePos.x / state.window_w;
	double ym = (double)mousePos.y / state.window_h;

	if (scroll < 0) { // zoom out
		state.xs = state.xs - xm * state.dx * (state.ZOOM_FACTOR - 1.0);
		state.ys = state.ys - ym * state.dy * (state.ZOOM_FACTOR - 1.0);

		state.dx *= state.ZOOM_FACTOR;
		state.dy *= state.ZOOM_FACTOR;
	}
	else if (scroll > 0) { // zoom in
		state.dx /= state.ZOOM_FACTOR;
		state.dy /= state.ZOOM_FACTOR;

		state.xs = state.xs + xm * state.dx * (state.ZOOM_FACTOR - 1.0);
		state.ys = state.ys + ym * state.dy * (state.ZOOM_FACTOR - 1.0);
	}
	printf("dx / dy : %lf\n", state.dx / state.dy);
	state.upload_and_redraw_board = true;
}
void handle_key_release(GOL_params& state, sf::Keyboard::Key key) {
	switch (key)
	{
	case sf::Keyboard::Space:
		state.paused = !state.paused;
		break;
	case sf::Keyboard::O:
		state.xs = 0.0;
		state.ys = 0.0;
		state.dx = 1.0;
		state.dy = state.dx * state.window_h / state.window_w;
		state.upload_and_redraw_board = true;
		break;
	case sf::Keyboard::Left:
		state.simulation_fps /= 1.5;
		printf("simulation Rate: %lf\n", state.simulation_fps);
		break;
	case sf::Keyboard::Right:
		state.simulation_fps *= 1.5;
		printf("simulation Rate: %lf\n", state.simulation_fps);
		break;
	case sf::Keyboard::R:
	case sf::Keyboard::Num0:
	case sf::Keyboard::Num1:
	case sf::Keyboard::Num2:
	case sf::Keyboard::Num3:
	case sf::Keyboard::Num4:
		handle_board_reset_pattern(state, key);
		break;
	case sf::Keyboard::F11:
		toggle_fullscreen(state);
		break;
	}
}
void draw_settings_sprite(GOL_params& state) {
	if (state.showSettings) {
		const char* settings_options[] = {
			"Board Size:",
			"Start Pattern 1",
			"Start Pattern 2",
			"Start Pattern 3",
			"Start Pattern 4",
			"Simulation Speed:"
		};

		sf::FloatRect settings_button_coords = state.settings_button.getGlobalBounds();
		float y_cord = settings_button_coords.top + settings_button_coords.height;
		float start_y_cord = y_cord;

		const int text_y_size = sizeof(settings_options) / sizeof(settings_options[0]);
		float text_width = 0.0f;

		sf::Text option_texts[text_y_size];

		for (int i = 0; i < text_y_size; ++i) {
			sf::Text option_text = sf::Text(settings_options[i], state.font, 15);
			option_text.setPosition(5, y_cord);
			y_cord += 18.0;
			option_texts[i] = option_text;
			float cur_text_width = option_text.getGlobalBounds().width;
			text_width = text_width > cur_text_width ? text_width : cur_text_width;
		}
		sf::RectangleShape settings_menu = sf::RectangleShape(sf::Vector2f(text_width + 10, text_y_size * 18 + 5));
		settings_menu.setFillColor(sf::Color(128, 128, 128, 128));
		settings_menu.setPosition(0, start_y_cord);

		state.window.draw(settings_menu);

		for (int i = 0; i < text_y_size; i++) {
			state.window.draw(option_texts[i]);
		}
	}
	state.window.draw(state.settings_button);
	state.window.draw(state.settings_button_text);
}
void gameOfLifeMain()
{
	printf("started\n");

	GOL_params state;
	printf("sizeof state:%d\n", sizeof(GOL_params));

	setup(state);
	board_reset_and_draw_pattern(state, 0);
	state.mainTexture.create(state.window_w, state.window_h);

	while (state.window.isOpen())
	{
		sf::Event event;
		while (state.window.pollEvent(event))
		{
			switch (event.type)
			{
			case sf::Event::Closed:
				state.window.close();
				break;
			case sf::Event::Resized:
				handle_window_resize(state, event.size.width, event.size.height, false);
				break;
			case sf::Event::MouseButtonPressed:
				handle_mouse_press(state, event.mouseButton.button);
				break;
			case sf::Event::MouseButtonReleased:
				handle_mouse_release(state, event.mouseButton.button);
				break;
			case sf::Event::MouseMoved:
				handle_mouse_move(state);
				break;
			case sf::Event::MouseWheelScrolled:
				handle_mouse_scroll(state, event.mouseWheelScroll.delta);
				break;
			case sf::Event::KeyReleased:
				handle_key_release(state, event.key.code);
				break;
			}
		}

		state.cur_time = clock();
		double ms_passed = (double)(state.cur_time - state.last_frame_time);
		double target_frame_time_ms = 1000.0 / state.simulation_fps;

		bool paused = !state.paused && state.drawMode == -1;
		bool frame_time_passed = ms_passed >= target_frame_time_ms;
		bool get_next_frame_and_draw = frame_time_passed && paused;

		bool redrawing = get_next_frame_and_draw || state.upload_and_redraw_board || state.redraw_board;

		if (redrawing) {
			cudaError_t cudaStatus;
			if (get_next_frame_and_draw && !state.upload_and_redraw_board) {
				state.last_frame_time = state.cur_time;
				cudaStatus = get_next_frame_cuda(state);

				clock_t time_after_drawing_frame = clock();
				double frame_time = (double)(time_after_drawing_frame - state.last_frame_time);
				printf("frame time: %lf / %lf\n", frame_time, ms_passed);
			}

			if (state.upload_and_redraw_board && !state.MMousePressed) {
				cudaStatus = cudaMemcpy(state.dev_board_IN, state.board, state.BOARD_AREA, cudaMemcpyHostToDevice);
				state.upload_and_redraw_board = false;
			}

			cudaStatus = update_pixels_cuda(state);

			state.mainTexture.update(state.pixels, state.window_w, state.window_h, 0, 0);
			state.redraw_board = false;
		}

		state.mainSprite.setTexture(state.mainTexture, state.resetRect);
		state.resetRect = false;

		state.window.clear(sf::Color::Green);
		state.window.draw(state.mainSprite);

		draw_settings_sprite(state);

		state.window.display();
	}
}