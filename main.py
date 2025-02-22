import pygame
import random
import numpy as np

width, height = 640, 480

black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)
white = (255, 255, 255)

block_size = 20
speed = 15


class SnakeGame:
    def __init__(self, width, height, block_size):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.reset()

    def reset(self):
        initial_position = [self.width // 2, self.height // 2]
        self.snake = [
            initial_position,
            [initial_position[0] - self.block_size, initial_position[1]],
            [initial_position[0] - 2 * self.block_size, initial_position[1]]
        ]
        self.direction = 'UP'
        self.food = [random.randrange(1, self.width // self.block_size) * self.block_size,
                     random.randrange(1, self.height // self.block_size) * self.block_size]
        self.score = 0
        self.done = False
        return self.get_state()

    def step(self, action):
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        if self.direction != 'UP' and action == 1:
            self.direction = 'DOWN'
        elif self.direction != 'DOWN' and action == 0:
            self.direction = 'UP'
        elif self.direction != 'LEFT' and action == 3:
            self.direction = 'RIGHT'
        elif self.direction != 'RIGHT' and action == 2:
            self.direction = 'LEFT'

        x, y = self.snake[0]
        if self.direction == 'UP':
            y -= block_size
        elif self.direction == 'DOWN':
            y += block_size
        elif self.direction == 'LEFT':
            x -= block_size
        elif self.direction == 'RIGHT':
            x += block_size
        new_head = [x, y]

        if x < 0 or x >= width or y < 0 or y >= height:
            self.done = True
            reward = -500  # penalty of hit walls
            return self.get_state(), reward, self.done
        if new_head in self.snake:
            self.done = True
            reward = -200  # penalty of hit self
            return self.get_state(), reward, self.done

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward = 500  # reward of eating
            self.food = [random.randrange(1, width // block_size) * block_size,
                         random.randrange(1, height // block_size) * block_size]
        else:
            self.snake.pop()
            reward = -1

        return self.get_state(), reward, self.done

    def distance(self, point1, point2):
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

    def get_state(self):
        head = self.snake[0]
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]

        state = (
            head[0] // self.block_size, head[1] // self.block_size,
            self.food[0] // self.block_size, self.food[1] // self.block_size,
            int(self.direction == 'LEFT'), int(self.direction == 'RIGHT'),
            int(self.direction == 'UP'), int(self.direction == 'DOWN'),
            int(self.is_collision(point_l)), int(self.is_collision(point_r)),
            int(self.is_collision(point_u)), int(self.is_collision(point_d))
        )

        return state

    def is_collision(self, point):
        return point[0] < 0 or point[0] >= self.width or point[1] < 0 or point[1] >= self.height or point in self.snake

    def display(self, screen, action, reward):
        screen.fill(black)
        for part in self.snake:
            pygame.draw.rect(screen, green, pygame.Rect(part[0], part[1], self.block_size, self.block_size))
        pygame.draw.rect(screen, red, pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))

        # Display the action and reward
        font = pygame.font.SysFont("monospace", 15)
        action_text = font.render(f"Action: {['UP', 'DOWN', 'LEFT', 'RIGHT'][action]}", True, white)
        reward_text = font.render(f"Reward: {reward}", True, white)
        score_text = font.render(f"Score: {self.score}", True, white)
        screen.blit(action_text, (5, 5))
        screen.blit(reward_text, (5, 25))
        screen.blit(score_text, (5, 45))

        pygame.display.flip()


class QLearningAgent:
    def __init__(self, q_table):
        self.q_table = q_table

    def get_action(self, state):
        state_key = str(state)
        if state_key not in self.q_table:
            return random.randint(0, 3)  # random action
        else:
            return np.argmax(self.q_table[state_key])


def load_q_table():
    # load the pre-trained q-table
    q_table = np.load('q_table.npy', allow_pickle=True).item()
    return q_table


def run_game(q_table):
    pygame.init()
    pygame.display.set_caption("Snake Game")
    screen = pygame.display.set_mode((width, height))
    game = SnakeGame(width, height, block_size)
    agent = QLearningAgent(q_table)

    state = game.reset()
    done = False
    total_reward = 0
    clock = pygame.time.Clock()

    while not done:
        for event in pygame.event.get():  # Process the event
            if event.type == pygame.QUIT:
                done = True

        action = agent.get_action(state)
        next_state, reward, done = game.step(action)
        game.display(screen, action, reward)  # Display the action and reward
        state = next_state
        total_reward += reward
        clock.tick(12)  # Frame Control


if __name__ == "__main__":
    q_table = load_q_table()
    run_game(q_table)