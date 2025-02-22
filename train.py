import pygame
import random
import numpy as np

width, height = 640, 480

black = (0, 0, 0)
green = (0, 255, 0)
red = (255, 0, 0)

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
            reward = -500  # 撞墙的惩罚
            return self.get_state(), reward, self.done
        if new_head in self.snake:
            self.done = True
            reward = -200  # 撞到自身的严重惩罚
            return self.get_state(), reward, self.done

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward = 500  # 吃到食物的奖励
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

    def display(self, screen):
        screen.fill(black)
        for part in self.snake:
            pygame.draw.rect(screen, green, pygame.Rect(part[0], part[1], self.block_size, self.block_size))
        pygame.draw.rect(screen, red, pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        pygame.display.flip()


class QLearningAgent:
    def __init__(self, learning_rate=0.5, discount_rate=0.8, exploration_rate=0.45):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.exploration_rate = exploration_rate
        self.q_table = {}

    def get_action(self, state):
        state_key = str(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(4)

        if np.random.rand() < self.exploration_rate:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state_key])

    def update_q_table(self, state, action, reward, next_state, done):
        state_key = str(state)
        next_state_key = str(next_state)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(4)

        next_max = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] = (1 - self.learning_rate) * self.q_table[state_key][action] + \
                                          self.learning_rate * (reward + self.discount_rate * next_max)

        if done:
            self.exploration_rate *= 0.9999
            if self.exploration_rate < 0.1:
                self.exploration_rate = 0.1


def train_agent(num_episodes):
    game = SnakeGame(width, height, block_size)
    agent = QLearningAgent()
    for episode in range(num_episodes):
        state = game.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = game.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        print(f"Episode {episode}, Total reward: {total_reward}")
    return agent


def run_game(agent):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    game = SnakeGame(width, height, block_size)
    state = game.reset()
    done = False
    total_reward = 0

    while not done:
        game.display(screen)
        action = agent.get_action(state)
        next_state, reward, done = game.step(action)
        state = next_state
        total_reward += reward
        pygame.time.Clock().tick(12)

    pygame.quit()
    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    num_episodes = 10000000
    agent = train_agent(num_episodes)
    np.save('q_table.npy', agent.q_table)  # 保存Q-table
    run_game(agent)