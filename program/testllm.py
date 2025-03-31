import pygame
import sys
import math
import numpy as np

# 初始化Pygame
pygame.init()

# 屏幕设置
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("弹力小球游戏")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# 物理参数
gravity = 0.2  # 重力加速度
friction = 0.99  # 摩擦系数
elasticity = 0.8  # 弹性系数

# 六边形参数
hex_radius = 250  # 六边形半径
hex_center = (WIDTH // 2, HEIGHT // 2)  # 六边形中心
rotation_angle = 0  # 初始旋转角度
rotation_speed = 0.5  # 旋转速度(度/帧)

# 小球参数
ball_radius = 15
ball_pos = np.array([WIDTH // 2, HEIGHT // 3], dtype=float)
ball_vel = np.array([2.0, 0.0], dtype=float)


# 计算六边形的顶点
def calculate_hex_vertices(center, radius, rotation):
    vertices = []
    for i in range(6):
        angle_deg = 60 * i + rotation
        angle_rad = math.radians(angle_deg)
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        vertices.append((x, y))
    return vertices


# 计算点到线段的最短距离和最近点
def point_to_segment_distance(p, v1, v2):
    segment = np.array(v2) - np.array(v1)
    segment_length_squared = np.sum(segment ** 2)

    if segment_length_squared == 0:
        return np.linalg.norm(np.array(p) - np.array(v1)), np.array(v1)

    t = max(0, min(1, np.dot(np.array(p) - np.array(v1), segment) / segment_length_squared))
    projection = np.array(v1) + t * segment
    distance = np.linalg.norm(np.array(p) - projection)

    return distance, projection


# 检查小球与六边形的碰撞
def check_collision(ball_pos, ball_radius, hex_vertices):
    for i in range(6):
        v1 = hex_vertices[i]
        v2 = hex_vertices[(i + 1) % 6]

        distance, closest_point = point_to_segment_distance(ball_pos, v1, v2)

        if distance <= ball_radius:
            # 计算碰撞法线（从墙壁指向小球的方向）
            normal = np.array(ball_pos) - closest_point
            if np.linalg.norm(normal) > 0:
                normal = normal / np.linalg.norm(normal)

            # 计算反弹方向（反射向量）
            dot_product = np.dot(ball_vel, normal)
            reflection = ball_vel - 2 * dot_product * normal

            # 应用弹性和反弹
            ball_vel[0] = reflection[0] * elasticity
            ball_vel[1] = reflection[1] * elasticity

            # 防止小球卡在墙内
            overlap = ball_radius - distance
            ball_pos[0] += normal[0] * overlap
            ball_pos[1] += normal[1] * overlap

            return True
    return False


# 游戏主循环
clock = pygame.time.Clock()
running = True

while running:
    # 事件处理
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 更新旋转角度
    rotation_angle = (rotation_angle + rotation_speed) % 360

    # 计算六边形顶点
    hex_vertices = calculate_hex_vertices(hex_center, hex_radius, rotation_angle)

    # 应用重力
    ball_vel[1] += gravity

    # 应用摩擦力
    ball_vel *= friction

    # 更新小球位置
    ball_pos += ball_vel

    # 检测碰撞
    check_collision(ball_pos, ball_radius, hex_vertices)

    # 清屏
    screen.fill(WHITE)

    # 绘制六边形
    pygame.draw.polygon(screen, BLUE, hex_vertices, 2)

    # 绘制小球
    pygame.draw.circle(screen, RED, (int(ball_pos[0]), int(ball_pos[1])), ball_radius)

    # 更新显示
    pygame.display.flip()

    # 控制帧率
    clock.tick(60)

# 退出游戏
pygame.quit()
sys.exit()