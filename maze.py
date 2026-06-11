"""
======================================================================
專案名稱：基於圖論演算法之動態網格拓撲與智能尋路視覺化系統
======================================================================
核心技術：
  - 演算法動態對評：整合並對比非權重圖寻路（DFS, BFS）與啟發式最佳化（A*）
  - 非阻塞式協程機制：利用 Python Generator 實作 Randomized DFS 迷宮生成動畫
  - 動態拓撲路徑重算：結合道具破壞機制，即時更新網格連通性並觸發尋路引擎重算
  - 響應式佈局：依據動態矩陣維度（11x11 ~ 101x101）進行網格 Auto-scaling
======================================================================
"""

import pygame
import random
import heapq
import sys
import time
from collections import deque
from enum import Enum, auto

# ──────────────────────────────────────────────────────────────
#  全域常數與組態配置（Configuration）
# ──────────────────────────────────────────────────────────────
WIN_MAZE_W   = 700          # 迷宮渲染區塊固定寬度（像素）
HUD_H        = 90           # 下方資訊資訊欄（HUD）高度
FPS          = 60

MIN_DIM      = 11           # 最小網格維度（必須為奇數）
MAX_DIM      = 101          # 最大網格維度
DIM_STEP     = 10           # 維度動態調整步長

GEN_SPEED    = 20           # 生成演算法幀率（每秒更新網格數）
MON_SPEED    = 4            # 巡邏怪物移動步長（Hz）
NUM_BOMBS    = 3            # 靜態地圖中初始分佈之炸彈物件數

# 配色面板（Color Palette） ────────────────────────────────────
C_BG         = (15,  15,  22)   
C_WALL       = (38,  40,  58)   
C_PATH       = (222, 224, 232)  
C_GOAL_BG    = (40, 180,  90)   
C_GOAL_DOT   = (15, 100,  50)

C_PLAYER     = (230,  55,  55)  
C_MONSTER    = (180,  50, 225)  
C_BOMB       = (255, 145,  30)  

C_PATH_DFS   = (255, 165,  55)
C_PATH_BFS   = ( 80, 225, 175)
C_PATH_ASTAR = ( 75, 175, 255)

C_VIS_DFS    = ( 70,  45,  18)
C_VIS_BFS    = ( 20,  65,  52)
C_VIS_ASTAR  = ( 22,  48,  80)

C_GEN_STACK  = (160, 135,  35)  
C_GEN_HEAD   = (255, 235,  70)  

C_HUD_BG     = (  8,   8,  18)
C_HUD_BASE   = (190, 192, 208)

ALGO_NAMES   = {0: "無", 1: "DFS 搜尋", 2: "BFS 搜尋", 3: "A* 搜尋"}
ALGO_VIS     = {1: C_VIS_DFS,    2: C_VIS_BFS,    3: C_VIS_ASTAR}
ALGO_PATH    = {1: C_PATH_DFS,   2: C_PATH_BFS,   3: C_PATH_ASTAR}


class Phase(Enum):
    """ 系統狀態機（System State Machine）列舉 """
    GENERATING = auto()
    PLAYING    = auto()
    WON        = auto()
    CAUGHT     = auto()


# ══════════════════════════════════════════════════════════════
#  核心模組：隨機迷宮生成器（Randomized DFS Generator）
# ══════════════════════════════════════════════════════════════
class MazeGenerator:
    """
    基於深度優先搜尋（DFS）與堆疊回溯之迷宮生成器。
    採用非阻塞式單步設計，用以支援外部渲染迴圈之動態視覺化。
    """
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.maze: list[list[int]] = [[1] * cols for _ in range(rows)]
        self.stack: list[tuple[int, int]] = []
        self.stack_set: set[tuple[int, int]] = set() # 用於開銷 O(1) 的包含檢查
        self.done = False

        # 初始化空間起點
        sr, sc = 1, 1
        self.maze[sr][sc] = 0
        self.stack.append((sr, sc))
        self.stack_set.add((sr, sc))

    def _unvisited_neighbors(self, r: int, c: int) -> list[tuple[int, int, int, int]]:
        """ 計算指定網格距離為 2 的未訪問鄰居節點以及其中間牆壁座標 """
        result = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 1 <= nr <= self.rows - 2 and 1 <= nc <= self.cols - 2:
                if self.maze[nr][nc] == 1:
                    result.append((nr, nc, r + dr // 2, c + dc // 2))
        return result

    def step(self) -> None:
        """ 執行單步生成。驅動狀態機前進一個節點 """
        if self.done:
            return
        if not self.stack:
            self.done = True
            return

        r, c = self.stack[-1]
        nbrs = self._unvisited_neighbors(r, c)
        if nbrs:
            nr, nc, wr, wc = random.choice(nbrs)
            self.maze[wr][wc] = 0       # 打通連通牆面
            self.maze[nr][nc] = 0       # 宣告新通道節點
            self.stack.append((nr, nc))
            self.stack_set.add((nr, nc))
        else:
            popped = self.stack.pop()   # 無路可走，執行回溯（Backtrack）
            self.stack_set.discard(popped)

        if not self.stack:
            self.done = True

    def finish(self) -> None:
        """ 抑制動態動畫，直接收斂至演算法最終狀態 """
        while not self.done:
            self.step()


# ══════════════════════════════════════════════════════════════
#  圖論尋路模組（Graph Search Algorithms）
# ══════════════════════════════════════════════════════════════
def _reconstruct(came_from: dict[tuple[int, int], tuple[int, int] | None], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """ 溯源追蹤父節點指標，重構完整最優路徑陣列 """
    if goal not in came_from:
        return []
    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path


def search_dfs(maze: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """ 空間複雜度優化之迭代式深度優先搜尋（DFS） """
    rows, cols = len(maze), len(maze[0])
    visited: set[tuple[int, int]] = set()
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    stack = [start]

    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if (r, c) == goal:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and (nr, nc) not in visited:
                came_from.setdefault((nr, nc), (r, c))
                stack.append((nr, nc))

    return _reconstruct(came_from, goal), visited


def search_bfs(maze: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """ 基於雙端佇列（Deque）之廣度優先搜尋（BFS），保證非權重圖之最短路徑 """
    rows, cols = len(maze), len(maze[0])
    visited: set[tuple[int, int]] = {start}
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                came_from[(nr, nc)] = (r, c)
                queue.append((nr, nc))

    return _reconstruct(came_from, goal), visited


def search_astar(maze: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    """ A* 尋路演算法，採用曼哈頓距離（Manhattan Distance）作為一致性啟發標記 """
    rows, cols = len(maze), len(maze[0])

    def h(r: int, c: int) -> int:
        return abs(r - goal[0]) + abs(c - goal[1])

    # 最小優先佇列優化：(F, G, r, c)
    open_heap = [(h(*start), 0, start[0], start[1])]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    g_score = {start: 0}
    visited: set[tuple[int, int]] = set()

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)
        if (r, c) in visited:
            continue
        visited.add((r, c))
        if (r, c) == goal:
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr][nc] == 0:
                ng = g + 1
                if (nr, nc) not in g_score or ng < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + h(nr, nc), ng, nr, nc))

    return _reconstruct(came_from, goal), visited


ALGO_FUNCS = {1: search_dfs, 2: search_bfs, 3: search_astar}


# ══════════════════════════════════════════════════════════════
#  動態實體模組：自主巡邏怪物（Patrolling Agent）
# ══════════════════════════════════════════════════════════════
class Monster:
    """ 實作非決策型隨機巡邏智能體 """
    def __init__(self, maze: list[list[int]], forbidden: set[tuple[int, int]]):
        rows, cols = len(maze), len(maze[0])
        self.mov_interval = 1.0 / MON_SPEED
        # 生成於與起點具備拓撲距離之安全網格
        min_dist = max(rows, cols) // 3
        candidates = [
            (r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)
            if maze[r][c] == 0 and (r, c) not in forbidden and abs(r - 1) + abs(c - 1) > min_dist
        ]
        if not candidates:
            candidates = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1) if maze[r][c] == 0 and (r, c) not in forbidden]
        
        self.pos: tuple[int, int] = random.choice(candidates) if candidates else (rows // 2, cols // 2)
        self._timer = 0.0

    def update(self, dt: float, maze: list[list[int]]) -> None:
        """ 基於動態時間增量（dt）控制怪物之移動狀態 """
        self._timer += dt
        if self._timer < self.mov_interval:
            return
        self._timer -= self.mov_interval
        r, c = self.pos
        rows, cols = len(maze), len(maze[0])
        options = [
            (r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < rows and 0 <= c + dc < cols and maze[r + dr][c + dc] == 0
        ]
        if options:
            self.pos = random.choice(options)


# ══════════════════════════════════════════════════════════════
#  系統整合控制器（Main System Controller）
# ══════════════════════════════════════════════════════════════
class MazeGame:
    def __init__(self):
        pygame.init()
        self.rows = 21
        self.cols = 21
        self.screen = pygame.display.set_mode((WIN_MAZE_W, WIN_MAZE_W + HUD_H))
        pygame.display.set_caption("專案展示 ─ 動態演算法視覺化尋路系統")
        self.clock = pygame.time.Clock()
        self._init_fonts()
        self._new_game()

    def _init_fonts(self) -> None:
        for name in ("microsoftjhenghei", "mingliu", "pingfangtc", "arial", None):
            try:
                self.font_sm  = pygame.font.SysFont(name, 14, bold=True)
                self.font_mid = pygame.font.SysFont(name, 18, bold=True)
                self.font_big = pygame.font.SysFont(name, 28, bold=True)
                break
            except Exception:
                continue

    def _calc_layout(self) -> None:
        """ 響應式網格計算（Auto-scaling Layout Engine） """
        dim = max(self.rows, self.cols)
        unit = WIN_MAZE_W // dim
        unit = max(3, unit)
        margin = max(1, unit // 8)
        cell = unit - margin
        self.margin = margin
        self.cell = max(2, cell)
        self.maze_pw = self.cols * (self.cell + margin) + margin
        self.maze_ph = self.rows * (self.cell + margin) + margin

    def _new_game(self) -> None:
        """ 重設狀態機與重置遊戲內容 """
        self._calc_layout()
        self.WIN_W = self.maze_pw
        self.WIN_H = self.maze_ph + HUD_H
        self.screen = pygame.display.set_mode((self.WIN_W, self.WIN_H))

        self.gen = MazeGenerator(self.rows, self.cols)
        self.gen_acc = 0.0
        self.phase = Phase.GENERATING

        self.player = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)
        self.steps = 0

        # 演算法資料緩衝區
        self.algo_mode = 0
        self.algo_path: list[tuple[int, int]] = []
        self.algo_vis: set[tuple[int, int]] = set()
        self.algo_ms = 0.0
        self.algo_count = 0

        # 動態拓撲實體
        self.bombs: list[tuple[int, int]] = []
        self.held_bombs = 0
        self.monster = None

    def _post_gen_setup(self) -> None:
        """ 拓撲完整後之靜態物件撒佈與動態 agent 注入 """
        maze = self.gen.maze
        maze[self.goal[0]][self.goal[1]] = 0   

        passages = [(r, c) for r in range(1, self.rows - 1) for c in range(1, self.cols - 1) if maze[r][c] == 0 and (r, c) not in ((1, 1), self.goal)]
        random.shuffle(passages)
        self.bombs = passages[:NUM_BOMBS]

        forbidden = {(1, 1), self.goal} | set(self.bombs)
        try:
            self.monster = Monster(maze, forbidden)
        except Exception:
            self.monster = None

    def _cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = self.margin + c * (self.cell + self.margin)
        y = self.margin + r * (self.cell + self.margin)
        return pygame.Rect(x, y, self.cell, self.cell)

    def _draw_dot(self, r: int, c: int, color: tuple, ratio: float = 0.38) -> None:
        rect = self._cell_rect(r, c)
        radius = max(1, int(self.cell * ratio))
        pygame.draw.circle(self.screen, color, rect.center, radius)

    def _run_algo(self, mode: int) -> None:
        """ 調用選定之圖論尋路引擎並測量其時間、空間開銷 """
        if mode == 0:
            self.algo_mode, self.algo_path, self.algo_vis, self.algo_ms, self.algo_count = 0, [], set(), 0.0, 0
            return
        
        fn = ALGO_FUNCS[mode]
        t0 = time.perf_counter()
        path, visited = fn(self.gen.maze, self.player, self.goal)
        elapsed = (time.perf_counter() - t0) * 1000   
        
        self.algo_mode = mode
        self.algo_path = path
        self.algo_vis = visited
        self.algo_ms = round(elapsed, 3)
        self.algo_count = len(visited)

    def _use_bomb(self) -> None:
        """ 觸發動態拓撲修改（炸牆），並同步要求尋路引擎即時更新（Path Re-planning） """
        if self.held_bombs <= 0:
            return
        maze = self.gen.maze
        pr, pc = self.player
        destroyed = False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = pr + dr, pc + dc
            if 0 < nr < self.rows - 1 and 0 < nc < self.cols - 1: # 保護最外層邊界牆
                if maze[nr][nc] == 1:
                    maze[nr][nc] = 0
                    destroyed = True
        if destroyed:
            self.held_bombs -= 1
            if self.algo_mode:
                self._run_algo(self.algo_mode)

    def _move(self, dr: int, dc: int) -> None:
        nr, nc = self.player[0] + dr, self.player[1] + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols) or self.gen.maze[nr][nc] != 0:
            return
        
        self.player = (nr, nc)
        self.steps += 1
        
        if (nr, nc) in self.bombs:
            self.bombs.remove((nr, nc))
            self.held_bombs += 1
            
        if self.algo_mode:
            self._run_algo(self.algo_mode)
            
        if (nr, nc) == self.goal:
            self.phase = Phase.WON

    def handle_events(self) -> None:
        """ 集中式事件分發系統（Centralized Event Dispatcher） """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type != pygame.KEYDOWN:
                continue

            key = event.key

            if key == pygame.K_r:
                self._new_game()
                return

            # 地圖維度動態放大
            if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                new_dim = min(MAX_DIM, self.rows + DIM_STEP)
                self.rows = self.cols = new_dim if new_dim % 2 != 0 else new_dim + 1
                self._new_game()
                return

            # 地圖維度動態縮小
            if key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                new_dim = max(MIN_DIM, self.rows - DIM_STEP)
                self.rows = self.cols = new_dim if new_dim % 2 != 0 else new_dim - 1
                self._new_game()
                return

            if self.phase == Phase.GENERATING:
                if key == pygame.K_RETURN:
                    self.gen.finish()
                    self._post_gen_setup()
                    self.phase = Phase.PLAYING
                return   

            if self.phase != Phase.PLAYING:
                return

            if key == pygame.K_1:   self._run_algo(0 if self.algo_mode == 1 else 1)
            elif key == pygame.K_2: self._run_algo(0 if self.algo_mode == 2 else 2)
            elif key == pygame.K_3: self._run_algo(0 if self.algo_mode == 3 else 3)
            elif key == pygame.K_b: self._use_bomb()
            else:
                direction = {
                    pygame.K_UP: (-1, 0), pygame.K_w: (-1, 0),
                    pygame.K_DOWN: (1, 0), pygame.K_s: (1, 0),
                    pygame.K_LEFT: (0, -1), pygame.K_a: (0, -1),
                    pygame.K_RIGHT: (0, 1), pygame.K_d: (0, 1)
                }.get(key)
                if direction:
                    self._move(*direction)

    def update(self, dt: float) -> None:
        """ 更新狀態機之核心邏輯 """
        if self.phase == Phase.GENERATING:
            self.gen_acc += dt
            interval = 1.0 / GEN_SPEED
            while self.gen_acc >= interval:
                self.gen_acc -= interval
                self.gen.step()
            if self.gen.done:
                self._post_gen_setup()
                self.phase = Phase.PLAYING
            return

        if self.phase == Phase.PLAYING and self.monster:
            self.monster.update(dt, self.gen.maze)
            if self.monster.pos == self.player:
                self.phase = Phase.CAUGHT

    def draw(self) -> None:
        """ 多圖層雙緩衝渲染管線（Rendering Pipeline） """
        self.screen.fill(C_BG)
        maze = self.gen.maze

        # Layer 1: 基礎靜態網格
        for r in range(self.rows):
            for c in range(self.cols):
                color = C_WALL if maze[r][c] == 1 else C_PATH
                pygame.draw.rect(self.screen, color, self._cell_rect(r, c))

        # Layer 2: 迷宮建構動畫層
        if self.phase == Phase.GENERATING:
            for pos in self.gen.stack_set:
                pygame.draw.rect(self.screen, C_GEN_STACK, self._cell_rect(*pos))
            if self.gen.stack:
                pygame.draw.rect(self.screen, C_GEN_HEAD, self._cell_rect(*self.gen.stack[-1]))

        # Layer 3: 尋路演算法閉鎖區與最短路徑追蹤
        if self.phase != Phase.GENERATING and self.algo_mode:
            vc, pc = ALGO_VIS[self.algo_mode], ALGO_PATH[self.algo_mode]
            excl = {self.player, self.goal}
            for pos in self.algo_vis:
                if pos not in excl:
                    pygame.draw.rect(self.screen, vc, self._cell_rect(*pos))
            for pos in self.algo_path:
                if pos not in excl:
                    self._draw_dot(*pos, pc, ratio=0.28)

        # Layer 4: 目標拓撲與靜態道具渲染
        gr, gc = self.goal
        pygame.draw.rect(self.screen, C_GOAL_BG, self._cell_rect(gr, gc))
        self._draw_dot(gr, gc, C_GOAL_DOT, ratio=0.26)
        for br, bc in self.bombs:
            self._draw_dot(br, bc, C_BOMB, ratio=0.30)

        # Layer 5: 動態 Agent（玩家與追逐實體）
        if self.monster and self.phase != Phase.GENERATING:
            self._draw_dot(self.monster.pos[0], self.monster.pos[1], C_MONSTER, ratio=0.40)
        if self.phase != Phase.GENERATING:
            self._draw_dot(self.player[0], self.player[1], C_PLAYER, ratio=0.42)

        # Layer 6: 使用者介面疊加層（HUD & Overlay）
        self._draw_hud()
        if self.phase in (Phase.WON, Phase.CAUGHT):
            self._draw_overlay()

        pygame.display.flip()

    def _draw_hud(self) -> None:
        """ 繪製包含即時演算法複雜度效能之控制面板 """
        hud_top = self.maze_ph
        pygame.draw.rect(self.screen, C_HUD_BG, pygame.Rect(0, hud_top, self.WIN_W, HUD_H))
        pygame.draw.line(self.screen, (55, 58, 78), (0, hud_top), (self.WIN_W, hud_top), 1)

        lx, ly = 10, hud_top + 8

        if self.phase == Phase.GENERATING:
            surf = self.font_sm.render(f"生成中… Stack 深度: {len(self.gen.stack)}   [Enter] 跳過動畫", True, C_GEN_HEAD)
        else:
            if self.algo_mode:
                surf = self.font_sm.render(f"[ {ALGO_NAMES[self.algo_mode]} ]  探索空間(V): {self.algo_count} 格 | 運算耗時: {self.algo_ms:.3f} ms", True, ALGO_PATH[self.algo_mode])
            else:
                surf = self.font_sm.render("按鍵控制 1:DFS | 2:BFS | 3:A* 尋路（再按一次關閉提示）", True, C_HUD_BASE)
        self.screen.blit(surf, (lx, ly))

        line2 = f"玩家步數: {self.steps}   持有炸彈: {self.held_bombs} (按B使用)   矩陣維度: {self.rows}x{self.cols} (+/-調整)   R:重開"
        self.screen.blit(self.font_sm.render(line2, True, C_HUD_BASE), (lx, ly + 22))

        if self.algo_mode and self.phase != Phase.GENERATING:
            s1 = self.font_sm.render("■ 拓撲最優路徑", True, ALGO_PATH[self.algo_mode])
            vc = ALGO_VIS[self.algo_mode]
            s2 = self.font_sm.render("  ■ 演算法閉鎖區(Closed Set)", True, (min(255, vc[0]*3), min(255, vc[1]*3), min(255, vc[2]*3)))
            self.screen.blit(s1, (lx, ly + 44))
            self.screen.blit(s2, (lx + 130, ly + 44))

    def _draw_overlay(self) -> None:
        """ 狀態終止覆蓋層 """
        overlay = pygame.Surface((self.WIN_W, self.maze_ph), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 165))
        self.screen.blit(overlay, (0, 0))
        cx, cy = self.WIN_W // 2, self.maze_ph // 2

        if self.phase == Phase.WON:
            surf = self.font_big.render(f"SUCCESS! 共耗費 {self.steps} 步", True, (255, 215, 45))
        else:
            surf = self.font_big.render("GAME OVER! 被怪物突襲", True, (255, 75, 75))
        
        self.screen.blit(surf, surf.get_rect(center=(cx, cy - 20)))
        self.screen.blit(self.font_mid.render("請按 R 鍵重新初始化地圖", True, (200, 200, 205)), self.font_mid.render("請按 R 鍵重新初始化地圖", True, (200, 200, 205)).get_rect(center=(cx, cy + 26)))

    def run(self) -> None:
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    MazeGame().run()
