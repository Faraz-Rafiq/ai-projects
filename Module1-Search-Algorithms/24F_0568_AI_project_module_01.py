# MODULE 1
# Intelligent Urban Delivery Robot
# Name: M Faraz Rafiq
# Roll No: 24F-0568
#
# this program simulates a delivery robot on a 15x15 city grid
# robot starts from base and delivers 5 packages using different search algorithms
# buildings are obstacles, traffic zones have high cost
# algorithms used: BFS, DFS, UCS, Greedy Best First, A*
import random
import time
import heapq
import math
import tkinter as tk
from tkinter import ttk
from collections import deque

# grid is 15x15
GRID_SIZE = 15
CELL_SIZE = 44
PADDING = 10

# cell type numbers
ROAD = 0
BUILDING = 1
TRAFFIC = 2
BASE = 3
DELIVERY = 4

# colors for drawing the grid
COLORS = {
    "road": "#1a1a2e",
    "building": "#16213e",
    "traffic": "#8b0000",
    "base": "#006400",
    "delivery": "#4b0082",
    "path": "#ffd700",
    "visited": "#1e90ff",
    "robot": "#ff6347",
    "bg": "#0f0f1a",
    "panel": "#16213e",
    "text": "#e0e0e0",
    "border": "#2d4a7a",
}

# GridEnvironment class
# this creates and stores the city grid
class GridEnvironment:
    # constructor, sets up empty grid then calls setup function
    def __init__(self):
        self.grid = [[ROAD] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.cost_grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.base_station = (0, 0)
        self.delivery_points = []
        self._setup_grid()

    def _setup_grid(self):
        # this function fills the grid with buildings, traffic, delivery points
        random.seed(42)

        # place buildings randomly, roughly 22% of total cells
        buildings_placed = 0
        target_buildings = int(GRID_SIZE * GRID_SIZE * 0.22)
        while buildings_placed < target_buildings:
            r = random.randint(0, GRID_SIZE - 1)
            c = random.randint(0, GRID_SIZE - 1)
            if (r, c) != (0, 0) and self.grid[r][c] == ROAD:
                self.grid[r][c] = BUILDING
                buildings_placed += 1

        # make sure borders are roads so path always exists
        for r in range(GRID_SIZE):
            self.grid[r][0] = ROAD
            self.grid[r][GRID_SIZE - 1] = ROAD
        for c in range(GRID_SIZE):
            self.grid[0][c] = ROAD
            self.grid[GRID_SIZE - 1][c] = ROAD

        # place traffic zones, roughly 12% of cells
        traffic_placed = 0
        target_traffic = int(GRID_SIZE * GRID_SIZE * 0.12)
        while traffic_placed < target_traffic:
            r = random.randint(0, GRID_SIZE - 1)
            c = random.randint(0, GRID_SIZE - 1)
            if self.grid[r][c] == ROAD and (r, c) != (0, 0):
                self.grid[r][c] = TRAFFIC
                traffic_placed += 1

        # assign cost to each cell
        # road = 1 to 5, traffic = 10 to 20, building = 0 (cant go there)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell_type = self.grid[r][c]
                if cell_type in (ROAD, BASE):
                    self.cost_grid[r][c] = random.randint(1, 5)
                elif cell_type == TRAFFIC:
                    self.cost_grid[r][c] = random.randint(10, 20)
                else:
                    self.cost_grid[r][c] = 0

        # base station is at top left corner
        self.grid[0][0] = BASE
        self.cost_grid[0][0] = 1

        # pick 5 delivery locations randomly
        possible = []
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if self.grid[r][c] in (ROAD, TRAFFIC) and (r, c) != (0, 0):
                    possible.append((r, c))

        random.shuffle(possible)
        for pos in possible[:5]:
            self.delivery_points.append(pos)
            self.grid[pos[0]][pos[1]] = DELIVERY
            self.cost_grid[pos[0]][pos[1]] = random.randint(1, 5)

    def get_neighbors(self, row, col):
        # returns neighbors in 4 directions, skips buildings
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_r = row + dr
            new_c = col + dc
            if 0 <= new_r < GRID_SIZE and 0 <= new_c < GRID_SIZE:
                if self.grid[new_r][new_c] != BUILDING:
                    move_cost = self.cost_grid[new_r][new_c]
                    neighbors.append(((new_r, new_c), move_cost))
        return neighbors

# heuristic functions for greedy and astar
def manhattan_distance(pos1, pos2):
    # sum of horizontal + vertical distance
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance(pos1, pos2):
    # straight line distance using pythagorean theorem
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

# helper to trace path back from goal to start using parent dictionary
def rebuild_path(parent_map, start, goal):
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent_map[current]
    path.append(start)
    path.reverse()
    return path

# BFS - Breadth First Search
# uses a queue, explores level by level
# not optimal for weighted graphs but works fine
def bfs(env, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    nodes_explored = 0
    start_time = time.perf_counter()

    while queue:
        current = queue.popleft()
        nodes_explored += 1
        if current == goal:
            break
        for neighbor, _ in env.get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    elapsed = time.perf_counter() - start_time

    if goal not in parent:
        return None, 0, nodes_explored, elapsed, list(visited)

    path = rebuild_path(parent, start, goal)
    total_cost = sum(env.cost_grid[r][c] for r, c in path)
    return path, total_cost, nodes_explored, elapsed, list(visited)

# DFS - Depth First Search
# uses a stack, goes deep first
# can find very long paths, not optimal
def dfs(env, start, goal):
    stack = [start]
    visited = {start}
    parent = {start: None}
    nodes_explored = 0
    start_time = time.perf_counter()

    while stack:
        current = stack.pop()
        nodes_explored += 1
        if current == goal:
            break
        for neighbor, _ in env.get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                stack.append(neighbor)

    elapsed = time.perf_counter() - start_time

    if goal not in parent:
        return None, 0, nodes_explored, elapsed, list(visited)

    path = rebuild_path(parent, start, goal)
    total_cost = sum(env.cost_grid[r][c] for r, c in path)
    return path, total_cost, nodes_explored, elapsed, list(visited)

# UCS - Uniform Cost Search
# uses priority queue sorted by cost
# always expands cheapest path, so it finds optimal path
def ucs(env, start, goal):
    priority_queue = [(0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    visited = set()
    nodes_explored = 0
    start_time = time.perf_counter()

    while priority_queue:
        cost, current = heapq.heappop(priority_queue)
        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1
        if current == goal:
            break
        for neighbor, step_cost in env.get_neighbors(*current):
            new_cost = cost + step_cost
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                parent[neighbor] = current
                heapq.heappush(priority_queue, (new_cost, neighbor))

    elapsed = time.perf_counter() - start_time

    if goal not in parent:
        return None, 0, nodes_explored, elapsed, list(visited)

    path = rebuild_path(parent, start, goal)
    return path, g_cost.get(goal, 0), nodes_explored, elapsed, list(visited)

# Greedy Best First Search
# uses heuristic to move toward goal
# fast but may not find cheapest path
def greedy_search(env, start, goal, heuristic_type="manhattan"):
    if heuristic_type == "manhattan":
        h_func = manhattan_distance
    else:
        h_func = euclidean_distance

    priority_queue = [(h_func(start, goal), start)]
    visited = {start}
    parent = {start: None}
    nodes_explored = 0
    start_time = time.perf_counter()

    while priority_queue:
        _, current = heapq.heappop(priority_queue)
        nodes_explored += 1
        if current == goal:
            break
        for neighbor, _ in env.get_neighbors(*current):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                h_val = h_func(neighbor, goal)
                heapq.heappush(priority_queue, (h_val, neighbor))

    elapsed = time.perf_counter() - start_time

    if goal not in parent:
        return None, 0, nodes_explored, elapsed, list(visited)

    path = rebuild_path(parent, start, goal)
    total_cost = sum(env.cost_grid[r][c] for r, c in path)
    return path, total_cost, nodes_explored, elapsed, list(visited)

# A* Search
# f(n) = g(n) + h(n)
# g(n) = actual cost so far, h(n) = heuristic estimate to goal
# best algorithm, finds optimal path efficiently
def astar_search(env, start, goal, heuristic_type="manhattan"):
    if heuristic_type == "manhattan":
        h_func = manhattan_distance
    else:
        h_func = euclidean_distance

    start_h = h_func(start, goal)
    priority_queue = [(start_h, 0, start)]
    g_cost = {start: 0}
    parent = {start: None}
    visited = set()
    nodes_explored = 0
    start_time = time.perf_counter()

    while priority_queue:
        _, cost, current = heapq.heappop(priority_queue)
        if current in visited:
            continue
        visited.add(current)
        nodes_explored += 1
        if current == goal:
            break
        for neighbor, step_cost in env.get_neighbors(*current):
            new_g = cost + step_cost
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                parent[neighbor] = current
                h_val = h_func(neighbor, goal)
                f_new = new_g + h_val
                heapq.heappush(priority_queue, (f_new, new_g, neighbor))

    elapsed = time.perf_counter() - start_time

    if goal not in parent:
        return None, 0, nodes_explored, elapsed, list(visited)

    path = rebuild_path(parent, start, goal)
    return path, g_cost.get(goal, 0), nodes_explored, elapsed, list(visited)

# all algorithms in one dict to call easily
ALL_ALGORITHMS = {
    "BFS": lambda e, s, g: bfs(e, s, g),
    "DFS": lambda e, s, g: dfs(e, s, g),
    "UCS": lambda e, s, g: ucs(e, s, g),
    "Greedy (Manhattan)": lambda e, s, g: greedy_search(e, s, g, "manhattan"),
    "Greedy (Euclidean)": lambda e, s, g: greedy_search(e, s, g, "euclidean"),
    "A* (Manhattan)": lambda e, s, g: astar_search(e, s, g, "manhattan"),
    "A* (Euclidean)": lambda e, s, g: astar_search(e, s, g, "euclidean"),
}

ALGO_COLORS = {
    "BFS": "#38bdf8",
    "DFS": "#f87171",
    "UCS": "#34d399",
    "Greedy (Manhattan)": "#fb923c",
    "Greedy (Euclidean)": "#fbbf24",
    "A* (Manhattan)": "#a78bfa",
    "A* (Euclidean)": "#c084fc",
}

# main GUI class
# this handles all the tkinter window and widgets
class RobotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AL-2002 - Urban Delivery Robot")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)

        self.env = GridEnvironment()
        self.display_grid = [row[:] for row in self.env.grid]
        self.robot_pos = self.env.base_station
        self.delivery_index = 0
        self.is_running = False
        self.results_list = []

        self.selected_algo = tk.StringVar(value="A* (Manhattan)")
        self.anim_speed = tk.IntVar(value=60)

        self._build_interface()
        self._redraw_grid()

    def _build_interface(self):
        # builds all widgets

        # top header bar
        header = tk.Frame(self.root, bg="#050d1a")
        header.pack(fill="x")
        tk.Frame(header, bg=COLORS["border"], height=2).pack(fill="x")

        header_inner = tk.Frame(header, bg="#050d1a")
        header_inner.pack(fill="x", padx=15, pady=8)
        tk.Label(header_inner,
                 text="INTELLIGENT URBAN DELIVERY ROBOT",
                 font=("Courier New", 12, "bold"),
                 bg="#050d1a", fg=COLORS["text"]).pack(side="left")

        self.status_label = tk.Label(header_inner, text="STATUS: IDLE",
                                     font=("Courier New", 9),
                                     bg="#050d1a", fg="#888888")
        self.status_label.pack(side="right")

        # main area, grid on left and controls on right
        main_body = tk.Frame(self.root, bg=COLORS["bg"])
        main_body.pack(padx=10, pady=8)

        canvas_size = GRID_SIZE * CELL_SIZE + 2 * PADDING
        self.canvas = tk.Canvas(main_body,
                                width=canvas_size, height=canvas_size,
                                bg=COLORS["bg"], highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=(0, 10))

        right_panel = tk.Frame(main_body, bg=COLORS["bg"], width=290)
        right_panel.grid(row=0, column=1, sticky="ns")
        right_panel.grid_propagate(False)

        # helper to create section boxes
        def make_section(label, color="#2d4a7a"):
            outer = tk.Frame(right_panel, bg=COLORS["bg"])
            outer.pack(fill="x", pady=(0, 7))
            title_row = tk.Frame(outer, bg=COLORS["bg"])
            title_row.pack(fill="x", pady=(0, 2))
            tk.Label(title_row, text=label,
                     font=("Courier New", 8, "bold"),
                     bg=COLORS["bg"], fg=color).pack(side="left")
            inner = tk.Frame(outer, bg=COLORS["panel"],
                             highlightbackground=COLORS["border"],
                             highlightthickness=1)
            inner.pack(fill="x")
            return inner

        # algorithm selection using radio buttons
        algo_frame = make_section("[ ALGORITHM ]", "#38bdf8")
        for algo_name in ALL_ALGORITHMS:
            dot_color = ALGO_COLORS.get(algo_name, "#ffffff")
            row_frame = tk.Frame(algo_frame, bg=COLORS["panel"], cursor="hand2")
            row_frame.pack(fill="x", padx=6, pady=2)
            tk.Canvas(row_frame, width=8, height=8,
                      bg=dot_color, highlightthickness=0).pack(side="left", padx=(4, 6))
            rb = tk.Radiobutton(row_frame, text=algo_name,
                                variable=self.selected_algo, value=algo_name,
                                font=("Courier New", 8),
                                bg=COLORS["panel"], fg=COLORS["text"],
                                selectcolor=COLORS["panel"],
                                activebackground=COLORS["panel"],
                                activeforeground=COLORS["text"],
                                cursor="hand2")
            rb.pack(side="left")

        # speed slider
        speed_frame = make_section("[ SPEED ]", "#fb923c")
        speed_row = tk.Frame(speed_frame, bg=COLORS["panel"])
        speed_row.pack(fill="x", padx=6, pady=6)
        tk.Label(speed_row, text="Slow", font=("Courier New", 7),
                 bg=COLORS["panel"], fg="#666666").pack(side="left")
        tk.Scale(speed_row, from_=250, to=10, orient="horizontal",
                 variable=self.anim_speed,
                 bg=COLORS["panel"], fg=COLORS["text"],
                 troughcolor=COLORS["bg"],
                 highlightthickness=0, length=175,
                 showvalue=False).pack(side="left", padx=3)
        tk.Label(speed_row, text="Fast", font=("Courier New", 7),
                 bg=COLORS["panel"], fg="#666666").pack(side="left")

        # control buttons
        btn_frame = make_section("[ CONTROLS ]", "#34d399")
        buttons_inner = tk.Frame(btn_frame, bg=COLORS["panel"])
        buttons_inner.pack(fill="x", padx=6, pady=6)

        self.run_btn = tk.Button(buttons_inner,
                                 text="RUN DELIVERY",
                                 font=("Courier New", 9, "bold"),
                                 bg="#006400", fg="white",
                                 relief="flat", cursor="hand2", pady=7,
                                 command=self._start_delivery)
        self.run_btn.pack(fill="x", pady=(0, 4))

        tk.Button(buttons_inner, text="RESET",
                  font=("Courier New", 9, "bold"),
                  bg=COLORS["panel"], fg=COLORS["text"],
                  relief="flat", cursor="hand2", pady=6,
                  highlightbackground=COLORS["border"],
                  highlightthickness=1,
                  command=self._reset).pack(fill="x", pady=(0, 4))

        tk.Button(buttons_inner, text="COMPARE ALGORITHMS",
                  font=("Courier New", 8, "bold"),
                  bg="#4b0082", fg="white",
                  relief="flat", cursor="hand2", pady=6,
                  command=self._show_comparison).pack(fill="x")

        # live stats display
        stats_frame = make_section("[ LIVE STATS ]", "#38bdf8")
        stats_grid = tk.Frame(stats_frame, bg=COLORS["panel"])
        stats_grid.pack(fill="x", padx=6, pady=6)
        stats_grid.columnconfigure(0, weight=1)
        stats_grid.columnconfigure(1, weight=1)

        self.stat_vars = {}
        stat_defs = [
            ("delivery", "Delivery", "#34d399", 0, 0),
            ("cost", "Path Cost", "#fb923c", 0, 1),
            ("nodes", "Nodes", "#38bdf8", 1, 0),
            ("time_ms", "Time(ms)", "#a78bfa", 1, 1),
        ]
        for key, label, col, r2, c2 in stat_defs:
            cell = tk.Frame(stats_grid, bg=COLORS["panel"])
            cell.grid(row=r2, column=c2, padx=4, pady=3, sticky="ew")
            v = tk.StringVar(value="--")
            self.stat_vars[key] = v
            tk.Label(cell, text=label, font=("Courier New", 7),
                     bg=COLORS["panel"], fg="#666666").pack(anchor="w")
            tk.Label(cell, textvariable=v,
                     font=("Courier New", 13, "bold"),
                     bg=COLORS["panel"], fg=col).pack(anchor="w")

        # log box to show delivery messages
        log_frame = make_section("[ LOG ]", "#fb923c")
        log_inner = tk.Frame(log_frame, bg=COLORS["panel"])
        log_inner.pack(fill="both", padx=4, pady=4)

        self.log_box = tk.Text(log_inner, height=8,
                               bg="#050d1a", fg=COLORS["text"],
                               font=("Courier New", 7),
                               relief="flat", state="disabled", wrap="word")
        self.log_box.pack(fill="both")
        self.log_box.tag_config("green", foreground="#34d399")
        self.log_box.tag_config("yellow", foreground="#ffd700")
        self.log_box.tag_config("blue", foreground="#38bdf8")
        self.log_box.tag_config("gray", foreground="#666666")

        # legend
        legend_frame = make_section("[ LEGEND ]", "#888888")
        leg_inner = tk.Frame(legend_frame, bg=COLORS["panel"])
        leg_inner.pack(fill="x", padx=6, pady=6)
        leg_inner.columnconfigure(0, weight=1)
        leg_inner.columnconfigure(1, weight=1)

        legend_items = [
            (COLORS["base"], "Base Station"),
            (COLORS["delivery"], "Delivery Point"),
            (COLORS["building"], "Building"),
            (COLORS["traffic"], "Traffic Zone"),
            (COLORS["road"], "Road Cell"),
            (COLORS["path"], "Path"),
            (COLORS["visited"], "Visited"),
            (COLORS["robot"], "Robot"),
        ]
        for i, (col, lbl) in enumerate(legend_items):
            item_row = tk.Frame(leg_inner, bg=COLORS["panel"])
            item_row.grid(row=i // 2, column=i % 2, sticky="w", padx=3, pady=1)
            tk.Canvas(item_row, width=9, height=9,
                      bg=col, highlightthickness=0).pack(side="left", padx=(0, 4))
            tk.Label(item_row, text=lbl, font=("Courier New", 7),
                     bg=COLORS["panel"], fg="#888888").pack(side="left")

    def _redraw_grid(self, visited_cells=None, path_cells=None, robot_at=None):
        # redraws canvas with current grid state
        self.canvas.delete("all")

        v_set = set(visited_cells) if visited_cells else set()
        p_set = set(path_cells) if path_cells else set()

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x1 = PADDING + c * CELL_SIZE
                y1 = PADDING + r * CELL_SIZE
                x2 = x1 + CELL_SIZE - 1
                y2 = y1 + CELL_SIZE - 1
                cx = x1 + CELL_SIZE // 2
                cy = y1 + CELL_SIZE // 2
                pos = (r, c)
                cell_type = self.display_grid[r][c]

                # color depends on what kind of cell it is
                if pos in p_set:
                    fill_color = COLORS["path"]
                    border_col = "#ffed4a"
                elif pos in v_set:
                    fill_color = COLORS["visited"]
                    border_col = "#0066cc"
                elif cell_type == BUILDING:
                    fill_color = COLORS["building"]
                    border_col = "#1a3050"
                elif cell_type == TRAFFIC:
                    fill_color = COLORS["traffic"]
                    border_col = "#660000"
                elif cell_type == BASE:
                    fill_color = COLORS["base"]
                    border_col = "#004d00"
                elif cell_type == DELIVERY:
                    fill_color = COLORS["delivery"]
                    border_col = "#380060"
                else:
                    fill_color = COLORS["road"]
                    border_col = "#111122"

                self.canvas.create_rectangle(x1, y1, x2, y2,
                                             fill=fill_color,
                                             outline=border_col, width=1)

                # draw icon or text inside cell
                if pos in p_set:
                    self.canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                            fill="#ffed4a", outline="")
                elif pos in v_set:
                    self.canvas.create_oval(cx - 2, cy - 2, cx + 2, cy + 2,
                                            fill="#66aaff", outline="")
                elif cell_type == BUILDING:
                    self.canvas.create_text(cx, cy, text="B",
                                            fill="#2d4a63",
                                            font=("Courier New", 9, "bold"))
                elif cell_type == TRAFFIC:
                    self.canvas.create_text(cx, cy, text="T",
                                            fill="#ff6666",
                                            font=("Courier New", 9, "bold"))
                elif cell_type == BASE:
                    self.canvas.create_text(cx, cy, text="BASE",
                                            fill="white",
                                            font=("Courier New", 7, "bold"))
                elif cell_type == DELIVERY:
                    d_num = self.env.delivery_points.index(pos) + 1 if pos in self.env.delivery_points else "?"
                    self.canvas.create_text(cx, cy, text=str(d_num),
                                            fill="white",
                                            font=("Courier New", 11, "bold"))
                elif cell_type == ROAD:
                    self.canvas.create_text(cx, cy,
                                            text=str(self.env.cost_grid[r][c]),
                                            fill="#2a3a5a",
                                            font=("Courier New", 7))

        # draw robot on top
        bot_pos = robot_at if robot_at else self.robot_pos
        bx = PADDING + bot_pos[1] * CELL_SIZE + CELL_SIZE // 2
        by = PADDING + bot_pos[0] * CELL_SIZE + CELL_SIZE // 2
        self.canvas.create_oval(bx - 14, by - 14, bx + 14, by + 14,
                                fill="#3d0a00", outline=COLORS["robot"], width=2)
        self.canvas.create_text(bx, by, text="R",
                                fill=COLORS["robot"],
                                font=("Courier New", 10, "bold"))

    def _write_log(self, msg, tag=""):
        # adds message to log box
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n", tag)
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _update_status(self, text, color):
        self.status_label.configure(text=f"STATUS: {text}", fg=color)

    def _animate_visited(self, visited_nodes, path, on_done):
        # shows explored nodes one by one, then moves robot along path
        path_set = set(path) if path else set()
        explore_nodes = [n for n in visited_nodes if n not in path_set]
        shown_so_far = []

        def do_step(i):
            if i < len(explore_nodes):
                shown_so_far.append(explore_nodes[i])
                self._redraw_grid(visited_cells=shown_so_far,
                                  path_cells=path,
                                  robot_at=self.robot_pos)
                delay = max(3, self.anim_speed.get() // 6)
                self.root.after(delay, lambda: do_step(i + 1))
            else:
                self._animate_robot(path, 0, on_done)

        do_step(0)

    def _animate_robot(self, path, step_idx, on_done):
        # moves robot step by step along the path
        if not path or step_idx >= len(path):
            on_done()
            return
        self.robot_pos = path[step_idx]
        self._redraw_grid(path_cells=set(path), robot_at=self.robot_pos)
        delay = self.anim_speed.get()
        self.root.after(delay, lambda: self._animate_robot(path, step_idx + 1, on_done))

    def _start_delivery(self):
        # starts the delivery process when button is clicked
        if self.is_running:
            return
        self.is_running = True
        self.delivery_index = 0
        self.robot_pos = self.env.base_station
        self.results_list = []
        self.run_btn.configure(state="disabled")
        self._clear_log()
        self._update_status("RUNNING", "#34d399")
        self._write_log(f"> Algorithm: {self.selected_algo.get()}", "blue")
        self._write_log("> Starting 5 deliveries...", "gray")
        self._next_delivery()

    def _next_delivery(self):
        # handles each delivery one at a time
        if self.delivery_index >= len(self.env.delivery_points):
            self._all_done()
            return

        goal_pos = self.env.delivery_points[self.delivery_index]
        start_pos = self.robot_pos
        algo_fn = ALL_ALGORITHMS[self.selected_algo.get()]

        self._write_log(f"\n> Delivery {self.delivery_index + 1}/5", "green")
        self._write_log(f"  From {start_pos} to {goal_pos}", "gray")

        path, total_cost, nodes, elapsed, visited = algo_fn(self.env, start_pos, goal_pos)

        if path is None:
            self._write_log("  WARNING: No path found! Skipping.", "yellow")
            self.delivery_index += 1
            self.root.after(300, self._next_delivery)
            return

        self._write_log(f"  Cost:{total_cost}  Nodes:{nodes}  {elapsed*1000:.2f}ms", "gray")

        self.stat_vars["delivery"].set(f"{self.delivery_index + 1}/5")
        self.stat_vars["cost"].set(str(total_cost))
        self.stat_vars["nodes"].set(str(nodes))
        self.stat_vars["time_ms"].set(f"{elapsed * 1000:.1f}")

        self.results_list.append({
            "delivery": self.delivery_index + 1,
            "algo": self.selected_algo.get(),
            "cost": total_cost,
            "nodes": nodes,
            "time_ms": elapsed * 1000,
            "steps": len(path) - 1,
        })

        def after_done():
            self.display_grid[goal_pos[0]][goal_pos[1]] = ROAD
            self.delivery_index += 1
            self.root.after(300, self._next_delivery)

        self._animate_visited(visited, path, after_done)

    def _all_done(self):
        # called when all 5 deliveries finished
        total_c = sum(r["cost"] for r in self.results_list)
        total_n = sum(r["nodes"] for r in self.results_list)
        total_t = sum(r["time_ms"] for r in self.results_list)

        self._write_log("\n> ALL DELIVERIES DONE!", "green")
        self._write_log(f"  Total Cost  : {total_c}", "blue")
        self._write_log(f"  Total Nodes : {total_n}", "blue")
        self._write_log(f"  Total Time  : {total_t:.2f} ms", "blue")

        self.stat_vars["cost"].set(str(total_c))
        self.stat_vars["nodes"].set(str(total_n))
        self.stat_vars["time_ms"].set(f"{total_t:.1f}")
        self._update_status("DONE", "#34d399")
        self.is_running = False
        self.run_btn.configure(state="normal")

    def _reset(self):
        # resets grid and robot to initial state
        self.is_running = False
        self.env = GridEnvironment()
        self.display_grid = [row[:] for row in self.env.grid]
        self.robot_pos = self.env.base_station
        self.delivery_index = 0
        self.results_list = []
        for key in self.stat_vars:
            self.stat_vars[key].set("--")
        self._clear_log()
        self._redraw_grid()
        self.run_btn.configure(state="normal")
        self._update_status("IDLE", "#888888")
        self._write_log("> Grid reset.", "gray")

    def _show_comparison(self):
        # opens popup window comparing all algorithms on same grid
        test_env = GridEnvironment()
        comparison_results = []

        for name, fn in ALL_ALGORITHMS.items():
            current_pos = test_env.base_station
            total_cost = 0
            total_nodes = 0
            total_time = 0
            deliveries_done = 0

            for goal in test_env.delivery_points:
                path, cost, nodes, elapsed, _ = fn(test_env, current_pos, goal)
                if path is not None:
                    total_cost += cost
                    total_nodes += nodes
                    total_time += elapsed * 1000
                    deliveries_done += 1
                    current_pos = goal

            comparison_results.append((name, total_cost, total_nodes,
                                       round(total_time, 3), deliveries_done))

        popup = tk.Toplevel(self.root)
        popup.title("Algorithm Comparison")
        popup.configure(bg=COLORS["bg"])
        popup.resizable(False, False)

        tk.Frame(popup, bg=COLORS["border"], height=2).pack(fill="x")
        tk.Label(popup, text="ALGORITHM COMPARISON TABLE",
                 font=("Courier New", 11, "bold"),
                 bg=COLORS["bg"], fg="#a78bfa",
                 pady=10, padx=15).pack(anchor="w")

        col_names = ("Algorithm", "Total Cost", "Nodes", "Time(ms)", "Deliveries")
        table_frame = tk.Frame(popup, bg=COLORS["bg"])
        table_frame.pack(padx=15, pady=(0, 8))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("Cmp.Treeview",
                        background=COLORS["panel"],
                        foreground=COLORS["text"],
                        fieldbackground=COLORS["panel"],
                        rowheight=28,
                        font=("Courier New", 8))
        style.configure("Cmp.Treeview.Heading",
                        background=COLORS["border"],
                        foreground="#38bdf8",
                        font=("Courier New", 8, "bold"),
                        relief="flat")
        style.map("Cmp.Treeview", background=[("selected", "#1e3a5f")])

        tree = ttk.Treeview(table_frame, columns=col_names,
                            show="headings", height=len(comparison_results),
                            style="Cmp.Treeview")

        col_widths = [180, 100, 100, 100, 100]
        for col, w in zip(col_names, col_widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center")

        valid_costs = [r[1] for r in comparison_results if r[4] == 5]
        best_cost = min(valid_costs) if valid_costs else None
        best_nodes = min(r[2] for r in comparison_results)
        best_time = min(r[3] for r in comparison_results)

        for row_data in comparison_results:
            tags = []
            if row_data[1] == best_cost:
                tags.append("best_cost")
            if row_data[2] == best_nodes:
                tags.append("best_nodes")
            if row_data[3] == best_time:
                tags.append("best_time")
            tree.insert("", "end", values=row_data, tags=tags)

        tree.tag_configure("best_cost", background="#064e3b")
        tree.tag_configure("best_nodes", background="#0c4a6e")
        tree.tag_configure("best_time", background="#3b0764")
        tree.pack()

        legend_row = tk.Frame(popup, bg=COLORS["bg"])
        legend_row.pack(padx=15, pady=(3, 8), anchor="w")
        for col, lbl in [("#064e3b", "Best cost"),
                         ("#0c4a6e", "Fewest nodes"),
                         ("#3b0764", "Fastest")]:
            lr = tk.Frame(legend_row, bg=COLORS["bg"])
            lr.pack(side="left", padx=8)
            tk.Canvas(lr, width=11, height=11, bg=col,
                      highlightthickness=0).pack(side="left", padx=(0, 4))
            tk.Label(lr, text=lbl, font=("Courier New", 7),
                     bg=COLORS["bg"], fg="#888888").pack(side="left")

        summary = tk.Text(popup, height=10,
                          bg=COLORS["panel"], fg=COLORS["text"],
                          font=("Courier New", 7),
                          relief="flat", padx=8, pady=6)
        summary.pack(fill="x", padx=15, pady=(0, 15))
        summary.insert("end", "SUMMARY\n" + "-" * 50 + "\n")
        for name, tc, tn, tt, succ in comparison_results:
            summary.insert("end",
                f"  {name:<25} cost={tc:<6} nodes={tn:<6} time={tt:<8.3f}ms [{succ}/5]\n")
        summary.configure(state="disabled")

# entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = RobotApp(root)
    root.mainloop()
