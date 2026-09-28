# Arena 设计 — snake-nn-lab

> **状态：已实现并验证通过（2026-09-22）。**
> 计划与决策背景见 [docs/60-执行计划.md](60-执行计划.md)；实际实现见文末第 8 节。

---

## 1. 要解决什么问题

**「一份游戏规则现在要写两遍。」**

| 改动类型 | 现在要改几处 | 说明 |
|---|---|---|
| 实体 / 物理规则<br>（蛇怎么转、吃食物、碰撞、饥饿） | **1 处** | `Snake` / `FoodManager` / `CollisionSystem` 本来就共用 |
| 观测编码 | **1 处** | 刚由 E2 收敛（原先 2 处） |
| **对局循环**（每帧顺序、死亡处理、感知填充） | **2 处** | `GameScene::fixedUpdate()` ⇄ `train_probe::runEpisode()` |
| **食物怎么生成** | **2 处** | `GameScene` 构造 ⇄ `train_probe::World` 构造 |

后两行是问题所在。而负责人要加的**「食物圈」玩法**恰好落在这两类里。

**两份不一致的后果**（负责人的原话）：

> 「以后如果我想微调一下游戏本体的规则，就必须修改两份代码，并且很有可能出现 bug？」

**是的**，而且刚刚就发生过一次微缩版：仅仅一个扇区边界判断的写法不同，
就造成了 **0.96** 的观测差异（见 `devlogs/2026-09-22.md` 第 7 节）。
食物圈逻辑比这复杂得多，靠人工同步两份是不可能的。

**验收判据**：改食物圈规则时，`grep` 圈半径这个参数，
**全项目只应该有一个地方定义它**。

---

## 2. 职责划分

```
┌───────────────────────────────────────────────────────────────┐
│  GameScene（表现层，可以碰 SFML）                              │
│    渲染 · 输入采集 · 相机 · 小地图 · HUD · 观战 · 限时 · 结算   │
│                          │                                    │
│                          ▼ 喂输入 / 取快照                     │
├───────────────────────────────────────────────────────────────┤
│  Arena（世界层，不碰窗口/纹理/字体）                            │
│    agents（蛇 + 大脑 + 感知缓冲）· FoodManager · CollisionSystem│
│    PerceptionBuilder · 出生布局 · 【每帧顺序】· 决策调用        │
└───────────────────────────────────────────────────────────────┘
                          ▲
                          │ 同一份
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
  snake_game.exe                      train_probe.exe
  （GameScene 驱动）                   （训练器驱动）
```

### Arena **负责**

- 持有 `std::vector<agent::Agent>`、`FoodManager`、`CollisionSystem`、
  `PerceptionBuilder`、`WorldView`、`tick`
- **出生布局**（各队从均分圆周的方位出生、朝向地图中心）
- **每帧的固定步长 1~6 步**（见第 3 节）
- 调用每条蛇的 `brain->think()` / `controller`（见第 4 节）
- 饥饿判定、死亡 → 尸体转食物（`deathHandled` 只做一次）
- `reset()` —— 重开一局

### Arena **不负责**

| 不负责 | 留在哪 |
|---|---|
| 渲染、纹理、字体 | `GameScene` |
| 相机、小地图 | `GameScene` |
| 鼠标键盘 → 输入 | `GameScene` 采集，**喂**给 Arena |
| 观战切换 | `GameScene`（每帧查一次玩家蛇是否已死） |
| 队伍统计 | `GameScene`（只读 Arena 的 agents） |
| **限时判定 / 胜负 / 结算** | `GameScene` |
| 对局结束冻结世界 | `GameScene`（不调 `step()` 就是冻结） |

> 后面几条「不负责」是有意的边界。将来要把结算也搬进来，是**给 Arena 加方法**，
> 而不是重构调用方。

### ⚠️ 关于「无头」的准确含义

`Arena` 要持有 `Snake`，而 `Snake.h` include 了 SFML
（`sf::Color`、`sf::Vector2f`、`sf::Texture`、`sf::Sprite`）。

所以本项目的「无头」= **不创建窗口、不加载资源**，不是「不依赖 SFML」。

**验收判据**：`Arena.h` / `Arena.cpp` 里**不出现** `sf::RenderWindow`、
`sf::Texture`、`sf::Font`。（`sf::Color` 允许 —— 它只是数据。）

彻底剥离 SFML（自定义 `Vec2`/`Color` + `Snake::render()` 拆成 `SnakeRenderer`）
是**后续独立任务**，本轮不做。负责人的指示：

> 「训练效率优先于架构洁癖，不必把 SFML 从 `Snake` 里彻底剥离。」

---

## 3. 每帧固定步长（顺序与现在**完全一致**）

```cpp
void Arena::step(float dt) {
    if (m_agents.empty()) return;
    ++m_tick;

    // 1. 移动
    for (auto& a : m_agents) a.body.update(dt);

    // 2. 碰撞（撞墙 / 撞蛇 / 吃食物）
    m_collision.checkAll(m_ptrs, m_food);

    // 3. 饥饿
    for (auto& a : m_agents)
        if (a.isAlive() && a.body.getTimeSinceEat() > m_cfg.starvationTime)
            a.body.die();

    // 4. 死亡 → 尸体转食物（每条蛇只做一次）
    for (auto& a : m_agents) {
        if (a.isAlive() || a.deathHandled) continue;
        m_food.spawnCorpseFood(a.body);
        a.deathHandled = true;
        a.brain->onDeath();
    }

    // 5. 感知 → 决策（★ 由 controller 决定"怎么决策"）
    m_worldView.tick = static_cast<int>(m_tick);
    m_worldView.dt   = dt;

    for (std::size_t i = 0; i < m_agents.size(); ++i) {
        agent::Agent& a = m_agents[i];
        if (!a.isAlive()) continue;

        m_builder.build(m_worldView, i, a.perception);
        // 饥饿剩余依赖配置，感知层拿不到，这里补上
        a.perception.self.hungerLeft =
            m_cfg.starvationTime - a.body.getTimeSinceEat();

        // ★ 决策委托给 controller —— 这是"游戏"与"训练"唯一的分岔点
        const arena::Action act = m_controller ? m_controller(i, a.perception)
                                               : a.brain->think(a.perception);
        a.body.setTargetAngle(a.body.getAngle() + a.body.getSteerLead(act.steer, dt));
        a.body.setBoosting(act.boost);
    }

    // 6. 食物更新
    m_food.update(dt);
}
```

**必须原样保留、不许"顺手优化"的几处**（都是踩过坑的地方）：

- 指针表 `m_ptrs` 在**全部 agent 建完之后**填一次，之后不增删
- 出生布局细节：散开系数 `0.22`、半径系数 `0.82 / 0.18`、朝向指向地图中心
- `steer → 目标朝向` 必须用 **`Snake::getSteerLead()`**，
  不许写 `steer × turnRate × dt`（少了 `1/ease` 因子，实际角速度只有 1/10）
- 尸体只转一次（`deathHandled`）
- 感知在**全部世界推进完成之后**生成（当前就是这样，先保持不变）

---

## 4. 决策从哪来 —— `controller` 回调

这是「游戏」和「训练」唯一的分岔点。

```cpp
// 每条蛇该出什么动作。返回 Action。
// index = 该蛇在 agents() 里的下标；perc = 刚为它填好的感知。
using Controller = std::function<arena::Action(std::size_t index,
                                              const arena::Perception& perc)>;
void setController(Controller c);
```

| 场景 | controller 是什么 |
|---|---|
| **游戏** | `nullptr` → Arena 退回默认行为：`a.brain->think(perc)` |
| **训练** | 一个 lambda：编码观测 → 采样动作 → **把 (obs, act) 记进轨迹** → 返回动作 |

**为什么用回调而不是把决策留给调用方**：决策必须发生在**感知刚刚填好**的那一刻，
而感知与决策之间没有其它世界状态变化。把决策留在 `step()` 内部，
调用方就不可能插进"改世界"的代码，也就**不可能让训练环境和游戏环境跑偏**。

### 人类玩家的输入怎么进去

现在是 `HumanBrain::think()` 直接读窗口鼠标键盘（`GameScene::pushHumanInput()`
把输入塞进 `HumanBrain` 再由它读）。**保留这个做法**，`Arena` 不碰输入。

代价：`Arena` 里的 `HumanBrain` 是个"会读窗口的 Brain"。
将来若要在无窗口环境下跑带人类的对局，需要改成注入输入 —— **本轮不做**。

---

## 5. 公开接口

```cpp
namespace arena {

struct ArenaConfig {
    // 地图
    float mapWidth = 2000.0f, mapHeight = 2000.0f;

    // 队伍
    int teamCount = 1;
    int snakesPerTeam = 1;
    int playerTeam = -1;                       // -1 = 没有玩家
    const std::string* teamBrains = nullptr;   // 长度 MAX_TEAMS，来自 GameConfig

    // 蛇
    float snakeSpeed = 250.0f, snakeBoostSpeed = 450.0f, snakeTurnRate = 8.0f;
    float segmentSpacing = 20.0f;
    int   snakeInitSegs = 12;
    int   foodSegments = 5, corpseSegments = 2;

    // 食物
    int   foodMaxCount = 120, foodSpawnCount = 2, corpseScore = 3;
    float foodSpawnInterval = 2.0f;

    // 规则
    float starvationTime = 20.0f, collisionForgive = 5.0f;
    bool  friendlyFire = false;

    // 感知
    float visionCellSize = 32.0f;
    int   allyVisibleCount = 2;
};

class Arena {
public:
    using Controller = std::function<Action(std::size_t, const Perception&)>;

    Arena(const ArenaConfig& cfg, sf::RenderWindow* window);
    ~Arena();

    // 重建全部 agent（首次 / 重开一局）
    void reset();

    void setController(Controller c);          // 见第 4 节
    void step(float dt);

    std::vector<agent::Agent>&       agents();
    const std::vector<agent::Agent>& agents() const;
    const std::vector<Snake*>&       snakePtrs() const;
    const std::vector<Food>&         foods() const;

    FoodManager&       food();
    const FoodManager& food() const;

    float mapWidth() const;
    float mapHeight() const;
    std::uint64_t tick() const;

    // 玩家那条蛇在 agents() 里的下标；没有玩家时 -1
    int humanIndex() const;
};

}  // namespace arena
```

---

## 6. 迁移步骤（每步单独编译+验证）

| 步 | 内容 | 验证 |
|---|---|---|
| **A1** | 本文档 + 修正 `CLAUDE.md` 的「无头」说法 | 负责人确认 |
| **A5** | 先加 `tools/replay_check.cpp`「世界指纹」工具，**录下重构前的基线** | 指纹文件留档 |
| **A2** | 新增 `src/arena/Arena.{h,cpp}`，把 `buildTeams()` + `fixedUpdate()` 的 1~6 步**逐行搬过去** | 编译通过（暂时还没人用） |
| **A3** | `GameScene` 改为持有 `Arena`；`fixedUpdate` 变成 `step()` + 相机 + 小地图 + 统计 + 限时；观战改成**每帧查询**玩家蛇是否已死 | **`replay_check` 指纹与 A5 录的逐位一致** |
| **A4** | `train_probe` 删掉 `struct World`，改用真 `Arena` + controller | 训练能跑；**如实记录新基线**（与旧 World 本来就可能不同） |

### A5 为什么必须排在 before A2

`GameScene` 是图形程序，没法在无头环境里自动比对"改动前后世界是否一致"。
所以先做一个小工具：给定固定的 `ArenaConfig` + 固定种子 + 固定动作序列，
跑 N 帧后输出一个**确定性世界指纹**（每条蛇的 x/y/角度/长度/分数/存活 + 食物数 + 哈希）。

这样 A3 改完之后，只要指纹与改动前**逐位一致**，就能硬证明「搬家没有改变行为」。
这也顺便给项目建立「无头回归」这个能力 —— `bench_sim` 的「标墙对拍」是同一套做法。

---

## 7. 这一步**不做**什么

| 不做 | 理由 |
|---|---|
| 把 SFML 从 `Snake` 里剥离 | 训练效率优先于架构洁癖（负责人指示） |
| 把限时 / 胜负 / 结算搬进 Arena | 与「世界推进」正交；将来要搬是加方法，不用重构调用方 |
| 改奖励、改观测编码 | 本轮只做重构，**不改行为** |
| 食物圈玩法 | **排在 Arena 之后** —— 否则圈逻辑要写两遍 |
| 多线程并行环境 | 需要 Arena 先到位，且需要先有训练可视化 |

---

## 8. 实际实现（2026-09-22 完成）

### 8.1 与本文档设计的差异

| 设计稿 | 实际实现 | 原因 |
|---|---|---|
| `Arena(const ArenaConfig&, sf::RenderWindow*)` | **没有窗口参数** | `HumanBrain` 自己读窗口输入，Arena 不需要窗口指针。少一个依赖更好 |
| 只有 `reset()` | 另有 `reset(const ArenaConfig&)` 与默认构造 `Arena()` | GameScene 需要"构造后按配置建局"；换配置必须重新分发到子系统（食物上限、碰撞参数、视距都是构造期生效的），只调 `reset()` 不够 |
| `teamBrains` 由调用方塞进 config | 同左 | — |
| 未提及出生模式 | 新增 `SpawnMode { Ring, Random }` | 游戏要"均分圆周"（否则开局全挤中心互撞），训练要"随机撒点"（否则过拟合到固定出生方位）。这个差别**以前是两份世界各写各的**，现在是一个配置项 |
| 未提及配置种子 | 新增 `spawnSeed`（0 = 不重播种） | 支持"同一种子跑出同一局"，`replay_check` 需要 |
| 未提及动作量化 | 量化搬进 `Arena::step()` | 它属于世界规则（对局里 AI 动作要量化到 5 档，否则规则 AI 对只能出 5 档的网络有不公平优势），必须和世界一起只有一份 |

### 8.2 实际验证结果

| 验证项 | 结果 |
|---|---|
| 五个目标编译 | ✅ 全绿 |
| **`replay_check` manual vs arena** | ✅ **30 个 `t=` 行 + 累积哈希 `3930d05f42b36af4` 逐字节一致** |
| `nn_selftest` | ✅ 6 项全通过 |
| `bench_sim` 标墙对拍 | ✅ 400 例 × 409600 格全一致 |
| 游戏本体启动 | ✅ 正常，stderr 无输出 |

**`replay_check manual` vs `arena` 的逐字节一致，是"搬家没有改变行为"的硬证据。**
两份实现（手写循环 / `Arena::step()`）跑同一局，每帧每条蛇的
x / y / 角度 / 速度 / 长度 / 分数 / 存活 + 食物数 + 全剧哈希全部相同。

### 8.3 `train_probe` 的新基线（**与旧数字不可比**）

训练世界从"自己写的极简世界"换成了**和游戏同一份 Arena**，所以数字变了是预期内的：

| | 平均回报 | 平均存活 | 平均吃 | 死亡率 |
|---|---|---|---|---|
| 随机基线 | 3.79 → **4.09** | 360.2 → **348.2 帧** | 1.17 → **1.43 个** | 100% → **98%** |
| 训练后贪心 | 16.52 → **12.06** | 1200 → **1200 帧** | 3.87 → **0.03 个** | 0% → **0%** |

> ⚠️ **"平均吃 0.03 个"值得警惕，但按负责人指示暂不深挖** ——
> 当前优先把框架搭好、游戏本体做干净；训练效果等玩法（食物圈）定型后再调。
> 已记录在 `docs/30` 的已知不足里。

### 8.4 踩到的坑

1. **`replay_check` 忘了 `registerBuiltinBrains()`**
   → `BrainRegistry` 是空的 → `Arena::reset()` 里 `create()` 返回 `nullptr` → 解引用
   → 症状是 `0xC0000005` 访问冲突，**一句提示都没有**，而且只在"进程存活 4 秒"这类
   检查里表现为"提前退出"。
   游戏本体是在 `main.cpp` 里注册的，工具程序必须自己注册。

2. **`applyAction` 里的动作量化差点被漏掉**
   搬 `fixedUpdate` 时只看了 1~6 步，量化写在 `applyAction`（另一个函数）里。
   靠"逐行对照旧代码"才发现。教训：**搬家时要连被调用的辅助函数一起清点**。

3. **`TEAM_COLORS` 的归属**
   颜色是 Arena 造蛇时给的（`Agent` 构造要 `sf::Color`），但 HUD 也要用。
   处理成 `Arena::teamColor(i)` 静态查询，避免"渲染代码另抄一份颜色表"。

---

## 9. 下一步（阶段 B / C）

- **阶段 B**：游戏内「训练可视化」——能在游戏里看 AI 训练的数据
- **阶段 C**：支持多种算法 / 训练多个 AI 版本
- 之后才实现**食物圈玩法**（`docs/70`）—— 圈逻辑现在只需要写在 `FoodManager` 一处
