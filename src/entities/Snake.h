#pragma once

#include <SFML/Graphics.hpp>
#include <deque>
#include <vector>
#include <string>

// ==========================================
// 蛇实体 — 玩家和AI共用
//
// 核心算法：Trail Buffer（轨迹缓冲）
//   1. 头部沿任意方向平滑移动（360°）
//   2. 移动过程中持续将头部位置记录到 m_trail（双端队列）
//   3. 体节沿 m_trail 等距采样，形成丝滑跟随效果
//   4. 转向受 turnRate 限制，不会瞬间变向
//
// 这个算法天然支持变速（加速时轨迹点变稀疏，体节自动拉开），
// 且只需遍历体节数 + 轨迹长度，性能远优于每节独立物理模拟。
// ==========================================
class Snake {
public:
    // 构造函数：在 (startX, startY) 创建蛇，初始朝向 startAngle
    Snake(float startX, float startY, float startAngle,
          const sf::Color& bodyColor, bool isPlayer);

    // ---- 每帧更新 ----

    // 核心更新：转向 → 变速 → 移动头部 → 记录轨迹 → 计算体节 → 清理旧轨迹
    void update(float dt);

    // 设置目标转向角度（弧度），由玩家输入或 AI 决定
    void setTargetAngle(float angle) { m_targetAngle = angle; }

    // 设置是否加速（加速时速度提升，体节消耗待后续实现）
    void setBoosting(bool boosting) { m_isBoosting = boosting; }

    // ---- 转向"个性"（主要给AI用，让每条蛇的转头手感都不一样） ----
    //
    // turnEase   — 缓动系数：转向速度 = 角度差 × 它（越大越急）
    // wobbleAmp  — 最大转速的起伏幅度：0 = 匀速转；0.5 = 转速在 50%~150% 之间摆动
    // wobbleSpeed— 起伏快慢（弧度/秒）
    void setTurnPersonality(float turnEase, float wobbleAmp, float wobbleSpeed);

    // ---- 状态查询 ----

    float getX() const { return m_x; }
    float getY() const { return m_y; }
    float getAngle() const { return m_angle; }
    float getSpeed() const { return m_currentSpeed; }
    // 最大转向速率（弧度/秒）。智能体层把算法的 steer ∈ [-1,1] 换算成
    // 目标朝向时要用到它。
    float getTurnRate() const { return m_turnRate; }

    // ★ 把"转向量 steer"换算成【目标朝向应该领先当前朝向多少弧度】
    //
    //  为什么不能直接写 目标 = 当前 + steer × turnRate × dt ？
    //  因为蛇的转向是【缓动】的：每 tick 只闭合到目标的 ease = turnEase×dt 倍。
    //  如果目标只领先 0.133 弧度，实际每 tick 只转 0.133×0.1 = 0.0133 弧度，
    //  即 0.8 rad/s —— 比设定的 turnRate=8 慢了整整 10 倍，玩家会完全控不住。
    //
    //  要打出 steer × turnRate 的角速度，需要：
    //      turn = diff × ease = steer × turnRate × dt
    //   => diff = steer × turnRate × dt / ease          ← 就是下面这个
    float getSteerLead(float steer, float dt) const {
        const float ease = (m_turnEase * dt < 1.0f) ? (m_turnEase * dt) : 1.0f;
        if (ease <= 0.0001f) return 0.0f;
        return steer * m_turnRate * dt / ease;
    }
    float getHeadRadius() const { return m_headRadius; }
    float getSegmentRadius() const { return m_segmentRadius; }
    sf::Vector2f getHeadPosition() const { return sf::Vector2f(m_x, m_y); }
    const std::vector<sf::Vector2f>& getSegments() const { return m_segmentPositions; }
    int getSegmentCount() const { return m_segmentCount; }
    const sf::Color& getColor() const { return m_color; }
    bool isAlive() const { return m_isAlive; }
    bool isPlayer() const { return m_isPlayer; }
    bool isBoosting() const { return m_isBoosting; }
    int getScore() const { return m_score; }
    int getKills() const { return m_kills; }
    float getTimeSinceEat() const { return m_timeSinceEat; }

    // 队伍编号（0 起；-1 表示不属于任何队伍）。
    // 感知层靠它区分"敌人 / 队友"，碰撞系统靠它决定队友之间能否穿过。
    int  getTeamId() const { return m_teamId; }
    void setTeamId(int id) { m_teamId = id; }

    // ---- 状态修改 ----

    void addFoodEaten();              // 吃普通食物计数
    void addCorpseEaten();            // 吃尸体食物计数
    void setSegmentCount(int count);  // 直接设置体节数（AI初始化用）
    void addKill();                   // 击杀+1
    void die();                       // 标记为死亡
    void addScore(int points);        // 增加分数
    void resetStarvation();           // 重置饥饿计时（吃东西时调用）

    // -----------------------------------------------------------------------
    //  ★ 出生时把整条身体堆在【一个点】上，而不是笔直铺开
    //
    //  ★ 负责人提的（原话："不要把蛇做成出生直接拉直，蛇刚出生时只有一点
    //    （身体全部堆在一起）"）。这条不只是好看，它解决三个真问题：
    //
    //    1) 【学习课程更干净】
    //       身体堆在一个点 = 出生时【不可能撞到自己】。
    //       于是"学会找食物"和"学会不撞自己"这两件事被分开了 ——
    //       先学前者，等身体长起来再自然面对后者。
    //       笔直铺开时，蛇一转身就可能蹭到自己的尾巴，两个变量混在一起。
    //
    //    2) 【小地图上不再一出生就撞墙】
    //       笔直铺开会让身体向后延伸 (体节数 × 间距 × 1.2) ——
    //       12 节 × 20px × 1.2 = 288px。在 400x400 的地图上，
    //       蛇头离中心才 128px，尾巴直接捅到地图外面去了。
    //       堆成一个点就完全没这个问题。
    //
    //    3) 【视觉上更符合直觉】
    //       蛇刚出生本来就是一小团，长起来才拉成一条。
    //
    //  ★ 实现：把整条轨迹都放在 (m_x, m_y)。updateSegments() 会把各体节
    //    都插值到同一个位置，渲染出来就是一个点（准确说是一串同心圆）。
    //    之后蛇一开始移动，轨迹自然拉开，身体就正常展开了。
    //
    //  ⚠️ 与游戏本体的关系：默认【关】，只有训练任务会打开它。
    //     游戏里保持原来的"笔直铺开"，避免改动既有手感和 replay_check。
    // -----------------------------------------------------------------------
    void setCompactSpawn(bool on) { m_compactSpawn = on; }
    bool compactSpawn() const { return m_compactSpawn; }

    // 应用配置参数（速度、转向率等），构造后调用
    void applyConfig(float speed, float boost, float turn, float spacing,
                     int foodNeeded, int corpseNeeded);

    // ---- 渲染 ----

    // 设置头部贴图（玩家蛇用图片，AI蛇用圆形）
    void setHeadTexture(const sf::Texture& texture);

    // 在窗口上绘制整条蛇：体节 → 头部 → 眼睛
    void render(sf::RenderWindow& window) const;

private:
    // 从 trail 中沿轨迹等距采样，计算每个体节的当前世界坐标
    void updateSegments();

    // 删除 trail 尾端不再被任何体节需要的旧轨迹点，防止内存无限增长
    void trimTrail();

    // 在出生点用一条直线轨迹把身体铺满（出生和改变体节数时调用）
    void prefillTrail();

    // ---- 位置与方向 ----
    float m_x, m_y;              // 头部世界坐标
    float m_angle;               // 当前实际朝向（弧度）
    float m_targetAngle;         // 目标朝向（玩家输入或AI决策给出）
    float m_currentSpeed;        // 当前实际速度（平滑过渡后的值）
    float m_age = 0.0f;          // 存活时间（秒），仅用于转向起伏的相位

    // ---- 尺寸参数 ----
    float m_headRadius;          // 头部圆形半径（比体节略大）
    float m_segmentRadius;       // 每个体节的圆形半径
    float m_segmentSpacing;      // 相邻体节圆心之间的距离

    // ---- 移动参数 ----
    float m_normalSpeed;         // 正常速度（像素/秒）
    float m_boostSpeed;          // 加速速度（像素/秒）
    float m_turnRate;            // 最大转向速率（弧度/秒）—— 机动性上限
    float m_turnEase;            // 转向缓动系数（转向速度 ∝ 角度差）
    float m_turnWobbleAmp = 0.0f;    // 最大转速的起伏幅度（0 = 匀速转）
    float m_turnWobbleSpeed = 0.0f;  // 起伏快慢（弧度/秒）
    float m_turnWobblePhase = 0.0f;  // 起伏初相位（每条蛇随机，避免整齐划一）

    // ---- 身体数据 ----
    int m_segmentCount;                              // 当前体节总数
    std::deque<sf::Vector2f> m_trail;                // 头部历史轨迹（前端=最新）
    std::vector<sf::Vector2f> m_segmentPositions;    // 每个体节的当前世界坐标（每帧重算）

    // 出生时身体堆成一个点（见 setCompactSpawn 的说明）
    bool m_compactSpawn = false;

    // ---- 状态标记 ----
    sf::Color m_color;           // 蛇身颜色
    int  m_teamId = -1;          // 队伍编号（-1 = 无队伍）
    bool m_isAlive = true;       // 是否存活
    bool m_isPlayer = false;     // 是否由玩家控制
    bool m_isBoosting = false;   // 是否正在加速
    int m_score = 0;             // 当前得分
    int m_kills = 0;             // 击杀数
    int m_foodCounter=0;         // 普通食物累积计数
    int m_corpseCounter = 0;     // 尸体食物累积计数
    int m_foodNeeded = 5;        // 吃几个普通食物+1体节
    int m_corpseNeeded = 2;      // 吃几个尸体食物+1体节
    float m_timeSinceEat = 0.0f; // 距上次进食的时间（秒），超过20秒饿死

    // ---- 玩家头部贴图（可选，仅玩家蛇使用） ----
    const sf::Texture* m_headTexture = nullptr;  // 不拥有纹理所有权，仅引用
    mutable sf::Sprite m_headSprite;             // mutable 因为 render() 是 const
};
