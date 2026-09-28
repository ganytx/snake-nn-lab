#include "RuleBrain.h"

#include <algorithm>
#include <cmath>

// ★ 算法层铁律：不 include 任何 SFML 头文件。
//   所以这里不用 utils/Math.h（它 include 了 SFML），角度处理就地实现。
namespace {

constexpr float PI = 3.14159265358979323846f;

inline float normalizeAngle(float a) {
    while (a >  PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// `Perception.bearing` 已经是 [−π, π]，但切线方向加了 ±π/2 之后会出界，
// 所以这里要再折回来。就地实现，不 include utils/Math.h（那个依赖 SFML）。
inline float wrapPiLocal(float a) {
    return normalizeAngle(a);
}

// 把"相对自己朝向的方位角"折算成转向量 steer ∈ [-1, 1]
//   0     → 正前方，不用转
//   ±π/2  → 正左/正右，满舵
//   |β|>π/2 → 也按满舵处理（掉头就交给蛇自己的转向速度上限）
inline float bearingToSteer(float bearing) {
    const float half = PI * 0.5f;
    return clampf(bearing / half, -1.0f, 1.0f);
}

}  // namespace

namespace arena {

// ---------------------------------------------------------------------------
RuleBrain::RuleBrain() {
    reset();
}

void RuleBrain::reset() {
    m_timer = 0.0f;
    m_reactionTime = randRange(0.05f, 0.30f);
    m_wanderAngle = randRange(-PI, PI);
    m_last = Action{};
}

const BrainInfo& RuleBrain::info() const {
    // displayName / description 会被画到界面上。
    // ⚠️ 只用 ASCII 括号，不要用全角（）—— SimHei 可能缺这些字形，会画成方块。
    static const BrainInfo s_info{
        "rule",                                   // id：UI 和存档只认它
        "规则 AI (基线)",                          // displayName
        "行为树: 保命 > 找食物 > 找圈 > 攻击 > 漫游", // description
        "",                                       // author
        /*needsWeights=*/false,
        /*trainable=*/false,
    };
    return s_info;
}

void RuleBrain::onEpisodeStart(std::uint32_t seed) {
    m_rng = seed ? seed : 1u;
    reset();
}

// ---------------------------------------------------------------------------
//  每 tick 调用。反应时间没到就沿用上次的动作 —— 模拟生物的反应延迟，
//  顺便（在真实对局里）省掉大量重复计算。
// ---------------------------------------------------------------------------
Action RuleBrain::think(const Perception& p) {
    m_timer += p.dt;
    if (m_timer < m_reactionTime) return m_last;
    m_timer = 0.0f;
    m_last = decide(p);
    return m_last;
}

// ---------------------------------------------------------------------------
//  行为树主循环（优先级从高到低，命中即返回）
//
//  ★ 2026-09-26 重排：加了"找圈"与"攻击"两级
//    —— 旧版只有 躲障碍 > 找食物 > 漫游，
//       而食物只在圈里、视野只有 512px，所以蛇一旦不在圈附近
//       就只会"漫游"，表现为"傻"。
//
//  ★ 2026-09-26 晚：所有分支都接上【加速】判定
//    —— 以前规则 AI 从不加速，白扔了 1.8 倍速度。
// ---------------------------------------------------------------------------
Action RuleBrain::decide(const Perception& p) {
    Action action;   // 默认直行、不加速
    const bool boost = wantBoost(p);

    // ---- 优先级 0：保命（躲障碍）----
    if (avoidObstacles(p, action)) { action.boost = boost; return action; }

    // ---- 优先级 1：找食物（视野内）----
    if (seekFood(p, action)) { action.boost = boost; return action; }

    // ---- 优先级 2：圈内搜索（已在圈里但附近没食物 → 沿圈扫）----
    if (scrubZone(p, action)) { action.boost = boost; return action; }

    // ---- 优先级 3：找圈（★ 新增，用全局的圈方位）----
    if (seekZone(p, action)) { action.boost = boost; return action; }

    // ---- 优先级 4：攻击敌人（★ 新增，只在占优势时动手）----
    if (attackEnemy(p, action)) { action.boost = boost; return action; }

    // ---- 优先级 5：漫游 ----
    m_wanderAngle = normalizeAngle(m_wanderAngle + randRange(-0.5f, 0.5f));
    action.steer = bearingToSteer(normalizeAngle(m_wanderAngle - p.self.heading));
    action.boost = boost;
    return action;
}

// ===========================================================================
//  ★★ 是否该加速（2026-09-26 晚新增）
//
//  ★ 为什么以前"从不加速"是个明显的浪费
//
//    加速把速度从 250 提到 450 px/s（**1.8 倍**），而当前实现里
//    **它没有任何代价** —— `Snake.h` 第 34 行的注释自己写着
//    "（加速时速度提升，体节消耗待后续实现）"，也就是说消耗还没实现。
//    附带好处：体节沿轨迹按固定间距采样，加速时轨迹点变稀疏，
//    于是**身体被拉长、更松散 → 更容易别住敌人**。
//
//  ★ 为什么不能无脑一直加速
//
//    速度越快，**一个反应时间内走过的距离越大**。
//    反应时间 0.05~0.30 秒，250px/s 时是 12~75px，450px/s 时是 22~135px ——
//    后者会让"看到危险再躲"变得明显更难。
//    所以只在**没有近处威胁**时加速。
//
//  ★ 三个条件（全部满足才加速）
//      ① 饥饿余量足够：剩余饥饿 > kBoostMinHungerSec
//      ② 正前方足够空：连"三个反应时间"的路程都没有障碍物
//         用射线距离比较，比猜一个固定像素数更稳（视野格子边长会变）
//      ③ 没有敌人在近处：贴上去会被扣分、也容易撞死
//
//  ★★★ 为什么【① 饥饿余量】是必须的（这一条我曾经漏掉，负责人指出）
//
//    加速不是纯赚 —— `Snake::update()` 里写着：
//        m_timeSinceEat += dt * (m_isBoosting ? 1.5f : 1.0f);
//    也就是说：
//        速度    250 → 450 px/s（1.8 倍）
//        饿死窗口  20 → 13.3 秒（×2/3）
//        **每像素的饥饿成本 ×1.5**
//
//    净收益：加速让你在饿死前多走 20% 的路（6000px vs 5000px），
//    但每像素贵 50%。所以：
//        目标在 4180px 内  → 加速稳赚（省 44% 时间，抢食直接优势）
//        4180~5000px       → 只有加速能到
//        5000px 外         → 两种都饿死，加速只是死得快
//
//    ⚠️ **还有一层更隐蔽的影响**：`hungerLeft` 也决定 ⑤「吃饱余韵」的窗口
//       （余韵 = starvationTime − hungerLeft，见 EpisodeRunner）。
//       所以加速不只"饿得快"，它**同时把占回报 57~62% 的那笔正收益缩短到 2/3**。
//
//  ★ 门限取 8 秒（负责人 2026-09-26 选定："更激进：剩 > 8s 就加速"）
//    —— 激进侧的取舍：吃得饱时基本一直在加速（速度快、抢食占优），
//       代价是"刚进食后只有约 5 秒的加速窗口"。
//    ⚠️ 想保守就把它调大（13.3 秒是"再加速就回不来"的临界）。
// ---------------------------------------------------------------------------
bool RuleBrain::wantBoost(const Perception& p) const {
    // ---- ① 饥饿余量（最先查：它最便宜，也最致命）----
    //
    //  ⚠️ hungerLeft 是 Arena 每帧算好的【剩余秒数】。
    //     它不是"距离上次进食多久"，而是"还有多久饿死"。
    constexpr float kBoostMinHungerSec = 8.0f;
    if (p.self.hungerLeft < kBoostMinHungerSec) return false;

    const VisionGrid& g = p.grid;

    // ---- ② 正前方要足够空 ----
    //
    //  需要"反应时间 + 一点余量"的路程是空的。
    //  取 3 倍反应时间：留出"看到 → 决策 → 转向生效"的余量。
    //  ⚠️ 这个 3 倍是保守值，不是精确推导 —— 加速撞死时先往下调它。
    constexpr float kReactMargin = 3.0f;
    const float seconds = m_reactionTime * kReactMargin;
    // 用"要走多少格"来比，而不是像素 —— 视野格子边长是配置项，会变
    const float gridCells = (450.0f * seconds) / (p.cellSize > 1.0f ? p.cellSize : 32.0f);
    const float need = gridCells / static_cast<float>(VISION_ROWS);
    if (g.rayDistance(0) < need) return false;

    // ---- ③ 近处不能有敌人 ----
    //
    //  ⚠️ headToEnemyBodyDist 只在视野内有效，视野外是哨兵 1e9
    //     —— 那种情况算"没有敌人"，这是安全的（真看不到就没法躲，加速与否一样）
    constexpr float kBoostEnemyClear = 200.0f;
    if (p.headToEnemyBodyDist < kBoostEnemyClear) return false;

    // ---- ④ 两侧也不能贴太近（加速时来不及修正侧向）----
    const float sideNeed = need * 0.5f;
    if (g.rayDistance(2) < sideNeed) return false;
    if (g.rayDistance(6) < sideNeed) return false;

    return true;
}

// ===========================================================================
//  优先级 1：躲障碍
//
//  用视野网格的 8 条方向射线：0=前，2=右，4=后，6=左（每 45° 一档）。
//  rayDistance() 返回的是【归一化到网格步数】的距离，1 表示该方向一路畅通。
//
//  规则：
//    · 正前方很危险 → 往左右两边中"更宽敞"的那一侧满舵
//    · 正前方尚可，但某一侧贴得太近 → 轻微往反方向修正
//    · 左右都被堵死 → 只能直行（听天由命，或者交给蛇自己的碰撞判定）
// ===========================================================================
bool RuleBrain::avoidObstacles(const Perception& p, Action& action) {
    const VisionGrid& g = p.grid;

    // 归一化距离阈值：网格一共有 VISION_ROWS 步，
    // "前方 N 格以内有东西"就算危险。取 6 格 ≈ 192px（约 0.77 秒的行程）。
    const float dangerSteps = 6.0f / static_cast<float>(VISION_ROWS);

    const float front = g.rayDistance(0);
    const float right = g.rayDistance(2);
    const float left  = g.rayDistance(6);

    if (front < dangerSteps) {
        // 前方有东西：往宽敞的一侧让
        if (left > right)      action.steer = -1.0f;   // 左转（steer 负 = 左）
        else if (right > left) action.steer =  1.0f;   // 右转
        else                   action.steer = (randUnit() < 0.5f) ? -1.0f : 1.0f;
        return true;
    }

    // 前方通畅，但侧边贴太近 → 轻微修正，避免贴着障碍蹭过去
    const float sideDanger = 4.0f / static_cast<float>(VISION_ROWS);
    if (left < sideDanger && right > left) {
        action.steer = 0.4f;
        return true;
    }
    if (right < sideDanger && left > right) {
        action.steer = -0.4f;
        return true;
    }

    return false;
}

// ===========================================================================
//  优先级 2：找食物
//
//  在视野网格里逐格扫，找最近的 FOOD / CORPSE 格，
//  再用 cellToRelative() 把格坐标反算成"相对蛇头"的世界偏移，
//  最后折算成转向量。
//
//  逐格扫描是 32x32 = 1024 次，看着不少，但：
//    · 决策被反应时间节流（每秒 3~20 次，不是 60 次）
//    · 只读字节，缓存友好
//  真成为瓶颈的话再说 —— 现在先要正确和可读。
// ===========================================================================
bool RuleBrain::seekFood(const Perception& p, Action& action) {
    const VisionGrid& g = p.grid;
    const float cell = p.cellSize;

    int bestRow = -1, bestCol = -1;
    int bestDist2 = 0;   // 用"格坐标的曼哈顿/欧氏平方"比较，省掉开方

    for (int r = 0; r < VISION_ROWS; ++r) {
        for (int col = 0; col < VISION_COLS; ++col) {
            const std::uint8_t v = g.cells[r][col];
            if (!(v & (CELL_FOOD | CELL_CORPSE))) continue;

            const int dr = r - VISION_CENTER_ROW;
            const int dc = col - VISION_CENTER_COL;
            const int d2 = dr * dr + dc * dc;
            if (bestRow < 0 || d2 < bestDist2) {
                bestRow = r; bestCol = col; bestDist2 = d2;
            }
        }
    }

    if (bestRow < 0) return false;   // 视野里没食物

    float dx = 0.0f, dy = 0.0f;
    g.cellToRelative(bestRow, bestCol, p.self.heading, cell, dx, dy);

    const float bearing = normalizeAngle(std::atan2(dy, dx));
    action.steer = bearingToSteer(bearing);
    return true;
}

// ===========================================================================
//  ★★ 优先级 2.5：圈内搜索（2026-09-26 晚新增）
//
//  ★ 为什么单独需要它
//
//    `seekFood` 只能在**视野网格**里找食物（半径 ≈512px）。
//    而食物是**在圈里随机位置**生成的，圈半径 150px —— 蛇进了圈之后
//    很可能"附近没有食物，但圈的另一侧有"，于是 `seekFood` 返回 false。
//
//    旧逻辑这时会掉到"漫游"或"找圈"，而"找圈"看的是**圈心方位**，
//    于是它往圈心走 → 圈心没食物 → 继续抖 → **在圈里打转却没吃到东西**。
//
//  ★ 做法：绕圈扫（向圈的切线方向转）
//
//    已经在圈里时，朝**切线方向**走（而不是朝圈心），沿圆周巡一圈，
//    食物自然会被视野扫到。这比"原地打转"有目的性得多。
//
//    ★ 切线的两个方向选哪个：挑"离圈心更近的一侧"那半边先扫 ——
//      即让速度方向偏向圈心一侧，走螺旋而不是正圆，
//      这样连"圈心附近"也覆盖到。
// ===========================================================================
bool RuleBrain::scrubZone(const Perception& p, Action& action) {
    const int n = static_cast<int>(p.zones.size());
    if (n <= 0) return false;

    // 找到"我在哪一个圈里"（活跃或不活跃都算 —— 上几轮的食物还在圈里）
    int inside = -1;
    float bestGap = 0.0f;
    for (int i = 0; i < n; ++i) {
        const FoodZoneView& z = p.zones[static_cast<std::size_t>(i)];
        if (z.distance < z.radius) {
            // 在多个圈的重叠区里时，选"最靠里"的那个
            const float gap = z.radius - z.distance;
            if (inside < 0 || gap > bestGap) { inside = i; bestGap = gap; }
        }
    }
    if (inside < 0) return false;   // 不在任何圈里 → 交给找圈/漫游

    const FoodZoneView& z = p.zones[static_cast<std::size_t>(inside)];

    // ★ 圈里没东西了就别在这儿磨 —— 交给"找圈"去下一个
    if (z.foodCount <= 0) return false;

    // 切线方向 = 圈心方位 ± 90°。
    //   bearing 是"到圈心的相对方位"，所以切线 = bearing ± π/2。
    //
    // ★ 选哪一侧：让前进方向稍微偏向圈心（走螺旋而不是纯圆），
    //   这样圈心附近的食物也会被扫到。偏 25°（≈0.44 rad）。
    constexpr float kInward = 0.44f;   // 弧度：向圈心偏一点
    const float tangentPlus  = wrapPiLocal(z.bearing + PI * 0.5f + kInward);
    const float tangentMinus = wrapPiLocal(z.bearing - PI * 0.5f - kInward);

    // 挑离当前朝向更近的那一侧（转向更小、更快生效）
    const float da = std::fabs(tangentPlus);
    const float db = std::fabs(tangentMinus);
    action.steer = bearingToSteer(da <= db ? tangentPlus : tangentMinus);
    return true;
}


//
//  ★ 为什么这是本次重写最重要的一条
//
//    食物只在圈里生成，而视野半径 ≈512px、圈半径 150px。
//    蛇一旦不在圈附近，视野里就**什么都没有** ——
//    旧版到这里只能"漫游"，也就是**靠运气撞到圈**。
//
//    而 `Perception.zones` 一直提供**全局**的圈方位与距离
//    （`FoodZoneView::bearing` / `distance`，不受视野限制）。
//    **"该往哪走"这个信息一直都在，只是旧版从来没用过。**
//
//  ★ 选圈规则（2026-09-26 晚加了"提前出发"）
//      · 当前圈还在供食、而且剩余时间还够 → 去当前圈
//      · 当前圈快没产出了，或者根本到不了 → 去【下一个圈】（提前出发）
//      · 已经在圈内 → 让更低优先级接管（在圈里瞎找圈心会让它原地抖）
//    这正好利用了"当前 + 下一个"这两个槽位（见 encoder.h 的设计）。
//
//  ★★★ 为什么必须"提前出发"（这是本次最关键的改进）
//
//    圈周期 = 供食 9s + 空窗 5s。如果只在"当前圈空了"之后才动身去下一个圈，
//    那么在新圈开门的**那一刻**它还在路上 —— 永远迟到。
//    而空窗只有 5 秒，对手可能已经先到了。
//
//    判据（见下面的 earlyLeaveSeconds）：留给赶路的时间
//        = 当前圈剩余供食时间 + 空窗时间
//    如果这个总时间 < 走过去需要的时间，就该**现在**动身。
// ===========================================================================
bool RuleBrain::seekZone(const Perception& p, Action& action) {
    const int n = static_cast<int>(p.zones.size());
    if (n <= 0) return false;   // 没有圈（比如 Scatter 任务）→ 交给下一级

    const int cur = p.activeZoneIndex;
    const int nxt = p.nextZoneIndex;

    // ---- 先判断"去下一个圈是否更划算" ----
    //
    //  两个条件都考虑：
    //    ① 下一个圈比我到当前圈更近（当前圈太远，来不及）
    //    ② 当前圈剩余供食时间不多，而下一个圈近得多（提前出发）
    int pick = -1;
    bool haveCur = (cur >= 0 && cur < n);
    bool haveNxt = (nxt >= 0 && nxt < n && nxt != cur);

    if (haveCur) {
        const FoodZoneView& zc = p.zones[static_cast<std::size_t>(cur)];
        // 正在供食且有东西 → 默认去当前圈
        if (zc.active && zc.foodCount > 0) pick = cur;

        if (haveNxt) {
            const FoodZoneView& zn = p.zones[static_cast<std::size_t>(nxt)];

            // 走过去还要多久（用最大速度估，保守：加速不是总能开）
            constexpr float kAssumedSpeed = 250.0f;   // px/s（基础速度）
            const float travelCur = zc.distance / kAssumedSpeed;
            const float travelNxt = zn.distance / kAssumedSpeed;

            // 当前圈还能"指望"多久 = 剩余供食 + 空窗（空窗期不算完全没机会，
            //   上一波没吃完的食物还在，但价值降低 —— 这里只看时间够不够走到）
            const float usable = zc.supplySecondsLeft + 5.0f;

            // ① 到当前圈已经来不及（走过去它就空了）
            const bool tooLateForCur = (travelCur > usable);
            // ② 下一个圈明显更近，而且当前圈供食快结束了
            const bool betterGoNext =
                (travelNxt + 1.5f < travelCur) && (zc.supplySecondsLeft < travelCur);

            if (tooLateForCur || betterGoNext) {
                if (zn.foodCount > 0 || zn.supplySecondsLeft > 0.0f || tooLateForCur) {
                    pick = nxt;
                }
            }
        }
    }

    // 当前圈不可用 → 退到下一个圈
    if (pick < 0 && haveNxt) pick = nxt;
    if (pick < 0 && haveCur) pick = cur;      // 兜底：至少还有个当前圈
    if (pick < 0 || pick >= n) return false;

    const FoodZoneView& z = p.zones[static_cast<std::size_t>(pick)];

    // 已经在圈里了就不必再"去"—— 让更低优先级的行为接管
    //   （否则蛇会在圈心附近来回抖动，因为 bearing 在近距离极不稳定）
    //
    //  ★ 为什么用 0.9 倍半径而不是 1.0：贴着边缘时食物可能刚好在圈外，
    //    往里走一点更稳。这个系数不敏感，取 0.9 是为了留一点余量。
    if (z.distance < z.radius * 0.9f) return false;

    // `bearing` 已经是【相对自己朝向】的方位角（0 = 正前方），直接用
    action.steer = bearingToSteer(z.bearing);
    return true;
}

// ===========================================================================
//  ★★ 优先级 3：攻击敌人（2026-09-26 新增）
//
//  ★ 为什么必须带优势判定，而不是"看到敌人就冲"
//
//    规则是「**我的头**碰到**对方身体**才死」。所以对主动方来说，
//    "靠近敌人"本身是**危险**的 —— 无条件冲上去等于自杀。
//
//  ★ 围猎的几何条件（canHunt）
//        · 我的头离对方身体**远**（我还安全）
//        · 对方的头离我的身体**近**（它更危险）
//    两条同时成立时，我处于"用身体别住它、同时把头让开"的位置 ——
//    这正是"围猎"。任何一条不满足就不动手。
//
//  ★ 为什么只在【没有任何食物可去】时才会走到这里
//    负责人给的优先级是「食物 > 圈 > 攻击」。攻击排在圈之后，
//    所以它只在"世界里没有圈、或者找不到圈"时才生效。
//    这是刻意的：**活着并吃到东西，比杀掉别人重要**。
// ===========================================================================
bool RuleBrain::attackEnemy(const Perception& p, Action& action) {
    if (!p.hasEnemyInView) return false;
    if (!canHunt(p)) return false;

    // canHunt 保证 p.snakes 里至少有敌人；取最近的那条（列表按距离排序）
    const int n = static_cast<int>(p.snakes.size());
    const VisibleSnake* target = nullptr;
    for (int i = 0; i < n; ++i) {
        const VisibleSnake& s = p.snakes[static_cast<std::size_t>(i)];
        if (!s.alive) continue;
        if (s.teamId == p.self.teamId) continue;   // 队友不算敌人
        target = &s;                                // 第一条就是最近的
        break;
    }
    if (target == nullptr) return false;

    // 朝它走。注意这是"接近敌人" —— 之所以敢这么做，
    // 是因为 canHunt 已经确认了"我的头离它的身体还远"。
    action.steer = bearingToSteer(target->bearing);
    return true;
}

// ---------------------------------------------------------------------------
//  "我是否有围猎优势" —— 纯几何判定
//
//  ★ 两个阈值都取【保守】值，宁可不动手：
//      huntSafeDist  ：我的头至少离对方这么远，才算"我还安全"
//      huntTrapDist  ：对方的头离我这么近，才算"它被我别住"
//
//  ★ 为什么取 120 / 90 这两个数
//      头半径 + 身体半径约 20px 量级，碰撞还有 5px 宽容（collisionForgive）；
//      120px ≈ 6 节体长，是"我一个反应时间（0.05~0.3s，约 12~75px 行程）
//      转不出去"的距离余量。
//      90px 比它小，保证"它比我更危险"这件事成立。
//    ⚠️ 这两个是**可调常数**，效果不好就调它们，不要改判定结构。
// ---------------------------------------------------------------------------
bool RuleBrain::canHunt(const Perception& p) const {
    constexpr float huntSafeDist = 120.0f;   // 我的头 → 对方身体：要【大于】这个
    constexpr float huntTrapDist =  90.0f;   // 对方头 → 我的身体：要【小于】这个

    if (p.headToEnemyBodyDist > huntSafeDist) {
        // 我的头还安全 —— 再看有没有把对方别住
        return (p.enemyHeadToMyBodyDist < huntTrapDist);
    }
    return false;
}

// ---------------------------------------------------------------------------
//  可复现的伪随机（xorshift32）
//  不用全局 rand()：那会让"同一 seed 跑出同一局"的保证失效，
//  而跑分对比正依赖这个保证。
// ---------------------------------------------------------------------------
float RuleBrain::randUnit() {
    m_rng ^= m_rng << 13;
    m_rng ^= m_rng >> 17;
    m_rng ^= m_rng << 5;
    return static_cast<float>(m_rng & 0xFFFFFFu) / static_cast<float>(0x1000000u);
}

float RuleBrain::randRange(float lo, float hi) {
    return lo + randUnit() * (hi - lo);
}

}  // namespace arena
