#include "PerceptionBuilder.h"

#include "../utils/Math.h"

#include <algorithm>
#include <cmath>

namespace agent {

namespace {

// 视线半径（像素）：网格覆盖的方形半边长。
// 中心行/列是 25，所以前方最远 25 格，横向最远 25 格。
inline float visionHalfExtent(float cellSize) {
    return static_cast<float>(arena::VISION_CENTER_ROW) * cellSize;
}

// 把一条蛇的头信息转成 arena::VisibleSnake
arena::VisibleSnake makeVisible(const Snake& other, int id,
                                float dx, float dy, float selfHeading) {
    arena::VisibleSnake v;
    v.id       = id;
    v.teamId   = other.getTeamId();
    v.headRel  = { dx, dy };
    v.distance = std::sqrt(dx * dx + dy * dy);
    // bearing：把世界方向转到"以自己朝向为 0"的坐标系
    const float worldAngle = std::atan2(dy, dx);
    v.bearing    = Math::normalizeAngle(worldAngle - selfHeading);
    v.headingRel = Math::normalizeAngle(other.getAngle() - selfHeading);
    v.length     = other.getSegmentCount();
    v.boosting   = other.isBoosting();
    v.alive      = other.isAlive();
    return v;
}

bool nearer(const arena::VisibleSnake& a, const arena::VisibleSnake& b) {
    return a.distance < b.distance;
}

// 把角度归一化到 [-π, π]
inline float wrapPi(float a) {
    return Math::normalizeAngle(a);
}

}  // namespace

// ===========================================================================
void PerceptionBuilder::build(const WorldView& world, std::size_t selfIndex,
                              arena::Perception& out) const {
    out.clearLists();      // 只清空，容量保留 → 预热后零分配

    // ★★★ 2026-09-26 修：必须同时重置【本帧派生标量】
    //
    //  ⚠️ `clearLists()` 只清列表，不清 `hasEnemyInView` /
    //     `headToEnemyBodyDist` / `enemyHeadToMyBodyDist`。
    //     少了这一行，**敌人一出视野那几个读数就冻结在最后一次看到的值** ——
    //     后果是 `RuleBrain` 的 escape 奖励恒为 0（进得去、出不来），
    //     以及 `wantBoost` / `canHunt` 全都被污染。
    //     完整分析见 `Perception::resetFrameState()` 的注释。
    out.resetFrameState();

    if (world.snakes == nullptr || selfIndex >= world.snakes->size()) {
        // ★ 注意：即使这里提前返回，食物圈也要填 ——
        //   WorldView 目前没有任何"提前返回"的路径会走到（self 一定存在），
        //   但把这条注释留在这里，提醒以后加分支的人：圈是全局信息，
        //   不该跟着"这条蛇看不看得见"走。
        return;
    }

    const Snake& self = *(*world.snakes)[selfIndex];
    if (!self.isAlive()) {
        out.self.alive = false;
        return;
    }

    const float selfX   = self.getX();
    const float selfY   = self.getY();
    const float heading = self.getAngle();
    const float cell    = m_cfg.cellSize;

    // ---- 自身状态 ----
    out.self.id         = static_cast<int>(selfIndex);
    out.self.teamId     = self.getTeamId();
    out.self.pos        = { selfX, selfY };
    out.self.heading    = heading;
    out.self.speed      = self.getSpeed();
    out.self.length     = self.getSegmentCount();
    out.self.boosting   = self.isBoosting();
    out.self.headRadius = self.getHeadRadius();
    out.self.bodyRadius = self.getSegmentRadius();
    out.self.alive      = true;
    // 饥饿剩余：这里拿不到配置，由调用方在 build 之后填（见 Perception.dt 的用法）

    out.mapWidth  = world.mapWidth;
    out.mapHeight = world.mapHeight;
    out.cellSize  = cell;
    out.tick      = world.tick;
    out.dt        = world.dt;

    // ★ 当前 / 下一个圈的下标 —— 直接透传，**不在感知层重推调度规则**
    //   （"下一个是谁"在顺序模式和随机模式下不一样，唯一实现是 ZoneSchedule）
    out.activeZoneIndex = world.activeZoneIndex;
    out.nextZoneIndex   = world.nextZoneIndex;

    // ---- 食物圈（辅助通道 3，不受视野限制）----
    //
    // ★ 为什么圈走"全局通道"而不是视野网格：见 Perception.h 里 FoodZoneView 的注释。
    //   简单说：圈是【地图结构】不是"看到的东西"，而且圈半径远小于视野，
    //   只让它在视野里可见的话，蛇永远学不会"提前往下一个圈移动"。
    if (world.zones != nullptr) {
        out.zones.reserve(world.zones->size());
        for (const arena::FoodZone& z : *world.zones) {
            arena::FoodZoneView v;
            v.pos       = { z.x, z.y };
            v.radius    = z.radius;
            v.foodCount = z.foodCount;

            const float dx = z.x - selfX;
            const float dy = z.y - selfY;
            v.distance = std::sqrt(dx * dx + dy * dy);
            // 相对自身朝向的方位角：0 = 正前方
            v.bearing  = wrapPi(std::atan2(dy, dx) - heading);
            v.active   = (&z == world.activeZone);
            // ★ 本轮还能供食几秒（2026-09-26 新增）—— 供"提前出发"用
            //
            //  ⚠️ 只在【活跃圈】上算，不活跃的圈恒为 0。
            //     别的圈现在本来就没产出，给"剩余时间"没有意义。
            v.supplySecondsLeft = 0.0f;

            // -------------------------------------------------------------------
            //  ★★ 可见性：「下一个圈」只在【预告期】可见（2026-09-27 定稿）
            //
            //  ★ 负责人定义的规格（原话）
            //    "不要一开始就有预告圈，预告圈是在活跃圈产食物**还剩 3s** 后出现的，
            //      然后中间有一定时间 gap，然后再被激活。"
            //
            //  ★ 时间线（供食 9s + 空窗 5s + 预告提前 3s）
            //
            //      0s ────────── 6s ─────── 9s ─────── 14s ─────── 23s
            //      │← 只有活跃圈 →│←── 预告期 ──→│← 空窗 →│← 新活跃圈供食 →│
            //                      ↑
            //                下一个圈在这里出现
            //
            //    预告圈从出现到被激活有 `previewLead + gapDuration` = **8 秒**
            //    给蛇赶路（圈间距 600~900px，按 250px/s 要跑 2.4~3.6 秒）。
            //
            //  ★ 为什么"开局就预告"不行
            //    那等于提前一整轮（14 秒）就告诉它下一站在哪 —— 不自然，
            //    而且让"什么时候该出发"失去时间上的紧迫感。
            //
            //  ★ 判据的唯一实现是 `ZoneSchedule::previewing()`。
            //    感知层【不重推】这个规则（那是本项目栽过两次的"两份实现"）。
            //
            //  ⚠️⚠️ 活跃圈【永远可见】—— 只有"下一个圈"受预告期约束。
            //     这个 `|| v.active` 不能省：少了它，供食期的前 6 秒
            //     连**当前圈**都会被藏掉，AI 就彻底看不到食物圈了。
            // -------------------------------------------------------------------
            v.visible = v.active || world.zonePreviewing;
            if (world.zoneSchedule != nullptr) {
                const float sd = world.zoneSchedule->spawnDuration;
                const float left = sd - world.cycleElapsed;
                v.supplySecondsLeft = (left > 0.0f) ? left : 0.0f;
            }
            out.zones.push_back(v);
        }
    }

    // ---- 地图边界 ----
    //
    // ★ 早退优化（数学上等价，只是不做无用功）
    //
    // 视野是一个边长 2*half 的【旋转正方形】，离蛇头最远的格子中心距离是
    // half*sqrt(2)。如果蛇离最近的地图边界比这个距离还远，
    // 就不可能有任何一个格子落在图外 —— markWalls 跑一遍一个墙也不会标。
    //
    // 实测感知填充占每帧 63%~68%，而 markWalls 的逐格扫描是其中的大头，
    // 所以这条早退省下来的正是最贵的那部分。
    const float half     = visionHalfExtent(cell);
    const float farthest = half * 1.4143f + cell;   // +cell 留一格余量（用格心判断内外）
    const float dEdge    = std::min(std::min(selfX, selfY),
                                    std::min(out.mapWidth  - selfX,
                                             out.mapHeight - selfY));

    // ★★ 2026-09-26：把"头到最近边界"的距离写进感知（⑪ 危险距离扣分要用）
    //
    //  ★ 为什么复用 dEdge 而不是另算一遍
    //    它就是同一个量（到四边距离的最小值）。这里本来是为了
    //    "要不要跑 markWalls"的早退判断而算的 —— 顺手写出去，
    //    既不多一次计算，也保证"奖励用的墙距离"和"视野用的墙距离"
    //    永远是同一个数（不会出现两份实现算出两个值）。
    out.headToWallDist = dEdge;

    if (dEdge <= farthest) {
        markWalls(selfX, selfY, heading, out);
    }

    // 用外接圆做粗筛；真正是否落在方形网格内由 mark() 决定。
    // half 在上面做"是否需要标墙"的早退判断时已经算过了，这里直接复用。
    const float viewR2 = (half + cell) * (half + cell);

    // ---- 自己的身体 ----
    for (const auto& seg : self.getSegments()) {
        const float dx = seg.x - selfX;
        const float dy = seg.y - selfY;
        out.grid.mark(dx, dy, heading, cell, arena::CELL_SELF_BODY);
    }

    // ---- 其他蛇 ----
    const std::size_t n = world.snakes->size();
    for (std::size_t i = 0; i < n; ++i) {
        if (i == selfIndex) continue;
        const Snake& other = *(*world.snakes)[i];
        if (!other.isAlive()) continue;

        const float dx = other.getX() - selfX;
        const float dy = other.getY() - selfY;
        const bool  isAlly = (other.getTeamId() == self.getTeamId());

        // 队友走全局通道（不受视野限制），所以先收进 allies
        if (isAlly) {
            out.allies.push_back(makeVisible(other, static_cast<int>(i),
                                             dx, dy, heading));
        }

        if (dx * dx + dy * dy > viewR2) continue;   // 视野外

        // 蛇头
        out.grid.mark(dx, dy, heading, cell,
                      isAlly ? arena::CELL_ALLY_HEAD : arena::CELL_ENEMY_HEAD);

        // 身体（每节都画；体节间距 20px < 格边长 48px，身体不会断开）
        const std::uint8_t bodyBit = isAlly ? arena::CELL_ALLY_BODY
                                            : arena::CELL_ENEMY_BODY;
        for (const auto& seg : other.getSegments()) {
            const float sx = seg.x - selfX;
            const float sy = seg.y - selfY;
            if (sx * sx + sy * sy > viewR2) continue;
            out.grid.mark(sx, sy, heading, cell, bodyBit);
        }

        // 辅助列表：给算法一份【精确】的（网格是量化过的）
        out.snakes.push_back(makeVisible(other, static_cast<int>(i),
                                         dx, dy, heading));

        // -------------------------------------------------------------------
        // ★ 碰撞风险 / 猎杀优势（只算敌人，队友之间能互相穿过所以不算）
        //
        //  规则是"我的头碰到对方的身体才死"，所以两个方向要分开：
        //      我的头 → 对方身体 ：危险
        //      对方头 → 我的身体 ：猎杀优势（包围就是这么形成的）
        //
        //  ★ 记录"取自哪条敌人"（riskEnemyId）：
        //    包围要用【同一条】敌人来衡量 —— 对 X 包围、同时躲 Y，
        //    不能拿 A 的危险和 B 的优势凑在一起，那样策略没有意义。
        // -------------------------------------------------------------------
        if (!isAlly) {
            out.hasEnemyInView = true;

            const sf::Vector2f selfHead(selfX, selfY);
            const sf::Vector2f otherHead = other.getHeadPosition();

            // (a) 我的头 → 这条敌人的身体或头
            float myHeadToIt = Math::distance(selfHead, otherHead);
            for (const auto& seg : other.getSegments()) {
                const float d = Math::distance(selfHead, seg);
                if (d < myHeadToIt) myHeadToIt = d;
            }

            if (myHeadToIt < out.headToEnemyBodyDist) {
                out.headToEnemyBodyDist = myHeadToIt;
                // ★ 危险来自这条 —— 就用【同一条】敌人衡量猎杀优势。
                //   不能拿 A 的危险配 B 的优势，那样策略没有意义。
                out.riskEnemyId = static_cast<int>(i);

                // (b) 这条敌人的头 → 我的身体
                float itsHeadToMe = Math::distance(otherHead, selfHead);
                for (const auto& seg : self.getSegments()) {
                    const float d = Math::distance(otherHead, seg);
                    if (d < itsHeadToMe) itsHeadToMe = d;
                }
                out.enemyHeadToMyBodyDist = itsHeadToMe;
            }
        }
    }

    // ---- 食物 ----
    if (world.foods != nullptr) {
        for (const Food& f : *world.foods) {
            const float dx = f.position.x - selfX;
            const float dy = f.position.y - selfY;
            if (dx * dx + dy * dy > viewR2) continue;
            out.grid.mark(dx, dy, heading, cell,
                          f.isCorpse ? arena::CELL_CORPSE : arena::CELL_FOOD);
        }
    }

    // ---- 排序 + 截断 ----
    // 按距离由近到远：算法可以只看前 N 个，"谁更近"本身就是重要信息
    std::sort(out.snakes.begin(), out.snakes.end(), nearer);
    if (static_cast<int>(out.snakes.size()) > arena::limits::MAX_VISIBLE_SNAKES) {
        out.snakes.resize(static_cast<std::size_t>(arena::limits::MAX_VISIBLE_SNAKES));
    }

    // 队友只保留最近的 allyVisibleCount 个
    std::sort(out.allies.begin(), out.allies.end(), nearer);
    if (m_cfg.allyVisibleCount <= 0) {
        out.allies.clear();
    } else if (static_cast<int>(out.allies.size()) > m_cfg.allyVisibleCount) {
        out.allies.resize(static_cast<std::size_t>(m_cfg.allyVisibleCount));
    }
}

// ===========================================================================
//  标记地图外的格子为墙 —— 解析法
//
//  朴素做法是逐格把网格坐标反算成世界坐标再判断内外，1024 格 x 约 10 次运算。
//  实测感知占每帧 63%~68%，其中大头就在这里。
//
//  ★ 关键数学事实：地图是【凸矩形】，网格的每一行在世界空间里是一条【直线】，
//    所以"这一行里哪些格子在图内"必然是一个【连续区间】。
//    于是可以按行解析求出列区间，逐格试探就没必要了：
//        1024 次逐格  →  32 次按行
//
//  推导（第 r 行，局部前向坐标 f 固定）：
//      世界点 = (bx - rr·s, by + rr·c)
//      其中 bx = selfX + f·c, by = selfY + f·s, rr = (col - CENTER_COL)·cell
//    四条边界各给出一个关于 rr 的半平面约束：
//        wx ≥ 0     →  rr·(−s) ≥ −bx
//        wx ≤ mapW  →  rr·(−s) ≤  mapW − bx
//        wy ≥ 0     →  rr·c    ≥ −by
//        wy ≤ mapH  →  rr·c    ≤  mapH − by
//    求交集得到 [lo, hi]，再换算成列区间。
//
//  ⚠️ 退化情况：k = ±s 或 ±c 接近 0 时，约束变成"与 rr 无关"，
//     此时要么恒成立、要么恒不成立，必须单独判断（见 consGE / consLE）。
// ===========================================================================

namespace {

constexpr double WALL_EPS = 1e-6;

// 约束 rr * k >= b
void consGE(double k, double b, double& lo, double& hi) {
    if (k > WALL_EPS)       lo = std::max(lo, b / k);
    else if (k < -WALL_EPS) hi = std::min(hi, b / k);
    else if (b > 0.0)       { lo = 1.0; hi = -1.0; }   // 0 >= 正数，不成立
}

// 约束 rr * k <= b
void consLE(double k, double b, double& lo, double& hi) {
    if (k > WALL_EPS)       hi = std::min(hi, b / k);
    else if (k < -WALL_EPS) lo = std::max(lo, b / k);
    else if (b < 0.0)       { lo = 1.0; hi = -1.0; }   // 0 <= 负数，不成立
}

// rr 换算成列号，并夹到一个安全范围（避免天文数字转 int 溢出）
//
// ★ 下界必须用 ceil，上界必须用 floor —— 这里错过一次，被对拍测试抓出来了：
//   内区间的定义是  lo <= (col-C)·cell <= hi
//   左边  ⟺  col >= C + lo/cell   →  colLo = ceil(lo/cell) + C
//   右边  ⟺  col <= C + hi/cell   →  colHi = floor(hi/cell) + C
//   两边都用 floor 的话，下界会少算一格。
int rrToCol(double rr, double cell, bool isLower) {
    const double q = rr / cell;
    const double v = (isLower ? std::ceil(q) : std::floor(q))
                   + arena::VISION_CENTER_COL;
    if (v < -1000000.0) return -1000000;
    if (v >  1000000.0) return  1000000;
    return static_cast<int>(v);
}

}  // namespace

void PerceptionBuilder::markWalls(float selfX, float selfY, float heading,
                                  arena::Perception& out) const {
    const double c = std::cos(static_cast<double>(heading));
    const double s = std::sin(static_cast<double>(heading));
    const double cell = static_cast<double>(m_cfg.cellSize);
    const double mapW = static_cast<double>(out.mapWidth);
    const double mapH = static_cast<double>(out.mapHeight);

    for (int r = 0; r < arena::VISION_ROWS; ++r) {
        const double f  = static_cast<double>(arena::VISION_CENTER_ROW - r) * cell;
        const double bx = static_cast<double>(selfX) + f * c;
        const double by = static_cast<double>(selfY) + f * s;

        double lo = -1.0e30;
        double hi =  1.0e30;
        consGE(-s, -bx,        lo, hi);
        consLE(-s,  mapW - bx, lo, hi);
        consGE( c, -by,        lo, hi);
        consLE( c,  mapH - by, lo, hi);

        int colLo, colHi;
        if (lo > hi) {
            colLo = 1; colHi = 0;          // 空区间 → 整行都是墙
        } else {
            colLo = rrToCol(lo, cell, /*isLower=*/true);
            colHi = rrToCol(hi, cell, /*isLower=*/false);
        }

        // ★ 必须把两端都夹住：
        //   colLo / colHi 可能来自 b/k（k 只是"略大于阈值"时结果会非常大），
        //   如果只夹下界不夹上界，a 会变成天文数字，
        //   下面的循环就会写到 cells[r][a] 之外 —— 直接把堆写坏（0xC0000374）。
        const int a = std::max(0, std::min(arena::VISION_COLS, colLo));
        const int b = std::max(-1, std::min(arena::VISION_COLS - 1, colHi));

        for (int col = 0; col < a; ++col) {
            out.grid.cells[r][col] |= arena::CELL_WALL;
        }
        for (int col = b + 1; col < arena::VISION_COLS; ++col) {
            out.grid.cells[r][col] |= arena::CELL_WALL;
        }
    }
}

}  // namespace agent
