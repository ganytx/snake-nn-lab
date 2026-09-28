#pragma once

// ============================================================================
//  FoodZone.h — 食物圈
//
//  ★ 玩法（详见 docs/70-食物圈玩法设计.md）
//
//  地图上有 4 个固定的圈，按【固定顺序】轮流供食：
//
//      第 k 轮：只在圈 (k mod 4) 里生成食物
//      生成：每秒 蛇总数/9 个，持续 9 秒   →  共「蛇总数」个
//      空窗：5 秒（谁都不供，逼蛇吃完就走 / 落后方有机会追）
//      然后轮到下一个圈
//
//  ★ 为什么位置固定、只有圈内位置随机
//
//  网络是纯 MLP：没有记忆、没有时钟。位置固定 → 它能靠观测里的
//  sin/cos(heading) 把「某个绝对方位」和「食物多」绑定，这是**空间记忆**，
//  MLP 学得会。位置每局随机就变成「看一眼记住、等会儿再用」，
//  那需要工作记忆，MLP 结构上做不到。
//
//  ★ 为什么是「逐渐生成」而不是「一次生成完」
//
//  一次性生成 → 最优行为是「冲进去扫完就走」。
//  逐渐生成 → 最优行为是「待着不走」——吃掉一个、原地附近等、下一个又出现。
//  这是「占住地盘」这个行为能学出来的前提，不是手感问题。
// ============================================================================

#include <cmath>
#include <cstdint>
#include <vector>

namespace arena {

// ---------------------------------------------------------------------------
//  一个圈
// ---------------------------------------------------------------------------
struct FoodZone {
    float x = 0.0f;        // 圈心世界坐标
    float y = 0.0f;
    float radius = 150.0f; // 圈半径

    // 该圈里当前有多少食物（每帧由 FoodManager 统计，供感知使用）
    int foodCount = 0;

    // 点是否落在圈内
    bool contains(float px, float py) const {
        const float dx = px - x;
        const float dy = py - y;
        return dx * dx + dy * dy <= radius * radius;
    }
};

// ---------------------------------------------------------------------------
//  圈的布局
//
//  全部按地图尺寸【比例】推导，换地图自动适配 —— 不写死像素。
//
//      固定朝向   第 i 个圈在 baseAngle + i * 90°
//      baseAngle  默认为 0（正右），所以四个圈在 右 / 下 / 左 / 上
//      圈心距中心  mapMin * centerRatio
//      圈半径      mapMin * radiusRatio
//
//  2000x2000 地图 + 默认比例：圈心距中心 620px、圈半径 150px
//  4000x4000 地图 + 默认比例：圈心距中心 1240px、圈半径 300px
//
//  ⚠️ 注意半径是跟着地图缩放的。如果大图上想让圈保持 150px（更容易守住），
//     就把 radiusRatio 调小。
// ---------------------------------------------------------------------------
struct ZoneLayout {
    int   count       = 4;
    float centerRatio = 0.31f;   // 圈心距地图中心的距离 / min(mapW,mapH)
    float radiusRatio = 0.075f;  // 圈半径 / min(mapW,mapH)
    float baseAngle   = 0.0f;    // 第一个圈的角度（弧度）

    // -----------------------------------------------------------------------
    //  ★★ 位置随机模式（负责人要求）
    //
    //  ★ 它和"固定四方位"是【两件不同的事】，别混
    //
    //      固定位置（randomPosition = false）：4 个圈心永远在 右/下/左/上 四个方位。
    //          蛇可以靠"背下地图结构"取胜 —— 它不需要看观测也知道该去哪。
    //
    //      位置随机（randomPosition = true）：每一轮在地图上【随机挑一个新的
    //          合法坐标】当圈心。于是：
    //            · 蛇【不可能】背下位置，只能读观测里的"当前圈/下一个圈"
    //            · 观测里那个"下一个圈在哪"的方位才有了真实价值（提前出发）
    //
    //  ★ 合法性约束（三个都必须满足）
    //      ① 整圆在图内：x ∈ [r, W−r]，y ∈ [r, H−r]
    //      ② 离上一轮那个圈足够远 —— 否则"跑过去"这个行为没有意义，
    //         而且两圈重叠会让"去下一个圈"和"留在原地"变成同一件事
    //      ③ 离墙留出蛇头活动余量（复用 ①，r 已经保证了）
    // -----------------------------------------------------------------------
    // -----------------------------------------------------------------------
    //  ★ 和【上一个圈】的距离必须落在 [minDistRatio, maxDistRatio] × 短边
    //
    //  ★ 负责人要求（原话）
    //    "每次食物圈的生成不是完全随机，而是保持一定范围的距离。" → 40%~60%
    //
    //  ★ 为什么必须有这个约束
    //    纯随机撒点时，两圈可能挨着（那样"跑过去"根本不算本事），
    //    也可能跳到地图对角（最远 2800px，要跑 11 秒）——
    //    **难度方差巨大**，每一轮都是全新考验，于是什么都学不稳。
    //    夹在一个区间里，每一轮的"跑图"难度就基本恒定，
    //    这才是能形成学习课程的前提。
    //
    //  ★ 为什么用【极坐标】而不是先挑 x,y 再筛
    //    旧写法是先随机 x,y，再检查距离够不够远 —— 那样"距离"只是下限，
    //    而且不满足就重试，很容易退化到"退回地图中心"。
    //    改成先定【角度 + 距离】再算坐标，距离区间就是【硬保证】，
    //    而且在圆的范围内一定能找到合法点（见下面的角度试探）。
    // -----------------------------------------------------------------------
    bool  randomPosition = false;
    float minSeparationRatio = 0.45f;  // 已废弃：被 minDistRatio/maxDistRatio 取代

    // -----------------------------------------------------------------------
    //  ★★ 与上一个圈的距离区间（2026-09-27 第三次调整）
    //
    //  ★ 改动史（来回调过一次，第三次才定下来）
    //      最初     0.40~0.60 —— 负责人："保持一定范围的距离"
    //      第二次   0.20~0.30 —— 负责人："只把距离上限缩再短一些吧"
    //      第三次   0.40~0.60 —— 负责人："下一个圈的距离限制调远，是现有大概两倍"
    //                            （0.20×2 = 0.40、0.30×2 = 0.60，正好回到最初的值）
    //
    //  ★ 换算成实际数字（3000² 地图、蛇速 250px/s）
    //      0.40 × 3000 = 1200px  → 跑 4.8 秒
    //      0.60 × 3000 = 1800px  → 跑 7.2 秒
    //
    //  ★ 赶路时间够不够（这次调远之后必须重新核对）
    //    预告圈在【供食还剩 3s】时出现，加上整个空窗期 5s
    //    ⇒ 从看到它到它开始供食有 **8 秒**（见 ZoneSchedule::previewLeadSeconds）。
    //    而最远 7.2 秒 —— 仍然赶得上，但余量只剩 0.8 秒，比原来紧。
    //    ⚠️ 如果再往上调（0.7 以上），"跑到就快换圈了"，那条"永远在赶路、
    //       一口都吃不上"的蛇就会重新出现（第二次调整就是为了修它）。
    //
    //  ★ 为什么下限也跟着动（而不是只动上限）
    //    只调上限会把区间压成一个点（上限比下限还小），
    //    代码里 `dHi = max(dLo, min(dMax, saneMax))` 会把它俩拉平，
    //    结果是"每一轮距离都一样"—— 那比原来更糟。
    //    所以区间整体缩放：**保持 1.5:1 的跨度比例不变**。
    // -----------------------------------------------------------------------
    float minDistRatio = 0.40f;        // 与上一个圈的最小距离 / min(W,H)
    float maxDistRatio = 0.60f;        // 与上一个圈的最大距离 / min(W,H)

    // -----------------------------------------------------------------------
    //  位置随机种子
    //
    //  ★ 为什么是"确定性哈希"而不是 rand()
    //    位置必须只依赖 (cycle, seed)，否则 replay_check 的世界指纹
    //    每次跑都不一样、同一个 seed 也复现不出同一局。
    //    "对蛇随机"、但对调试完全可复现。
    //
    //  ★★ posSeedOffset：让【每一局】不一样（2026-09-27 负责人要求）
    //    只用固定 posSeed 时，位置序列**每次开新局都完全相同** ——
    //    负责人看到的"种子似乎固定"就是这个。
    //    现在由调用方（GameScene）在每局开始时给一个随机偏移，
    //    于是每局的圈位序列都不同；而把偏移设成 0 就回到完全可复现，
    //    replay_check 用的就是它自己的固定配置（不设这个字段）。
    // -----------------------------------------------------------------------
    unsigned posSeed = 20260924u;      // 位置随机种子（确定性，可复现）
    unsigned posSeedOffset = 0u;       // ★ 每局偏移（0 = 完全可复现）

    // 按地图尺寸算出所有圈（固定位置模式）
    std::vector<FoodZone> build(float mapW, float mapH) const {
        std::vector<FoodZone> zones;
        if (count <= 0) return zones;

        const float m  = (mapW < mapH) ? mapW : mapH;
        const float cx = mapW * 0.5f;
        const float cy = mapH * 0.5f;
        const float ringR = m * centerRatio;
        const float zr    = m * radiusRatio;

        zones.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            const float a = baseAngle
                          + 6.283185307179586f * static_cast<float>(i)
                          / static_cast<float>(count);
            FoodZone z;
            z.radius = zr;

            if (!randomPosition) {
                z.x = cx + std::cos(a) * ringR;
                z.y = cy + std::sin(a) * ringR;
            } else {
                // ★★ 位置随机模式下，这里【不挑位置】，只把槽位搭出来
                //
                //  ★ 为什么不在 build 里挑
                //    "每个槽位该在哪儿"依赖【别的槽位在哪】（距离约束要求两圈相距
                //    短边的 20%~30%），而那要等所有槽位都建好才能算。
                //    build 里边建边挑会读到还没 push 的 `zones[0]` —— 顺序一错就崩。
                //
                //  ★ 所以位置统一由 `Arena::reset()` 显式布置，规则和
                //    `Arena::step` 第 0 步【完全一致】：
                //        槽位 0 = 第 0 轮的活跃圈
                //        槽位 1 = 第 1 轮的活跃圈（相对第 0 轮就是"预告"）
                //    这样开局那一处也满足"预告圈 = 下一轮活跃圈"。
                //    先给一个地图中心的安全值，真正的坐标随 reset() 覆盖。
                z.x = cx;
                z.y = cy;
            }
            zones.push_back(z);
        }
        return zones;
    }

    // -----------------------------------------------------------------------
    //  ★ 把某个圈挪到一个新的随机合法位置
    //
    //  调用时机：这个圈【成为当前活跃圈】的那一轮（见 Arena::step 里的轮换检测）。
    //  也就是说"A 正在供食时，B 的位置已经定好了" —— 这是负责人明确要的：
    //  上一轮还没消失、下一轮已经选好，中间仍然有 gap。
    //
    //  cycle  —— 用哪一轮的随机数（确定性：同一 cycle 永远同一个位置）
    //  prevX/Y —— 上一个活跃圈的圈心，用来保证两圈不重叠
    //
    //  ★ 为什么用【确定性哈希】而不是 rand()
    //    位置必须只依赖 (cycle, seed)，否则 replay_check 的世界指纹
    //    每次跑都不一样、同一个 seed 也复现不出同一局。
    //    "对蛇随机"、但对调试完全可复现。
    // -----------------------------------------------------------------------
    void relocateRandom(FoodZone& z, float mapW, float mapH,
                        int cycle, float prevX, float prevY) const {
        pickRandomCenter(z, mapW, mapH, cycle, prevX, prevY);
    }

    // -----------------------------------------------------------------------
    //  给一个圈挑一个新的合法圈心
    //
    //  cycle  —— 用哪一轮的随机数（确定性：同一 cycle 永远同一个位置）
    //  prevX/Y —— 上一个圈心，用来保证两圈不重叠
    //
    //  ★ 为什么用【确定性哈希】而不是 rand()
    //    和 ZoneSchedule 的随机顺序同理：位置必须只依赖 (cycle, seed)，
    //    否则 replay_check 的世界指纹每次都不一样、同一个 seed 也复现不出同一局。
    //    "对蛇随机"、但对调试完全可复现。
    // -----------------------------------------------------------------------
    void pickRandomCenter(FoodZone& z, float mapW, float mapH,
                          int cycle, float prevX, float prevY) const {
        const float m  = (mapW < mapH) ? mapW : mapH;
        const float zr = m * radiusRatio;
        z.radius = zr;

        // 合法范围：整圆在图内
        const float loX = zr, hiX = mapW - zr;
        const float loY = zr, hiY = mapH - zr;
        if (hiX <= loX || hiY <= loY) {   // 地图太小装不下这个圈 → 退到中心
            z.x = mapW * 0.5f;
            z.y = mapH * 0.5f;
            return;
        }

        // ---- 1) 先定【距离】（硬保证落在区间内）----
        const float dMin = m * minDistRatio;
        const float dMax = m * maxDistRatio;
        // 距离也不能大到"无论往哪个方向都出界"：夹到"能塞进图里的最大距离"
        const float saneMax = 0.5f * std::sqrt(mapW * mapW + mapH * mapH);
        const float dLo = std::min(dMin, saneMax);
        const float dHi = std::max(dLo, std::min(dMax, saneMax));

        const float u = hash01(cycle, 0);
        const float d = dLo + u * (dHi - dLo);

        // ---- 2) 再挑【角度】，要求落点在地图内 ----
        //
        //  ★ 为什么不是"挑个方向，出界就退回中心"
        //    退回中心会让"距离约束"名存实亡（地图中心往往离上一个圈很远/很近）。
        //    所以这里在 24 个方向上找【第一个合法】的；
        //    真的一个都没有（上一个圈贴着角落、而 d 又很大）时，
        //    才按下面第 3 步把距离收进来。
        auto inside = [&](float x, float y) -> bool {
            return x >= loX && x <= hiX && y >= loY && y <= hiY;
        };

        auto tryAt = [&](float dist, int attempt) -> bool {
            const float a0 = hash01(cycle, 1 + attempt * 2) * 6.283185307179586f;
            // 每个 attempt 换一个起始角，再扫一圈小步长 ——
            // 这样既不遗漏"只有窄扇区合法"的情况，也保持确定性
            for (int k = 0; k < 12; ++k) {
                const float a = a0 + static_cast<float>(k) * 0.5235988f;  // +30°
                const float x = prevX + std::cos(a) * dist;
                const float y = prevY + std::sin(a) * dist;
                if (inside(x, y)) { z.x = x; z.y = y; return true; }
            }
            return false;
        };

        for (int attempt = 0; attempt < 4; ++attempt) {
            if (tryAt(d, attempt)) return;
        }

        // ---- 3) 兜底：把距离逐步收进来，直到能塞进图里 ----
        //     （保持方向随机，只是缩短距离；语义仍是"离上一个圈有一定距离"）
        for (float f = 0.85f; f >= 0.15f; f -= 0.14f) {
            if (tryAt(d * f, 0)) return;
        }

        // 连很近都不行（极端小的地图）→ 直接放地图中心，至少是合法的
        z.x = mapW * 0.5f;
        z.y = mapH * 0.5f;
    }

    // (cycle, slot) → [0,1) 的确定性伪随机数
    //
    // ★ 加上 posSeedOffset：同一个 cycle 在不同"局"里给出不同的值。
    //   偏移为 0 时行为和以前完全一致（replay_check 就是这种情况）。
    float hash01(int cycle, int slot) const {
        unsigned x = static_cast<unsigned>(cycle) * 0x9E3779B9u
                   + static_cast<unsigned>(slot) * 0x85EBCA6Bu
                   + posSeed
                   + posSeedOffset;      // ★ 每局偏移
        x ^= x >> 16; x *= 0x7FEB352Du;
        x ^= x >> 15; x *= 0x846CA68Bu;
        x ^= x >> 16;
        return static_cast<float>(x & 0xFFFFFFu) / static_cast<float>(0x1000000u);
    }
};

// ---------------------------------------------------------------------------
//  供食时间表
//
//  把「第几秒该干什么」算清楚，游戏和训练都读这一份。
// ---------------------------------------------------------------------------
struct ZoneSchedule {
    float spawnDuration = 9.0f;   // 持续生成多少秒
    float gapDuration   = 5.0f;   // 空窗多少秒

    // -----------------------------------------------------------------------
    //  ★★ 预告多久出现（负责人要求，2026-09-27）
    //
    //  ★ 负责人原话
    //    "不要一开始就有预告圈，预告圈是在活跃圈产食物**还剩 3s** 后出现的，
    //      然后中间有一定时间 gap，然后再被激活。"
    //
    //  ★ 时间线（默认值：供食 9s + 空窗 5s + 预告提前 3s）
    //
    //      0s ──────────── 6s ──────── 9s ──────── 14s ──────── 23s
    //      │←── 只有活跃圈 ──→│←─ 预告出现 ─→│←─ 空窗 ─→│←─ 新的活跃圈供食 ─→
    //      │   在产食物       │  还在产食物   │  都不产   │
    //                          ↑
    //                    预告圈在这里出现（spawnDuration − previewLead）
    //
    //    所以预告圈从出现到被激活一共有 `previewLead + gapDuration` 秒
    //    （默认 3 + 5 = **8 秒**）给蛇赶路。
    //
    //  ★ 为什么不是"开局就出现"
    //    开局就预告等于**提前一整轮（14 秒）**告诉它下一站在哪 ——
    //    那既不自然（还没决定要不要走就先看到了），
    //    也让"什么时候该出发"这件事失去了时间上的紧迫感。
    //
    //  ★ 为什么是 3 秒
    //    既要够短（"快结束了才预告"，有紧迫感），
    //    又要让 3s + gap 的总时长 ≥ 赶路时间（圈间距 600~900px，
    //    按 250px/s 要跑 2.4~3.6 秒）—— 8 秒留了充足余量。
    // -----------------------------------------------------------------------
    float previewLeadSeconds = 3.0f;   // 供食期还剩几秒时，把下一个圈显示出来

    // 现在是不是【预告期】（下一个圈该被看到/画出来了吗）
    //
    //   定义：本轮已过的秒数 >= spawnDuration − previewLeadSeconds
    //   也就是"供食期最后 previewLead 秒" + "整个空窗期"。
    bool previewing(float t) const {
        const float c = cycleDuration();
        if (c <= 0.0f) return false;
        float local = t - std::floor(t / c) * c;
        if (local < 0.0f) local += c;

        float lead = previewLeadSeconds;
        if (lead < 0.0f) lead = 0.0f;
        if (lead > spawnDuration) lead = spawnDuration;   // 不能提前到上一轮去
        return local >= (spawnDuration - lead);
    }

    // -----------------------------------------------------------------------
    //  ★ 下一个供食圈怎么选：顺序 还是 随机
    //
    //  ★ 为什么要有"随机"这一档（负责人要求）
    //
    //    固定顺序 0→1→2→3 有个隐蔽的副作用：
    //    **"看着 active 标志走"和"背下固定路线"会产生一模一样的行为** ——
    //    因为顺序永远一样，一个只会往"固定的下一个圈"走的策略，
    //    看起来和"真的在看信号"完全无法区分。
    //
    //    改成随机之后，**背路线这条路被彻底堵死**：
    //    下一个圈不可预测，蛇唯一的出路就是真的去读观测里的"当前圈/下一个圈"。
    //
    //  ★ 为什么用【确定性哈希】而不是 rand()
    //    `activeZone(t)` 必须只依赖 t（和圈数），否则：
    //      · replay_check 的世界指纹每次跑都不一样 → 回归工具失效
    //      · 同一个 seed 复现不出同一局
    //    所以用一个便宜的整数哈希把 (轮次, seed) 映射成圈下标。
    //    这样"随机"是【对蛇随机】的，但对调试是【完全可复现】的。
    // -----------------------------------------------------------------------
    bool randomOrder = false;
    unsigned orderSeed = 20260924u;

    // 每轮（一个圈）的总时长
    float cycleDuration() const { return spawnDuration + gapDuration; }

    // 给定累计时间 t 和圈数，当前轮到第几个圈
    int activeZone(float t, int zoneCount) const {
        if (zoneCount <= 0) return -1;
        const float c = cycleDuration();
        if (c <= 0.0f) return 0;
        const int cycle = static_cast<int>(t / c);
        if (!randomOrder) {
            return ((cycle % zoneCount) + zoneCount) % zoneCount;
        }
        return randomZone(cycle, zoneCount);
    }

    // 第 cycle 轮该供食哪个圈（确定性哈希 → 对蛇不可预测、对调试可复现）
    int randomZone(int cycle, int zoneCount) const {
        if (zoneCount <= 0) return -1;
        // splitmix32 风格的一轮混合：便宜、雪崩够好、无需维护状态
        unsigned x = static_cast<unsigned>(cycle) * 0x9E3779B9u + orderSeed;
        x ^= x >> 16; x *= 0x7FEB352Du;
        x ^= x >> 15; x *= 0x846CA68Bu;
        x ^= x >> 16;
        return static_cast<int>(x % static_cast<unsigned>(zoneCount));
    }

    // 当前是否处于【生成阶段】（相对于本轮开始过了多久）
    bool spawning(float t) const {
        const float c = cycleDuration();
        if (c <= 0.0f) return false;
        float local = t - std::floor(t / c) * c;
        if (local < 0.0f) local += c;
        return local < spawnDuration;
    }

    // 本轮已经过了几秒
    float elapsedInCycle(float t) const {
        const float c = cycleDuration();
        if (c <= 0.0f) return 0.0f;
        float local = t - std::floor(t / c) * c;
        if (local < 0.0f) local += c;
        return local;
    }

    // -----------------------------------------------------------------------
    //  ★【下一个】要供食的圈是哪个（负责人要求：观测给"当前圈 + 下一个圈"）
    //
    //  ★ 为什么必须放在这里，而不是让调用方写 `(active+1)%n`
    //    "下一个是谁"取决于调度模式：顺序模式是 +1，
    //    随机模式要用哈希算下一轮的圈。**调度规则的唯一实现是本结构**，
    //    在别处重推一遍就是第二份实现 —— 这个项目已经因此栽过两次
    //    （观测编码分叉 0.96、世界推进两份）。
    //
    //  ★ 为什么需要知道"本轮已过几秒"
    //    因为"下一个"的定义是【下一个要开始供食的圈】：
    //      · 本轮还在供食（前 spawnDuration 秒）→ 下一个就是下一轮的圈
    //      · 本轮已进空窗（后 gapDuration 秒）  → 下一个是【本轮的下一轮】，
    //        也就是即将开始供食的那个 —— 对蛇来说这是"马上就该去的地方"
    //    两种情况的答案都是 `cycle + 1` 那一轮，所以实现上统一。
    // -----------------------------------------------------------------------
    int nextZone(float t, int zoneCount) const {
        if (zoneCount <= 0) return -1;
        const float c = cycleDuration();
        if (c <= 0.0f) return 0;
        const int cycle = static_cast<int>(t / c);
        if (!randomOrder) {
            const int nxt = cycle + 1;
            return ((nxt % zoneCount) + zoneCount) % zoneCount;
        }
        return randomZone(cycle + 1, zoneCount);
    }
};

}  // namespace arena
