#pragma once

// ============================================================================
//  PerceptionBuilder.h — 智能体层：把世界"翻译"成算法能看的 Perception
//
//  这是分层里的中间那一层：
//
//      仿真层（Snake / Food / SpatialGrid）
//            │  本文件负责这一段
//            ▼
//      arena::Perception  ← 算法层唯一能看到的东西
//
//  为什么不把光栅化交给算法自己做？
//    · 让算法从实体列表自己光栅化，需要几千次空间查询，慢得离谱
//    · 而且每个算法都写一遍光栅化，不但重复，还容易出现"某个算法看到的
//      世界和别人不一样"这种破坏公平性的 bug
//  所以统一由这里生成，所有算法拿到的是【完全相同】的感知。
// ============================================================================

#include "../brain/Perception.h"
#include "../entities/Food.h"
#include "../entities/FoodZone.h"
#include "../entities/Snake.h"

#include <cstddef>
#include <vector>

namespace agent {

// ---------------------------------------------------------------------------
//  感知参数（来自对局配置）
// ---------------------------------------------------------------------------
struct PerceptionConfig {
    // 视野网格的格边长（像素）。32 格 x 32px = 视野 1024 x 1024px
    //
    // ⚠️ 注意：光栅化成本 = 格数 x 每格约 10 次运算，**和格边长无关**。
    //    想把感知调便宜，要改的是【格数】（VisionGrid.h 里的 VISION_COLS/ROWS），
    //    改格边长只会改变视野范围和精度，不改变开销。
    float cellSize = 32.0f;

    // 最近的多少个队友走"全局通道"（不受视野限制）
    // 默认 2；0 表示完全关闭队友全局感知
    int allyVisibleCount = 2;
};

// ---------------------------------------------------------------------------
//  世界快照 —— 感知层需要知道的全部
//  做成一个独立结构，就不用让算法层去认识 GameScene / FoodManager
// ---------------------------------------------------------------------------
struct WorldView {
    const std::vector<Snake*>* snakes = nullptr;   // 世界里的所有蛇
    const std::vector<Food>*   foods  = nullptr;   // 世界里的所有食物
    float mapWidth  = 0.0f;
    float mapHeight = 0.0f;
    int   tick = 0;
    float dt   = 0.0f;

    // ★ 食物圈（辅助通道 3）。nullptr = 这个世界没有圈。
    //   activeZone 指向当前正在供食的那个圈（没有则为 nullptr）。
    //   由 Arena 填 —— FoodManager 持有 zones，Arena 每帧把指针接过来。
    const std::vector<arena::FoodZone>* zones      = nullptr;
    const arena::FoodZone*              activeZone = nullptr;

    // ★ 当前 / 下一个要供食的圈的下标（−1 = 没有圈）。
    //
    //  ★ 为什么用【下标】而不是再给一个"下一个圈"的指针
    //    "下一个"的算法在各个 ZoneSchedule 模式（顺序 / 随机）下不一样，
    //    与其让感知层学一遍，不如让 Arena 直接给答案 ——
    //    **这套调度规则的唯一实现是 ZoneSchedule**，别处不许重推。
    int activeZoneIndex = -1;
    int nextZoneIndex   = -1;

    // ★ 本轮供食【已经过了几秒】（2026-09-26 新增）
    //
    //  ★ 为什么需要它
    //    规则 AI（以及任何算法）要判断"现在这个圈还能供多久"，
    //    才能在供食结束【之前】就动身去下一个圈（提前出发）。
    //    没有它就只能等圈空了才走 —— 永远迟到一步。
    //
    //  ★ 为什么由 Arena 填，而不是让算法自己推
    //    "本轮过了多久"由 ZoneSchedule 的周期决定（顺序 / 随机 / 不同时长），
    //    让每个算法各推一遍就是**调度规则的第二份实现** ——
    //    本项目为此栽过两次。所以透传 Arena 给的答案。
    //
    //  ⚠️ 这是【时间】量，不是"看到的"东西，所以它属于世界常识，不是作弊。
    float cycleElapsed = 0.0f;

    // ★ 圈的调度规则（只读）—— 给"本轮还能供食几秒"用（2026-09-26）
    //
    //  ★ 为什么直接给指针，而不是让 Arena 把剩余秒数算好
    //    剩余秒数 = spawnDuration − cycleElapsed，是**一行纯算术**，
    //    不含任何调度决策（"轮到谁"才是决策，那个已经由 activeZoneIndex 给了）。
    //    为了这一行再往 WorldView 加一个字段，反而让"谁负责算什么"变模糊。
    //  ⚠️ 但**绝不能用它去推"下一个是谁"** —— 那必须用 nextZoneIndex。
    const arena::ZoneSchedule* zoneSchedule = nullptr;

    // ★ 当前是否处于【供食阶段】（2026-09-26 新增）
    bool zoneSpawning = true;

    // ★★ 当前是否处于【预告期】（2026-09-27 新增）
    //
    //  ★ 用途：决定"下一个圈"可不可见
    //      预告期  → 下一个圈**可见**（告知位置，留出赶路时间）
    //      其余时间 → 下一个圈**不可见**（只有当前圈，避免"挂着却不产食物的圈"）
    //
    //  ★ 预告期是什么时候（负责人定义）
    //    "预告圈是在活跃圈产食物**还剩 3s** 后出现的，然后中间有一定时间 gap，
    //      然后再被激活。"
    //    即：供食期最后 `previewLeadSeconds` 秒 + **整个空窗期**。
    //    默认 3 + 5 = 8 秒。判据的唯一实现是 `ZoneSchedule::previewing()`。
    //
    //  ★ 为什么不让感知层自己算
    //    它是**调度规则**的一部分（和"轮到谁"同级），唯一实现必须在 ZoneSchedule。
    //    感知层重推一遍就是这个项目栽过两次的"两份实现"。
    bool zonePreviewing = false;
};

// ---------------------------------------------------------------------------
class PerceptionBuilder {
public:
    void setConfig(const PerceptionConfig& cfg) { m_cfg = cfg; }
    const PerceptionConfig& config() const { return m_cfg; }

    // 为 world.snakes[selfIndex] 这条蛇填充感知。
    // out 由调用方（Agent）持有并复用，本函数只做清空 + 重填，不做堆分配。
    void build(const WorldView& world, std::size_t selfIndex,
               arena::Perception& out) const;

    // 把地图边界外的格子标记为墙。
    //
    // public 是刻意的：bench_sim 里的【对拍测试】需要直接调它，
    // 拿它和一份"逐格朴素实现"逐格比对，确认解析法没有标错。
    // 这段代码是"看起来对、容易错一格"的类型，没有对拍不敢合并。
    void markWalls(float selfX, float selfY, float heading,
                   arena::Perception& out) const;

private:
    PerceptionConfig m_cfg;
};

}  // namespace agent
