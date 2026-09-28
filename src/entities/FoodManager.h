#pragma once

#include "Food.h"
#include "FoodZone.h"
#include "Snake.h"

#include <cstddef>
#include <vector>

// ==========================================
// 食物管理器 — 管理地图上所有食物的生命周期
//
// 职责：
//   1. 【食物圈】按固定顺序轮流供食（见 FoodZone.h 与 docs/70）
//   2. 蛇死亡时将尸体转化为食物
//   3. 被吃掉的食物从列表中移除
//   4. 食物超时后自动消失（普通食物与尸体食物都有寿命）
//   5. 渲染所有食物
//
// ★ 这是【唯一的食物生成实现】
//
//   抽 Arena 之前，食物生成有两套：GameScene 构造里配一套、
//   train_probe::World 构造里配另一套，靠人工同步。
//   现在只有这一份 —— 改食物规则时改这里就够了，训练和游戏同时生效。
//
//   ⚠️ 随之而来的硬约束：本文件**不能依赖 GameConfig**（那是 core/ 层的），
//      所有参数都要通过 applyConfig / setZones 显式传进来。
// ==========================================
class FoodManager {
public:
    FoodManager(float mapWidth, float mapHeight);

    // 应用配置参数。
    //
    //   foodPerSecond      每秒生成多少食物。**这个值随蛇数缩放** ——
    //                      调用方按「蛇数 / 9」算好传进来（见 docs/70）。
    //                      0 = 不生成（只靠尸体）
    //   foodSegs           吃几个普通食物长一节体节
    //   corpseRatio        每隔多少体节生成一个尸体食物
    //   corpseScore        尸体食物分值
    //   corpseSegs         吃几个尸体食物长一节
    //   foodLifetime       普通食物寿命（秒），到点自动消失
    //   corpseLifetime     尸体食物寿命（秒）
    void applyConfig(float foodPerSecond, int foodSegs, int corpseRatio, int corpseScore,
                     int corpseSegs, float foodLifetime = 30.0f,
                     float corpseLifetime = 30.0f);

    // 设置食物圈与供食时间表。
    //   zones 为空 → 退化成"全图随机撒"（老行为，调试/对照用）
    void setZones(const std::vector<arena::FoodZone>& zones, const arena::ZoneSchedule& sched) {
        m_zones = zones;
        m_sched = sched;
    }
    const std::vector<arena::FoodZone>& zones() const { return m_zones; }
    const arena::ZoneSchedule&          schedule() const { return m_sched; }

    // ★★ 只更新调度表（不动圈）—— 给"每局换种子"用（2026-09-27）
    //
    //  ★ 为什么需要它
    //    `FoodManager` 持的是 `ZoneSchedule` 的【副本】（`m_sched`），
    //    而其中 `orderSeed`（圈轮换顺序的种子）会**每局变**。
    //    `Arena` 里那个 `m_sched` 改了之后**必须同步到这个副本**，
    //    否则"食物到底在哪个圈生成"和"感知说轮到哪个圈"会用两个不同的种子 ——
    //    那种不一致极难查（现象是"食物长在了错误的圈里"）。
    void setSchedule(const arena::ZoneSchedule& sched) { m_sched = sched; }

    // ★ 可写访问 —— 给"位置随机"模式用（Arena 在轮换时把新的活跃圈挪到新坐标）。
    //
    //  ★ 为什么允许外部改圈的位置
    //    "圈在哪"属于【世界状态】，而世界状态的唯一所有者是 Arena。
    //    FoodManager 只负责"在圈里生成食物"和"统计圈里有多少食物"。
    //    如果让 FoodManager 自己去随机位置，它就同时承担了调度职责 ——
    //    而调度规则（第几轮轮到谁）的唯一实现是 ZoneSchedule。
    std::vector<arena::FoodZone>& mutableZones() { return m_zones; }

    // 生成速率可以每帧改（Arena 按【存活蛇数】动态算，见 Arena::step()）
    void  setFoodPerSecond(float v) { m_foodPerSecond = (v > 0.0f) ? v : 0.0f; }
    float foodPerSecond() const { return m_foodPerSecond; }

    // -----------------------------------------------------------------------
    //  ★★ 普通食物的数量上限（2026-09-27 恢复，负责人要求）
    //
    //  ★ 为什么需要它
    //    `foodPerSecond` 是【蛇数 × 每蛇速率】算出来的，而无限模式要"食物管够"
    //    （每秒几十个）。只靠"速率 × 寿命"来限制总量是不可控的：
    //      上限 = 速率 × 寿命 = 40 × 30 = 1200 个
    //    1200 个食物铺在 3000² 上已经不是"管够"，而是"地图变红"——
    //    而且蛇根本吃不完，等于把"觅食"再次变成"随便走都能撞到"。
    //    有了上限，就可以放心地把【速率】调高（看起来源源不断），
    //    同时把【场上存量】控制在一个能看清、也吃得完的量。
    //
    //  ★ 口径：只算【普通食物】，**不含尸体**。
    //    尸体是别的蛇死了留下的，属于战利品，不该被"自然生成上限"卡住 ——
    //    否则大混战里尸体会被自然食物挤掉，而它恰恰是那个模式的核心食物来源。
    //
    //  ★ 取值：0 或负数 = 不限制（老行为，训练任务保持这个语义）
    // -----------------------------------------------------------------------
    void  setFoodCap(int n) { m_foodCap = (n > 0) ? n : 0; }
    int   foodCap() const { return m_foodCap; }

    // 当前【普通食物】个数（不含尸体）。由 update() 每帧重算，是准确值。
    int   foodCount() const { return m_foodCount; }

    // 累计时间（秒）。圈轮换就是按它算的。
    float zoneTimer() const { return m_zoneTimer; }

    // 每帧调用：分发食物生成 + 超时清理 + 重新统计各圈食物数
    void update(float dt);

    // 清空这一局留下的【世界状态】：食物、尸体、生成计时器、圈内统计。
    //
    // ★ 为什么必须单独有这么一个函数
    //   `Arena::reset()` 会重建全部蛇，但食物不归蛇管 —— 少了这一步，
    //   "重开一局"时上一局的食物和尸体全都留在场上。
    //   （这是抽 Arena 时引入的回归：以前 GameScene 是 `m_foodManager =
    //     FoodManager(...)` 整个重建，顺手就清掉了。）
    //
    // ⚠️ 只清"这一局产生的状态"，【不清配置】：圈的定义、食物寿命、
    //    生成速率都是配置，必须留着。
    void reset() {
        m_foods.clear();
        m_zoneTimer  = 0.0f;
        m_spawnAccum = 0.0f;
        m_foodCount  = 0;
        for (auto& z : m_zones) z.foodCount = 0;
    }

    // 在【全图随机位置】撒 n 个普通食物。
    //
    // ★ 用途：给"没有食物圈"的世界一个初始食物量（老玩法的行为）。
    //   食物圈玩法不需要它 —— 圈会自己逐秒生成。
    //
    // ⚠️ 受 `setFoodCap()` 限制：要到上限了就少撒（不会硬塞进去）。
    //    这是为了让"初始就来一大片"的无限模式仍然守上限 ——
    //    否则开局那一下就冲破了上限，之后的限制也就没有意义。
    void seedUniform(int n);

    // 在【指定位置】放一个普通食物。
    //
    // ★ 用途：最小学习验证（"朝前觅食"任务）需要把食物放在一条【已知的
    //   固定位置】上 —— 比如蛇出生点正前方 300px。
    //   这样最优策略是显而易见的"直行"，用来判断学习链路本身通不通。
    //   详见 docs/50 与 docs/85-最小学习验证.md。
    void spawnAt(float x, float y);

    // 当蛇死亡时调用：将其体节转化为尸体食物
    void spawnCorpseFood(const Snake& snake);

    // 获取所有食物列表（供碰撞检测遍历）
    std::vector<Food>& getFoods() { return m_foods; }
    const std::vector<Food>& getFoods() const { return m_foods; }

    // 移除指定索引的食物（被吃掉时调用）
    void removeFood(size_t index);

    // 渲染所有食物
    void render(sf::RenderWindow& window) const;

private:
    // 在某个圈内随机位置生成一个普通食物
    void spawnFoodInZone(const arena::FoodZone& z);

    // 全图随机位置生成一个普通食物（只在没有圈时用）
    void spawnRandomFood();

    // 重新统计每个圈里有多少食物（感知要用）
    void recountZoneFood();

    float m_mapWidth, m_mapHeight;
    std::vector<Food> m_foods;

    // ---- 圈与供食时间表 ----
    std::vector<arena::FoodZone> m_zones;
    arena::ZoneSchedule          m_sched;
    float                 m_zoneTimer = 0.0f;   // 累计时间（秒）
    float                 m_spawnAccum = 0.0f;  // 生成累加器（支持「每秒 0.22 个」这种小数）
    float                 m_foodPerSecond = 0.0f;

    // ★★ 普通食物存量上限，0 = 不限制（见 setFoodCap 的说明）
    int   m_foodCap   = 0;
    // 当前普通食物个数（不含尸体），update() 每帧重算
    int   m_foodCount = 0;

    // ---- 尸体参数 ----
    int   m_foodSegs    = 1;   // 吃几个普通食物长一节（游戏配置覆盖）
    int   m_corpseRatio = 1;
    int   m_corpseScore = 3;
    int   m_corpseSegs  = 2;

    // ---- 食物寿命 ----
    float m_foodLifetime   = 30.0f;
    float m_corpseLifetime = 30.0f;
};
