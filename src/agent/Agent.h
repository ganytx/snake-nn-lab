#pragma once

// ============================================================================
//  Agent.h — 一条蛇 + 一个大脑 + 一份感知缓冲
//
//  ★ 为什么要有这个结构体？
//
//  重构之前，GameScene 里有【三个按下标一一对应的容器】：
//      std::vector<Snake>        m_aiSnakes
//      std::vector<AIController> m_aiControllers
//      std::vector<float>        m_aiBoostCooldown
//  删一条蛇必须三个一起删，漏掉任何一个，"反应时间""加速冷却"就会串到
//  别的蛇身上。而且 m_allSnakes 是一张裸指针表，vector 一扩容就全部失效，
//  全靠手动调 rebuildSnakeList() 兜着。这两个隐患都属于"出一次就够你查半天"。
//
//  Agent 把它们合成【一个容器】：
//      std::vector<Agent> agents;
//  于是那两类 bug 从根上消失了。
//
//  ★ 玩家也是一个 Agent，只是 brain 挂的是 HumanBrain。
//    这样 GameScene 里不再有"玩家 vs AI"的分支 ——
//    可视化对局和训练走的是完全相同的世界代码。
// ============================================================================

#include "../brain/Brain.h"
#include "../entities/Snake.h"

#include <memory>
#include <utility>

namespace agent {

struct Agent {
    Snake                         body;
    std::unique_ptr<arena::Brain> brain;
    arena::Perception             perception;      // 复用缓冲，每帧清空重填
    int                           teamId = -1;
    bool                          deathHandled = false;  // 尸体是否已经转成食物（只做一次）

    Agent() = delete;   // Snake 没有默认构造函数，必须给定出生参数

    Agent(float x, float y, float angle, const sf::Color& color,
          bool isPlayer, std::unique_ptr<arena::Brain> b, int team)
        : body(x, y, angle, color, isPlayer)
        , brain(std::move(b))
        , teamId(team)
    {
        body.setTeamId(team);
    }

    // 只可移动，不可拷贝（unique_ptr 成员）。
    // std::vector<Agent> 扩容时走移动构造 —— 因为 Snake 里没有任何
    // "指向自身内部"的裸指针，移动是安全的。
    Agent(Agent&&) noexcept = default;
    Agent& operator=(Agent&&) noexcept = default;
    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    bool isAlive() const { return body.isAlive(); }

    // 是否是玩家控制的（调试面板 / 队伍统计用）
    bool isHuman() const { return body.isPlayer(); }
};

}  // namespace agent
