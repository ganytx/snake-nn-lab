#pragma once

// ============================================================================
//  HumanBrain.h — "玩家"也是一个算法
//
//  ★ 为什么玩家要走 Brain 接口？
//
//  需求方明确说过："这只是给用户可视化体验的，实际训练过程无需玩家参与。"
//  把玩家做成一个 Brain 实现之后：
//
//      可视化对局：某队里有一个 agent 挂 HumanBrain，其余挂 AI
//      训练：      全部挂 AI，没有 HumanBrain
//
//  于是 GameScene 里不再有"玩家 vs AI"的分支，全是一视同仁的 Agent。
//  而且训练和观战走【完全相同】的世界代码，只是最后一个 agent 换了 brain，
//  不会出现"训练时好好的，一加玩家就崩"这类问题。
//
//  ★ 输入怎么进来？
//  鼠标/键盘是 SFML 的东西，而算法层禁止 include SFML。
//  所以游戏层负责采集，填进 HumanInput，再 setInput() 交给本类。
//  —— 这条边界必须守住，否则无头训练模式会莫名其妙拖进图形依赖。
// ============================================================================

#include "Brain.h"

namespace arena {

// 玩家这一帧的输入快照（由游戏层从 SFML 采集后填入）
struct HumanInput {
    bool  hasAim    = false;   // 鼠标位置是否有效
                               // （玩家第一次移动鼠标前为 false，避免开局
                               //   鼠标停在窗口角落导致蛇朝那个方向冲出去）
    float aimX      = 0.0f;    // 鼠标所指的【世界坐标】（游戏层已经做过屏幕→世界换算）
    float aimY      = 0.0f;
    bool  boost     = false;   // 加速键（空格 / 鼠标左键）
    bool  turnLeft  = false;   // 键盘备选：相对当前朝向左转 90°
    bool  turnRight = false;
};

class HumanBrain : public Brain {
public:
    const BrainInfo& info() const override;

    // 游戏层每帧调用一次，把这一帧的输入交给它
    void setInput(const HumanInput& in) { m_input = in; }

    Action think(const Perception& p) override;

private:
    HumanInput m_input;
};

}  // namespace arena
