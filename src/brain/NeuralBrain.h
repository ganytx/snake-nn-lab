#pragma once

// ============================================================================
//  NeuralBrain.h — 把训练好的神经网络接成游戏里的一个算法
//
//  它和 RuleBrain 的唯一区别就是"怎么把感知变成动作"：
//      RuleBrain    : 手写的优先级行为树
//      NeuralBrain  : 20 维观测 → MLP → 3 个动作的概率 → 取最大的那个
//
//  两者看到的世界完全一样（都是 arena::Perception），所以能公平对战。
//
//  ★ 显示时用 argmax（贪心），不采样。
//    训练时必须采样（否则没有探索），但给人看的时候要的是"它现在最好的判断"。
//
//  ★ 渲染时也【不加速】：这个网络的动作空间里没有加速，
//    让它加速等于喂它没学过的东西。
// ============================================================================

#include "Brain.h"
#include "../rl/encoder.h"

#include <string>

namespace arena {

class NeuralBrain : public Brain {
public:
    // 构造时自动去几个常见路径找权重文件；找不到就用随机权重，并如实标注
    NeuralBrain();

    const BrainInfo& info() const override;
    Action think(const Perception& p) override;

    // ★ 每局开始时清空帧堆叠历史（避免新一局带着上一局的"记忆"）
    void onEpisodeStart(std::uint32_t seed) override;

    // 权重是否加载成功（UI / 调试用）
    bool weightsLoaded() const { return m_loaded; }
    const std::string& weightsPath() const { return m_path; }

    // -----------------------------------------------------------------------
    //  ★ 指定要加载的权重文件（负责人要求：能自己选加载哪个模型）
    //
    //  ★ 为什么用"待定设置"而不是构造函数参数
    //    `NeuralBrain` 是由 `BrainRegistry::create(id)` 造出来的，
    //    那个工厂函数【只认 id】—— 加参数就要改注册表接口，
    //    而注册表的全部意义就是"UI 只认 id，新增算法不用改 UI"。
    //
    //    所以沿用本项目已经验证过的做法（同 TrainingSetupScene::pendingPreload）：
    //    界面先把"要加载哪份"记在一个静态待定值里，
    //    NeuralBrain 构造时读它。**空 = 走原来的自动查找**。
    //
    //  ★ 为什么是全局而不是每队一份
    //    观测维度由 OBS_SIZE 决定，现在是 29 维。不同维度训练出来的权重
    //    根本无法互相加载 —— 所以"每队一个模型"目前没有意义，
    //    只会让局面变成"某个队必然加载失败"。一份就够。
    // -----------------------------------------------------------------------
    static void setPendingWeights(const std::string& name);   // 空 = 自动查找
    static const std::string& pendingWeights();

    // -----------------------------------------------------------------------
    //  ★★ 敌方神经网络：从"冻结快照"加载（自对弈）
    //
    //  ★ 负责人要求："我想把敌人也改成神经网络"
    //
    //  ★ 语义和 pendingWeights 的区别
    //      pendingWeights  = 这是【我方】要用的权重（用户在界面选的）
    //      pendingEnemySnapshot = 这是【敌方】要用的权重（训练器定期写的冻结拷贝）
    //
    //  ★ 为什么敌方失败时【不退回自动查找】
    //    敌方如果加载失败就变成"随机蛇"，那这局训练就白跑了，而且很难察觉。
    //    所以宁可让它"不加载"（保持随机初始化）并在描述里明确写出来 ——
    //    至少界面上看得见。更重要的：「选了 A 却偷偷用 B」是最难查的一类问题。
    // -----------------------------------------------------------------------
    static void setPendingEnemySnapshot(const std::string& pathOrName);
    static const std::string& pendingEnemySnapshot();

private:
    rl::Policy m_policy;
    bool        m_loaded = false;
    std::string m_path;
    BrainInfo   m_info;      // 成员而非 static：displayName 要反映"有没有加载到权重"

    // ★ 帧堆叠历史 ——【每条蛇私有】
    //   共用会让多条蛇互相污染"记忆"，现象极难查。
    rl::FrameStack m_frameStack;
};

}  // namespace arena
