#pragma once

// ============================================================================
//  TrainingTeamScene.h — 训练模式的【队伍设置】（负责人要求）
//
//  ★ 为什么训练模式要有第二个设置界面
//
//  原来只有一个「训练设置」，里面挤了 10 行：任务 / 蛇数 / 食物 / 时长 /
//  出生 / 学习率 / 折扣 / 探索率 / 镜头 / 加载模型。
//  其中"几条蛇"和"什么队伍"是【组队】这件事，和"任务难度 / 学习超参"
//  是两类不同的东西，混在一页只会越堆越长。
//
//  所以拆成两个界面（和游戏模式一致 —— 那边也是"对局设置"专门管组队）：
//      ① 训练设置（TrainingSetupScene）—— 加载模型 / 任务 / 食物 / 学习超参
//      ② 队伍设置（本界面）            —— 几队 / 每队几条 / 每队用什么算法
//
//  ★ 两个界面共享【同一份】待定配置
//    `TrainingSetupScene::pendingConfig()`。所以在这边改完蛇数，
//    回到那边"训练蛇数"的显示会跟着变（回去时调 refreshRows()）。
//
//  操作：上下键 选择   左右键 调整   Enter 返回训练设置   ESC 返回主菜单
// ============================================================================

#include "../core/Scene.h"
#include "../core/Game.h"
#include "../rl/TrainTask.h"

#include <functional>
#include <string>
#include <vector>

class TrainingTeamScene : public Scene {
public:
    explicit TrainingTeamScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    // -----------------------------------------------------------------------
    //  ★ 可选的"每队算法"
    //
    //  ★ 负责人要求：像主模式（对局设置）那样，能选每队是【神经网络】还是
    //    【固定规则策略】。
    //
    //  ★ 在训练语境下这两个选项意味着什么（必须说清楚，否则这行会让人误解）
    //      神经网络 —— 这一队的【第 0 条蛇】就是正在学习的那条网络。
    //                  全训练场只允许存在一条（EpisodeRunner 固定学 index 0），
    //                  所以选它会把"学习中的那条"挪到这一队。
    //      规则策略 —— 这一队全是固定行为的规则 AI：
    //                  队 0 → 队友（可以互相穿过）
    //                  其它队 → 对手（会撞死你）
    //
    //  ★ 为什么"神经网络"只能出现在一个队上
    //    一条网络 + 一条轨迹是当前训练器的全部能力。让两支队伍都挂网络，
    //    需要两套网络和两条轨迹 —— 那是另一个量级的改动，不在本轮范围。
    //    所以这里做成【全局唯一】：选了某一队是神经网络，其它队自动回规则 AI。
    // -----------------------------------------------------------------------
    std::vector<std::string> m_teamBrain;   // 每队的算法 id："nn" / "rule"

    bool teamUsesNN(int team) const;
    void setTeamBrain(int team, const std::string& id);
    void cycleTeamBrain(int team, int delta);

    // -----------------------------------------------------------------------
    //  行表
    //
    //  ★ 第 0、1 行是"几队 / 每队几条"（改它们会重建 roles/teams 表），
    //    之后的每一行对应一支队伍的算法。所以行数是【动态的】。
    // -----------------------------------------------------------------------
    struct Row {
        std::wstring label;
        std::function<std::wstring()> get;
        std::function<void(int delta)> adjust;
        bool readOnly = false;      // 只读行：不上高亮、左右键不响应
    };
    std::vector<Row> m_rows;

    void buildRows();

    sf::Font m_font;
    int      m_row = 0;
};
