#pragma once

// ============================================================================
//  ModeSetupScene.h — 某个模式的【这一局设置】（小设置）
//
//  ★ 它和「主菜单 → 设置」（大设置）的分工（负责人 2026-09-27 明确）
//
//      大设置：改【通用世界规则】+【三个模式的默认值】→ 存盘 → 影响以后每一局
//      小设置（本页）：只改【这一局】→ 不存盘 → 离开这一局就没了
//
//  ★ 所以本页【只】放这一局的配置，不放世界规则
//
//      队数 / 每队几条 / 每队算法 / 每队权重
//
//    速度、饥饿时间、碰撞宽容那些属于"世界规则"，只能去大设置改。
//    ⚠️ 把它们放进本页会直接破坏上面那条分工 —— 那正是负责人要避免的。
//
//  ★ 它改的是【第三层配置】
//    `Game::getMatchConfig().modes[该模式]` —— 也就是"这一局的配置"。
//    ⚠️ 不是 `Game::getConfig().modes[...]`（那是模式默认值）。
//       改错了对象的现象是"这次改的东西下次开局还在"，与"只影响这一局"矛盾。
//
//  ★ 前提：进本页之前必须先 `Game::beginMatch(mode)`
//    因为第三层配置是那一步才被初始化的。`ModeSelectScene` 打开本页时
//    会先调它（见那里的说明），所以本页可以假定 ③ 已经就绪。
//
//  操作：上下键 选择   左右键 调整   R 恢复该模式默认值
//        Enter 保存并开始    ESC 返回（放弃这一局的改动）
// ============================================================================

#include "../core/Scene.h"
#include "../core/Game.h"
#include "../core/GameModes.h"

#include <functional>
#include <string>
#include <vector>

class ModeSetupScene : public Scene {
public:
    ModeSetupScene(Game* game, game::GameMode mode);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    // 一行：名字 + 当前值 + 左右键怎么调
    //
    // ★ 沿用 `TrainingTeamScene` / `TrainingSetupScene` 已经验证过的行表模式
    //   （get/adjust 都是 std::function，所以 lambda 能就地写在 buildRows 里）。
    struct Row {
        std::wstring label;
        std::function<std::wstring()> get;
        std::function<void(int delta)> adjust;
        bool readOnly = false;   // 只读行：不上高亮、左右键不响应
    };
    std::vector<Row> m_rows;

    void buildRows();

    // 本页操作的"这一局配置"（= 第三层）
    //
    // ★ 收成一个函数，避免调用方把它和全局配置搞混（那是这个页面最容易犯的错）
    GameConfig::ModeConfig& mc();
    const GameConfig::ModeConfig& mc() const;

    // 可选权重文件列表（"自动查找" + saves/ 里的存档）
    std::vector<std::string> m_weightFiles;

    sf::Font m_font;
    int      m_row = 0;
    game::GameMode m_mode;
};
