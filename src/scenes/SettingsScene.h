#pragma once

// ============================================================================
//  SettingsScene.h — 主菜单的【游戏设置】（大设置）
//
//  ★★ 2026-09-27 整页重写。它和「模式选择页 → 设置」的分工是：
//
//      大设置（本页）：改【通用世界规则】+【三个模式的默认值】→ 存盘 → 影响以后每一局
//      小设置（ModeSetupScene）：只改【这一局】→ 不存盘
//
//  ★ 本页有哪些内容（负责人指定）
//
//      ① 通用世界规则 —— 三个模式共用
//           蛇速 / 加速 / 转向 / 初始体节 / 体节间距 / 饥饿时间 /
//           碰撞宽容 / 尸体分值 / 友军伤害 / 友军可见数
//         ⚠️ 地图尺寸【已撤掉】：三个模式统一 3000×3000，让玩家改它
//            会造出"某个模式地图不一样"的不一致。
//
//      ② 三个模式各自的默认值
//           队数 / 每队几条 / 食物速率（每蛇每秒）
//         每个模式一"页"，用上下键切页（TAB 跳到下一页也行）。
//         权重与算法属于"每队的临时选择"，只在【模式选择页 → 设置】里改 ——
//         但如果玩家在大设置里改，那就是"下次开局默认用什么"。
//         所以本页也给【每队算法】和【每队权重】——负责人要求"大设置能改
//         三个游戏的默认设置"，而"默认哪队用什么算法"正是默认设置的一部分。
//
//  ★ 操作
//      上下键 选择    左右键 调整    Enter 输入数字
//      TAB / [ ] 切换模式页
//      S / Ctrl+S 保存     R 恢复初始默认设置（全部）
//      ESC 返回（自动保存）
//
//  ★ 「R 恢复初始默认设置」= 全部字段回【代码里写死的基线】
//    包括三个模式的默认值。这和小设置的 R（只回该模式默认值）不同 ——
//    负责人明确区分了这两者。
//
//  ★ 本页改的是【第二层配置】（`Game::getConfig()`）
//    ⚠️ 不是这一局的配置。开新局时它会被拷进第三层。
// ============================================================================

#include "../core/Scene.h"
#include "../core/Game.h"
#include "../core/GameModes.h"

#include <functional>
#include <string>
#include <vector>

class SettingsScene : public Scene {
public:
    explicit SettingsScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    // -----------------------------------------------------------------------
    //  一行：名字 + 显示的当前值 + 怎么调
    //
    //  ★ 为什么用 get/adjust 两个 std::function 而不是裸指针
    //    以前的版本存的是 `float*`，所以只能表达"结构体的某个标量字段"。
    //    现在要调的既有标量（蛇速），也有"模式数组里的字段"（第 2 页的队数）、
    //    还有字符串（每队算法/权重）—— 裸指针表达不了后面两类。
    // -----------------------------------------------------------------------
    struct Row {
        std::wstring label;
        std::function<std::wstring()> get;
        std::function<void(int delta)> adjust;
        std::function<void(const std::string&)> setText;   // 非空 = 可输入
        bool readOnly = false;
    };

    // 重建行表（切模式页 / 改队数之后要重建）
    void buildRows();

    // 「恢复初始默认设置」：把所有字段回代码基线，然后重建行表
    void resetAllToDefaults();

    // 保存（写 m_game->getConfig() 到磁盘）
    void doSave();
    void applyInput();          // Enter 输入模式里确认

    // 行表的变化部分 = 当前模式页的那几行
    void buildWorldRows(std::vector<Row>& out);
    void buildModeRows(std::vector<Row>& out, int modeIndex);

    sf::Font m_font;

    std::vector<Row> m_rows;
    int  m_selected = 0;

    // ---- 模式页 ----
    //
    // ★ 为什么分页而不是一页塞下
    //   三个模式 × (队数+每条+食物+每队算法+每队权重) ≈ 30 行，
    //   一页显示不下；而缩小字号会让"每队权重"那种长文件名挤成一团。
    int  m_page = 0;             // 0 = 通用世界规则，1..3 = 模式 0..2

    // ---- 数字输入模式 ----
    bool        m_inputMode = false;
    std::string m_inputBuffer;

    // ---- 保存提示 ----
    float m_saveNotifyTimer = 0.0f;
    bool  m_saveFlash = false;
};
