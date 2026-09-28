#include "SettingsScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "../brain/BrainRegistry.h"
#include "../rl/Checkpoint.h"       // ★ ckpt::list() 列出 saves/ 里的存档
#include "../utils/Constants.h"
#include "../utils/Text.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

const sf::Color kSelect = sf::Color(78, 205, 196);
const sf::Color kNormal = sf::Color(190, 190, 200);
const sf::Color kHint   = sf::Color(115, 115, 128);
const sf::Color kPageOn = sf::Color(255, 210, 90);

// 可选算法 id / 显示名（从注册表读，新增算法不改本页）
//
// ★ 排除内部 id 的理由和 ModeSetupScene 一致：
//   "nn-mate"/"nn-enemy" 是 Arena::reset() 按队自动分配的，
//   让玩家手选等于绕过那条规则。而 HumanBrain 不是"给 AI 选的算法"。
std::vector<std::string> g_brainIds;
std::vector<std::string> g_brainNames;

void ensureBrainList() {
    if (!g_brainIds.empty()) return;
    for (const auto& info : arena::BrainRegistry::instance().list()) {
        if (info.id == "nn-mate" || info.id == "nn-enemy") continue;
        if (info.id == "human") continue;
        g_brainIds.push_back(info.id);
        g_brainNames.push_back(info.displayName);
    }
    if (g_brainIds.empty()) {
        g_brainIds.push_back("rule");
        g_brainNames.push_back("规则策略");
    }
}

// 页标题
std::wstring pageTitle(int page) {
    if (page == 0) return L"通用世界规则（三个模式共用）";
    return std::wstring(L"模式默认值：") + game::modeName(static_cast<game::GameMode>(page - 1));
}

}  // namespace

// ===========================================================================
SettingsScene::SettingsScene(Game* game)
    : Scene(game)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "SettingsScene");
    }

    ensureBrainList();
    buildRows();
}

// ===========================================================================
//  行表
//
//  ★ 一处细节：所有 get/adjust 都直接读写 `m_game->getConfig()` 的【真实字段】，
//    不经过副本。
//
//  ★ 为什么不再用"临时副本 + 保存时拷回"
//    旧版用副本，于是【退出时不保存 = 改动被静默丢弃】——
//    玩家改了参数、按 ESC 返回、进游戏发现没变化，而界面上没有任何提示。
//    这是负责人明确提过的坑。
//    现在的做法：编辑即生效（改的是真配置），ESC 返回时自动写盘。
//    "放弃改动"这个能力由 R（恢复默认）提供 —— 语义比"忘了按保存就白改"清楚。
// ===========================================================================
void SettingsScene::buildRows() {
    m_rows.clear();

    if (m_page == 0) {
        buildWorldRows(m_rows);
    } else {
        buildModeRows(m_rows, m_page - 1);
    }

    if (m_selected >= static_cast<int>(m_rows.size())) {
        m_selected = static_cast<int>(m_rows.size()) - 1;
    }
    if (m_selected < 0) m_selected = 0;
}

// ---------------------------------------------------------------------------
//  第 0 页：通用世界规则
// ---------------------------------------------------------------------------
void SettingsScene::buildWorldRows(std::vector<Row>& out) {
    GameConfig& cfg = m_game->getConfig();

    // 小工具：把 float 字段做成一行
    auto fRow = [&](const wchar_t* label,
                    float GameConfig::*field,
                    float lo, float hi, float step, int prec) {
        out.push_back({
            label,
            [this, field, prec] {
                std::wostringstream o;
                o << std::fixed << std::setprecision(prec) << (m_game->getConfig().*field);
                return o.str();
            },
            [this, field, lo, hi, step](int d) {
                GameConfig& c = m_game->getConfig();
                float v = (c.*field) + step * static_cast<float>(d);
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                c.*field = v;
            },
            nullptr,
            false
        });
    };
    auto iRow = [&](const wchar_t* label,
                    int GameConfig::*field,
                    int lo, int hi, int step) {
        out.push_back({
            label,
            [this, field] { return std::to_wstring(m_game->getConfig().*field); },
            [this, field, lo, hi, step](int d) {
                GameConfig& c = m_game->getConfig();
                int v = (c.*field) + step * d;
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                c.*field = v;
            },
            nullptr,
            false
        });
    };
    auto bRow = [&](const wchar_t* label, bool GameConfig::*field) {
        out.push_back({
            label,
            [this, field] { return (m_game->getConfig().*field) ? std::wstring(L"开")
                                                               : std::wstring(L"关"); },
            [this, field](int) { GameConfig& c = m_game->getConfig(); c.*field = !(c.*field); },
            nullptr,
            false
        });
    };

    // ---- 蛇 ----
    fRow(L"蛇速 (像素/秒)",        &GameConfig::snakeSpeed,      50.0f, 1000.0f, 10.0f, 0);
    fRow(L"加速速度",              &GameConfig::snakeBoostSpeed, 50.0f, 1500.0f, 10.0f, 0);
    fRow(L"转向速率 (弧度/秒)",    &GameConfig::snakeTurnRate,    1.0f,   20.0f,  0.5f, 1);
    iRow(L"初始体节数",            &GameConfig::snakeInitSegs,       3,     50,     1);
    fRow(L"体节间距 (像素)",       &GameConfig::segmentSpacing,   5.0f,   50.0f,  1.0f, 0);

    // ---- 饥饿与碰撞 ----
    fRow(L"饥饿时间 (秒)",         &GameConfig::starvationTime,   3.0f,  300.0f,  5.0f, 0);
    fRow(L"碰撞宽容 (像素)",       &GameConfig::collisionForgive, 0.0f,   20.0f,  0.5f, 1);
    iRow(L"友军可见数",            &GameConfig::allyVisibleCount,    0,      8,     1);

    // ---- 食物与尸体 ----
    iRow(L"吃几个食物长一节",      &GameConfig::foodSegments,        1,     50,     1);
    fRow(L"食物寿命 (秒)",         &GameConfig::foodLifetime,     1.0f,  120.0f,  5.0f, 0);
    iRow(L"尸体食物分值",          &GameConfig::corpseScore,         1,    100,     1);
    iRow(L"吃几个尸体长一节",      &GameConfig::corpseSegments,      1,     10,     1);
    fRow(L"尸体存在时间 (秒)",     &GameConfig::corpseLifetime,   1.0f,  300.0f,  5.0f, 0);

    // ---- 对局 ----
    bRow(L"友军伤害（开了会自相残杀）", &GameConfig::friendlyFire);

    // ---- 只读说明：这两个字段为什么不在可调列表里 ----
    //
    // ⚠️ 2026-09-27：`单局限时` 从"可调"改成【只读说明】。
    //
    //  ★ 为什么它不能再调
    //    限时是【按模式不同】的规则：模式 1/3 不限时，而模式 2 **必须** 0
    //    （负责人："无限模式永不结束"）。
    //    现在 `game::applyModeParams()` 每次开局都会写这个字段，
    //    所以在设置里改它**不会生效** —— 那正是本项目最忌讳的"设了没用"。
    //    与其留一个假控件，不如如实说明它由模式决定。
    out.push_back({
        L"单局限时",
        [] { return std::wstring(L"由模式决定：模式 1/3 不限时，模式 2 必须无限"); },
        [](int) {},
        nullptr,
        true
    });
    out.push_back({
        L"地图尺寸",
        [] { return std::wstring(L"3000 x 3000（三个模式统一，不可调）"); },
        [](int) {},
        nullptr,
        true
    });
}

// ---------------------------------------------------------------------------
//  第 1..3 页：某个模式的默认值
// ---------------------------------------------------------------------------
void SettingsScene::buildModeRows(std::vector<Row>& out, int mi) {
    const int mid = std::max(0, std::min(game::MODE_COUNT - 1, mi));
    const auto mode = static_cast<game::GameMode>(mid);

    // -----------------------------------------------------------------------
    //  ★★ 无限模式没有队伍（负责人 2026-09-27 两次指出）
    //
    //      "我告诉过你，无限模式没有队伍数，但是设置里还是有。"
    //
    //  ★ 实现手段 ≠ 界面说法
    //    内部：每条蛇自己占一个"队"槽位（`snakesPerTeam == 1`），
    //          因为"会不会互相撞死"由队号决定 —— 这是实现手段。
    //    界面：一个"队"字都不能出现。所以本模式下：
    //          · "默认队数"    → 改叫"默认初始蛇数"
    //          · "默认每队几条" → **整行删掉**（恒等于 1，显示了只会让人以为能改）
    //          · "N 队默认算法" → 改叫"N 号蛇的脑子"
    // -----------------------------------------------------------------------
    const bool noTeams = (mode == game::GameMode::Infinite);

    // ---- 队数 / 初始蛇数 ----
    out.push_back({
        noTeams ? L"默认初始蛇数" : L"默认队数",
        [this, mid, noTeams] {
            const int n = m_game->getConfig().modes[mid].teamCount;
            return noTeams ? (std::to_wstring(n) + L" 条")
                           : (std::to_wstring(n) + L" 队");
        },
        [this, mid](int d) {
            GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
            int v = mc.teamCount + d;
            if (v < 1) v = 1;
            if (v > GameConfig::MAX_TEAMS) v = GameConfig::MAX_TEAMS;
            mc.teamCount = v;
            if (mid == static_cast<int>(game::GameMode::Infinite)) {
                // ★ 不变式：无限模式永远是"一蛇一队"
                mc.snakesPerTeam = 1;
            }
            buildRows();     // ★ 队数变了 → 每队那几行要重建
        },
        nullptr,
        false
    });

    // ---- 每队几条（无限模式没有这一行）----
    if (!noTeams) {
        out.push_back({
            L"默认每队几条",
            [this, mid] {
                const GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
                return std::to_wstring(mc.snakesPerTeam) + L" 条   （共 "
                     + std::to_wstring(mc.totalSnakes()) + L" 条）";
            },
            [this, mid](int d) {
                GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
                int v = mc.snakesPerTeam + d;
                if (v < 1) v = 1;
                if (v > 8) v = 8;
                mc.snakesPerTeam = v;
            },
            nullptr,
            false
        });
    } else {
        // 只读说明：把"内部用队号"这件事如实写出来，免得以后又有人加回"队数"
        out.push_back({
            L"组队",
            [] { return std::wstring(L"无。每条蛇互为敌人，可以互相撞死"); },
            [](int) {},
            nullptr,
            true
        });
    }

    // ---- 食物速率 ----
    //
    // ★ 显示"每蛇速率"和"总速率"两个数
    //   负责人曾经误会过这个字段的口径（以为单位是"每圈"）。
    //   把两个数一起显示出来，口径就不需要靠记忆了。
    out.push_back({
        L"默认食物速率",
        [this, mid] {
            const GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
            std::wostringstream o;
            o << std::fixed << std::setprecision(3) << mc.foodPerSnake
              << L" /蛇/秒   → 总计 "
              << std::setprecision(2)
              << (mc.foodPerSnake * static_cast<float>(mc.totalSnakes()))
              << L" 个/秒";
            return o.str();
        },
        [this, mid](int d) {
            GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
            // 步进 0.02/蛇/秒 ≈ 9 条蛇时每步 0.18 个/秒
            float v = mc.foodPerSnake + 0.02f * static_cast<float>(d);
            if (v < 0.0f) v = 0.0f;
            if (v > 20.0f) v = 20.0f;     // ⚠️ 和 sanitizeModes() 的上限保持一致
            mc.foodPerSnake = v;
        },
        nullptr,
        false
    });

    // ---- 食物存量（只有"没有食物圈"的模式用得到）----
    //
    // ★ 为什么只给模式 2/3 显示这两行
    //   模式 1 的食物由圈逐秒产，`initialFoodCount` / `foodMaxCount` 恒为 0
    //   且不参与任何计算。给它显示两个永远为 0 的行，就是本项目最忌讳的
    //   "设了没用" —— 玩家会以为改了有效，其实完全没人读。
    if (noTeams || mode == game::GameMode::Battle) {
        out.push_back({
            L"开局食物数",
            [this, mid] {
                return std::to_wstring(m_game->getConfig().modes[mid].initialFoodCount)
                     + L" 个   （开局在全图随机撒）";
            },
            [this, mid](int d) {
                GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
                int v = mc.initialFoodCount + 25 * d;
                if (v < 0) v = 0;
                if (v > 2000) v = 2000;
                mc.initialFoodCount = v;
            },
            nullptr,
            false
        });

        out.push_back({
            L"食物存量上限",
            [this, mid] {
                const int n = m_game->getConfig().modes[mid].foodMaxCount;
                if (n <= 0) return std::wstring(L"不限制");
                return std::to_wstring(n) + L" 个   （到量就暂停生成）";
            },
            [this, mid](int d) {
                GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
                int v = mc.foodMaxCount + 25 * d;
                if (v < 0) v = 0;
                if (v > 5000) v = 5000;
                mc.foodMaxCount = v;
            },
            nullptr,
            false
        });
    }

    // ---- 每条蛇（模式 2）/ 每队（模式 1、3）的算法 + 权重 ----
    //
    // ★ 权重和算法**合并成一行**显示。
    //   原来"算法"和"权重"各占一行，8 组就是 16 行 —— 加上上面几行会超出
    //   屏幕（每行约 28px，720p 放不下）。合并之后 8 组只有 8 行。
    for (int t = 0; t < m_game->getConfig().modes[mid].effectiveTeamCount(); ++t) {
        const int team = t;

        out.push_back({
            noTeams ? (std::to_wstring(team + 1) + L" 号蛇的脑子")
                    : (std::to_wstring(team + 1) + L" 队默认算法与权重"),
            [this, mid, team] {
                const GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
                const std::string id = mc.teamBrains[team];
                std::wstring w = textutil::utf8ToWide(id) + L"  (未知)";
                for (std::size_t i = 0; i < g_brainIds.size(); ++i) {
                    if (g_brainIds[i] == id) {
                        w = textutil::utf8ToWide(g_brainNames[i]);
                        break;
                    }
                }
                // 只有神经网络才有权重可挑（规则 AI 没有权重文件）
                if (id == "nn") {
                    const std::string& f = mc.teamWeights[team];
                    w += L"   ";
                    w += f.empty() ? std::wstring(L"[权重：随机/自动查找]")
                                   : textutil::utf8ToWide(f);
                }
                return w;
            },
            [this, mid, team](int d) {
                GameConfig::ModeConfig& mc = m_game->getConfig().modes[mid];
                std::string& id = mc.teamBrains[team];

                // 左右键按"算法 → 该算法下的各份权重"排成一条链，来回循环。
                if (id != "nn") {
                    const int n = static_cast<int>(g_brainIds.size());
                    int idx = 0;
                    for (int i = 0; i < n; ++i) {
                        if (g_brainIds[static_cast<std::size_t>(i)] == id) { idx = i; break; }
                    }
                    idx = ((idx + d) % n + n) % n;
                    id = g_brainIds[static_cast<std::size_t>(idx)];
                    if (id != "nn") mc.teamWeights[team].clear();
                    buildRows();     // 权重那一行要不要显示，取决于本队是不是 nn
                    return;
                }

                const auto files = ckpt::list();
                const int n = 1 + static_cast<int>(files.size());
                int cur = 0;
                for (std::size_t i = 0; i < files.size(); ++i) {
                    if (files[i] == mc.teamWeights[team]) { cur = static_cast<int>(i) + 1; break; }
                }
                cur = ((cur + d) % n + n) % n;
                mc.teamWeights[team] = (cur == 0)
                    ? std::string()
                    : files[static_cast<std::size_t>(cur - 1)];
            },
            nullptr,
            false
        });
    }

    // 参数说明（只读）
    out.push_back({
        L"本模式说明",
        [mid] { return std::wstring(game::modeParamNote(static_cast<game::GameMode>(mid))); },
        [](int) {},
        nullptr,
        true
    });
}

// ===========================================================================
//  恢复初始默认设置（全部）
//
//  ★ 这是负责人指定的"大设置 R = 全部回初始值"
//    ⚠️ 它包括【三个模式的默认值】—— 和小设置的 R（只回当前模式默认）不同。
//
//  ★ 实现：直接调 game::resetWorldRules + 逐个 applyModeParams
//    **不重写一份默认值** —— 默认值的唯一来源是 GameModes.cpp。
//    这里如果自己写一份，两边迟早不一致（本项目栽过多次）。
// ===========================================================================
void SettingsScene::resetAllToDefaults() {
    GameConfig& cfg = m_game->getConfig();

    game::resetWorldRules(cfg);
    for (int mi = 0; mi < game::MODE_COUNT; ++mi) {
        game::applyModeParams(cfg, static_cast<game::GameMode>(mi));
    }

    // ★ 队伍算法/权重也是"默认设置"的一部分，也要回基线
    //
    //  ⚠️ 这份基线必须和 `GameModes.cpp::applyModeParams()` 里设的
    //     teamCount / snakesPerTeam **保持一致** —— 两处都改才算对。
    //     （那边管"几队几条"，这边管"每队用什么算法"。）
    //
    //  ⚠️ "nn-mate" 不在玩家可选列表里（它是 Arena 按队自动分配的），
    //     所以模式 0 里同队第 2、3 条写 "nn"，由 Arena 自己换成 nn-mate。
    auto setBrains = [&](int mi, std::initializer_list<const char*> ids) {
        GameConfig::ModeConfig& mc = cfg.modes[mi];
        int i = 0;
        for (const char* s : ids) {
            if (i >= GameConfig::MAX_TEAMS) break;
            mc.teamBrains[i++] = s;
        }
        for (; i < GameConfig::MAX_TEAMS; ++i) mc.teamBrains[i] = "rule";
        for (int t = 0; t < GameConfig::MAX_TEAMS; ++t) mc.teamWeights[t].clear();
    };

    // 模式 0「随机食物圈」= 3 队 × 3 条：
    //   队 0 神经网络（学习/自对弈用的那队）、队 1 神经网络、队 2 规则 AI
    setBrains(0, { "nn", "nn", "rule" });

    // 模式 1「无限模式」= 5 队 × 1 条 = 5 条蛇，**没有组队**
    //   每条蛇自己一队 → "每队的算法"就是"每条蛇的算法"
    setBrains(1, { "nn", "nn", "nn", "nn", "nn" });

    // 模式 2「大混战」= 5 队 × 3 条 = 15 条，全部神经网络
    setBrains(2, { "nn", "nn", "nn", "nn", "nn" });

    cfg.sanitize();
    buildRows();
    doSave();          // 恢复也要落盘，否则下次启动又变回来
}

// ===========================================================================
//  保存
// ===========================================================================
void SettingsScene::doSave() {
    if (m_saveNotifyTimer > 0.0f) m_saveFlash = true;
    m_game->saveConfig();
    m_saveNotifyTimer = 2.5f;
}

// ===========================================================================
//  事件
// ===========================================================================
void SettingsScene::handleEvent(const sf::Event& event) {
    // ---- 数字输入模式下：优先吃掉所有按键 ----
    if (m_inputMode) {
        if (event.type == sf::Event::TextEntered) {
            const wchar_t ch = static_cast<wchar_t>(event.text.unicode);
            if ((ch >= L'0' && ch <= L'9') || ch == L'.' || ch == L'-') {
                if (m_inputBuffer.size() < 20) m_inputBuffer += static_cast<char>(ch);
            } else if (ch == 8) {   // Backspace
                if (!m_inputBuffer.empty()) m_inputBuffer.pop_back();
            }
            return;
        }
        if (event.type == sf::Event::KeyPressed) {
            if (event.key.code == sf::Keyboard::Enter) {
                applyInput();
                m_inputMode = false;
            } else if (event.key.code == sf::Keyboard::Escape) {
                m_inputMode = false;   // 取消，不改
            }
        }
        return;
    }

    // ---- Ctrl+S：任意位置保存 ----
    if (event.type == sf::Event::KeyPressed &&
        event.key.code == sf::Keyboard::S && event.key.control) {
        doSave();
        return;
    }

    if (event.type != sf::Event::KeyPressed) return;
    if (m_rows.empty()) return;

    const int n = static_cast<int>(m_rows.size());

    switch (event.key.code) {
        case sf::Keyboard::Up:
            m_selected = (m_selected - 1 + n) % n;
            break;
        case sf::Keyboard::Down:
            m_selected = (m_selected + 1) % n;
            break;

        case sf::Keyboard::Left:
        case sf::Keyboard::Right: {
            Row& r = m_rows[static_cast<std::size_t>(m_selected)];
            if (!r.readOnly && r.adjust) {
                r.adjust((event.key.code == sf::Keyboard::Right) ? 1 : -1);
            }
            break;
        }

        // ---- 切换模式页 ----
        case sf::Keyboard::Tab:
        case sf::Keyboard::RBracket:
            m_page = (m_page + 1) % (game::MODE_COUNT + 1);
            m_selected = 0;
            buildRows();
            break;
        case sf::Keyboard::LBracket:
            m_page = (m_page - 1 + game::MODE_COUNT + 1) % (game::MODE_COUNT + 1);
            m_selected = 0;
            buildRows();
            break;

        // ---- 数字输入 ----
        case sf::Keyboard::Enter: {
            Row& r = m_rows[static_cast<std::size_t>(m_selected)];
            if (!r.readOnly && r.setText) {
                m_inputBuffer = "";
                m_inputMode = true;
            }
            break;
        }

        // ---- 恢复初始默认设置（全部）----
        case sf::Keyboard::R:
            resetAllToDefaults();
            break;

        // ---- 保存 ----
        case sf::Keyboard::S:
            doSave();
            break;

        // ---- 返回：自动保存 ----
        //
        // ★ 这里就是那个"改了参数游戏里没变化"的坑的修法：
        //   编辑即时生效（改的是真配置），返回时再写一次盘。
        //   所以不存在"忘了按保存 = 白改"这件事。
        case sf::Keyboard::Escape:
            doSave();
            m_game->popScene();
            break;

        default: break;
    }
}

void SettingsScene::applyInput() {
    if (m_inputBuffer.empty()) return;
    if (m_selected < 0 || m_selected >= static_cast<int>(m_rows.size())) return;
    Row& r = m_rows[static_cast<std::size_t>(m_selected)];
    if (r.setText) r.setText(m_inputBuffer);
    m_inputBuffer.clear();
}

void SettingsScene::fixedUpdate(float) {}

void SettingsScene::update(float dt) {
    if (m_saveNotifyTimer > 0.0f) {
        m_saveNotifyTimer -= dt;
        if (m_saveFlash && m_saveNotifyTimer < 2.3f) m_saveFlash = false;
    }
}

// ===========================================================================
//  渲染
// ===========================================================================
void SettingsScene::render(sf::RenderWindow& window) {
    const float W = static_cast<float>(Constants::WINDOW_WIDTH);

    // ---- 标题 ----
    {
        sf::Text t(L"游戏设置", m_font, 32);
        t.setFillColor(kSelect);
        t.setStyle(sf::Text::Bold);
        sf::FloatRect b = t.getLocalBounds();
        t.setOrigin(b.width / 2.0f, 0.0f);
        t.setPosition(W / 2.0f, 18.0f);
        window.draw(t);
    }

    // ---- 页签：通用 + 三个模式 ----
    {
        const float y = 62.0f;
        const float tabW = 190.0f;
        const float totalW = tabW * static_cast<float>(game::MODE_COUNT + 1);
        float x = (W - totalW) / 2.0f;

        for (int p = 0; p <= game::MODE_COUNT; ++p) {
            const bool on = (p == m_page);
            sf::RectangleShape box(sf::Vector2f(tabW - 8.0f, 30.0f));
            box.setPosition(x, y);
            box.setFillColor(on ? sf::Color(78, 205, 196, 55) : sf::Color(40, 40, 52));
            box.setOutlineColor(on ? kSelect : sf::Color(80, 80, 95));
            box.setOutlineThickness(1.5f);
            window.draw(box);

            std::wstring label = (p == 0) ? std::wstring(L"通用规则")
                                          : game::modeName(static_cast<game::GameMode>(p - 1));
            sf::Text t(label, m_font, 16);
            t.setFillColor(on ? kPageOn : sf::Color(150, 150, 160));
            sf::FloatRect b = t.getLocalBounds();
            t.setOrigin(b.width / 2.0f, 0.0f);
            t.setPosition(x + (tabW - 8.0f) / 2.0f, y + 6.0f);
            window.draw(t);

            x += tabW;
        }

        sf::Text hint(L"[ ] 或 TAB 切换页", m_font, 13);
        hint.setFillColor(kHint);
        hint.setPosition(24.0f, y + 7.0f);
        window.draw(hint);

        sf::Text pg(pageTitle(m_page), m_font, 18);
        pg.setFillColor(sf::Color(230, 230, 240));
        sf::FloatRect b = pg.getLocalBounds();
        pg.setOrigin(b.width / 2.0f, 0.0f);
        pg.setPosition(W / 2.0f, y + 40.0f);
        window.draw(pg);
    }

    // ---- 行表 ----
    {
        const float left  = 150.0f;
        const float right = W - 150.0f;
        float y = 150.0f;
        const float lineH = 28.0f;

        // 行数多时自动缩一点间距，保证不越出窗口
        const int nrow = static_cast<int>(m_rows.size());
        float lh = lineH;
        const float avail = static_cast<float>(Constants::WINDOW_HEIGHT) - 150.0f - 70.0f;
        if (static_cast<float>(nrow) * lh > avail) lh = avail / static_cast<float>(nrow);

        for (int i = 0; i < nrow; ++i) {
            const bool sel = (i == m_selected);
            const Row& r = m_rows[static_cast<std::size_t>(i)];

            if (sel && !r.readOnly) {
                sf::RectangleShape bar(sf::Vector2f(right - left + 28.0f, lh - 3.0f));
                bar.setPosition(left - 14.0f, y - 2.0f);
                bar.setFillColor(sf::Color(78, 205, 196, 40));
                bar.setOutlineColor(sf::Color(78, 205, 196, 120));
                bar.setOutlineThickness(1.0f);
                window.draw(bar);
            }

            const int fsize = (lh < 24.0f) ? 14 : 16;

            sf::Text label(r.label, m_font, fsize);
            label.setFillColor(r.readOnly ? sf::Color(125, 125, 138)
                                : (sel ? kSelect : kNormal));
            label.setPosition(left, y);
            window.draw(label);

            // 输入模式下，正在编辑的那一行用输入缓冲显示
            std::wstring valText;
            if (m_inputMode && sel) {
                valText = textutil::utf8ToWide(m_inputBuffer) + L"_";
            } else {
                valText = r.get ? r.get() : std::wstring();
            }
            sf::Text val(valText, m_font, fsize);
            val.setFillColor(r.readOnly ? sf::Color(155, 155, 168)
                                        : sf::Color(240, 240, 245));
            val.setPosition(left + 330.0f, y);
            window.draw(val);

            y += lh;
        }
    }

    // ---- 底部提示 ----
    {
        std::wstring hint =
            L"上下键 选择    左右键 调整    Enter 输入数字    "
            L"S/Ctrl+S 保存    R 恢复初始默认设置    ESC 返回(自动保存)";
        sf::Text t(hint, m_font, 14);
        t.setFillColor(kHint);
        sf::FloatRect b = t.getLocalBounds();
        t.setOrigin(b.width / 2.0f, 0.0f);
        t.setPosition(W / 2.0f, static_cast<float>(Constants::WINDOW_HEIGHT) - 34.0f);
        window.draw(t);
    }

    // ---- 保存提示 ----
    if (m_saveNotifyTimer > 0.0f) {
        sf::Text t(L"已保存", m_font, 18);
        t.setFillColor(m_saveFlash ? sf::Color(255, 255, 180) : sf::Color(150, 230, 150));
        t.setPosition(W - 120.0f, 24.0f);
        window.draw(t);
    }
}
