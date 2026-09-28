#include "ModeSetupScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "GameScene.h"
#include "ModeSelectScene.h"
#include "../brain/BrainRegistry.h"
#include "../rl/Checkpoint.h"      // ★ ckpt::list() 列出 saves/ 里的存档
#include "../utils/Constants.h"
#include "../utils/Text.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

namespace {

const sf::Color kSelect = sf::Color(78, 205, 196);
const sf::Color kNormal = sf::Color(185, 185, 185);
const sf::Color kHint   = sf::Color(120, 120, 130);

// 每队算法可选项（第一个是"自动查找权重"的哨兵，不参与算法循环）
//
// ★ 三个选项的含义
//     rule      固定规则 AI —— 行为确定，不会学
//     nn        神经网络（本队用权重文件里那份）
//     nn-enemy  神经网络（用训练器写的冻结快照）—— 留给自对弈场景
//
// ⚠️ 这些 id 必须和 `BrainRegistry` 里注册的完全一致，否则
//    `BrainRegistry::create()` 返回 nullptr，Arena 会**静默退回规则 AI** ——
//    现象是"选了神经网络但行为像规则 AI"，很难查。
//    所以这里不写字面量，而是从注册表读（见 collectBrains）。
std::vector<std::string> g_brainIds;
std::vector<std::string> g_brainNames;

}  // namespace

// ===========================================================================
//  取"这一局的配置"（第三层）
//
//  ★ 这是本页唯一允许改的配置对象。
//    ⚠️ 不要在这里改成 `m_game->getConfig()` —— 那是模式【默认值】，
//       改了它意味着"这次改的东西下次开局还在"，与"只影响这一局"矛盾。
// ===========================================================================
GameConfig::ModeConfig& ModeSetupScene::mc() {
    return m_game->getMatchConfig().modes[static_cast<int>(m_mode)];
}
const GameConfig::ModeConfig& ModeSetupScene::mc() const {
    return m_game->getMatchConfig().modes[static_cast<int>(m_mode)];
}

ModeSetupScene::ModeSetupScene(Game* game, game::GameMode mode)
    : Scene(game), m_mode(mode)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "ModeSetupScene");
    }

    // ---- 从注册表读可用算法（新增算法不用改本页）----
    if (g_brainIds.empty()) {
        for (const auto& info : arena::BrainRegistry::instance().list()) {
            // "nn-mate"/"nn-enemy" 是【按队自动分配】的内部 id（见 Arena::reset），
            // 不该让玩家手选 —— 手选等于绕过那条规则，行为会难以预料。
            if (info.id == "nn-mate" || info.id == "nn-enemy") continue;
            if (info.id == "human") continue;      // 玩家不是"给 AI 选的算法"
            g_brainIds.push_back(info.id);
            g_brainNames.push_back(info.displayName);
        }
        if (g_brainIds.empty()) {                  // 兜底：注册表没初始化也要能用
            g_brainIds.push_back("rule");
            g_brainNames.push_back("规则策略");
        }
    }

    // ---- 可选权重文件 ----
    m_weightFiles = ckpt::list();

    buildRows();
}

// ===========================================================================
//  行表
// ===========================================================================
void ModeSetupScene::buildRows() {
    m_rows.clear();
    GameConfig::ModeConfig& c = mc();

    // -----------------------------------------------------------------------
    //  ★★ "无限模式没有队伍"（负责人 2026-09-27 两次指出）
    //
    //  ★ 负责人原话
    //      "我告诉过你，无限模式没有队伍数，但是设置里还是有。"
    //
    //  ★ 内部实现 vs 界面上该说什么（这是两件事，别混）
    //    内部：每条蛇自己占一个"队"槽位（`snakesPerTeam == 1`）——
    //          因为"会不会互相撞死"是由队号决定的，这是实现手段。
    //    界面：**一个"队"字都不能出现**。出现"分成几队"就是在说
    //          "这个模式有队伍"，而它没有 —— 所有蛇互为敌人。
    //    所以本模式下这两行改成"初始蛇数"，并且没有"每队几条蛇"
    //    （它恒等于 1，显示出来只会让人以为可以改）。
    // -----------------------------------------------------------------------
    const bool noTeams = (m_mode == game::GameMode::Infinite);

    if (noTeams) {
        // ---- 无限模式：初始蛇数（借用 teamCount 槽位，语义是"蛇数"）----
        m_rows.push_back({
            L"初始蛇数",
            [this] { return std::to_wstring(mc().teamCount) + L" 条"; },
            [this](int d) {
                GameConfig::ModeConfig& x = mc();
                int v = x.teamCount + d;
                if (v < 1) v = 1;
                if (v > GameConfig::MAX_TEAMS) v = GameConfig::MAX_TEAMS;
                if (v == x.teamCount) return;
                x.teamCount = v;
                x.snakesPerTeam = 1;      // ★ 不变式：无限模式恒为"一蛇一队"
                buildRows();              // 蛇数变了 → 每蛇那几行要重建
            },
            false
        });

        // ★ 说明"队号只是实现手段"，免得以后又有人在界面上加回"队"
        m_rows.push_back({
            L"组队",
            [] { return std::wstring(L"无。所有蛇互为敌人，可以互相撞死"); },
            [](int) {},
            true
        });
    } else {
        // ---- 第 0 行：几队 ----
        m_rows.push_back({
            L"分成几队",
            [this] { return std::to_wstring(mc().teamCount) + L" 队"; },
            [this](int d) {
                GameConfig::ModeConfig& x = mc();
                const int lo = 1;
                const int hi = GameConfig::MAX_TEAMS;
                int v = x.teamCount + d;
                if (v < lo) v = lo;
                if (v > hi) v = hi;
                if (v == x.teamCount) return;
                x.teamCount = v;
                buildRows();          // ★ 队数变了 → 每队那几行要重建
            },
            false
        });

        // ---- 第 1 行：每队几条蛇 ----
        m_rows.push_back({
            L"每队几条蛇",
            [this] {
                return std::to_wstring(mc().snakesPerTeam) + L" 条"
                     + L"   （共 " + std::to_wstring(mc().totalSnakes()) + L" 条）";
            },
            [this](int d) {
                GameConfig::ModeConfig& x = mc();
                int v = x.snakesPerTeam + d;
                if (v < 1) v = 1;
                if (v > 8) v = 8;
                x.snakesPerTeam = v;
            },
            false
        });

        // ---- 第 2 行：本模式的队伍一览（只读，帮玩家看清"N 队"到底意味着什么）----
        m_rows.push_back({
            L"本局队伍规模",
            [this] {
                std::wostringstream o;
                o << mc().effectiveTeamCount() << L" 队 x "
                  << mc().effectiveSnakesPerTeam() << L" 条 = "
                  << mc().totalSnakes() << L" 条";
                return o.str();
            },
            [](int) {},
            true
        });
    }

    // ---- 之后：每队（无限模式下是"每条蛇"）的算法 + 权重 ----
    //
    // ★ 权重和算法**合并成一行**显示。
    //   原来"算法"和"权重"各占一行，8 组就是 16 行 —— 加上上面几行会超出
    //   屏幕（每行 34px，720p 只能放 17 行）。
    //   合并之后 8 组只有 8 行，怎么调都不会溢出，而且"这条蛇用什么脑子"
    //   本来就该在一行里看全。
    for (int t = 0; t < c.effectiveTeamCount(); ++t) {
        const int team = t;

        m_rows.push_back({
            noTeams ? (std::to_wstring(team + 1) + L" 号蛇的脑子")
                    : (std::to_wstring(team + 1) + L" 队算法与权重"),
            [this, team, noTeams] {
                const std::string id = mc().teamBrains[team];
                std::wstring w = L"(未知算法)";
                for (std::size_t i = 0; i < g_brainIds.size(); ++i) {
                    if (g_brainIds[i] == id) {
                        // ⚠️ 用 utf8ToWide 而不是 utf8 ——
                        //    `utf8()` 返回的是 sf::String，不能拼进 std::wstring。
                        w = textutil::utf8ToWide(g_brainNames[i]);
                        break;
                    }
                }
                // 只有神经网络才有权重可挑；规则 AI 没有权重文件
                // （给它显示一个权重名会让人以为能改，实际改了没用）
                if (id == "nn") {
                    const std::string& f = mc().teamWeights[team];
                    w += L"   ";
                    w += f.empty() ? std::wstring(L"[权重：自动查找]")
                                   : textutil::utf8ToWide(f);
                }
                if (!noTeams && team == game::MODE_PLAYER_TEAM)
                    w += L"   （你在这一队）";
                if (noTeams && team == game::MODE_PLAYER_TEAM)
                    w += L"   （你）";
                return w;
            },
            [this, team](int d) {
                GameConfig::ModeConfig& x = mc();
                std::string& id = x.teamBrains[team];

                // 左右键按"算法 → 该算法下的各份权重"排成一条链，来回循环。
                //  ★ 为什么不做成两行（算法一行、权重一行）
                //    见上面的行数说明 —— 8 组两行会溢出屏幕。
                if (id != "nn") {
                    // 当前不是神经网络 → 左右键换算法
                    const int n = static_cast<int>(g_brainIds.size());
                    int idx = 0;
                    for (int i = 0; i < n; ++i) {
                        if (g_brainIds[static_cast<std::size_t>(i)] == id) { idx = i; break; }
                    }
                    idx = ((idx + d) % n + n) % n;
                    id = g_brainIds[static_cast<std::size_t>(idx)];
                    if (id != "nn") x.teamWeights[team].clear();
                    buildRows();
                    return;
                }

                // 当前是神经网络 → 左右键在 [自动查找, 存档1..N] 里换权重
                const int n = 1 + static_cast<int>(m_weightFiles.size());
                int cur = 0;
                for (std::size_t i = 0; i < m_weightFiles.size(); ++i) {
                    if (m_weightFiles[i] == x.teamWeights[team]) {
                        cur = static_cast<int>(i) + 1; break;
                    }
                }
                cur = ((cur + d) % n + n) % n;
                x.teamWeights[team] = (cur == 0)
                    ? std::string()
                    : m_weightFiles[static_cast<std::size_t>(cur - 1)];
            },
            false
        });
    }
}

// ===========================================================================
//  事件
// ===========================================================================
void ModeSetupScene::handleEvent(const sf::Event& event) {
    if (event.type != sf::Event::KeyPressed) return;
    if (m_rows.empty()) return;

    const int n = static_cast<int>(m_rows.size());

    switch (event.key.code) {
        case sf::Keyboard::Up:
            m_row = (m_row - 1 + n) % n;
            break;
        case sf::Keyboard::Down:
            m_row = (m_row + 1) % n;
            break;

        case sf::Keyboard::Left:
        case sf::Keyboard::Right: {
            Row& r = m_rows[static_cast<std::size_t>(m_row)];
            if (!r.readOnly && r.adjust) {
                r.adjust((event.key.code == sf::Keyboard::Right) ? 1 : -1);
            }
            break;
        }

        case sf::Keyboard::R: {
            // ---------------------------------------------------------------
            //  ★ 恢复【该模式的默认值】（负责人指定：小设置的 R 只回当前模式默认）
            //
            //  ★ 为什么不是"回代码里的初始值"
            //    大设置里玩家改过模式默认值。小设置的 R 应该回到【玩家设定的默认】，
            //    而不是把玩家在大设置里的设置也一并推翻 ——
            //    后者会让人以为"我在大设置里改的东西被吃掉了"。
            //
            //  ★ 做法：从 `m_config.modes[mid]`（第二层：模式默认值）拷一份过来。
            //    顺手把世界规则也拷一份（小设置不改世界规则，但保持一致不会有坏处）。
            // ---------------------------------------------------------------
            const GameConfig& def = m_game->getConfig();
            m_game->getMatchConfig() = def;      // ① + ② 全部回到默认
            buildRows();
            break;
        }

        case sf::Keyboard::Enter:
        case sf::Keyboard::Space: {
            // ★ 保存并开始：这一局的配置已经就地改在 m_matchCfg 里了，
            //   直接进 GameScene 即可（它读的就是这一份）。
            //
            //  ⚠️ 【不调 beginMatch】—— 那会把 m_matchCfg 重新拷一遍，
            //     本页刚做的改动会被覆盖掉。这是本页最容易犯的错。
            m_game->switchScene(std::make_unique<GameScene>(m_game, m_mode));
            break;
        }

        case sf::Keyboard::Escape:
            // 返回模式选择页。
            // ★ 这一局的改动【留在 m_matchCfg 里】—— 再次按 Enter 开始会带着它。
            //   只有下次 beginMatch 才会被重置，符合"只影响这一局"的语义。
            m_game->switchScene(std::make_unique<ModeSelectScene>(m_game));
            break;

        default: break;
    }
}

void ModeSetupScene::fixedUpdate(float) {}
void ModeSetupScene::update(float) {}

// ===========================================================================
//  渲染
// ===========================================================================
void ModeSetupScene::render(sf::RenderWindow& window) {
    const float W = static_cast<float>(Constants::WINDOW_WIDTH);

    // 标题
    {
        sf::Text t(L"这一局的设置", m_font, 34);
        t.setFillColor(kSelect);
        t.setStyle(sf::Text::Bold);
        sf::FloatRect b = t.getLocalBounds();
        t.setOrigin(b.width / 2.0f, 0.0f);
        t.setPosition(W / 2.0f, 26.0f);
        window.draw(t);

        // 模式名（让玩家清楚自己在配哪个模式）
        sf::Text m(game::modeName(m_mode), m_font, 20);
        m.setFillColor(sf::Color(255, 210, 90));
        sf::FloatRect mb = m.getLocalBounds();
        m.setOrigin(mb.width / 2.0f, 0.0f);
        m.setPosition(W / 2.0f, 68.0f);
        window.draw(m);
    }

    // 说明：只影响这一局
    {
        sf::Text t(L"只影响【这一局】—— 不会改变「主菜单 - 设置」里的模式默认值",
                   m_font, 15);
        t.setFillColor(kHint);
        sf::FloatRect b = t.getLocalBounds();
        t.setOrigin(b.width / 2.0f, 0.0f);
        t.setPosition(W / 2.0f, 98.0f);
        window.draw(t);
    }

    // 行表
    {
        const float left  = 160.0f;
        const float right = W - 160.0f;
        float y = 140.0f;
        const float lineH = 34.0f;

        for (std::size_t i = 0; i < m_rows.size(); ++i) {
            const bool sel = (static_cast<int>(i) == m_row);
            const Row& r = m_rows[i];

            // 高亮条
            if (sel && !r.readOnly) {
                sf::RectangleShape bar(sf::Vector2f(right - left, lineH - 4.0f));
                bar.setPosition(left - 14.0f, y - 3.0f);
                bar.setFillColor(sf::Color(78, 205, 196, 40));
                bar.setOutlineColor(sf::Color(78, 205, 196, 120));
                bar.setOutlineThickness(1.0f);
                window.draw(bar);
            }

            sf::Text label(r.label, m_font, 19);
            label.setFillColor(r.readOnly ? sf::Color(130, 130, 140)
                                : (sel ? kSelect : kNormal));
            label.setPosition(left, y);
            window.draw(label);

            sf::Text val(r.get ? r.get() : std::wstring(), m_font, 19);
            val.setFillColor(r.readOnly ? sf::Color(160, 160, 170)
                                        : sf::Color(235, 235, 240));
            val.setPosition(left + 300.0f, y);
            window.draw(val);

            y += lineH;
        }
    }

    // 底部操作提示
    {
        sf::Text t(L"上下键 选择    左右键 调整    R 恢复该模式默认值    "
                   L"Enter 保存并开始    ESC 返回",
                   m_font, 15);
        t.setFillColor(kHint);
        sf::FloatRect b = t.getLocalBounds();
        t.setOrigin(b.width / 2.0f, 0.0f);
        t.setPosition(W / 2.0f, static_cast<float>(Constants::WINDOW_HEIGHT) - 40.0f);
        window.draw(t);
    }
}
