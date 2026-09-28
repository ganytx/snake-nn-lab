#include "TrainingTeamScene.h"
#include "../utils/GameFont.h"   // ★ 字体加载的唯一实现（候选路径见该文件）
#include "TrainingSetupScene.h"
#include "MenuScene.h"
#include "../core/Game.h"
#include "../utils/Constants.h"
#include "../utils/Text.h"

#include <algorithm>
#include <sstream>

namespace {

const sf::Color kSelect = sf::Color(78, 205, 196);
const sf::Color kNormal = sf::Color(185, 185, 185);
const sf::Color kTitle  = sf::Color(255, 210, 90);
const sf::Color kDim    = sf::Color(130, 130, 130);
const sf::Color kNn     = sf::Color(150, 220, 150);   // 神经网络用绿色，显眼

// 可选档位
const int kTeamCounts[] = {1, 2, 3, 4};
const int kPerTeam[]    = {1, 2, 3, 4};

template <typename T, int N>
int cycleIndex(const T (&opts)[N], T cur, int delta) {
    int idx = 0;
    for (int i = 0; i < N; ++i) if (opts[i] == cur) { idx = i; break; }
    idx = ((idx + delta) % N + N) % N;
    return idx;
}

}  // namespace

// ===========================================================================
TrainingTeamScene::TrainingTeamScene(Game* game)
    : Scene(game)
{
    // ★ 字体加载的唯一实现（候选路径与兜底策略见 src/utils/GameFont.h）
    if (!textutil::loadGameFont(m_font)) {
        // 加载失败不会崩，但中文会变方块 —— 留个记号方便排查
        std::printf("[%s] 字体加载失败，中文会显示为方块\n", "TrainingTeamScene");
    }

    // 初始化每队算法：默认全部规则 AI。
    // 保留 TaskConfig 里已经选过的值（可能是从任务预设带过来的）。
    {
        const rl::TaskConfig& cfg = TrainingSetupScene::pendingConfigRead();
        m_teamBrain.resize(static_cast<std::size_t>(std::max(1, cfg.teamCount)));
        for (int i = 0; i < static_cast<int>(m_teamBrain.size()); ++i) {
            const std::string& a = cfg.algorithmOf(i);
            m_teamBrain[static_cast<std::size_t>(i)] = (a == "nn") ? "nn" : "rule";
        }
    }

    buildRows();
}

bool TrainingTeamScene::teamUsesNN(int team) const {
    return team >= 0 && team < static_cast<int>(m_teamBrain.size())
        && m_teamBrain[static_cast<std::size_t>(team)] == "nn";
}

void TrainingTeamScene::setTeamBrain(int team, const std::string& id) {
    if (team < 0 || team >= static_cast<int>(m_teamBrain.size())) return;

    // ★ "学习中的那个"全局唯一：某队用了 nn，其它队的 nn 自动降回规则 AI
    //   （"nn-enemy" 不受影响 —— 敌方快照不是学习对象）
    if (id == "nn") {
        for (auto& a : m_teamBrain) {
            if (a == "nn") a = "rule";
        }
    }
    m_teamBrain[static_cast<std::size_t>(team)] = id;

    rl::TaskConfig& cfg = TrainingSetupScene::pendingConfig();

    // ★ 让"学习中的那条"落在用神经网络的那一队：
    //   约定队 0 的第 0 条是学习者，所以把 nn 队和队 0 交换。
    if (id == "nn" && team != 0) {
        for (auto& tv : cfg.teams) {
            if (tv == 0) tv = team;
            else if (tv == team) tv = 0;
        }
        std::swap(m_teamBrain[0], m_teamBrain[static_cast<std::size_t>(team)]);
    }

    cfg.teamAlgorithms = m_teamBrain;
}

void TrainingTeamScene::cycleTeamBrain(int team, int delta) {
    if (team < 0 || team >= static_cast<int>(m_teamBrain.size())) return;

    // -----------------------------------------------------------------------
    //  ★ 三档循环：固定规则策略 → 神经网络（学习） → 神经网络（敌方快照）
    //
    //  ★ 负责人要求："我想把敌人也改成神经网络"
    //    所以第三档就是它：敌方用【学习网络的冻结快照】决策，只决策不学习，
    //    训练器每 100 局把当前权重同步给它 —— 于是对手会跟着一起变强。
    //
    //  ★ 为什么"神经网络（学习）"全局只能一个
    //    当前训练器只有一条轨迹、一次更新（REINFORCE）。多个队同时学
    //    需要两套网络 + 两条轨迹 + 两套奖励 —— 那是重做训练器。
    //    所以选了学习的那个队会自动把别的队降回规则 AI，并交换到队 0
    //    （因为 EpisodeRunner 固定学 index 0）。
    // -----------------------------------------------------------------------
    const std::string cur = m_teamBrain[static_cast<std::size_t>(team)];
    std::string next;
    if (delta > 0) {
        next = (cur == "rule")     ? "nn"
             : (cur == "nn")       ? "nn-enemy"
                                   : "rule";
    } else {
        next = (cur == "rule")     ? "nn-enemy"
             : (cur == "nn-enemy") ? "nn"
                                   : "rule";
    }
    setTeamBrain(team, next);
}

// ===========================================================================
void TrainingTeamScene::buildRows() {
    m_rows.clear();
    rl::TaskConfig& cfg = TrainingSetupScene::pendingConfig();

    // ---- 第 0 行：几队 ----
    m_rows.push_back({
        L"分成几队",
        [this] {
            const rl::TaskConfig& c = TrainingSetupScene::pendingConfigRead();
            return std::to_wstring(c.teamCount) + L" 队";
        },
        [this](int d) {
            rl::TaskConfig& c = TrainingSetupScene::pendingConfig();
            c.teamCount = kTeamCounts[cycleIndex(kTeamCounts, c.teamCount, d)];
            c.rebuildTeamTables();
            // ★ 队数变了 → 每队算法表要跟着调长度，行表也要重建
            m_teamBrain.resize(static_cast<std::size_t>(c.teamCount), "rule");
            c.teamAlgorithms = m_teamBrain;
            buildRows();
            if (m_row >= static_cast<int>(m_rows.size())) {
                m_row = static_cast<int>(m_rows.size()) - 1;
            }
        },
        false
    });

    // ---- 第 1 行：每队几条 ----
    m_rows.push_back({
        L"每队几条蛇",
        [this] {
            const rl::TaskConfig& c = TrainingSetupScene::pendingConfigRead();
            return std::to_wstring(c.snakesPerTeam) + L" 条"
                 + L"   （共 " + std::to_wstring(c.snakeCount()) + L" 条）";
        },
        [this](int d) {
            rl::TaskConfig& c = TrainingSetupScene::pendingConfig();
            c.snakesPerTeam = kPerTeam[cycleIndex(kPerTeam, c.snakesPerTeam, d)];
            c.rebuildTeamTables();
        },
        false
    });

    // ---- 之后每一行：一队的算法 ----
    //
    // ★ 三档：固定规则策略 / 神经网络（学习）/ 神经网络（敌方快照）
    //   第三档就是负责人要的"敌人也用神经网络"。
    for (int i = 0; i < cfg.teamCount; ++i) {
        const int team = i;
        m_rows.push_back({
            std::to_wstring(team + 1) + L" 队算法",
            [this, team]() -> std::wstring {
                const std::string a = m_teamBrain[static_cast<std::size_t>(team)];
                if (a == "nn")       return L"神经网络（这一队在学）";
                if (a == "nn-enemy") return L"神经网络（敌方快照，只决策不学）";
                // 规则 AI：把"这队是队友还是对手"写清楚 —— 它决定蛇会遇到什么
                return std::wstring(L"固定规则策略")
                     + (team == 0 ? L"   （你的队友，可穿过）"
                                  : L"   （对手，会撞死你）");
            },
            [this, team](int d) { cycleTeamBrain(team, d); },
            false
        });
    }
}

// ===========================================================================
void TrainingTeamScene::handleEvent(const sf::Event& event) {
    if (event.type != sf::Event::KeyPressed) return;

    const int n = static_cast<int>(m_rows.size());
    if (n == 0) return;

    switch (event.key.code) {
        case sf::Keyboard::Up:
            m_row = (m_row - 1 + n) % n;
            break;
        case sf::Keyboard::Down:
            m_row = (m_row + 1) % n;
            break;

        case sf::Keyboard::Left:
        case sf::Keyboard::Right: {
            if (m_row >= 0 && m_row < n) {
                Row& r = m_rows[static_cast<std::size_t>(m_row)];
                if (!r.readOnly && r.adjust) {
                    r.adjust((event.key.code == sf::Keyboard::Right) ? 1 : -1);
                }
            }
            break;
        }

        case sf::Keyboard::Enter:
        case sf::Keyboard::Space: {
            // ★ 回训练设置界面，并让它重建行表 ——
            //   否则那边"训练蛇数"那一行还显示旧值，用户会以为改的没生效
            auto* back = new TrainingSetupScene(m_game);
            back->refreshRows();
            m_game->switchScene(std::unique_ptr<Scene>(back));
            break;
        }

        case sf::Keyboard::Escape:
            m_game->switchScene(std::make_unique<MenuScene>(m_game));
            break;

        default: break;
    }
}

void TrainingTeamScene::fixedUpdate(float) {}
void TrainingTeamScene::update(float) {}

// ===========================================================================
void TrainingTeamScene::render(sf::RenderWindow& window) {
    sf::Text title(L"队伍设置", m_font, 40);
    title.setFillColor(kTitle);
    title.setPosition(60.0f, 30.0f);
    window.draw(title);

    sf::Text sub(L"训练模式 · 组队", m_font, 18);
    sub.setFillColor(kDim);
    sub.setPosition(300.0f, 46.0f);
    window.draw(sub);

    sf::Text hint(L"上下键 选择    左右键 调整    Enter 返回训练设置    ESC 返回主菜单",
                  m_font, 16);
    hint.setFillColor(kNormal);
    hint.setPosition(60.0f, 90.0f);
    window.draw(hint);

    // -----------------------------------------------------------------------
    //  行区：动态行距，保证不撞到底部说明（和训练设置界面同一个做法）
    // -----------------------------------------------------------------------
    const float kFooterTop = static_cast<float>(Constants::WINDOW_HEIGHT) - 110.0f;
    const float kRowsTop   = 140.0f;
    const int   nRows      = static_cast<int>(m_rows.size());
    const float rowGap = (nRows > 0)
        ? std::min(44.0f, (kFooterTop - kRowsTop - 8.0f) / static_cast<float>(nRows))
        : 44.0f;

    float y = kRowsTop;
    for (std::size_t i = 0; i < m_rows.size(); ++i) {
        const bool sel = (static_cast<int>(i) == m_row);
        const Row& row = m_rows[i];

        if (sel) {
            sf::RectangleShape bar(sf::Vector2f(1000.0f, 34.0f));
            bar.setPosition(50.0f, y - 4.0f);
            bar.setFillColor(sf::Color(78, 205, 196, 45));
            bar.setOutlineColor(sf::Color(78, 205, 196, 120));
            bar.setOutlineThickness(1.0f);
            window.draw(bar);

            sf::Text arrowL(L"<", m_font, 22);
            arrowL.setFillColor(kSelect);
            arrowL.setPosition(150.0f, y);
            window.draw(arrowL);
            sf::Text arrowR(L">", m_font, 22);
            arrowR.setFillColor(kSelect);
            arrowR.setPosition(1030.0f, y);
            window.draw(arrowR);
        }

        sf::Text label(row.label, m_font, 21);
        label.setFillColor(sel ? sf::Color::White : kNormal);
        label.setPosition(60.0f, y);
        window.draw(label);

        // 名称里有 L"神经网络" 这种宽字符串，直接构造最稳
        sf::Text value(row.get(), m_font, 21);
        value.setFillColor(sel ? kSelect : kNormal);
        if (i >= 2 && value.getString().find(L"神经网络") != sf::String::InvalidPos) {
            value.setFillColor(kNn);      // 神经网络用绿字，一眼看出哪队在学
        }
        value.setPosition(200.0f, y);
        window.draw(value);

        y += rowGap;
    }

    // ---- 底部说明（2 行，起点统一）----
    {
        float fy = kFooterTop + 10.0f;
        sf::Text f1(L"· 只有 1 条蛇在学习（算法选「神经网络」的那一队），其余都是固定行为的陪练",
                    m_font, 15);
        f1.setFillColor(kDim);
        f1.setPosition(60.0f, fy);
        window.draw(f1);
        fy += 21.0f;

        sf::Text f2(L"· 0 队是「你的队」：它的队友可以互相穿过；其它队是对手，会撞死你",
                    m_font, 15);
        f2.setFillColor(kDim);
        f2.setPosition(60.0f, fy);
        window.draw(f2);
    }
}
