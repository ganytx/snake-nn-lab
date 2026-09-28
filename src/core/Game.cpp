#include "Game.h"
#include "GameConfigFields.h"   // ★ 再展开一次字段表 → 生成下面读写的分支
#include "GameModes.h"          // ★ game::applyModeParams —— 补模式基线用
#include "../utils/Constants.h"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <windows.h>

// ===========================================================================
//  配置文件的位置
//
//  ★ 这里踩过一个很隐蔽的坑，值得写清楚：
//
//    run.bat 写的是 `start "" "%BUILD_DIR%\snake_game.exe"`。
//    **`start` 会把子进程的工作目录设成 exe 所在目录**，所以游戏实际是在
//    `build\` 下运行的。而 loadConfig() 读的是相对路径 `config/settings.cfg`
//    —— 于是读的是 `build\config\settings.cfg`，写也是写它。
//
//    后果：在设置界面保存的参数进了 `build\config\`，
//    而项目根那份 `config/settings.cfg`（文档里写明的权威配置）永远停在旧值；
//    反过来在根目录手改配置也【完全不生效】。两份副本各说各话。
//
//  ★ 修法：按【优先级】找配置目录，读写都用同一个结果：
//      1. 可执行文件旁边的 config\        ← 正常情况（run.bat 下就是 build\config\）
//      2. 当前工作目录的 config\          ← 从项目根直接跑 exe 时
//      3. 上一级目录的 config\            ← 万一从 build\ 的某个子目录跑
//    找到哪个就用哪个，所以不管从哪儿启动，行为和"改哪个文件"都是确定的。
//
//  ★ 约定：`build\config\settings.cfg` 是【运行时的那份】。
//    项目根的 `config/settings.cfg` 是模板，构建时拷进 build\。
//    想改默认值 → 改根目录那份，然后重新 build.bat。
// ===========================================================================
namespace {

std::string dirOf(const std::string& path) {
    const std::size_t pos = path.find_last_of("\\/");
    return (pos == std::string::npos) ? std::string() : path.substr(0, pos);
}

bool fileExists(const std::string& path) {
    const DWORD attr = GetFileAttributesA(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

// 返回带结尾反斜杠的配置目录，例如 "build\config\"
std::string findConfigDir() {
    char exePath[MAX_PATH] = {};
    const DWORD n = GetModuleFileNameA(nullptr, exePath, MAX_PATH);
    if (n > 0 && n < MAX_PATH) {
        const std::string exeDir = dirOf(std::string(exePath, n));
        if (!exeDir.empty()) {
            // 1) exe 旁边
            const std::string a = exeDir + "\\config\\";
            if (fileExists(a + "settings.cfg")) return a;

            // 3) 上一级（放在这里判，因为它是"更远"的兜底）
            const std::string parent = dirOf(exeDir);
            if (!parent.empty()) {
                const std::string c = parent + "\\config\\";
                if (fileExists(c + "settings.cfg")) return c;
            }
        }
    }

    // 2) 当前工作目录
    if (fileExists("config\\settings.cfg")) return "config\\";

    // 都没找到：默认用 exe 旁边那个（首次运行时会创建）
    char fallback[MAX_PATH] = {};
    const DWORD m = GetModuleFileNameA(nullptr, fallback, MAX_PATH);
    if (m > 0 && m < MAX_PATH) {
        const std::string d = dirOf(std::string(fallback, m));
        if (!d.empty()) return d + "\\config\\";
    }
    return "config\\";
}

std::string trim(const std::string& s) {
    std::size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) --e;
    return s.substr(b, e - b);
}

}  // namespace

// ===========================================================================
//  加载配置 — 从 <配置目录>settings.cfg 读取
//
//  用一张表驱动（见 GameConfigIO.h），所以：
//    · 加了新字段就不可能忘记读
//    · teamBrains 也一并读，不会再出现"选了算法一重启就丢"
// ===========================================================================
void Game::loadConfig() {
    const std::string path = findConfigDir() + "settings.cfg";

    // 每个模式是否在配置文件里出现过（见函数末尾的"补基线"）
    bool modeKeySeen[game::MODE_COUNT] = { false, false, false };

    std::ifstream file(path);
    if (!file.is_open()) {
        // 没有配置文件是正常的（第一次跑），用默认值即可。
        // 但要【打出来】，否则"我的配置怎么不生效"会查很久。
        std::printf("[config] 未找到 %s，使用内置默认值\n", path.c_str());
        // ★ 没有文件时 modes[] 全是成员初值 —— 必须补上模式基线，
        //   否则三个模式会跑同一套参数（这就是那个"三个模式一样"的 bug）。
        for (int mi = 0; mi < game::MODE_COUNT; ++mi) {
            game::applyModeParams(m_config, static_cast<game::GameMode>(mi));
        }
        m_config.sanitize();
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        const std::string key = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));
        if (key.empty()) continue;

        bool handled = false;

        // ---- 数值 / 布尔字段 ----
#define X_FLOAT(name, k, def) \
        if (key == #k) { m_config.name = static_cast<float>(std::atof(val.c_str())); handled = true; }
#define X_INT(name, k, def) \
        if (key == #k) { m_config.name = std::atoi(val.c_str()); handled = true; }
#define X_BOOL(name, k, def) \
        if (key == #k) { m_config.name = (std::atoi(val.c_str()) != 0); handled = true; }
        GAME_CONFIG_FIELDS(X_FLOAT, X_INT, X_BOOL)
#undef X_FLOAT
#undef X_INT
#undef X_BOOL

        // ---- 每模式配置：mode<M>_* （2026-09-27 新增）----
        //
        //  键名约定见 Game.h 的 ModeConfig 说明。这里只做"拆前缀 + 分发"。
        //  ⚠️ 键名格式必须和 saveConfig 写出的完全一致，否则就是"存了读不回来"。
        if (!handled) {
            const std::string mp = GameConfig::MODE_KEY_PREFIX;
            if (key.compare(0, mp.size(), mp) == 0) {
                // mode<M>_rest
                const std::size_t us = key.find('_', mp.size());
                if (us != std::string::npos) {
                    const int mi = std::atoi(key.substr(mp.size(), us - mp.size()).c_str());
                    if (mi >= 0 && mi < game::MODE_COUNT) {
                        GameConfig::ModeConfig& mc = m_config.modes[mi];
                        const std::string sub = key.substr(us + 1);

                        if (sub == "teams") {
                            mc.teamCount = std::atoi(val.c_str());
                            modeKeySeen[mi] = true;
                            handled = true;
                        } else if (sub == "per_team") {
                            mc.snakesPerTeam = std::atoi(val.c_str());
                            modeKeySeen[mi] = true;
                            handled = true;
                        } else if (sub == "food_per_snake") {
                            mc.foodPerSnake = static_cast<float>(std::atof(val.c_str()));
                            modeKeySeen[mi] = true;
                            handled = true;
                        } else if (sub == "initial_food") {
                            // ★ 开局撒多少个普通食物（2026-09-27 新增）
                            //  ⚠️ 键名必须和 saveConfig 写出的完全一致，
                            //     否则就是"存了读不回来"（本项目的经典坑）。
                            mc.initialFoodCount = std::atoi(val.c_str());
                            modeKeySeen[mi] = true;
                            handled = true;
                        } else if (sub == "food_cap") {
                            // ★ 普通食物存量上限（0 = 不限制）
                            mc.foodMaxCount = std::atoi(val.c_str());
                            modeKeySeen[mi] = true;
                            handled = true;
                        } else {
                            // brain_<T> / weight_<T>
                            const std::size_t us2 = sub.find('_');
                            if (us2 != std::string::npos) {
                                const std::string kind = sub.substr(0, us2);
                                const int ti = std::atoi(sub.substr(us2 + 1).c_str());
                                if (ti >= 0 && ti < GameConfig::MAX_TEAMS) {
                                    if (kind == "brain") {
                                        mc.teamBrains[ti] = val;
                                        modeKeySeen[mi] = true;
                                        handled = true;
                                    } else if (kind == "weight") {
                                        mc.teamWeights[ti] = val;
                                        handled = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    //  ★★ 给"配置文件里没出现过的模式"补上代码基线（2026-09-27 修）
    //
    //  ★ 这个 bug 是什么
    //    旧格式的 settings.cfg 里**没有 `mode<M>_*` 这些键**。
    //    于是 `m_config.modes[]` 从来没被写过，只有结构体成员初值 ——
    //    而模式定义（圈数/食物速率/队数）在 `GameModes.cpp` 里，
    //    不应用就等于**三个模式跑同一套参数**。
    //    负责人报的"三个游戏模式完全错误 / 看起来是同一个模式"就是它。
    //
    //  ★ 判据：逐模式判断"文件里有没有它的键"
    //    · 有（玩家在大设置里存过）→ 保留文件里的值，**不能覆盖**
    //    · 没有（首次运行 / 旧格式）→ 用代码基线
    //    这样两边都对：玩家的设置优先，缺省的也能跑。
    //
    //  ★ 为什么不干脆"总是应用基线"
    //    那会把玩家存过的队数/食物速率/权重全部冲掉 ——
    //    正是本项目最忌讳的"设置里改了没用"。
    // -----------------------------------------------------------------------
    for (int mi = 0; mi < game::MODE_COUNT; ++mi) {
        if (!modeKeySeen[mi]) {
            std::printf("[config] 配置里没有模式 %d 的设置，使用内置默认值\n", mi);
            game::applyModeParams(m_config, static_cast<game::GameMode>(mi));
        }
    }

    // ★ 手改配置文件可以绕过设置界面的 min/max，所以这里必须兜一道
    m_config.sanitize();
}

// ===========================================================================
//  保存配置 — 写回【同一个】目录，所以 load / save 永远指向同一份文件
// ===========================================================================
void Game::saveConfig() const {
    const std::string dir  = findConfigDir();
    const std::string path = dir + "settings.cfg";

    CreateDirectoryA(dir.c_str(), nullptr);
    // 如果配置目录是 exe 旁边的（例如 build\config\），父目录可能还不存在
    const std::string parent = dirOf(dir.substr(0, dir.size() - 1));
    if (!parent.empty()) CreateDirectoryA(parent.c_str(), nullptr);

    std::ofstream file(path);
    if (!file.is_open()) {
        // ★ 以前这里是静默 return —— 用户点了"保存"、界面提示"保存成功"，
        //   而磁盘上什么都没发生。现在必须喊出来。
        std::printf("[config] 【保存失败】无法写入 %s\n", path.c_str());
        return;
    }

    file << "# Snake Game Settings\n";
    file << "# 由游戏在设置界面保存时自动生成（这份是【运行时实际使用】的配置）\n";
    file << "# 想改默认值：改项目根的 config/settings.cfg，然后重新 build.bat\n";

#define X_FLOAT(name, k, def) file << #k << "=" << m_config.name << "\n";
#define X_INT(name, k, def)   file << #k << "=" << m_config.name << "\n";
#define X_BOOL(name, k, def)  file << #k << "=" << (m_config.name ? 1 : 0) << "\n";
    GAME_CONFIG_FIELDS(X_FLOAT, X_INT, X_BOOL)
#undef X_FLOAT
#undef X_INT
#undef X_BOOL

    // ---- 每模式配置（下标 = GameMode 的值）----
    //
    //  ★ 必须【每个模式每个队都写全】，包括空格 ——
    //    省掉空值会让"清空某队权重"这个操作无法被保存
    //    （读的时候键不存在 → 保留上一次的值 → 看起来"清不掉"）。
    file << "# --- 每模式默认值（0=随机食物圈 1=无限 2=大混战）---\n";
    for (int mi = 0; mi < game::MODE_COUNT; ++mi) {
        const GameConfig::ModeConfig& mc = m_config.modes[mi];
        file << GameConfig::MODE_KEY_PREFIX << mi << "_teams=" << mc.teamCount << "\n";
        file << GameConfig::MODE_KEY_PREFIX << mi << "_per_team=" << mc.snakesPerTeam << "\n";
        file << GameConfig::MODE_KEY_PREFIX << mi << "_food_per_snake="
             << mc.foodPerSnake << "\n";
        // ★ 食物存量（2026-09-27 新增）
        //
        //  ⚠️ 这两行【必须】写。它们能在大设置的模式页里调，
        //     而 saveConfig 原本只写 teams / per_team / food_per_snake / brain / weight ——
        //     不写它们的话，玩家调完"开局食物数"按 ESC 保存、
        //     重启游戏后就被 `applyModeParams` 的基线冲回去了。
        //     现象是"设置里改了没用"，而且**没有任何报错**。
        file << GameConfig::MODE_KEY_PREFIX << mi << "_initial_food="
             << mc.initialFoodCount << "\n";
        file << GameConfig::MODE_KEY_PREFIX << mi << "_food_cap="
             << mc.foodMaxCount << "\n";
        for (int t = 0; t < GameConfig::MAX_TEAMS; ++t) {
            file << GameConfig::MODE_KEY_PREFIX << mi << "_brain_" << t
                 << "=" << mc.teamBrains[t] << "\n";
            file << GameConfig::MODE_KEY_PREFIX << mi << "_weight_" << t
                 << "=" << mc.teamWeights[t] << "\n";
        }
    }

    if (!file.good()) {
        std::printf("[config] 【保存可能不完整】写入 %s 时出错\n", path.c_str());
        return;
    }
    std::printf("[config] 已保存到 %s\n", path.c_str());
}

Game::Game()
    : m_window(sf::VideoMode(Constants::WINDOW_WIDTH, Constants::WINDOW_HEIGHT),
               Constants::WINDOW_TITLE,
               sf::Style::Close | sf::Style::Titlebar)
{
    m_window.setVerticalSyncEnabled(true);
    loadConfig();  // 启动时加载配置文件（内部会补上缺失的模式基线）
}

// ==========================================
// 主循环
// ==========================================
void Game::run() {
    while (m_window.isOpen()) {
        processEvents();
        sf::Time dt = m_clock.restart();
        update(dt.asSeconds());
        render();
    }
}

// ==========================================
//  ★★★ 开始一局：世界规则 + 该模式参数 → 这一局的配置
//
//  ★★ 2026-09-27 修一个严重 bug：这里原来【只拷贝、不应用模式参数】
//
//    原实现：`m_matchCfg = m_config;` 就完事了。
//    后果：模式专属字段（地图/圈/食物速率）在 `m_config` 里是
//    `ModeConfig` 的**成员初值**，而模式定义表（`applyModeParams`）
//    **从来没被调用过** —— 于是三个模式跑的是同一套参数：
//        · 都是 3 队 × 3 条（模式 1/2 的队数根本没生效）
//        · 都是 `foodPerSnake = 1/9`（模式 1 的 2 个/秒、模式 3 的 1/15 没生效）
//        · 圈的设置来自配置文件里遗留的 `zone_count=4`（不是模式定义的 2 或 0）
//    现象就是负责人报的"**三个游戏模式完全错误 / 看起来是同一个模式**"。
//
//  ★ 修法：拷贝之后【显式应用该模式的参数】
//    `applyModeParams(cfg, mode)` 只碰"模式专属字段"
//    （地图 / 圈 / 食物速率 / 队数 / 每条），**不碰通用世界规则** ——
//    所以玩家在【主菜单 → 设置】里改过的速度/饥饿/碰撞不会被冲掉。
//    这正是 `applyModeParams` 当初被拆成两个函数的原因（见 GameModes.cpp）。
//
//  ⚠️ 顺序不能反：必须先拷贝（拿到玩家的世界规则），再应用（覆盖模式字段）。
// ==========================================
void Game::beginMatch(game::GameMode mode) {
    m_matchMode = mode;
    m_matchCfg  = m_config;                      // ① 通用世界规则 + ② 的容器

    // ② 该模式的专属参数（地图/圈/食物/队数）覆盖进这一局的配置
    game::applyModeParams(m_matchCfg, mode);

    m_matchCfg.sanitize();
}

// 当前这一局的模式配置（唯一入口）
//
//  ★ 为什么要一个访问函数而不是让调用方自己写 `modes[static_cast<int>(...)]`
//    那个下标表达式一旦写错（比如忘了 cast、或者用了全局配置而不是这一局的），
//    现象是"改了模式没生效"或者"读到别的模式的参数"，很难查。
//    收成一个函数之后，全项目只有一处做这个转换。
GameConfig::ModeConfig& Game::activeModeConfig() {
    // 越界保护：monitor 模式值损坏时退回模式 0，而不是越界访问
    int mi = static_cast<int>(m_matchMode);
    if (mi < 0 || mi >= game::MODE_COUNT) mi = 0;
    return m_matchCfg.modes[mi];
}

// ==========================================
// 事件分发
// ==========================================
void Game::processEvents() {
    sf::Event event;
    while (m_window.pollEvent(event)) {
        if (event.type == sf::Event::Closed) {
            m_window.close();
            return;
        }
        if (!m_sceneStack.empty()) {
            m_sceneStack.back()->handleEvent(event);
        }
    }
}

// ==========================================
// 双层更新
// ==========================================
void Game::update(float dt) {
    if (m_sceneStack.empty()) return;
    m_accumulator += dt;
    while (m_accumulator >= Constants::FIXED_DT) {
        m_sceneStack.back()->fixedUpdate(Constants::FIXED_DT);
        m_accumulator -= Constants::FIXED_DT;
    }
    m_sceneStack.back()->update(dt);
}

// ==========================================
// 渲染
// ==========================================
void Game::render() {
    m_window.clear(sf::Color(26, 26, 46));
    if (!m_sceneStack.empty()) {
        m_sceneStack.back()->render(m_window);
    }
    m_window.display();
}

// ==========================================
// 场景管理
// ==========================================
void Game::pushScene(std::unique_ptr<Scene> scene) {
    m_sceneStack.push_back(std::move(scene));
}
void Game::popScene() {
    if (!m_sceneStack.empty()) m_sceneStack.pop_back();
}
void Game::switchScene(std::unique_ptr<Scene> scene) {
    m_sceneStack.clear();
    m_sceneStack.push_back(std::move(scene));
}

