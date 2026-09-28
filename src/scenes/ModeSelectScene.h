#pragma once

// ============================================================================
//  ModeSelectScene.h — 模式选择页（主菜单 →「开始游戏」进这里）
//
//  ★ 它取代了原来的 MatchSetupScene
//
//  以前「开始游戏」直接进"对局设置"（队数/算法/权重/限时），那里只有一种玩法。
//  现在有三种模式，所以第一层先让玩家【选模式】，每种模式给：
//
//      上：模式名字 + 一句话标语
//      中：示意图（占较大篇幅，一眼看出这个模式长什么样）
//      下：小字文字介绍
//      底：【设置】进入该模式的"这一局配置"　【开始】直接开打
//
//  ★ 为什么示意图是"画出来的"而不是图片文件
//    三个模式的差别是【几何与规则】（圈 vs 全图撒 vs 没有食物），
//    画示意图就是把这些规则可视化 —— 用 SFML 基本图形几十行就够，
//    而换成 png 就要多三个资源文件、还要处理缺失时的兜底。
//
//  ★ 操作
//      左右键 切换模式        Enter 开始
//      S / 上键 进设置        ESC 回主菜单
//
//  ★ 三层配置里本页面负责什么
//    它自己【不】改任何配置。按【开始】时调 `Game::beginMatch(模式)`，
//    由那一步把 ①通用规则 + ②该模式默认值 拷成 ③这一局的配置。
//    「设置」进的那个页面才改 ③（见 ModeSetupScene）。
// ============================================================================

#include "../core/Scene.h"
#include "../core/GameModes.h"

class ModeSelectScene : public Scene {
public:
    explicit ModeSelectScene(Game* game);

    void handleEvent(const sf::Event& event) override;
    void fixedUpdate(float dt) override;
    void update(float dt) override;
    void render(sf::RenderWindow& window) override;

private:
    // 画三种模式的示意图（在给定矩形里画）
    void drawZoneDiagram(sf::RenderWindow& w, const sf::FloatRect& r);
    void drawInfiniteDiagram(sf::RenderWindow& w, const sf::FloatRect& r);
    void drawBattleDiagram(sf::RenderWindow& w, const sf::FloatRect& r);

    // -----------------------------------------------------------------------
    //  ★ 示意图只画【规则本身】，不画蛇、不画队色带
    //
    //  ★ 负责人对这张图的两次要求，都指向同一个方向：**再简单一些**
    //      ① "把蛇删了吧，太诡异了"
    //         —— 用几个圆模拟生物容易出戏
    //      ② "色带删掉，黄点删掉，简单一些"
    //         —— 队色带和黄点都是"额外信息"，它们让图变花，却没让规则更清楚
    //
    //  ★ 现在每张图只留【一样东西】：
    //      模式 1：两个圈（一亮一暗，亮的那轮换）
    //      模式 2：满图食物点 + 一排"蛇数在涨"的圆点
    //      模式 3：极少食物点 + 一排"蛇很多"的圆点 + 尸体点
    //    队数不再单独画 —— 它写在下方文字介绍里，那才是它该在的地方。
    // -----------------------------------------------------------------------

    // 画一个"模式卡片"的通用外框（标题/标语/介绍/按钮）
    void drawFrame(sf::RenderWindow& w, const sf::FloatRect& r);

    sf::Font m_font;
    int      m_index = 0;    // 当前选中的模式（= static_cast<int>(GameMode)）

    // ★ 示意图是"活着"的：用一个计时器让圈/蛇缓慢动起来
    //   —— 静态图看不出"圈会轮换""蛇会不断加入"这两件事。
    sf::Clock m_anim;

    game::GameMode mode() const {
        int i = m_index;
        if (i < 0) i = 0;
        if (i >= game::MODE_COUNT) i = game::MODE_COUNT - 1;
        return static_cast<game::GameMode>(i);
    }
};
