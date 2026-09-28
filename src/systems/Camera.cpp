#include "Camera.h"
#include "../utils/Math.h"
#include <cmath>

// ==========================================
// 构造函数 — 初始化摄像机在地图中心
// ==========================================
Camera::Camera(float mapWidth, float mapHeight, float viewWidth, float viewHeight)
    : m_center(mapWidth / 2.0f, mapHeight / 2.0f)
    , m_target(mapWidth / 2.0f, mapHeight / 2.0f)
    , m_followSpeed(3.0f)
    , m_mapWidth(mapWidth)
    , m_mapHeight(mapHeight)
    , m_viewWidth(viewWidth)
    , m_viewHeight(viewHeight)
{
    m_view.setSize(viewWidth, viewHeight);
    m_view.setCenter(m_center);
}

// ==========================================
// 每帧更新 — 摄像机追赶目标
//
// 使用指数衰减公式：center = lerp(center, target, 1 - e^(-speed*dt))
//
// 为什么不用常速追赶？
//   - 距离大时追得快（画面快速跟上来）
//   - 距离小时追得慢（微调时画面不抖）
//   - 整体自然流畅
// ==========================================
void Camera::update(float dt) {
    // 指数衰减：越远追得越快，越近追得越慢，永远不会超过目标
    float t = 1.0f - std::exp(-m_followSpeed * dt);
    m_center = Math::lerp(m_center, m_target, t);

    // 限制摄像机不超出地图边界（不显示地图外的空白区域）
    //
    // ★ 两个轴必须【分别判断】，而且要先判"地图比视口还小"这种情况：
    //
    //   地图 20x20、视口 1280x720 时，halfW = 640 > mapWidth - halfW = -630，
    //   夹取区间变成 [640, -630] —— 下界比上界还大，clamp 的结果会被钉死在
    //   640 上，摄像机跑到地图外，画面完全错误。
    //
    //   地图装不下视口时，"不显示地图外区域"这个目标本身就做不到，
    //   此时正确的行为是【把地图居中显示】，让人至少能看见地图在哪。
    const float halfW = m_viewWidth / 2.0f;
    const float halfH = m_viewHeight / 2.0f;

    m_center.x = (m_mapWidth  <= m_viewWidth)
               ? m_mapWidth  * 0.5f                                  // 装不下 → 居中
               : Math::clamp(m_center.x, halfW, m_mapWidth  - halfW);

    m_center.y = (m_mapHeight <= m_viewHeight)
               ? m_mapHeight * 0.5f                                  // 装不下 → 居中
               : Math::clamp(m_center.y, halfH, m_mapHeight - halfH);

    m_view.setCenter(m_center);
}
