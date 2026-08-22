# Visitor Analytics Service

这个 Cloudflare Worker + Pages Functions 服务为个人站提供无 Cookie 的全站访问聚合。

- `POST /v1/collect`：使用每日轮换 HMAC 去重访客，原始 IP 不写入 D1。
- `GET /v1/summary`：只返回累计访问、今日访客和达到阈值的国家级聚合。
- Pages Functions 提供 `pages.dev` 公网入口；独立 Worker 的每日 Cron 删除两天以前的临时访客哈希。

部署所需的 `VISITOR_HASH_SECRET` 只通过 Wrangler secret 保存，不能提交到 Git。生产入口为 `https://cosmicrealm-visitor-stats.pages.dev`。
