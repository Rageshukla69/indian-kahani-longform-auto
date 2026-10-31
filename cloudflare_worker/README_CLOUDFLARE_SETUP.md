# Free AI Image Generation API Setup (`saurav-z/free-image-generation-api`)

This repository integrates **[saurav-z/free-image-generation-api](https://github.com/saurav-z/free-image-generation-api)** out-of-the-box so every scene in your Hindi story video gets a custom 16:9 AI illustration matched to the exact spoken narration at **$0.00 cost**.

## Option A: 2-Minute Cloudflare Worker Setup (Recommended)
1. Sign in to your free [Cloudflare Dashboard](https://dash.cloudflare.com/) and open **Compute (Workers & Pages)**.
2. Click **Create application** -> **Create Worker** -> name it `free-image-generation-api` -> click **Deploy**.
3. Click **Edit code**, paste the contents of `cloudflare_worker/worker.js` from this repo, and click **Deploy**.
4. Go to **Settings** -> **Bindings** -> **Add binding** -> choose **Workers AI** -> set **Variable name** to `AI` -> click **Deploy**.
5. Go to **Settings** -> **Variables and Secrets** -> **Add** -> set **Name** to `API_KEY` and **Value** to any secret password (e.g. `my-secret-kahani-key`).
6. Copy your Worker URL (e.g. `https://free-image-generation-api.yourname.workers.dev`) and add two GitHub Repository Secrets:
   - `FREE_IMAGE_API_URL`: `https://free-image-generation-api.yourname.workers.dev`
   - `FREE_IMAGE_API_KEY`: `my-secret-kahani-key`

## Option B: Direct Cloudflare Account ID + API Token (Zero Worker Deployment Needed!)
If you don't want to create a Worker manually, our `src/media_fetcher.py` also supports calling Cloudflare Workers AI's REST endpoint directly! Just set:
- `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare Account ID
- `CLOUDFLARE_API_TOKEN`: A Cloudflare API Token with `Workers AI: Read/Edit` permission

## Option C: Automatic Multi-Provider Fallback Chain
If neither Option A nor Option B is configured (or if your daily free neuron quota resets at 00:00 UTC), `src/media_fetcher.py` automatically falls back to:
1. Optional Microsoft Foundry Image Endpoint (`FOUNDRY_ENDPOINT` / `FOUNDRY_API_KEY`) if provided
2. Free Pollinations FLUX 16:9 image endpoint
3. Wikimedia Commons / Openverse Indian art & illustration archives
4. Locally rendered 1920x1080 Rich Indian Storybook Canvas Painting with Devanagari scene callouts
