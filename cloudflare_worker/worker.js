/**
 * Free AI Image Generation API using Cloudflare Workers AI
 * Based on https://github.com/saurav-z/free-image-generation-api
 *
 * Supports:
 * - @cf/black-forest-labs/flux-1-schnell (fast, high quality, ~173 free images/day on 10,000 free daily neurons)
 * - @cf/stabilityai/stable-diffusion-xl-base-1.0 (classic SDXL)
 * - @cf/bytedance/stable-diffusion-xl-lightning (ultra-fast 4-step SDXL)
 *
 * Setup Instructions:
 * 1. Deploy this script to Cloudflare Workers.
 * 2. Add Workers AI Binding named `AI` (Settings -> Variables -> Service Bindings -> AI -> Workers AI).
 * 3. Add Environment Variable `API_KEY` (Settings -> Variables -> Environment Variables -> API_KEY = your-secret-key).
 * 4. Copy your Worker URL into `FREE_IMAGE_API_URL` and `FREE_IMAGE_API_KEY` in GitHub Secrets!
 */

export default {
  async fetch(request, env) {
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key",
    };

    if (request.method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method === "GET") {
      return new Response(
        JSON.stringify({
          status: "ok",
          service: "free-image-generation-api (Indian Kahani 16:9 Edition)",
          repo: "https://github.com/saurav-z/free-image-generation-api",
          models: [
            "@cf/black-forest-labs/flux-1-schnell",
            "@cf/bytedance/stable-diffusion-xl-lightning",
            "@cf/stabilityai/stable-diffusion-xl-base-1.0",
          ],
        }),
        { headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    if (request.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method Not Allowed. Use POST." }), {
        status: 405,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // Verify API_KEY if configured in Worker Environment Variables
    const expectedKey = (env.API_KEY || "").trim();
    if (expectedKey) {
      const authHeader = request.headers.get("Authorization") || "";
      const xApiKey = request.headers.get("x-api-key") || "";
      const token = authHeader.replace(/^Bearer\s+/i, "").trim() || xApiKey.trim();
      if (token !== expectedKey) {
        return new Response(JSON.stringify({ error: "Unauthorized: Invalid API_KEY" }), {
          status: 401,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
    }

    if (!env.AI) {
      return new Response(
        JSON.stringify({
          error: "Workers AI binding 'AI' is missing! Go to Worker Settings -> Variables -> Service Bindings -> Add binding named 'AI' linked to Workers AI.",
        }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    try {
      const body = await request.json();
      const prompt = (body.prompt || "").trim();
      if (!prompt) {
        return new Response(JSON.stringify({ error: "Missing 'prompt' in JSON body" }), {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }

      const model = body.model || "@cf/black-forest-labs/flux-1-schnell";
      const width = Number(body.width) || 1344;
      const height = Number(body.height) || 768;
      const steps = Number(body.num_steps || body.steps) || 4;

      let inputs = { prompt };
      if (model.includes("flux")) {
        inputs = { prompt, num_steps: Math.min(8, Math.max(1, steps)) };
      } else {
        inputs = { prompt, width, height, num_steps: Math.min(20, Math.max(4, steps)) };
      }

      const response = await env.AI.run(model, inputs);

      // Flux models return { image: "<base64>" } while SDXL models return a binary ReadableStream
      if (response && typeof response === "object" && "image" in response && typeof response.image === "string") {
        const binaryString = atob(response.image);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
          bytes[i] = binaryString.charCodeAt(i);
        }
        return new Response(bytes.buffer, {
          headers: { ...corsHeaders, "Content-Type": "image/jpeg" },
        });
      }

      return new Response(response, {
        headers: { ...corsHeaders, "Content-Type": "image/png" },
      });
    } catch (err) {
      return new Response(JSON.stringify({ error: String(err) }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }
  },
};
